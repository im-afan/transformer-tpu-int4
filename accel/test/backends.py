#!/usr/bin/env python3
"""backends.py — the three things a firmware kernel can be run on.

    ISSBackend   the native build of the .c, driving iss.py
    RTLBackend   the RISC-V build, through the whole core in Icarus
    TPUBackend   the same image, on the board over the UART

All three take a C source and a DRAM image and give back DRAM. They differ in
what else they can tell you: the ISS and the RTL see every byte of memory (so a
stray write is caught), the RTL and the board have cycle counters, and the ISS
and the RTL know the command stream. See accel/test/README.md.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
FW_DIR = os.path.join(REPO, "accel", "tpu", "fw")
TB_DIR = os.path.join(REPO, "accel", "tpu", "tb")
BUILD_ROOT = os.path.join(HERE, "build")

sys.path.insert(0, HERE)

from iss import TPU                                          # noqa: E402
from vector_generator import read_hex, write_hex             # noqa: E402

ROWS = COLS = 8
DRAM_BYTES = 1 << 19


@dataclass
class RunResult:
    """What one run of a kernel produced."""

    dram: dict = field(default_factory=dict)   # {addr: byte}, the read ranges
    cmds: list | None = None                   # [(unit, w0, w1, w2, w3), ...]
    counters: dict | None = None               # core clocks, per counter
    written: set | None = None                 # every DRAM byte the run wrote


class Backend:
    """Build a kernel, load DRAM once, then run it repeatedly."""

    name = "backend"
    sees_whole_dram = False

    def build(self, source: str, defines: dict, include_dirs=()) -> None:
        raise NotImplementedError

    def load(self, image: dict) -> None:
        """Seed DRAM. Called once, before any run."""
        raise NotImplementedError

    def run(self, patch: dict, read_ranges: list) -> RunResult:
        """Write `patch`, run the kernel, read back `read_ranges`."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def _flags(defines: dict) -> list:
    out = []
    for key, val in sorted(defines.items()):
        out.append(f"-D{key}" if val is None else f"-D{key}={val}")
    return out


def _slice(image, ranges: list) -> dict:
    out = {}
    for base, length in ranges:
        for a in range(base, base + length):
            out[a] = image[a]
    return out


# =============================================================================
# The ISS.
# =============================================================================
class ISSBackend(Backend):
    """The kernel compiled for the host, pushing its commands into iss.py.

    The binary is a co-process, not a captured trace: a kernel that reads its
    own results back through the scratchpad window puts the value it read into
    the address of a later command, so its command stream exists only while
    something is answering those reads. The ISS is that something.
    """

    name = "iss"
    sees_whole_dram = True

    def __init__(self, workdir: str | None = None, quiet: bool = True):
        self.workdir = workdir or os.path.join(BUILD_ROOT, "iss")
        self.quiet = quiet
        self.tpu = TPU(rows=ROWS, cols=COLS)
        self.exe: str | None = None

    def build(self, source: str, defines: dict, include_dirs=()) -> None:
        os.makedirs(self.workdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(source))[0]
        self.exe = os.path.join(self.workdir, f"{stem}.native")
        cc = shlex.split(os.environ.get("HOSTCC", "cc"))
        cmd = [*cc, "-DTPU_TRACE", "-O1", "-Wall",
               "-I", FW_DIR, "-I", os.path.dirname(os.path.abspath(source)),
               *[f"-I{d}" for d in include_dirs], *_flags(defines),
               "-o", self.exe, source, os.path.join(FW_DIR, "mock", "tpu_trace.c")]
        subprocess.run(cmd, check=True)

    def load(self, image: dict) -> None:
        for addr, byte in image.items():
            self.tpu.dram[addr] = byte
        self.tpu.dram_written.clear()

    def run(self, patch: dict, read_ranges: list) -> RunResult:
        for addr, byte in patch.items():
            self.tpu.dram[addr] = byte
        self.tpu.dram_written.clear()
        cmds = self._coexecute()
        return RunResult(dram=_slice(self.tpu.dram, read_ranges), cmds=cmds,
                         written=set(self.tpu.dram_written))

    def dram_image(self) -> dict:
        return {a: self.tpu.dram[a] for a in range(DRAM_BYTES)}

    def _coexecute(self) -> list:
        """Run the binary, executing each command as it arrives and answering
        its scratchpad reads (`SRD`) and writes (`SWR`) out of the model."""
        proc = subprocess.Popen([self.exe], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, text=True, bufsize=1)
        cmds = []
        try:
            for line in proc.stdout:
                f = line.split()
                if not f:
                    continue
                if f[0] == "CMD" and len(f) == 6:
                    unit, w = int(f[1]), [int(x, 16) for x in f[2:6]]
                    self.tpu.exec_command(unit, *w)
                    cmds.append((unit, *w))
                elif f[0] == "SRD" and len(f) == 2:
                    val = self.tpu.rd_u32(int(f[1], 16))
                    proc.stdin.write(f"{val:08x}\n")
                    proc.stdin.flush()
                elif f[0] == "SWR" and len(f) == 3:
                    # track=False: the CPU wrote this, not a unit.
                    self.tpu.wr_i32(int(f[1], 16), int(f[2], 16), track=False)
                elif f[0] != "WAIT":
                    raise SystemExit(f"iss: cannot parse {line!r}")
        finally:
            if proc.stdin:
                proc.stdin.close()
            rc = proc.wait()
        if rc != 0:
            raise SystemExit(f"iss: {self.exe} exited {rc}")
        return cmds


# =============================================================================
# The RISC-V image.
# =============================================================================
def build_firmware(source: str, defines: dict, include_dirs=(),
                   quiet: bool = True) -> str:
    """Cross-compile a kernel; return the path to its $readmemh image.

    One `make` per kernel into its own build directory, so two kernels built at
    different shapes cannot share a stale object.
    """
    stem = os.path.splitext(os.path.basename(source))[0]
    build = os.path.join(BUILD_ROOT, "fw", stem)
    os.makedirs(build, exist_ok=True)
    extra = " ".join([*[f"-I{d}" for d in include_dirs], *_flags(defines)])
    cmd = ["make", "-C", FW_DIR, "--no-print-directory",
           f"PROG={stem}", f"SRC={os.path.abspath(source)}",
           f"BUILD={os.path.abspath(build)}", f"EXTRA_CFLAGS={extra}"]
    try:
        subprocess.run(cmd, check=True,
                       **({"capture_output": True, "text": True} if quiet else {}))
    except FileNotFoundError:
        raise SystemExit("`make` is not on PATH")
    except subprocess.CalledProcessError as exc:
        if quiet and exc.stderr:
            print(exc.stdout or "", exc.stderr, file=sys.stderr)
        raise SystemExit(f"building {stem}.hex failed — a bare-metal RISC-V gcc "
                         f"is needed (brew install riscv64-elf-gcc, apt install "
                         f"gcc-riscv64-unknown-elf, or the xPack riscv-none-elf-gcc)")
    return os.path.join(build, f"{stem}.hex")


# =============================================================================
# The RTL.
# =============================================================================
_COUNTER_RE = re.compile(r"(\w+)=(\d+)")


class RTLBackend(Backend):
    """The RISC-V image through the whole core, in Icarus.

    The testbench checks two things this class does not have to: every byte of
    DRAM against the golden, and the command stream the CPU actually issued
    against the one the native build issued. Both goldens come from an
    ISSBackend running the same kernel here, which is what makes this an
    RTL-vs-ISS check rather than a second opinion from the same model.

    `uart=True` swaps the backdoor testbench for the one that loads the image
    and the operands over the simulated serial link, at about 6x the run time.
    """

    name = "rtl"
    sees_whole_dram = True

    def __init__(self, uart: bool = False, watchdog_ns: int = 2_000_000,
                 uart_cpb: int = 16, rerun: bool = False, quiet: bool = True):
        self.uart, self.watchdog_ns, self.uart_cpb = uart, watchdog_ns, uart_cpb
        self.name = "rtl-uart" if uart else "rtl"
        self.rerun, self.quiet = rerun, quiet
        self.iss = ISSBackend()
        self.workdir = os.path.join(BUILD_ROOT, "rtl")
        self.image: dict = {}
        self.vvp = self.hex = None
        self.stem = ""

    def build(self, source: str, defines: dict, include_dirs=()) -> None:
        os.makedirs(self.workdir, exist_ok=True)
        self.stem = os.path.splitext(os.path.basename(source))[0]
        self.iss.build(source, defines, include_dirs)
        self.hex = build_firmware(source, defines, include_dirs, self.quiet)

        if shutil.which("iverilog") is None:
            raise SystemExit("iverilog is not on PATH")
        tb = "fw_uart_tb" if self.uart else "fw_matmul_tb"
        self.vvp = os.path.join(self.workdir, f"{self.stem}_{tb}.vvp")
        # iverilog runs from tb/, which is what core.f's relative paths assume.
        cmd = ["iverilog", "-g2012", "-Wall", "-f", "core.f",
               f'-DFW_HEX="{self.hex}"', f'-DFW_NAME="{self.stem}"',
               f'-DFW_VEC_DIR="{self.workdir}"',
               f"-DWATCHDOG_NS={self.watchdog_ns}",
               *([f"-DUART_CPB={self.uart_cpb}"] if self.uart else []),
               "-o", self.vvp, f"{tb}.sv"]
        # iverilog is loud about the vendored picorv32's timescales and its
        # unsupported-but-harmless constructs; none of it is about this run.
        iv = subprocess.run(cmd, cwd=TB_DIR, capture_output=self.quiet, text=True)
        if iv.returncode:
            print(iv.stdout or "", iv.stderr or "", file=sys.stderr)
            raise SystemExit(f"iverilog failed on {tb}")

    def load(self, image: dict) -> None:
        self.image = dict(image)
        self.iss.load(image)

    def run(self, patch: dict, read_ranges: list) -> RunResult:
        self.image.update(patch)

        # The golden, from the same kernel on the ISS.
        iss_result = self.iss.run(patch, read_ranges)
        write_hex(os.path.join(self.workdir, "fw_dram_in.hex"), self.image,
                  f"{self.stem} operands")
        write_hex(os.path.join(self.workdir, "fw_dram_exp.hex"),
                  {a: self.iss.tpu.dram[a] for a in sorted(iss_result.written)},
                  f"{self.stem} golden DRAM (ISS)")
        self._write_cmds(iss_result.cmds)

        dump = os.path.join(self.workdir, f"{self.stem}_dram_out.hex")
        argv = ["vvp", self.vvp, f"+DRAMOUT={dump}"]
        if self.rerun:
            argv.append("+RERUN")
        proc = subprocess.run(argv, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        if not self.quiet:
            print(log)
        if "ALL TESTS PASSED" not in log:
            tail = "\n".join(log.splitlines()[-25:])
            raise SystemExit(f"rtl: {self.stem} failed in simulation:\n{tail}")

        out = read_hex(dump)
        self.image = out
        return RunResult(dram=_slice(out, read_ranges), cmds=iss_result.cmds,
                         counters=self._counters(log), written=iss_result.written)

    def _write_cmds(self, cmds: list) -> None:
        with open(os.path.join(self.workdir, "fw_cmds.hex"), "w") as f:
            for unit, w0, w1, w2, w3 in cmds:
                f.write(f"{unit:08x} {w0:08x} {w1:08x} {w2:08x} {w3:08x}\n")
            f.write("ffffffff 00000000 00000000 00000000 00000000\n")

    @staticmethod
    def _counters(log: str) -> dict | None:
        """The two lines the testbench prints out of the perf counter block."""
        found: dict = {}
        for line in log.splitlines():
            if "counters:" in line or "idlec=" in line:
                found.update({k: int(v) for k, v in _COUNTER_RE.findall(line)})
        return found or None


# =============================================================================
# The board.
# =============================================================================
class TPUBackend(Backend):
    """The same image on the FPGA, over the two serial pins.

    `load` costs the whole static image on the wire once (~35 s for the model's
    weights at 115200); a run after that is only its patch, so a sweep is
    dominated by the kernel, not the link.
    """

    name = "board"
    sees_whole_dram = False

    def __init__(self, port: str | None = None, baud: int = 115200,
                 run_timeout: float = 120.0, clk_mhz: float = 12.0,
                 quiet: bool = True):
        from tpu_uart import FW_BASE, TPUUart, autodetect_port

        self.FW_BASE = FW_BASE
        self.run_timeout, self.clk_mhz, self.quiet = run_timeout, clk_mhz, quiet
        self.uart = TPUUart(port or autodetect_port(), baud)
        self.words: list = []

    def build(self, source: str, defines: dict, include_dirs=()) -> None:
        from tpu_uart import parse_hex_program

        path = build_firmware(source, defines, include_dirs, self.quiet)
        with open(path) as f:
            self.words = parse_hex_program(f.read())

    def load(self, image: dict) -> None:
        from tpu_uart import contiguous_runs, wait_until_idle

        wait_until_idle(self.uart, self.run_timeout)
        self.uart.load_program(self.FW_BASE, self.words)
        for addr, blob in contiguous_runs(image):
            self.uart.write_mem(addr, blob)

    def run(self, patch: dict, read_ranges: list) -> RunResult:
        from tpu_uart import contiguous_runs, wait_until_idle

        for addr, blob in contiguous_runs(patch):
            self.uart.write_mem(addr, blob)
        self.uart.go(self.FW_BASE)
        wait_until_idle(self.uart, self.run_timeout)

        dram: dict = {}
        for base, length in read_ranges:
            for i, byte in enumerate(self.uart.read_mem(base, length)):
                dram[base + i] = byte
        return RunResult(dram=dram, counters=self.uart.read_counters())

    def close(self) -> None:
        self.uart.close()


BACKENDS = {"iss": ISSBackend, "rtl": RTLBackend, "board": TPUBackend}
