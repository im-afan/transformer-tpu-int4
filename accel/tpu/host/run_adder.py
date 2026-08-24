#!/usr/bin/env python3
"""run_adder.py — score the real adder checkpoint on the FPGA, problem by problem.

``run_fw_matmul.py`` answers "does the board compute what the ISS computes", on
a *synthetic* image. That is the right question for a kernel and the wrong one
for a model: it says nothing about whether the thing on the board still does
addition. This script asks the other question. It stages the **real checkpoint**
— the same DRAM image ``accel/tpulang/adder_export.py`` hands the ISS, produced
by the same :func:`adder_export.static_image` — runs one problem per ``G``, reads
the int32 logits back, argmaxes on the host, and scores exact sequence and
per-token accuracy over N problems.

The producer is ``accel/tpu/fw/adder.c`` on the PicoRV32: there is no assembler
and no ``.tpu`` program any more, so a run needs a firmware image the way it used
to need an assembled one.

**The requant table is compiled into that image.** The 16 ``{m0,n}`` words per
layer are literals in the macro-ops, so the device has no path by which it could
read them out of memory — a checkpoint's scales reach the board only through the
build. This script therefore derives them, writes ``fw/adder_rq_ckpt.h`` and
rebuilds ``fw/adder.hex`` against it before loading anything (``--fw`` takes a
prebuilt image instead, and then it is on you that the two agree: an image
carrying the checked-in synthetic ``adder_rq.h`` runs perfectly and scores
noise).

Everything that is not the transport is shared with the simulated path, so a
disagreement between ``--dry-run`` and a board run localizes to the hardware
rather than to a second copy of the staging logic. That mattered: the pipeline
has drifted from the model twice before (``adder_export.py``'s docstring).

**What moves over the wire, and when.** The weights are ~98 KB and identical for
every problem, so they are written once, before the loop, along with the 2 KB
firmware image. Per problem only the [T][D] input (2048 B) goes down and the
[T][VPAD] int32 logits (2048 B) come back — about 4 KB, or ~0.4 s at 115200 baud
including the header tax. The weights are *not* rewritten between problems, which
also means a run is a standing test that SRAM holds them across ``G`` boundaries,
and that re-running the firmware from reset is safe (``adder.c`` has no mutable
globals and ``start.S`` re-zeroes .bss).

    python accel/tpu/host/run_adder.py --dry-run -n 8      # ISS, no board
    python accel/tpu/host/run_adder.py -p COM5 -n 64
    python accel/tpu/host/run_adder.py -p COM5 -n 8 --compare-iss
    python accel/tpu/host/run_adder.py -p COM5 -n 64 --no-show   # progress only

By default every problem prints as it finishes — the expression, the expected
answer and what the device answered, in human digit order (the dataset stores
numbers reversed). ``--no-show`` falls back to the periodic progress counter,
which is what you want when N is large.

The device's performance counters are read after every problem (``--no-timing``
to skip the extra round trip), so each row carries that problem's *core* run
time and the summary at the end reports mean/min/max plus unit occupancy. The
counters reset at each ``G``, so one read per problem measures one forward pass
and nothing accumulates.

``--compare-iss`` runs each problem in *both* the ISS and the board and reports
the first differing logit, which is the diagnostic to reach for when the accuracy
comes out below the simulated number. It needs a host compiler (the ISS is driven
from ``adder.c``'s native trace, the same source the firmware is built from), so
it is not free the way ``run_program.py``'s ISS backend used to be.

The host keeps the token embedding and the final argmax — the ISA has no gather
and nothing returns an index — which is the same split ``fw/adder.c``'s header
defines, so the device's share is unchanged. (``fw/infer.c`` moves both onto the
device; that kernel has its own host script.)
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TPULANG_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "tpulang"))
FW_DIR = os.path.normpath(os.path.join(HERE, "..", "fw"))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
for _p in (HERE, TPULANG_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                 # noqa: E402

import adder_export as ax                                    # noqa: E402
import model.numbers_data as numbers_data                    # noqa: E402
from iss import TPU                                          # noqa: E402
from tpu_uart import (                                       # noqa: E402
    FW_AW, FW_BASE, ProtocolError, TPUUart, autodetect_port, contiguous_runs,
    parse_hex_program, probe_idle, report_run_time, wait_until_idle,
)

T = ax.T
KERNEL = "adder"
DEFAULT_FW = os.path.join(FW_DIR, f"{KERNEL}.hex")
RQ_HEADER = os.path.join(FW_DIR, "adder_rq_ckpt.h")     # generated, gitignored


# =============================================================================
# Display. The dataset writes every number least-significant digit first, so a
# raw token dump is unreadable — everything here goes through
# ``numbers_data.unreverse_expression`` to get human digit order, the same way
# ``model/tests/test_inference.py`` prints its samples.
# =============================================================================
def decode_answer(ids) -> str:
    """A row of answer-region token ids -> the number it spells, human order.

    Pads are dropped by ``detokenize``; a prediction that emits a pad *between*
    digits therefore shows up as a shorter string rather than as a hole, which
    is what we want — it is wrong either way and the length says so.
    """
    return numbers_data.unreverse_expression(
        numbers_data.detokenize([int(t) for t in ids]))


def decode_prompt(expr: str) -> str:
    """``'321+54=861'`` (generator order) -> ``'123 + 45'``, the question only."""
    lhs = numbers_data.unreverse_expression(expr).split("=")[0]
    return lhs.rstrip(numbers_data.PAD_TOKEN).replace("+", " + ")


def report_timing(counters: list[dict], clk_mhz: float) -> None:
    """Per-problem run time and unit occupancy, from the device's counters.

    ``run`` is the core's busy interval, so this is the cost of one forward pass
    on the hardware with the UART excluded — the wall-clock loop time is
    dominated by the link and says nothing about the design. The other counters
    are printed as a fraction of ``run``; they **overlap** and do not partition
    it (``mload`` is a subset of ``mxu``).

    The one worth reading every run is ``idlec``: clocks with no unit busy at
    all — what the CPU costs as a command producer. On the ISS-verified kernel
    it is ~7% of the run, and most of that is the ``tpu_wait`` barriers rather
    than building commands (``docs/picorv32_migration.md`` §9.9).

    ``swait`` and ``vmm`` are retired counter slots and always read 0.
    """
    if not counters:
        return
    runs = [c["run"] for c in counters]
    lo, hi = min(runs), max(runs)
    mean = sum(runs) / len(runs)
    us = lambda cyc: cyc / clk_mhz                       # clocks -> microseconds

    print(f"\ncore active time over {len(runs)} run(s) @ {clk_mhz:g} MHz:")
    print(f"  per problem   mean {us(mean) / 1e3:8.3f} ms   "
          f"min {us(lo) / 1e3:.3f} ms   max {us(hi) / 1e3:.3f} ms")
    print(f"                mean {mean:.0f} core clocks "
          f"({1e6 / us(mean):.1f} problems/s if the link were free)")

    units = [k for k in counters[0] if k != "run"]
    total = sum(runs)
    if total:
        share = "  ".join(f"{k} {100.0 * sum(c[k] for c in counters) / total:.1f}%"
                          for k in units)
        print(f"  unit busy     {share}")


# =============================================================================
# The firmware image. The requant table is a compile-time input, so a run on a
# real checkpoint means a build.
# =============================================================================
def build_firmware(rq_table: list, model_path: str, quiet: bool = False) -> str:
    """Write the checkpoint's requant header and rebuild ``fw/adder.hex``.

    The kernel's objects are deleted first on purpose: make does not track a
    changed ``-D``, and the only thing that moves between two runs of this
    script is exactly that.
    """
    ax.write_rq_header(RQ_HEADER, rq_table, f"from {model_path}")
    for ext in ("o", "d", "elf", "bin", "hex", "map"):
        stale = os.path.join(FW_DIR, f"{KERNEL}.{ext}")
        if os.path.exists(stale):
            os.remove(stale)

    cmd = ["make", "-C", FW_DIR, f"PROG={KERNEL}",
           f"RQ={os.path.basename(RQ_HEADER)}"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise SystemExit(f"cannot run make: {exc}")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(
            f"building {KERNEL}.hex failed. A bare-metal RISC-V gcc is needed "
            f"(-march=rv32ic_zmmul); build the image elsewhere and pass it with "
            f"--fw, but then build it with RQ={os.path.basename(RQ_HEADER)} or "
            f"it will carry the synthetic requant table.")
    if not quiet:
        print(f"built     : {KERNEL}.hex against "
              f"{os.path.relpath(RQ_HEADER, ROOT)}")
    return DEFAULT_FW


def load_firmware_words(path: str) -> list:
    try:
        with open(path) as fh:
            words = parse_hex_program(fh.read())
    except OSError as exc:
        raise SystemExit(f"{exc} — build it with `make -C accel/tpu/fw "
                         f"PROG={KERNEL}`")
    if len(words) > (1 << FW_AW):
        raise SystemExit(f"{len(words)} words does not fit the {1 << FW_AW}-word "
                         f"firmware RAM")
    return words


# =============================================================================
# The two backends. Same inputs, same outputs, different machine.
# =============================================================================
class IssBackend:
    """The ISS, driven through the same interface as the board.

    The command stream comes from ``adder.c`` compiled natively (``-DTPU_TRACE``,
    host cc), which is the same source the firmware image is built from — so
    this is not a second implementation of the kernel, it is the same one with
    the MMIO writes redirected.
    """

    name = "ISS"

    def __init__(self, records, weights):
        self.records = records
        self.tpu = TPU(rows=ax.ROWS, cols=ax.COLS)
        ax.stage_static(self.tpu, weights)
        self.counters: list[dict] = []      # the ISS has no clock; stays empty

    def run_problem(self, inputs: dict) -> torch.Tensor:
        for addr, byte in inputs.items():
            self.tpu.dram[addr] = byte
        self.tpu.run_trace(self.records)
        return ax.read_logits(self.tpu)


class BoardBackend:
    """The FPGA over UART. Firmware and weights once, then one G per problem."""

    name = "FPGA"

    def __init__(self, uart: TPUUart, words, args):
        self.u, self.words, self.args = uart, words, args
        self.counters: list[dict] = []
        self._timer_warned = False

    def read_counters(self) -> dict | None:
        """The device's per-run counters for the problem that just halted.

        Safe to call once per problem: the counters reset at each ``G`` and
        freeze at the halt, so what comes back describes *this* run and nothing
        accumulates across problems (``tpu_uart.read_counters``). ``run`` is the
        core's busy interval — 'G' to done — which is the honest cost of one
        forward pass; the wall clock is mostly UART.

        Non-fatal, and warns only once: 'T' is newer than the other commands and
        a bitstream flashed before it existed NAKs it, which should not cost a
        scoring run. See ``tpu_uart.report_run_time``.
        """
        try:
            ctr = self.u.read_counters()
        except (ProtocolError, OSError) as exc:
            if not self._timer_warned:
                print(f"  timer   : unavailable — {exc}")
                print("            (bitstream predates the 'T' command? "
                      "reflash board=cmod_a7)")
                self._timer_warned = True
            return None
        self.counters.append(ctr)
        return ctr

    def _idle(self) -> None:
        if not probe_idle(self.u):
            wait_until_idle(self.u, self.args.run_timeout, self.args.poll_interval)

    def load_static(self, weights: dict) -> None:
        """The firmware image and every byte of DRAM that outlives a problem."""
        self._idle()
        self.u.load_program(FW_BASE, self.words)
        runs = contiguous_runs(weights)
        for addr, blob in runs:
            self.u.write_mem(addr, blob)
        print(f"loaded    : {len(self.words)} firmware words, "
              f"{len(weights)} static bytes in {len(runs)} frame(s)")
        if self.args.verify_inputs:
            bad = sum(1 for addr, blob in runs
                      for i, b in enumerate(self.u.read_mem(addr, len(blob)))
                      if b != blob[i])
            if bad:
                raise SystemExit(f"weight readback FAILED: {bad} byte(s) differ")
            print(f"verify    : all {len(weights)} static bytes read back identical")

    def run_problem(self, inputs: dict) -> torch.Tensor:
        for addr, blob in contiguous_runs(inputs):
            self.u.write_mem(addr, blob)
        self.u.go(FW_BASE)
        wait_until_idle(self.u, self.args.run_timeout, self.args.poll_interval)
        if self.args.timing:
            self.read_counters()
        return ax.decode_logits(self.u.read_mem(ax.DR_LOG, ax.LOGITS_BYTES))


# =============================================================================
# Driver.
# =============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--port", help="serial port (default: autodetect the FTDI one)")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("-t", "--timeout", type=float, default=2.0)
    ap.add_argument("-n", "--problems", type=int, default=32)
    ap.add_argument("--ckpt", default="model/saved/int4_d64_f256_l4.pt",
                    help="the checkpoint to score (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0,
                    help="problem generator seed; 0 is adder_export.py's, so the "
                         "two score the same problems")
    ap.add_argument("--fw", metavar="HEX",
                    help="use this prebuilt firmware image instead of rebuilding "
                         "adder.hex — it must have been built with this "
                         "checkpoint's requant table")
    ap.add_argument("--poll-interval", type=float, default=0.05)
    ap.add_argument("--run-timeout", type=float, default=30.0)
    ap.add_argument("--clk-mhz", type=float, default=12.0)
    ap.add_argument("--verify-inputs", action="store_true",
                    help="read the weights back after writing them (bring-up check)")
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                    help="print the problem, the expected answer and the device's "
                         "answer for every problem (--no-show: progress only)")
    ap.add_argument("--timing", action=argparse.BooleanOptionalAction, default=True,
                    help="read the device's performance counters after every "
                         "problem and report the per-problem run time")
    ap.add_argument("--compare-iss", action="store_true",
                    help="also run every problem in the ISS and diff the logits")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the ISS as the backend; never open the serial port")
    args = ap.parse_args(argv)

    # ---- the checkpoint -> the device's integers ---------------------------
    print(f"checkpoint: {args.ckpt}")
    model = ax.load_model(args.ckpt)
    print(f"model     : d={model.d} f={model.f} layers={len(model.layers)} "
          f"q_heads={model.q_heads} head_dim={model.head_dim}")

    rq_table, weights = ax.derive(model)
    static = ax.static_image(weights)
    print(f"image     : {len(static)} static bytes "
          f"(weights, causal mask, output head)")

    # ---- the firmware, and the trace the ISS runs --------------------------
    # Both come from adder.c. The board needs the RISC-V build; the ISS needs
    # the native one, so a board-only run pays for neither of the other's
    # toolchain.
    need_iss = args.dry_run or args.compare_iss
    fw_path = None
    if not args.dry_run:
        fw_path = args.fw or build_firmware(rq_table, args.ckpt)

    iss = None
    if need_iss:
        workdir = tempfile.mkdtemp(prefix="run_adder_")
        header = os.path.join(workdir, "adder_rq_ckpt.h")
        ax.write_rq_header(header, rq_table, f"from {args.ckpt}")
        try:
            records = ax.build_trace(header, workdir)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"cannot build adder.c's native trace ({exc}) — "
                             f"the ISS backend needs a host compiler (HOSTCC)")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        n_cmds = sum(1 for r in records if r[0] == "CMD")
        print(f"trace     : {n_cmds} commands from adder.c (host build)")
        iss = IssBackend(records, weights)

    # ---- backends ----------------------------------------------------------
    uart = None
    if args.dry_run:
        dev = iss
        print("backend   : ISS (--dry-run; nothing is sent to a board)")
    else:
        words = load_firmware_words(fw_path)
        print(f"firmware  : {os.path.basename(fw_path)}, {len(words)} words "
              f"({4 * len(words)} bytes)")
        port = args.port or autodetect_port()
        uart = TPUUart(port, args.baud, args.timeout).__enter__()
        dev = BoardBackend(uart, words, args)
        dev.load_static(static)
        print(f"backend   : FPGA on {port} @ {args.baud} baud")

    # ---- the problems, and what the model itself answers -------------------
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    exprs, tokens, masks = numbers_data.create_addition_batch(args.problems, T)
    tok = torch.tensor(tokens)
    with torch.no_grad():
        ref_pred = model(tok, torch.stack(masks)).argmax(-1)
    x0 = ax.embed_int4(model, tok)

    ans = numbers_data.EQUALS_POS
    tgt = tok[:, ans:]

    # ---- the loop ----------------------------------------------------------
    print(f"\nscoring {args.problems} problems on the {dev.name} "
          f"(answer region [{ans}, {T})):")
    preds, first_diff = [], None
    wrong = ref_wrong = 0
    if args.show:
        print(f"  {'#':>4s}  {'problem':<16s} {'expected':>12s} {'device':>12s}"
              f"  {'':9s} {'run':>10s}")
    t0 = time.monotonic()
    try:
        for i in range(args.problems):
            inputs = ax.input_image(x0[i])
            logits = dev.run_problem(inputs)
            pred = logits.argmax(-1)[ans - 1:-1]
            preds.append(pred)

            ok = bool((pred == tgt[i]).all())
            wrong += not ok
            ref_wrong += not bool((ref_pred[i, ans - 1:-1] == tgt[i]).all())
            if args.show:
                # The device's own clocks for this problem, if it kept any.
                ms = (f"{dev.counters[-1]['run'] / args.clk_mhz / 1e3:.2f} ms"
                      if len(dev.counters) == i + 1 else "")
                print(f"  {i:>4d}  {decode_prompt(exprs[i]):<16s} "
                      f"{decode_answer(tgt[i]):>12s} {decode_answer(pred):>12s}"
                      f"  {'' if ok else '<-- WRONG':9s} {ms:>10s}")

            if args.compare_iss and dev is not iss:
                want = iss.run_problem(inputs)
                if first_diff is None and bool((logits != want).any()):
                    t, v = (logits != want).nonzero()[0].tolist()
                    first_diff = (i, t, v, int(logits[t, v]), int(want[t, v]))
            if not args.show and ((i + 1) % 8 == 0 or i + 1 == args.problems):
                print(f"  {i + 1}/{args.problems} problems "
                      f"({time.monotonic() - t0:.1f}s)  {wrong} wrong", flush=True)
    finally:
        if uart is not None:
            # Only fall back to the one-shot report when the loop collected
            # nothing — otherwise report_timing below covers every run, and
            # this would just print the last problem's counters twice.
            if isinstance(dev, BoardBackend) and not dev.counters:
                try:
                    report_run_time(uart, args.clk_mhz)
                except (ProtocolError, OSError):
                    pass
            uart.__exit__(None, None, None)

    if args.show:
        print(f"\n  {args.problems - wrong}/{args.problems} exact "
              f"({time.monotonic() - t0:.1f}s wall clock)")
    report_timing(dev.counters, args.clk_mhz)

    # ---- scoring -----------------------------------------------------------
    dev_pred = torch.stack(preds)
    ref_win = ref_pred[:, ans - 1:-1]
    n, n_tok = args.problems, tgt.numel()
    seq = (dev_pred == tgt).all(-1).sum().item() / n
    tokacc = (dev_pred == tgt).sum().item() / n_tok
    ref_seq = (ref_win == tgt).all(-1).sum().item() / n
    ref_tok = (ref_win == tgt).sum().item() / n_tok

    print(f"\naccuracy over {n} problems:")
    print(f"  {'':22s} {'exact-sequence':>15s} {'token':>9s}")
    print(f"  {dev.name:22s} {seq * 100:14.2f}% {tokacc * 100:8.2f}%")
    print(f"  {'QAT model (PyTorch)':22s} {ref_seq * 100:14.2f}% {ref_tok * 100:8.2f}%")
    diff = int((dev_pred != ref_win).sum())
    print(f"\nargmax vs. the model it came from: {diff} of {n_tok} scored "
          f"positions differ")

    if first_diff is not None:
        i, t, v, got, want = first_diff
        print(f"\n  !! problem {i} logit[{t}][{v}]: device {got}  ISS {want}")
        return 1
    if seq < ref_seq:
        print("\n  !! the device scored below the model it was derived from — "
              "rerun with --compare-iss to localize")
        return 1
    print(f"\nPASSED: the {dev.name} matches the checkpoint it was built from")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
