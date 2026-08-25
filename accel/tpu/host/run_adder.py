#!/usr/bin/env python3
"""run_adder.py — the adder checkpoint *generating* on the FPGA, problem by problem.

``run_fw_matmul.py`` answers "does the board compute what the ISS computes", on
a synthetic image. That is the right question for a kernel and the wrong one for
a model. This script asks the other one, in the shape the model would actually
be used: the device is handed a **prompt** and nothing else, prefills it, then
decodes its own output token by token against a KV cache, and what comes back is
a sequence it chose. A single wrong digit derails everything after it — the
property teacher forcing hides.

The producer is ``accel/tpu/fw/infer.c`` on the PicoRV32. (``fw/adder.c`` is the
same four layers in the training shape, every position at once, scored
teacher-forced; ``accel/tpulang/adder_export.py`` is the host for that one.)
There is no assembler and no ``.tpu`` program any more, so a run needs a
firmware image the way it used to need an assembled one.

**What the device owns here that a whole-sequence forward gave to the host.**
``cpu_subsys.sv`` maps the scratchpad at ``0x9xxx_xxxx``, so the head writes its
13 logits there, the CPU reads them back and argmaxes, and the embedding
"gather" is a DMA at ``DR_EMBED + token*D`` — an address the CPU computed from
the token it just chose. **The host tokenizes and nothing else.** The whole
autoregressive loop closes on the device: one ``G`` per problem, and a finished
sequence comes back out of ``DR_TOKENS``.

**The requant table is compiled into the image.** The 16 ``{m0,n}`` words per
layer are literals in the macro-ops, so the device has no path by which it could
read them out of memory — a checkpoint's scales reach the board only through the
build. So does ``--gen``: ``INFER_GEN`` is a compile-time constant. This script
therefore derives the table, writes ``fw/adder_rq_ckpt.h`` and rebuilds
``fw/infer.hex`` against it before loading anything (``--fw`` takes a prebuilt
image instead, and then it is on you that the two agree: an image carrying the
checked-in synthetic ``adder_rq.h`` runs perfectly and generates noise).

Everything that is not the transport is shared with the simulated path
(``infer_export.static_image``, which is ``adder_export``'s weights and head
plus this kernel's mask and embedding table, and its prompt and token codecs),
so a disagreement between ``--dry-run`` and a board run localizes to the
hardware rather than to a second copy of the staging logic. That mattered: the
pipeline has drifted from the model twice before (``adder_export.py``'s
docstring).

**What moves over the wire, and when.** Weights, causal mask, output head and
the embedding table are ~390 KB at ``d=128``, identical for every problem, and
go down once with the firmware image. Per problem the host sends **128 bytes**
(32 int32 token ids) and reads **128 back** (32 generated ids) — the loop is
device-bound for once, not link-bound.

**The scratchpad is not cleared between problems, on the board or in the ISS.**
The KV cache from problem n-1 is still there when problem n starts, and the
causal mask is what makes it harmless (a masked score is at most -1 before ReLU,
whatever the garbage was). Running the problems back to back is not a shortcut,
it is the test.

    python accel/tpu/host/run_adder.py --dry-run -n 4      # ISS, no board
    python accel/tpu/host/run_adder.py -p COM5 -n 64
    python accel/tpu/host/run_adder.py -p COM5 -n 4 --compare-iss
    python accel/tpu/host/run_adder.py -p COM5 -n 8 --gen 3    # 3 tokens, quick
    python accel/tpu/host/run_adder.py -p COM5 -n 1 --split    # prefill vs decode
    python accel/tpu/host/run_adder.py -p COM5 -n 8 --phase decode   # benchmark
    python accel/tpu/host/run_adder.py -p COM5 -n 8 --batch 2 --phase decode
    python accel/tpu/host/run_adder.py -p COM5 -n 64 --no-show  # progress only

By default every problem prints as it finishes — the prompt, what the device
generated and what the answer was, in human digit order (the dataset stores
numbers reversed). ``--no-show`` falls back to the periodic progress counter,
which is what you want when N is large.

The device's performance counters are read after every problem (``--no-timing``
to skip the extra round trip), so each row carries that problem's *core* run
time — a prefill plus ``--gen``-1 decode steps — and the summary reports
mean/min/max plus unit occupancy. The counters reset at each ``G``, so one read
per problem measures one generation and nothing accumulates.

**Benchmarking one half.** The counters cannot be read mid-run — they reset at
``G`` and freeze at the halt — so measuring the prefill and the decode
separately means two images. ``INFER_PREFILL`` / ``INFER_DECODE`` are
compile-time constants in ``infer.c``; ``--phase prefill`` and ``--phase
decode`` build and run one of them, and ``--split`` does both plus the whole
generation and tables the three.

**A benchmark run does not stage the weights.** What a phase costs does not
depend on what the weights are — the same bytes stream and the same commands
issue — so ``--phase prefill`` and ``--phase decode`` skip the ~390 KB static
upload entirely and score nothing. The ids they generate are noise, and the
summary says so instead of reporting an accuracy.

``--batch B`` runs B independent sequences per ``G``, sharing one weight stream:
the projections, ``Wo`` and both FFN matmuls take ``B*rows`` rows and the weight
is staged once for all of them, while attention stays per sequence because each
has its own KV cache. That cache is 48 KB per sequence, so B is bounded by DRAM
— the firmware build fails with ``the activation map runs into layer 0's
weights`` when it does not fit, which at this shape is B > 2 for a decode-only
image and B > 1 once the prefill's block-sized activations are also resident.

``--compare-iss`` runs each problem in *both* the ISS and the board and reports
the first differing logit. It needs a host compiler (the ISS is driven by
``infer.c`` compiled natively, the same source the firmware is built from) and
it is far slower than the board, so it is a bring-up check rather than the
default.
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
import infer_export as ix                                    # noqa: E402
import model.numbers_data as numbers_data                    # noqa: E402
from fw_vectors import coexecute                             # noqa: E402
from iss import TPU                                          # noqa: E402
from tpu_uart import (                                       # noqa: E402
    FW_AW, FW_BASE, ProtocolError, TPUUart, autodetect_port, contiguous_runs,
    parse_hex_program, probe_idle, report_run_time, wait_until_idle,
)

T, PROMPT, NGEN = ix.T, ix.PROMPT, ix.NGEN
KERNEL = "infer"
DEFAULT_FW = os.path.join(FW_DIR, f"{KERNEL}.hex")
RQ_HEADER = os.path.join(FW_DIR, "adder_rq_ckpt.h")     # generated, gitignored

# The generated ids, and the logits behind them. Both regions start where the
# prefill's own output does: the prefill produces the token at PROMPT out of
# position PROMPT-1's logits, and every decode step adds one of each. Both are
# per sequence, and so is everything below them in the map — see
# ``infer_export.layout``.
def tok_base(seq: int, batch: int) -> int:
    return ix.token_addr(seq, PROMPT, batch)


def log_base(seq: int, batch: int) -> int:
    return ix.logit_addr(seq, PROMPT - 1, batch)


# =============================================================================
# Display. The dataset writes every number least-significant digit first, so a
# raw token dump is unreadable — everything here goes through
# ``numbers_data.unreverse_expression`` to get human digit order, the same way
# ``model/tests/test_inference.py`` prints its samples.
# =============================================================================
def spell(ids) -> str:
    """A row of token ids -> the string it spells, human digit order.

    Pads are dropped by ``detokenize``; a generation that emits a pad *between*
    digits therefore shows up as a shorter string rather than as a hole, which
    is what we want — it is wrong either way and the length says so.
    """
    return numbers_data.unreverse_expression(
        numbers_data.detokenize([int(t) for t in ids]))


def decode_prompt(ids) -> str:
    """The prompt's ids -> ``'123 + 45'``, the question only."""
    return spell(ids).split("=")[0].replace("+", " + ")


def report_timing(counters: list[dict], clk_mhz: float, tokens_per_seq: int,
                  batch: int, phase: str) -> None:
    """Per-launch run time and unit occupancy, from the device's counters.

    ``run`` is the core's busy interval, so this is the cost of one ``G`` — a
    whole generation, or whichever half ``--phase`` built — with the UART
    excluded; the wall-clock loop time says nothing about the design. The other
    counters are printed as a fraction of ``run``; they **overlap** and do not
    partition it (``mload`` is a subset of ``mxu``).

    A launch carries ``batch`` sequences, so the per-problem and per-token rows
    divide by it. That is the number batching moves: the weights stream once per
    launch whatever ``batch`` is.

    Two counters are worth reading every run. ``dma`` should dominate: a decode
    step is one row of arithmetic against the same ~390 KB of weights the
    prefill streamed, so generation is DMA-bound by construction. ``idlec`` is
    clocks with no unit busy at all, i.e. what the CPU costs as a command
    producer, now including the argmax it does per token.
    """
    if not counters:
        return
    runs = [c["run"] for c in counters]
    lo, hi = min(runs), max(runs)
    mean = sum(runs) / len(runs)
    tokens = tokens_per_seq * batch
    us = lambda cyc: cyc / clk_mhz                       # clocks -> microseconds

    print(f"\ncore active time over {len(runs)} {phase} run(s) of {batch} "
          f"sequence(s) @ {clk_mhz:g} MHz:")
    print(f"  per launch    mean {us(mean) / 1e3:8.3f} ms   "
          f"min {us(lo) / 1e3:.3f} ms   max {us(hi) / 1e3:.3f} ms")
    print(f"  per problem   mean {us(mean / batch) / 1e3:8.3f} ms   "
          f"({batch} sequence(s) per launch)")
    if tokens:
        print(f"  per token     mean {us(mean / tokens) / 1e3:8.3f} ms   "
              f"({mean / tokens:.0f} core clocks over {tokens} token(s))")

    units = [k for k in counters[0] if k != "run"]
    total = sum(runs)
    if total:
        share = "  ".join(f"{k} {100.0 * sum(c[k] for c in counters) / total:.1f}%"
                          for k in units)
        print(f"  unit busy     {share}")

def report_split(prefill: dict, decode: dict, whole, gen: int, clk_mhz: float,
                 batch: int) -> None:
    """Prefill and one decode step, each measured by the image that runs it.

    The counters are cumulative over one `G` and reset at the next, so an image
    built with INFER_DECODE=0 IS the prefill in isolation and one built with
    INFER_PREFILL=0 IS the `gen`-1 decode steps. Nothing is subtracted and
    nothing is estimated: every counter is measured where it is reported.

    All three rows are the same prompts on the same staged weights; only the
    image differs. A step's cost is not data-dependent — the same weights stream
    and the same commands issue whatever is in the cache — which is also why the
    decode-only image may start from the prompt's last token.
    """
    steps = gen - 1
    step = {k: decode[k] / steps for k in decode} if steps else None

    rows = [(f"prefill ({PROMPT} rows x {batch})", prefill)]
    if step is not None:
        rows.append((f"one decode step (M={batch})", step))
    if whole is not None:
        rows.append(("whole generation", whole))

    units = [k for k in prefill if k != "run" and prefill[k]]
    print(f"\nprefill vs decode, one image each (batch {batch}, gen {gen}):")
    head = "".join(f"{u:>8s}" for u in units)
    print(f"  {'':24s} {'clocks':>10s} {'ms':>9s}  {head}")
    for name, ctr in rows:
        shares = "".join(f"{100.0 * ctr[u] / ctr['run']:7.1f}%" for u in units)
        print(f"  {name:24s} {ctr['run']:10.0f} "
              f"{ctr['run'] / clk_mhz / 1e3:9.3f}  {shares}")

    # What the cache buys, in one number: a prompt token is amortized over one
    # block, a generated one costs a whole step.
    if step is not None:
        per_prompt = prefill["run"] / (PROMPT * batch)
        per_gen = step["run"] / batch
        print(f"\n  a prompt token costs {per_prompt:.0f} clocks in the prefill, "
              f"{per_gen:.0f} in a decode step ({per_gen / per_prompt:.1f}x)")

# =============================================================================
# The firmware image. The requant table, the token count, the phase and the
# batch are all compile-time inputs, so a run means a build.
# =============================================================================
def build_firmware(rq_table: list, model_path: str, gen: int,
                   phase: str = "both", batch: int = 1, block=None) -> str:
    """Write the checkpoint's requant header and rebuild ``fw/infer.hex``.

    The kernel's objects are deleted first on purpose: make does not track a
    changed ``-D``, and everything that moves between two runs of this script
    (the table, ``INFER_GEN``, ``PHASE``, ``BATCH``, ``BLOCK``) is exactly that.
    """
    ax.write_rq_header(RQ_HEADER, rq_table, f"from {model_path}")
    for ext in ("o", "d", "elf", "bin", "hex", "map"):
        stale = os.path.join(FW_DIR, f"{KERNEL}.{ext}")
        if os.path.exists(stale):
            os.remove(stale)

    cmd = ["make", "-C", FW_DIR, f"PROG={KERNEL}",
           f"RQ={os.path.basename(RQ_HEADER)}", f"GEN={gen}",
           f"PHASE={phase}", f"BATCH={batch}"]
    if block:
        cmd.append(f"BLOCK={block}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise SystemExit(f"cannot run make: {exc}")
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(
            f"building {KERNEL}.hex failed. A bare-metal RISC-V gcc is needed "
            f"(-march=rv32ic_zmmul); build the image elsewhere and pass it with "
            f"--fw, but then build it with RQ={os.path.basename(RQ_HEADER)}, "
            f"GEN={gen}, PHASE={phase}, BATCH={batch} and BLOCK={block} or it "
            f"will not be the kernel this run reports on.\n"
            f"  'the activation map runs into layer 0's weights' -> the "
            f"activations do not fit the 128 KB under the weights: lower "
            f"--block, then --batch.\n"
            f"  'region RAM overflowed' -> the IMAGE does not fit the CPU's "
            f"16 KB instruction RAM: build one phase at a time (--phase).")
    print(f"built     : {KERNEL}.hex against "
          f"{os.path.relpath(RQ_HEADER, ROOT)}, GEN={gen}, PHASE={phase}, "
          f"BATCH={batch}, BLOCK={block or 'default'}")
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
# The two backends. Same prompts, same generated ids, different machine.
#
# Neither is re-initialized between launches: the KV cache and the token block
# are left exactly as the previous launch finished them, which is the state the
# kernel is designed to tolerate.
#
# A launch carries ``batch`` sequences, so both return one list of generated ids
# per sequence.
# =============================================================================
class IssBackend:
    """The ISS, driven through the same interface as the board.

    ``coexecute`` runs ``infer.c`` compiled natively (``-DTPU_TRACE``, host cc),
    executing each command as it is emitted and answering the kernel's
    scratchpad reads out of the model's memory. A kernel that argmaxes its own
    logits has no fixed command stream — the token it picks lands in the
    *address* of the next DMA — so there is no trace to replay, only a run.
    """

    name = "ISS"

    def __init__(self, exe: str, static, gen: int, batch: int):
        self.exe, self.gen, self.batch = exe, gen, batch
        self.tpu = TPU(rows=ax.ROWS, cols=ax.COLS)
        for addr, byte in (static or {}).items():
            self.tpu.dram[addr] = byte
        self.counters: list[dict] = []      # the ISS has no clock; stays empty
        self.commands = 0

    def run_problem(self, prompt: dict) -> tuple[list, list]:
        for addr, byte in prompt.items():
            self.tpu.dram[addr] = byte
        cmds, _ = coexecute(self.tpu, self.exe, quiet=True)
        self.commands = len(cmds)
        dram = self.tpu.dram
        tokens, logits = [], []
        for seq in range(self.batch):
            t0, l0 = tok_base(seq, self.batch), log_base(seq, self.batch)
            tokens.append(
                ix.decode_tokens(bytes(dram[t0:t0 + self.gen * 4])))
            logits.append(bytes(dram[l0:l0 + self.gen * ix.LOGIT_ROW]))
        return tokens, logits


class BoardBackend:
    """The FPGA over UART. Firmware and weights once, then one G per launch."""

    name = "FPGA"

    def __init__(self, uart: TPUUart, words, args, gen: int, batch: int):
        self.u, self.words, self.args = uart, words, args
        self.gen, self.batch = gen, batch
        self.counters: list[dict] = []
        self._timer_warned = False

    def read_counters(self) -> dict | None:
        """The device's per-run counters for the launch that just halted.

        Safe to call once per launch: the counters reset at each ``G`` and
        freeze at the halt, so what comes back describes *this* run and nothing
        accumulates across launches (``tpu_uart.read_counters``).

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
        return ctr

    def _idle(self) -> None:
        if not probe_idle(self.u):
            wait_until_idle(self.u, self.args.run_timeout, self.args.poll_interval)

    def load_firmware(self, words) -> None:
        """Replace the image in the CPU's RAM. DRAM is untouched.

        Swapping images mid-session is what --split needs: the same kernel built
        for one phase, against the same staged weights.
        """
        self._idle()
        self.words = words
        self.u.load_program(FW_BASE, words)

    def load_static(self, static) -> None:
        """The firmware image and every byte of DRAM that outlives a launch.

        ``static`` is None for a benchmark run: what a phase costs does not
        depend on what the weights are, so it is not worth ~390 KB of UART.
        """
        self.load_firmware(self.words)
        if static is None:
            print(f"loaded    : {len(self.words)} firmware words, no static "
                  f"image (benchmark run)")
            return
        runs = contiguous_runs(static)
        for addr, blob in runs:
            self.u.write_mem(addr, blob)
        print(f"loaded    : {len(self.words)} firmware words, "
              f"{len(static)} static bytes in {len(runs)} frame(s)")
        if self.args.verify_inputs:
            bad = sum(1 for addr, blob in runs
                      for i, b in enumerate(self.u.read_mem(addr, len(blob)))
                      if b != blob[i])
            if bad:
                raise SystemExit(f"static readback FAILED: {bad} byte(s) differ")
            print(f"verify    : all {len(static)} static bytes read back identical")

    def _launch(self, prompt: dict, timed: bool) -> dict | None:
        """Stage the prompts, press G, wait for the halt. Returns its counters."""
        for addr, blob in contiguous_runs(prompt):
            self.u.write_mem(addr, blob)
        self.u.go(FW_BASE)
        wait_until_idle(self.u, self.args.run_timeout, self.args.poll_interval)
        return self.read_counters() if timed else None

    def measure(self, prompt: dict) -> dict | None:
        """One timed run whose counters are NOT part of the scored summary.

        --split's two phase images: the same prompts against a different image,
        so they belong in the comparison table and nowhere else.
        """
        return self._launch(prompt, True)

    def run_problem(self, prompt: dict) -> tuple[list, list]:
        ctr = self._launch(prompt, self.args.timing)
        if ctr is not None:
            self.counters.append(ctr)
        if not self.args.score:
            return [], []                   # a benchmark run reads nothing back
        tokens, logits = [], []
        for seq in range(self.batch):
            tokens.append(ix.decode_tokens(
                self.u.read_mem(tok_base(seq, self.batch), self.gen * 4)))
            logits.append(
                self.u.read_mem(log_base(seq, self.batch),
                                self.gen * ix.LOGIT_ROW)
                if self.args.compare_iss else b"")
        return tokens, logits


def first_logit_diff(got: bytes, want: bytes) -> tuple | None:
    """``(position, vocab index, device, ISS)`` of the first int32 that differs."""
    for off in range(0, min(len(got), len(want)), 4):
        a = int.from_bytes(got[off:off + 4], "little", signed=True)
        b = int.from_bytes(want[off:off + 4], "little", signed=True)
        if a != b:
            pos, idx = divmod(off // 4, ix.VPAD)
            return (PROMPT - 1 + pos, idx, a, b)
    return None


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
    ap.add_argument("-g", "--gen", type=int, default=NGEN,
                    help=f"tokens to generate per problem (default {NGEN}, the "
                         f"whole answer field). Compiled into the image, so "
                         f"changing it rebuilds")
    ap.add_argument("--batch", type=int, default=1,
                    help="independent sequences per run, sharing one weight "
                         "stream. Compiled into the image; bounded by DRAM, "
                         "since the KV cache is 48 KB per sequence")
    ap.add_argument("--block", type=int, default=None,
                    help="prefill rows per sequence per pass (default: the "
                         "array's 32-row dispatch limit). Lowering it shrinks "
                         "every activation buffer at one weight stream per "
                         "extra pass, and is what makes --batch > 1 fit DRAM")
    ap.add_argument("--phase", choices=("both", "prefill", "decode"),
                    default="both",
                    help="which half of a generation the image runs. 'both' "
                         "(the default) is scored; the other two are benchmarks "
                         "and stage no weights")
    ap.add_argument("--ckpt", default="model/saved/int4_d128_f512_l4.pt",
                    help="the checkpoint to run (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0,
                    help="problem generator seed; 0 is infer_export.py's, so the "
                         "two run the same problems")
    ap.add_argument("--fw", metavar="HEX",
                    help="use this prebuilt firmware image instead of rebuilding "
                         "infer.hex — it must have been built with this "
                         "checkpoint's requant table and this --gen/--phase/--batch")
    ap.add_argument("--poll-interval", type=float, default=0.05)
    ap.add_argument("--run-timeout", type=float, default=30.0)
    ap.add_argument("--clk-mhz", type=float, default=12.0)
    ap.add_argument("--verify-inputs", action="store_true",
                    help="read the static image back after writing it (bring-up check)")
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                    help="print the prompt, what the device generated and the "
                         "answer for every problem (--no-show: progress only)")
    ap.add_argument("--timing", action=argparse.BooleanOptionalAction, default=True,
                    help="read the device's performance counters after every "
                         "launch and report the run time")
    ap.add_argument("--split", action="store_true",
                    help="also build and run a prefill-only and a decode-only "
                         "image, and report the two costs separately (board only)")
    ap.add_argument("--compare-iss", action="store_true",
                    help="also run every problem in the ISS and diff the logits")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the ISS as the backend; never open the serial port")
    args = ap.parse_args(argv)

    # A phase image generates ids nobody can score: the prefill-only one stops
    # after a single token and the decode-only one starts from the prompt's last
    # token rather than from what a prefill would have produced.
    args.score = (args.phase == "both")

    if not 1 <= args.gen <= NGEN:
        raise SystemExit(f"--gen must be in 1..{NGEN}")
    if args.batch < 1:
        raise SystemExit("--batch is a sequence count")
    if args.phase == "decode" and args.gen < 2:
        raise SystemExit("--phase decode runs --gen minus 1 steps: --gen >= 2")
    if not args.score and args.compare_iss:
        raise SystemExit("--compare-iss diffs logits against the ISS, which "
                         "needs the scored image: drop --phase")
    if args.split:
        # Every one of these would make the comparison meaningless rather than
        # wrong, so say so up front instead of printing a table of zeros.
        if args.dry_run:
            raise SystemExit("--split reads the device's counters; the ISS has "
                             "none. Drop --dry-run.")
        if not args.timing:
            raise SystemExit("--split needs the counters; drop --no-timing")
        if args.gen < 2:
            raise SystemExit("--split needs at least one decode step: --gen >= 2")
        if not args.score:
            raise SystemExit("--split builds its own phase images; run it with "
                             "--phase both")

    # ---- the checkpoint -> the device's integers ---------------------------
    print(f"checkpoint: {args.ckpt}")
    model = ax.load_model(args.ckpt)
    print(f"model     : d={model.d} f={model.f} layers={len(model.layers)} "
          f"q_heads={model.q_heads} head_dim={model.head_dim}")

    rq_table, weights = ax.derive(model)
    # Weights and output head from adder_export (the two kernels are one
    # checkpoint at one set of addresses), then infer.c's own causal mask and
    # the embedding table the device gathers from. Everything but the weights
    # moves with --batch, so the image is built at this batch's addresses.
    static = None
    if args.score:
        static = ix.static_image(model, weights, args.batch)
        print(f"image     : {len(static)} static bytes "
              f"(weights, causal mask, output head, embedding table)")
    else:
        print(f"image     : none — a {args.phase}-only run is a benchmark, and "
              f"what it costs does not depend on the weights")

    # ---- the firmware, and the binary the ISS co-executes ------------------
    # Both come from infer.c. The board needs the RISC-V build, the ISS the
    # native one, so a board-only run pays for neither of the other's toolchain.
    fw_path, split_words = None, {}
    if not args.dry_run:
        # The phase images go first so the image left in fw/ is the one whose
        # accuracy this run reports.
        if args.split:
            for ph in ("prefill", "decode"):
                split_words[ph] = load_firmware_words(
                    build_firmware(rq_table, args.ckpt, args.gen, ph,
                                   args.batch, args.block))
        fw_path = args.fw or build_firmware(rq_table, args.ckpt, args.gen,
                                            args.phase, args.batch, args.block)

    iss = workdir = None
    if args.dry_run or args.compare_iss:
        workdir = tempfile.mkdtemp(prefix="run_adder_")
        header = os.path.join(workdir, "adder_rq_ckpt.h")
        ax.write_rq_header(header, rq_table, f"from {args.ckpt}")
        try:
            exe = ix.build_kernel(header, workdir, args.gen, args.phase,
                                  args.batch, args.block)
        except (OSError, subprocess.CalledProcessError) as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise SystemExit(f"cannot build infer.c natively ({exc}) — the ISS "
                             f"backend needs a host compiler (HOSTCC)")
        iss = IssBackend(exe, static, args.gen, args.batch)
        print("iss       : infer.c built for the host, co-executing")

    # ---- backends ----------------------------------------------------------
    uart = None
    if args.dry_run:
        dev = iss
        print("backend   : ISS (--dry-run; nothing is sent to a board)")
        if not args.timing or not args.score:
            print("note      : the ISS keeps no clocks, so nothing here is timed")
    else:
        words = load_firmware_words(fw_path)
        print(f"firmware  : {os.path.basename(fw_path)}, {len(words)} words "
              f"({4 * len(words)} bytes)")
        port = args.port or autodetect_port()
        uart = TPUUart(port, args.baud, args.timeout).__enter__()
        dev = BoardBackend(uart, words, args, args.gen, args.batch)
        dev.load_static(static)
        print(f"backend   : FPGA on {port} @ {args.baud} baud")

    # ---- the problems ------------------------------------------------------
    # One launch carries `--batch` of them. A trailing group short of a full
    # batch repeats its last prompt into the spare sequences: the kernel always
    # runs BATCH of them, and only the real ones are scored.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _, tokens, _ = numbers_data.create_addition_batch(
        args.problems, T, max_digits=ix.MAX_DIGITS, equals_pos=PROMPT)

    groups = []
    for start in range(0, args.problems, args.batch):
        real = [tokens[i][:PROMPT]
                for i in range(start, min(start + args.batch, args.problems))]
        groups.append((start, real, real + [real[-1]] * (args.batch - len(real))))

    # ---- the loop ----------------------------------------------------------
    print(f"\nrunning {args.phase}, {args.gen} token(s) per problem, "
          f"{args.problems} problem(s) in {len(groups)} launch(es) of "
          f"{args.batch}, on the {dev.name}:")
    if args.show and args.score:
        print(f"  {'#':>4s}  {'problem':<16s} {'device':>12s} {'answer':>12s}"
              f"  {'':9s} {'run':>10s}")
    dev_seq = dev_tok = ref_seq = ref_tok = agree = 0
    n_tok = done = 0
    first_diff = prefill_ctr = decode_ctr = None
    t0 = time.monotonic()
    try:
        for start, real, padded in groups:
            image = ix.batch_prompt_image(padded, args.batch)
            timed_before = len(dev.counters)
            got_all, logits_all = dev.run_problem(image)
            done = start + len(real)

            # The device's own clocks for this launch, if it kept any, shared
            # out over the sequences it carried.
            ms = (f"{dev.counters[-1]['run'] / args.batch / args.clk_mhz / 1e3:.1f} ms"
                  if len(dev.counters) > timed_before else "")

            if args.score:
                for k, ids in enumerate(real):
                    i = start + k
                    target = tokens[i][PROMPT:PROMPT + args.gen]
                    got = got_all[k][:args.gen]
                    ref = ix.torch_generate(model, ids, args.gen)

                    ok = got == target
                    dev_seq += int(ok)
                    ref_seq += int(ref == target)
                    dev_tok += sum(a == b for a, b in zip(got, target))
                    ref_tok += sum(a == b for a, b in zip(ref, target))
                    agree += int(got == ref)
                    n_tok += len(target)

                    if args.show:
                        print(f"  {i:>4d}  {decode_prompt(ids):<16s} "
                              f"{spell(got):>12s} {spell(target):>12s}"
                              f"  {'' if ok else '<-- WRONG':9s} {ms:>10s}",
                              flush=True)

                if args.compare_iss and dev is not iss:
                    want_tok, want_log = iss.run_problem(image)
                    for k in range(len(real)):
                        if first_diff is None:
                            first_diff = first_logit_diff(logits_all[k],
                                                          want_log[k])
                        if (first_diff is None
                                and got_all[k][:args.gen]
                                != want_tok[k][:args.gen]):
                            # Identical logits and different tokens is
                            # impossible on the device, which argmaxes them —
                            # so this is a readback bug.
                            first_diff = ("tokens", got_all[k][:args.gen],
                                          want_tok[k][:args.gen])

            if not (args.show and args.score):
                wrong = f"  {done - dev_seq} wrong" if args.score else ""
                print(f"  {done}/{args.problems} problems "
                      f"({time.monotonic() - t0:.1f}s){wrong}  {ms}", flush=True)

        # ---- the two phase images ----
        # The same prompts and the same staged weights, an image built for one
        # half of the generation. They run last so the scored numbers above come
        # from the image this run reports on, and so a failure here cannot cost
        # the accuracy result.
        if args.split:
            image = ix.batch_prompt_image(groups[0][2], args.batch)
            dev.load_firmware(split_words["prefill"])
            prefill_ctr = dev.measure(image)
            dev.load_firmware(split_words["decode"])
            decode_ctr = dev.measure(image)
            print(f"\nsplit     : prefill-only and decode-only images, one run "
                  f"each on launch 0")
    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)
        if uart is not None:
            # Only fall back to the one-shot report when the loop collected
            # nothing — otherwise report_timing below covers every run, and
            # this would just print the last launch's counters twice.
            if isinstance(dev, BoardBackend) and not dev.counters:
                try:
                    report_run_time(uart, args.clk_mhz)
                except (ProtocolError, OSError):
                    pass
            uart.__exit__(None, None, None)

    if args.show and args.score:
        print(f"\n  {dev_seq}/{args.problems} exact "
              f"({time.monotonic() - t0:.1f}s wall clock)")

    tokens_per_seq = {"both": args.gen, "prefill": 1,
                      "decode": args.gen - 1}[args.phase]
    report_timing(dev.counters, args.clk_mhz, tokens_per_seq, args.batch,
                  args.phase)
    if prefill_ctr is not None and decode_ctr is not None:
        report_split(prefill_ctr, decode_ctr,
                     dev.counters[0] if dev.counters else None, args.gen,
                     args.clk_mhz, args.batch)

    # ---- scoring -----------------------------------------------------------
    n = args.problems
    if iss is not None and iss.commands:
        print(f"\nkernel    : {iss.commands} commands per launch "
              f"({args.phase}, batch {args.batch})")
    if not args.score:
        print(f"\nnothing scored: a {args.phase}-only image generates ids no "
              f"reference produces. Run without --phase to score.")
        return 0

    print(f"\naccuracy over {n} problems, generating {args.gen} token(s):")
    print(f"  {'':26s} {'exact-sequence':>15s} {'token':>9s}")
    print(f"  {dev.name + ', generating':26s} {100 * dev_seq / n:14.2f}% "
          f"{100 * dev_tok / n_tok:8.2f}%")
    print(f"  {'QAT model, greedy':26s} {100 * ref_seq / n:14.2f}% "
          f"{100 * ref_tok / n_tok:8.2f}%")
    print(f"\ndevice and model generated the same sequence on {agree} of {n} "
          f"problems")

    if first_diff is not None:
        if first_diff[0] == "tokens":
            print(f"\n  !! device generated {first_diff[1]} where the ISS "
                  f"generated {first_diff[2]}, on identical logits")
        else:
            pos, idx, mine, theirs = first_diff
            print(f"\n  !! logit[pos {pos}][{idx}]: device {mine}  ISS {theirs}")
        return 1
    if dev_seq < ref_seq:
        print("\n  !! the device generated worse than the model it was derived "
              "from — rerun with --compare-iss to localize")
        return 1
    print(f"\nPASSED: the {dev.name} generates what the checkpoint it was built "
          f"from generates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
