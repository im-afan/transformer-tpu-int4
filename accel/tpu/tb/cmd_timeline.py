#!/usr/bin/env python3
"""cmd_timeline.py — where a firmware kernel's clocks go, per command.

Consumes the CSV `fw_matmul_tb.sv` writes with `+CMDLOG=<path>` (see `make
fwtime`) and turns 534 rows of raw cycles into the three tables that answer
"what is this run bound by":

  1. **the split** — run clocks against MXU / VPU / DMA busy and the clocks with
     no unit busy at all. The last of those is what the CPU costs as a command
     producer, and it is the only overhead number that is not an estimate.
  2. **per phase** — the same split grouped by what the command was computing,
     decoded from its destination address. `idlec` says 4% of the run is CPU;
     this says *which* 4%, which is the difference between "spread evenly, ignore
     it" and "one phase is issue-bound, fix that one".
  3. **per command class** — mean exposed CPU clocks to get one command of each
     kind out. Exposed, not total: the CPU is free to build the next command
     while a unit runs, so only the clocks where nothing was busy are charged.
     A command pushed on top of a busy unit can therefore cost 0 here while
     costing the ~85 clocks §9.7 measured; that is the point of the metric, not
     a defect in it.

The two columns it leans on, `busy` and `gap`, are accumulated in the testbench
rather than derived from start/end here, and they reconstruct `tpu_top`'s own
perf counters exactly — which is the check that the attribution is real. The
tool verifies that on every run and says so.

    make fwtime FWPROG=adder
    python cmd_timeline.py vectors_fw/adder.timeline.csv -k adder
"""
from __future__ import annotations

import argparse
import collections
import sys

U_MXU, U_VPU, U_DMA = 0, 1, 2
UNIT = {U_MXU: "mxu", U_VPU: "vpu", U_DMA: "dma"}

MXU_GEOM, MXU_MM = 0x01, 0x02
VPU_OP, VPU_GEOM = 0x01, 0x02
DMA_MOVE = 0x01

# vpu.sv VOP_*
VOP = {0: "vecdot", 1: "add", 3: "relu", 10: "requant", 13: "vecmatmul",
       16: "dyt", 17: "quant4"}


class Cmd:
    __slots__ = ("idx", "unit", "op", "w0", "w1", "w2", "push", "start", "end",
                 "busy", "gap", "label", "phase", "waits")

    def __init__(self, row):
        (self.idx, self.unit, self.op, w0, w1, w2,
         self.push, self.start, self.end, self.busy, self.gap) = (
            int(row[0]), int(row[1]), int(row[2]), int(row[3], 16),
            int(row[4], 16), int(row[5], 16), int(row[6]), int(row[7]),
            int(row[8]), int(row[9]), int(row[10]))
        self.w0, self.w1, self.w2 = w0, w1, w2
        self.label = self.phase = "?"
        self.waits = 0

    # Field extraction, mirroring cmd_{mxu,vpu,dma}.sv.
    @property
    def dst(self):
        return (self.w0 >> 16) & 0xFFFF          # out / dst / scratchpad addr

    @property
    def vop(self):
        return (self.w0 >> 8) & 0x1F

    @property
    def dram(self):
        return self.w1 & 0x7FFFF

    @property
    def dma_write(self):
        return bool(self.w0 & (1 << 8))

    @property
    def dma_transpose(self):
        return bool(self.w0 & (1 << 9))

    def kind(self):
        """The command class, for the per-class table. Not the phase."""
        if self.unit == U_MXU:
            if self.op == MXU_GEOM:
                return "mxu geom"
            flags = []
            if self.w0 & (1 << 10):
                flags.append("tiled")
            if self.w0 & (1 << 9):
                flags.append("rq")
            if self.w0 & (1 << 8):
                flags.append("acc")
            return "mxu matmul" + ("." + ".".join(flags) if flags else "")
        if self.unit == U_VPU:
            if self.op == VPU_GEOM:
                return "vpu geom"
            return "vpu " + VOP.get(self.vop, f"op{self.vop}")
        if self.op == DMA_MOVE:
            d = "spill" if self.dma_write else "fill"
            return "dma " + d + (".t" if self.dma_transpose else "")
        return f"unit{self.unit} op{self.op}"


# =============================================================================
# Kernel-specific labelling.
#
# Decoded from the *destination address*, so the labels come out of the command
# stream itself rather than from a hand-kept list of command indices that a
# kernel edit would silently invalidate.
# =============================================================================
ADDER_ARENA = 0x2000               # fw/adder.c's staging arena, [0, ARENA)
ADDER_SP = {                       # ...and its resident tensors, above it
    0x2000: "X", 0x2800: "Q", 0x3000: "K", 0x3800: "V",
    0x4000: "KT", 0x4800: "KTP", 0x4C00: "VP", 0x5000: "MASK", 0x5400: "S",
    0x5800: "SM", 0x5C00: "P", 0x6000: "A", 0x6800: "O", 0x7000: "XO",
    0x7800: "X1", 0x8000: "H", 0xA000: "HR", 0xC000: "FF",
}
ADDER_DR = {0x00000: "X0", 0x00800: "mask", 0x00C00: "KT", 0x01400: "W_fc",
            0x01800: "logits"}
ADDER_LW = {0x0000: "Wq", 0x0800: "Wk", 0x1000: "Wv", 0x1800: "Wo",
            0x2000: "W1", 0x4000: "W2"}

# Which phase each destination tensor belongs to.
ADDER_PHASE = {
    "Q": "projections", "K": "projections", "V": "projections",
    "KT": "K transpose", "KTP": "pack K/V", "VP": "pack K/V",
    "S": "attention S", "SM": "attention mask", "P": "attention relu",
    "A": "attention A", "O": "Wo", "XO": "residual", "X1": "DyT norm1",
    "H": "FFN W1", "HR": "FFN relu", "FF": "FFN W2", "X": "DyT norm2",
    "LOG": "output head", "W": "weight fill", "T32": "?",
}


def _near(table, addr):
    """The table entry at or below `addr` — a head/column offset still lands."""
    best = None
    for base in table:
        if base <= addr and (best is None or base > best):
            best = base
    return table[best] if best is not None else None


def label_adder(cmds) -> None:
    for c in cmds:
        if c.unit == U_DMA:
            name = ADDER_DR.get(c.dram)
            if name is None:                       # inside a layer's block
                off = (c.dram - 0x02000) % 0x06000
                name = _near(ADDER_LW, off) or f"dram@{c.dram:05x}"
                c.phase = "weight fill"
            else:
                c.phase = {"X0": "operands in", "mask": "operands in",
                           "KT": "K transpose", "W_fc": "output head",
                           "logits": "logits out"}[name]
            c.label = f"{'spill' if c.dma_write else 'fill'} {name}"
            continue

        if c.dst < ADDER_ARENA:
            # The arena holds three different things and the address alone
            # cannot say which; the unit can. tpulib.h's elementwise pair stages
            # its int32 temp there, and `tpu_matmul` stages the output head's
            # int32 logits there because that tensor's destination is DRAM.
            tgt = "T32" if c.unit == U_VPU else "LOG"
        else:
            tgt = _near(ADDER_SP, c.dst)
        c.label = tgt or f"sp@{c.dst:04x}"
        c.phase = ADDER_PHASE.get(tgt, "?")

    # The elementwise ops come in pairs: a widening op into the int32 temp, then
    # the narrow that lands the real tensor. Charge the first to the second's
    # phase — on its own T32 is not a phase, it is a scratch buffer.
    for i, c in enumerate(cmds):
        if c.unit == U_VPU and c.label == "T32":
            nxt = next((n for n in cmds[i + 1:] if n.unit == U_VPU), None)
            if nxt is not None:
                c.phase = nxt.phase if nxt.label != "T32" else "?"

    # A GEOM carries no address of its own; it belongs to the matmul it precedes.
    for i, c in enumerate(cmds):
        if (c.unit == U_MXU and c.op == MXU_GEOM) or \
           (c.unit == U_VPU and c.op == VPU_GEOM):
            nxt = next((n for n in cmds[i + 1:] if n.unit == c.unit), None)
            if nxt is not None:
                c.phase, c.label = nxt.phase, "geom"


LABELLERS = {"adder": label_adder}


def label_generic(cmds) -> None:
    for c in cmds:
        c.label = f"{UNIT[c.unit]}@{c.dst:04x}"
        c.phase = c.kind()


# =============================================================================
def read_trace(path, cmds):
    """Mark each command with the number of `tpu_wait` barriers before its push.

    The trace is the kernel's own native run (`vectors_fw/<kernel>.trace.txt`),
    and its CMD records are 1:1 with the CSV rows — the testbench has already
    checked that the RTL issued exactly this stream, so aligning them by position
    is safe rather than a guess.

    This is what splits the exposed CPU cost into its two halves. A command that
    follows a barrier pays for building itself *and* for the CPU noticing that
    the unit retired; one pushed back-to-back pays only the first. The difference
    is the price of the fencing, which is the part double-buffering could get
    back.
    """
    pending = 0
    i = 0
    with open(path) as f:
        for line in f:
            tok = line.split("#", 1)[0].split()
            if not tok:
                continue
            if tok[0] == "WAIT":
                pending += 1
            elif tok[0] == "CMD":
                if i < len(cmds):
                    cmds[i].waits = pending
                pending = 0
                i += 1
    if i != len(cmds):
        print(f"  WARNING trace has {i} commands, timeline has {len(cmds)} — "
              f"barrier attribution skipped", file=sys.stderr)
        for c in cmds:
            c.waits = 0
        return False
    return True


def read_csv(path):
    cmds, meta = [], {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        if v.isdigit():
                            meta[k] = int(v)
                continue
            cmds.append(Cmd(line.split(",")))
    return cmds, meta


def bar(frac, width=24):
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-t", "--trace",
                    help="the kernel's native command trace, to split the CPU "
                         "cost into build+push and barrier-polling halves")
    ap.add_argument("-k", "--kernel", default="",
                    help=f"phase labeller ({', '.join(LABELLERS)}); "
                         f"anything else groups by command class")
    ap.add_argument("--per-command", action="store_true",
                    help="dump every command, not just the summaries")
    args = ap.parse_args()

    cmds, meta = read_csv(args.csv)
    if not cmds:
        raise SystemExit(f"{args.csv}: no commands")

    LABELLERS.get(args.kernel, label_generic)(cmds)
    have_waits = read_trace(args.trace, cmds) if args.trace else False

    # The perf counter's window, not the wall clock: `busy` is a registered
    # output, so the two differ by one and the reconstruction has to use the same
    # denominator the counters do.
    run = meta.get("perf_run") or meta.get("run_clocks", cmds[-1].end)
    tail = meta.get("tail_idle", 0)
    busy = {u: sum(c.busy for c in cmds if c.unit == u) for u in UNIT}
    idle = sum(c.gap for c in cmds) + tail
    union = run - idle                      # clocks with at least one unit up

    print(f"{args.csv}  —  {len(cmds)} commands, {run} clocks\n")
    print("  the split")
    print(f"    {'run':22s} {run:8d}  100.0%")
    for u in (U_MXU, U_VPU, U_DMA):
        print(f"    {UNIT[u] + ' busy':22s} {busy[u]:8d}  {100 * busy[u] / run:5.1f}%  "
              f"{bar(busy[u] / run)}")
    print(f"    {'any unit busy':22s} {union:8d}  {100 * union / run:5.1f}%")
    print(f"    {'CPU (no unit busy)':22s} {idle:8d}  {100 * idle / run:5.1f}%  "
          f"{bar(idle / run)}")
    overlap = sum(busy.values()) - union
    print(f"    {'overlap (2+ units)':22s} {overlap:8d}  {100 * overlap / run:5.1f}%"
          f"   <- sum of units minus the union")
    print(f"    {'  ...of which boot':22s} {cmds[0].gap:8d}  "
          f"(reset to the first command)")
    print(f"    {'  ...of which tail':22s} {tail:8d}  (last command to done)")

    print("\n  per phase, in first-appearance order")
    order, agg = [], collections.defaultdict(
        lambda: {"n": 0, "gap": 0, U_MXU: 0, U_VPU: 0, U_DMA: 0})
    for c in cmds:
        if c.phase not in agg:
            order.append(c.phase)
        a = agg[c.phase]
        a["n"] += 1
        a["gap"] += c.gap
        a[c.unit] += c.busy
    print(f"    {'phase':18s} {'cmds':>5s} {'mxu':>8s} {'vpu':>8s} {'dma':>8s} "
          f"{'cpu':>8s} {'total':>8s} {'share':>7s}")
    for ph in order:
        a = agg[ph]
        tot = a[U_MXU] + a[U_VPU] + a[U_DMA] + a["gap"]
        print(f"    {ph:18s} {a['n']:5d} {a[U_MXU]:8d} {a[U_VPU]:8d} "
              f"{a[U_DMA]:8d} {a['gap']:8d} {tot:8d} {100 * tot / run:6.1f}%")

    print("\n  per command class — `cpu` is the EXPOSED issue cost, i.e. clocks")
    print("  before this command's push with no unit busy at all")
    kagg = collections.defaultdict(lambda: {"n": 0, "gap": 0, "busy": 0})
    for c in cmds:
        a = kagg[c.kind()]
        a["n"] += 1
        a["gap"] += c.gap
        a["busy"] += c.busy
    print(f"    {'class':22s} {'cmds':>5s} {'unit busy':>10s} {'per cmd':>8s} "
          f"{'cpu':>8s} {'per cmd':>8s}")
    for k in sorted(kagg, key=lambda k: -kagg[k]["gap"]):
        a = kagg[k]
        print(f"    {k:22s} {a['n']:5d} {a['busy']:10d} {a['busy'] / a['n']:8.1f} "
              f"{a['gap']:8d} {a['gap'] / a['n']:8.1f}")

    if have_waits:
        after = [c for c in cmds[1:] if c.waits]        # boot is not a barrier
        b2b = [c for c in cmds[1:] if not c.waits]
        n_bar = sum(c.waits for c in cmds)
        print(f"\n  what the CPU's {idle} clocks are — {n_bar} tpu_wait barriers "
              f"over {len(cmds)} commands")
        if b2b:
            mean_b2b = sum(c.gap for c in b2b) / len(b2b)
            print(f"    {'pushed back-to-back':26s} {len(b2b):4d} cmds  "
                  f"{sum(c.gap for c in b2b):7d} clocks  {mean_b2b:6.1f}/cmd  "
                  f"<- exposed build+push only")
        else:
            mean_b2b = 0.0
        if after:
            mean_after = sum(c.gap for c in after) / len(after)
            print(f"    {'first after a barrier':26s} {len(after):4d} cmds  "
                  f"{sum(c.gap for c in after):7d} clocks  {mean_after:6.1f}/cmd  "
                  f"<- the same, plus polling the retired counter")
            if b2b:
                extra = (mean_after - mean_b2b) * len(after)
                print(f"    {'  the fencing itself':26s} {'':4s}       "
                      f"{extra:7.0f} clocks  {mean_after - mean_b2b:6.1f}/barrier"
                      f"  = {100 * extra / run:.1f}% of the run")
        print(f"    {'boot (reset to cmd 0)':26s} {'':4s}       "
              f"{cmds[0].gap:7d} clocks")
        print(f"    {'tail (last cmd to done)':26s} {'':4s}       {tail:7d} clocks")

    if args.per_command:
        print("\n  every command")
        print(f"    {'idx':>4s} {'unit':4s} {'class':22s} {'label':12s} "
              f"{'phase':18s} {'push':>8s} {'busy':>6s} {'cpu':>6s}")
        for c in cmds:
            print(f"    {c.idx:4d} {UNIT[c.unit]:4s} {c.kind():22s} {c.label:12s} "
                  f"{c.phase:18s} {c.push:8d} {c.busy:6d} {c.gap:6d} "
                  f"{'W' * c.waits}")

    print(f"\n  check: unit busy + CPU = {sum(busy.values()) - overlap + idle} "
          f"vs run {run}; compare `counters:` above — mxu/vpu/dma/idlec must match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
