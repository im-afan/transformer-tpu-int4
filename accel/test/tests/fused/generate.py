#!/usr/bin/env python3
"""fused: one tpu_matmul_wide_fused, DRAM to DRAM.

The same shape wide.c runs — more rows than one panel of the arena, an odd
number of column blocks, a ragged last one — with the residual add and the
activation folded into the matmul's visit to each output block. The golden is a
plain Python matmul followed by the same narrows the VPU does, so a fused block
that spills before it activates, or that stages the wrong block of `add`, fails
here rather than three layers deep in `infer`.

`--act` picks the activation: dyt reads the add tensor a second time (the
double residual, one VPU pass), relu and requant are unary, none leaves the add
alone. `--no-add` drops the add pass, which is the FFN's relu-only site.

    python accel/test/tests/fused/generate.py -b iss
    python accel/test/tests/fused/generate.py -b iss --sweep
    python accel/test/tests/fused/generate.py -b rtl --act relu --no-add
    python accel/test/tests/fused/generate.py -b rtl --transpose -M 64 -N 100
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (RQ_ONE, TPU_N, AddressMap, Case,  # noqa: E402
                              VectorGenerator, a_val, dyt, fit_rq, i4_row,
                              narrow, put_rowmajor_i4, w_val)

DRAM_BYTES = 1 << 19

ACTS = ("none", "relu", "dyt", "requant")
ACT_MACRO = {"none": "TPU_ACT_NONE", "relu": "TPU_V_RELU",
             "dyt": "TPU_V_DYT", "requant": "TPU_V_REQUANT"}


def add_val(m: int, n: int) -> int:
    """The tensor the matmul's output is added to — a residual stream, not a
    seed the kernel wrote: it is read and never written."""
    return ((m * 7 + n * 5) % 16) - 8


class FusedVectors(VectorGenerator):
    def __init__(self, rows: int = 20, depth: int = 512, cols: int = 52,
                 transpose: bool = False, act: str = "dyt", add: bool = True,
                 arena_banks: int = 4, prefetch: bool = True):
        if depth % TPU_N:
            raise SystemExit(f"K = {depth} must be a multiple of {TPU_N} — the "
                             f"contraction is taken in one dispatch")
        if cols % 2:
            raise SystemExit(f"N = {cols} must be even — an int4 row is two "
                             f"elements per byte")
        if act not in ACTS:
            raise SystemExit(f"--act must be one of {', '.join(ACTS)}")
        if not add and act == "none":
            raise SystemExit("--no-add with --act none is a plain matmul; "
                             "that is what tests/wide covers")
        self.rows, self.k, self.n = rows, depth, cols
        self.transpose, self.act, self.add = transpose, act, add
        self.arena_banks, self.prefetch = arena_banks, prefetch

        self.map = AddressMap(align=64, limit=DRAM_BYTES, what="fused")
        self.map.alloc("DR_A", rows * i4_row(self.k))
        self.map.alloc("DR_B", (self.n * i4_row(self.k) if transpose
                                else self.k * i4_row(self.n)))
        self.map.alloc("DR_ADD", rows * i4_row(self.n))
        self.map.alloc("DR_C", rows * i4_row(self.n))

        cells = [(i, j) for i in range(rows) for j in range(self.n)]
        acc32 = {(i, j): sum(a_val(i, t) * w_val(t, j) for t in range(self.k))
                 for i, j in cells}
        self.rq_c = fit_rq(acc32.values(), "RQ_C")

        # Each step's word comes from the values that step is about to narrow,
        # the same rule the unfused sequence's would.
        tile = {c: narrow(acc32[c], self.rq_c) for c in cells}
        if add:
            summed = {c: tile[c] + add_val(*c) for c in cells}
            self.rq_add = fit_rq(summed.values(), "RQ_ADD")
            tile = {c: narrow(summed[c], self.rq_add) for c in cells}
        else:
            self.rq_add = RQ_ONE

        if act == "dyt":
            pre = {c: tile[c] + add_val(*c) for c in cells}
            self.rq_act = fit_rq(pre.values(), "RQ_ACT")
            tile = {c: dyt(pre[c], self.rq_act) for c in cells}
        elif act in ("relu", "requant"):
            pre = ({c: max(tile[c], 0) for c in cells} if act == "relu"
                   else dict(tile))
            self.rq_act = fit_rq(pre.values(), "RQ_ACT")
            tile = {c: narrow(pre[c], self.rq_act) for c in cells}
        else:
            self.rq_act = RQ_ONE

        self.out = tile
        self.defines = {"M": rows, "K": self.k, "N": self.n,
                        "TRANSPOSE": int(transpose),
                        "ADD_OP": "TPU_V_ADD" if add else "TPU_ACT_NONE",
                        "ACT_OP": ACT_MACRO[act],
                        "ARENA_BANKS": f"{arena_banks}u", "SP_ARENA": "0x0000u",
                        "TPU_WGT_PREFETCH": int(prefetch),
                        "RQ_C": f"{self.rq_c}u", "RQ_ADD": f"{self.rq_add}u",
                        "RQ_ACT": f"{self.rq_act}u", **self.map.defines()}

    def dma_clocks(self) -> int:
        """A loose bound on what the run costs: every byte, a clock on a fill
        and two on a spill, with B read once per row block pessimistically."""
        panels = (self.rows + TPU_N - 1) // TPU_N
        reads_add = self.add or self.act == "dyt"
        fill = (self.rows * i4_row(self.k)
                + panels * self.k * i4_row(self.n)
                + (self.rows * i4_row(self.n) if reads_add else 0))
        return fill + 2 * self.rows * i4_row(self.n)

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.map["DR_A"], self.rows, self.k, a_val)
        if self.transpose:
            put_rowmajor_i4(img, self.map["DR_B"], self.n, self.k,
                            lambda j, t: w_val(t, j))
        else:
            put_rowmajor_i4(img, self.map["DR_B"], self.k, self.n, w_val)
        put_rowmajor_i4(img, self.map["DR_ADD"], self.rows, self.n, add_val)
        put_rowmajor_i4(img, self.map["DR_C"], self.rows, self.n,
                        lambda i, j: 0)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.map["DR_C"], self.rows, self.n,
                        lambda i, j: self.out[(i, j)])
        yield Case(name=f"{self.rows}x{self.k} @ {self.k}x{self.n}, fused"
                        + (", transposed" if self.transpose else "")
                        + (", + add" if self.add else "")
                        + (f", {self.act}" if self.act != "none" else ""),
                   golden=golden,
                   check_ranges=[(self.map["DR_C"],
                                  self.rows * i4_row(self.n))])


def program(backend, **kw):
    return TPUProgram(os.path.join(HERE, "fused.c"), backend, FusedVectors(**kw))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-M", "--rows", type=int, default=20,
                    help="more rows than one panel of the arena, so the row "
                         "panel loop runs twice")
    ap.add_argument("-K", "--depth", type=int, default=512,
                    help=f"the contraction, a multiple of {TPU_N}")
    ap.add_argument("-N", "--cols", type=int, default=52,
                    help="an odd number of column blocks, the last one ragged")
    ap.add_argument("--transpose", action="store_true",
                    help="store B as [N][K] and run the matmul with TPU_MM_T")
    ap.add_argument("--act", default="dyt", choices=ACTS,
                    help="the activation folded onto the block (default dyt, "
                         "which reads the add tensor a second time)")
    ap.add_argument("--no-add", dest="add", action="store_false",
                    help="drop the add pass — the FFN's relu-only site")
    ap.add_argument("--arena-banks", type=int, default=4,
                    help="scratchpad banks tpulib.h may spend: one each for A "
                         "and the C region and two for B, and 3 single-buffers")
    ap.add_argument("--single-buffer", action="store_true",
                    help="build with TPU_WGT_PREFETCH=0 — the A/B, same golden")
    ap.add_argument("--sweep", action="store_true",
                    help="run every add/activation combination in turn")
    args = ap.parse_args()

    combos = ([(a, ad) for ad in (True, False) for a in ACTS
               if ad or a != "none"] if args.sweep else [(args.act, args.add)])
    failed = 0

    for act, add in combos:
        gen = FusedVectors(rows=args.rows, depth=args.depth, cols=args.cols,
                           transpose=args.transpose, act=act, add=add,
                           arena_banks=args.arena_banks,
                           prefetch=not args.single_buffer)
        # The tb clock is 10 ns and a DMA byte is a clock (two on a spill), so a
        # shape past the default needs a watchdog that follows it.
        watchdog = max(2_000_000, 400 * gen.dma_clocks())
        backend = backend_from_args(args, watchdog_ns=watchdog)
        try:
            prog = TPUProgram(os.path.join(HERE, "fused.c"), backend, gen)
            prog.run_program(limit=args.cases)
            failed += report(prog, args.clk_mhz)
        finally:
            backend.close()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
