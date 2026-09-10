#!/usr/bin/env python3
"""wide: one tpu_matmul_wide, DRAM to DRAM.

The default shape is what the double buffer was written against — more rows
than one panel of the arena, so the row panel loop repeats, an odd number of
column blocks so the parity has to reset for the second panel, and a ragged
last block. The golden is a plain Python matmul over the same int4 codes the
image holds.

`--transpose` stores B as [N][K] and runs it with TPU_MM_T; `--acc` seeds DR_C
and runs with TPU_MM_ACC, so C = clip4(requant(A @ B) + C_old). Both are paths
tpu_matmul_wide did not used to have.

`--benchmark` writes no tensor data and checks nothing — the image is only
the firmware, the run is only the run, and all that comes back is the perf
counters. The clocks do not depend on the data, so it is the same measurement
without the load and read-back on the wire.

`--single-buffer` compiles the prefetch out (`TPU_WGT_PREFETCH=0`) and
`--arena-banks 3` leaves no room for the second half, so it single-buffers on
its own — the same problem and the same golden three ways.

    python accel/test/tests/wide/generate.py -b iss
    python accel/test/tests/wide/generate.py -b iss --transpose --acc
    python accel/test/tests/wide/generate.py -b rtl --single-buffer
    python accel/test/tests/wide/generate.py -b rtl -M 64 -K 128 -N 100
    python accel/test/tests/wide/generate.py -b board -p COM5 --benchmark
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (Q4_MAX, Q4_MIN, TPU_N, AddressMap,  # noqa: E402
                              Case, VectorGenerator, a_val, fit_rq, i4_row,
                              narrow, put_rowmajor_i4, w_val)

# DRAM to DRAM, so the map is bounded by the 512 KB part; the arena is the only
# scratchpad this pass spends.
DRAM_BYTES = 1 << 19


def c_seed(m: int, n: int) -> int:
    """What DR_C already holds, which --acc accumulates onto."""
    return ((m * 13 + n * 2) % 16) - 8


def clip4(v: int) -> int:
    return max(Q4_MIN, min(Q4_MAX, v))


class WideVectors(VectorGenerator):
    def __init__(self, rows: int = 20, depth: int = 512, cols: int = 52,
                 transpose: bool = False, acc: bool = False,
                 arena_banks: int = 4, prefetch: bool = True,
                 benchmark: bool = False):
        if depth % TPU_N:
            raise SystemExit(f"K = {depth} must be a multiple of {TPU_N} — the "
                             f"contraction is taken in one dispatch")
        if cols % 2:
            raise SystemExit(f"N = {cols} must be even — an int4 row is two "
                             f"elements per byte")
        self.rows, self.k, self.n = rows, depth, cols
        self.transpose, self.acc = transpose, acc
        self.arena_banks, self.prefetch = arena_banks, prefetch
        self.benchmark = benchmark

        self.map = AddressMap(align=64, limit=DRAM_BYTES, what="wide")
        self.map.alloc("DR_A", rows * i4_row(self.k))
        self.map.alloc("DR_B", (self.n * i4_row(self.k) if transpose
                                else self.k * i4_row(self.n)))
        self.map.alloc("DR_C", rows * i4_row(self.n))

        # The same product either way: --transpose swaps B's storage, not the
        # index function, which is what the kernel's TPU_MM_T says.
        # --benchmark has no golden to fit, so one row stands in for the shape:
        # the requant word does not change what the run costs.
        rq_rows = 1 if benchmark else rows
        self.acc32 = [[sum(a_val(i, t) * w_val(t, j) for t in range(self.k))
                       for j in range(self.n)] for i in range(rq_rows)]
        self.rq_c = fit_rq((v for row in self.acc32 for v in row), "RQ_C")

        self.defines = {"M": rows, "K": self.k, "N": self.n,
                        "TRANSPOSE": int(transpose), "ACC": int(acc),
                        "ARENA_BANKS": f"{arena_banks}u", "SP_ARENA": "0x0000u",
                        "TPU_WGT_PREFETCH": int(prefetch),
                        "RQ_C": f"{self.rq_c}u", **self.map.defines()}

    def result(self, i: int, j: int) -> int:
        v = narrow(self.acc32[i][j], self.rq_c)
        return clip4(v + c_seed(i, j)) if self.acc else v

    def dma_clocks(self) -> int:
        """A loose bound on what the run costs: every byte, a clock on a fill
        and two on a spill, with B read once per row panel and the panel count
        unknown here, so pessimistically once per row block."""
        panels = (self.rows + TPU_N - 1) // TPU_N
        fill = (self.rows * i4_row(self.k)
                + panels * self.k * i4_row(self.n)
                + (self.rows * i4_row(self.n) if self.acc else 0))
        return fill + 2 * self.rows * i4_row(self.n)

    def writable_ranges(self) -> list:
        """--benchmark checks nothing, so every byte the kernel touches is
        scratch as far as the stray-write check is concerned."""
        return [(0, self.map.next)] if self.benchmark else []

    def static(self) -> dict:
        if self.benchmark:
            return {}
        img: dict = {}
        put_rowmajor_i4(img, self.map["DR_A"], self.rows, self.k, a_val)
        if self.transpose:
            put_rowmajor_i4(img, self.map["DR_B"], self.n, self.k,
                            lambda j, t: w_val(t, j))
        else:
            put_rowmajor_i4(img, self.map["DR_B"], self.k, self.n, w_val)
        # --acc reads C before it writes it, so the image has to be dense over
        # it too; without it the kernel only writes.
        put_rowmajor_i4(img, self.map["DR_C"], self.rows, self.n,
                        c_seed if self.acc else (lambda i, j: 0))
        return img

    def cases(self):
        if self.benchmark:
            yield Case(name=f"{self.rows}x{self.k} @ {self.k}x{self.n}, "
                            f"wide, timing only")
            return
        golden: dict = {}
        put_rowmajor_i4(golden, self.map["DR_C"], self.rows, self.n,
                        self.result)
        yield Case(name=f"{self.rows}x{self.k} @ {self.k}x{self.n}, wide"
                        + (", transposed" if self.transpose else "")
                        + (", accumulating" if self.acc else ""),
                   golden=golden,
                   check_ranges=[(self.map["DR_C"],
                                  self.rows * i4_row(self.n))])


def program(backend, **kw):
    return TPUProgram(os.path.join(HERE, "wide.c"), backend, WideVectors(**kw))


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
    ap.add_argument("--acc", action="store_true",
                    help="seed DR_C and run with TPU_MM_ACC, so each C column "
                         "block is filled from DRAM before it is computed into")
    ap.add_argument("--arena-banks", type=int, default=4,
                    help="scratchpad banks tpulib.h may spend: one each for A "
                         "and C and two for B is what the prefetch costs, and "
                         "3 single-buffers on its own")
    ap.add_argument("--benchmark", action="store_true",
                    help="write no tensor data and check no result — load the "
                         "firmware, run it, read the perf counters back")
    ap.add_argument("--single-buffer", action="store_true",
                    help="build with TPU_WGT_PREFETCH=0 — the A/B, same golden")
    args = ap.parse_args()

    gen = WideVectors(rows=args.rows, depth=args.depth, cols=args.cols,
                      transpose=args.transpose, acc=args.acc,
                      arena_banks=args.arena_banks,
                      prefetch=not args.single_buffer,
                      benchmark=args.benchmark)
    # The tb clock is 10 ns and a DMA byte is a clock (two on a spill), so a
    # shape past the default needs a watchdog that follows it.
    watchdog = max(2_000_000, 400 * gen.dma_clocks())

    prog = TPUProgram(os.path.join(HERE, "wide.c"),
                      backend_from_args(args, watchdog_ns=watchdog), gen)
    prog.run_program(limit=args.cases)
    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
