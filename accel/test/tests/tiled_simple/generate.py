#!/usr/bin/env python3
"""tiled_simple: C = requant(A @ W), tiled by hand out of raw tpu.h commands.

Every operand lives in DRAM and only one tile of each is on chip at a time, so
the run is the tiling cost plus the array's — the floor `tiled`'s tpulib.h block
loops are measured against. The golden is a plain Python matmul over the same
int4 codes the image holds.

    python accel/test/tests/tiled_simple/generate.py -b iss
    python accel/test/tests/tiled_simple/generate.py -b rtl -M 32 -K 64 -N 32
    python accel/test/tests/tiled_simple/generate.py -b rtl -M 12 -K 256 -N 24
    python accel/test/tests/tiled_simple/generate.py -b iss -M 128 -K 128 -N 512 -O
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (TPU_BANK_BYTES, TPU_N,          # noqa: E402
                              TPU_SPAD_BYTES, AddressMap, Case,
                              VectorGenerator, a_val, fit_rq, i4_row, narrow,
                              put_rowmajor_i4, w_val)

# The operands are DRAM-resident, so the map is bounded by the 512 KB part; the
# scratchpad holds three staging regions and nothing else.
DRAM_BYTES = 1 << 19


def bank_up(nbytes: int) -> int:
    """What a tensor costs the scratchpad: whole banks, never zero."""
    step = max(nbytes, 1)
    return (step + TPU_BANK_BYTES - 1) // TPU_BANK_BYTES * TPU_BANK_BYTES


def fit_super_rows(rows: int, depth: int) -> int:
    """The most rows of A the optimized kernel can stage at once.

    It stages [super_rows][K] of A and the matching [super_rows][TPU_N] of C,
    against one [K][TPU_N] column block of W, and that is the whole scratchpad.
    A whole number of array tiles, so a partial last row block stays the only
    partial thing in the kernel.
    """
    w_cost = bank_up(depth * i4_row(TPU_N))
    most = (rows + TPU_N - 1) // TPU_N * TPU_N
    for super_rows in range(most, 0, -TPU_N):
        if (bank_up(super_rows * i4_row(depth)) + w_cost
                + bank_up(super_rows * i4_row(TPU_N))) <= TPU_SPAD_BYTES:
            return super_rows
    raise SystemExit(f"tiled_simple: a single {TPU_N}-row block of a "
                     f"contraction of {depth} does not fit the scratchpad")


class TiledSimpleVectors(VectorGenerator):
    def __init__(self, rows: int = 8, depth: int = 32, cols: int = 16,
                 optimized: bool = False):
        # A stride is elements/2 bytes and must be a whole scratchpad word, so
        # both of the strided extents are whole array tiles. Rows are not.
        for name, val in (("K", depth), ("N", cols)):
            if val % TPU_N:
                raise SystemExit(f"{name} = {val} must be a multiple of "
                                 f"{TPU_N} — a row stride of {val // 2} bytes "
                                 f"is not a whole scratchpad word")
        self.rows, self.k, self.n = rows, depth, cols
        self.optimized = optimized
        self.super_rows = fit_super_rows(rows, depth) if optimized else TPU_N

        self.dram = AddressMap(align=64, limit=DRAM_BYTES, what="tiled_simple")
        self.dram.alloc("DR_A", rows * i4_row(self.k))
        self.dram.alloc("DR_W", self.k * i4_row(self.n))
        self.dram.alloc("DR_C", rows * i4_row(self.n))

        # A bank boundary between the regions: the MXU reads A, B and C on the
        # same clock. A and C are sized for a whole row superblock, which is one
        # block unless --optimized asked for more.
        self.spad = AddressMap(what="tiled_simple staging")
        self.spad.alloc("SP_A", self.super_rows * i4_row(self.k))
        self.spad.alloc("SP_W", self.k * i4_row(TPU_N))
        self.spad.alloc("SP_C", self.super_rows * i4_row(TPU_N))

        self.acc = [[sum(a_val(i, t) * w_val(t, j) for t in range(self.k))
                     for j in range(self.n)] for i in range(rows)]
        self.rq_c = fit_rq((v for row in self.acc for v in row), "RQ_C")

        self.defines = {"M": rows, "K": self.k, "N": self.n,
                        "SUPER_ROWS": self.super_rows,
                        "TILED_OPTIMIZED": int(optimized),
                        "RQ_C": f"{self.rq_c}u",
                        **self.dram.defines(), **self.spad.defines()}

    def w_passes(self) -> int:
        """How many times the kernel reads the whole of W: once per row
        superblock. This is what the two kernels differ by."""
        return (self.rows + self.super_rows - 1) // self.super_rows

    def dma_bytes(self) -> tuple:
        """(fill, spill) bytes the kernel moves."""
        fill = (self.rows * i4_row(self.k)
                + self.w_passes() * self.k * i4_row(self.n))
        return fill, self.rows * i4_row(self.n)

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.dram["DR_A"], self.rows, self.k, a_val)
        put_rowmajor_i4(img, self.dram["DR_W"], self.k, self.n, w_val)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.dram["DR_C"], self.rows, self.n,
                        lambda i, j: narrow(self.acc[i][j], self.rq_c))
        yield Case(name=f"{self.rows}x{self.k} @ {self.k}x{self.n}, tiled",
                   golden=golden,
                   check_ranges=[(self.dram["DR_C"],
                                  self.rows * i4_row(self.n))])


def program(backend, rows: int = 8, depth: int = 32, cols: int = 16,
            optimized: bool = False):
    return TPUProgram(os.path.join(HERE, "tiled_simple.c"), backend,
                      TiledSimpleVectors(rows, depth, cols, optimized))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-M", "--rows", type=int, default=8,
                    help="token rows; a partial last row block is fine")
    ap.add_argument("-K", "--depth", type=int, default=32,
                    help=f"the contraction, a multiple of {TPU_N}")
    ap.add_argument("-N", "--cols", type=int, default=16,
                    help=f"output columns, a multiple of {TPU_N}")
    ap.add_argument("-O", "--optimized", action="store_true",
                    help="run tiled_matmul_optimized: stage as many rows of A "
                         "as the scratchpad holds, so a column block of W is "
                         "filled once per superblock instead of once per block")
    args = ap.parse_args()

    # The tb clock is 10 ns and a DMA byte is a clock (two on a spill), so the
    # watchdog has to follow the shape or a large one times out mid-run.
    gen = TiledSimpleVectors(args.rows, args.depth, args.cols, args.optimized)
    fill, spill = gen.dma_bytes()
    watchdog = max(2_000_000, 80 * (fill + 2 * spill))

    prog = TPUProgram(os.path.join(HERE, "tiled_simple.c"),
                      backend_from_args(args, watchdog_ns=watchdog), gen)
    prog.run_program(limit=args.cases)

    if prog.benchmark(clk_mhz=12):
        print(f"{gen.super_rows}-row blocks of A staged at once, "
              f"{gen.w_passes()} pass(es) over W")
        print(f"moved {fill} to spad, {spill} to DRAM")
        print(f"expected DMA clocks: {fill + 2 * spill}")

    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
