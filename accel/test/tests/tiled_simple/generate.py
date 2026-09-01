#!/usr/bin/env python3
"""tiled_simple: C = requant(A @ W), tiled by hand out of raw tpu.h commands.

Every operand lives in DRAM. The kernel stages one column block of W and as
many rows of A and C as the rest of the scratchpad holds, so the run is the
tiling cost plus the array's — the floor `tiled`'s tpulib.h block loops are
measured against. The golden is a plain Python matmul over the same int4 codes
the image holds.

`--transpose` stores W as [N][K] and runs the matmul with TPU_MM_T; `--acc`
seeds DR_C and runs with TPU_MM_ACC, so C = clip4(requant(A @ W) + C_old).

    python accel/test/tests/tiled_simple/generate.py -b iss
    python accel/test/tests/tiled_simple/generate.py -b rtl -M 32 -K 64 -N 32
    python accel/test/tests/tiled_simple/generate.py -b rtl -M 12 -K 256 -N 24
    python accel/test/tests/tiled_simple/generate.py -b iss -M 12 -K 64 -N 16 --transpose --acc
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (Q4_MAX, Q4_MIN, TPU_BANK_BYTES,  # noqa: E402
                              TPU_N, TPU_SPAD_BYTES, AddressMap, Case,
                              VectorGenerator, a_val, fit_rq, i4_row, narrow,
                              put_rowmajor_i4, w_val)

# The operands are DRAM-resident, so the map is bounded by the 512 KB part; the
# scratchpad holds three staging regions and nothing else.
DRAM_BYTES = 1 << 19


def c_val(m: int, n: int) -> int:
    """What DR_C already holds, which --acc accumulates onto."""
    return ((m * 7 + n * 11) % 9) - 4


def bank_up(nbytes: int) -> int:
    """The kernel's BANK_ROUND: whole banks, and zero stays zero."""
    return (nbytes + TPU_BANK_BYTES - 1) // TPU_BANK_BYTES * TPU_BANK_BYTES


def fit_super_rows(rows: int, depth: int) -> int:
    """tiled_simple.c's SUPER_ROWS, in Python.

    The kernel spends one bank-aligned region on a [K][TPU_N] block of W and
    the remainder on a row superblock of A and C, minus one bank for the round
    up to C's own bank. A whole number of array tiles, so a partial last row
    block stays the only partial thing in the kernel.
    """
    sp_a = bank_up(depth * i4_row(TPU_N))
    row_cost = i4_row(depth) + i4_row(TPU_N)
    spare = TPU_SPAD_BYTES - sp_a - TPU_BANK_BYTES
    fits = max(spare, 0) // row_cost // TPU_N * TPU_N
    if fits < TPU_N:
        raise SystemExit(f"tiled_simple: a single {TPU_N}-row block of a "
                         f"contraction of {depth} does not fit the scratchpad")
    need = (rows + TPU_N - 1) // TPU_N * TPU_N
    return min(fits, need)


class TiledSimpleVectors(VectorGenerator):
    def __init__(self, rows: int = 8, depth: int = 32, cols: int = 16,
                 transpose: bool = False, acc: bool = False):
        # A stride is elements/2 bytes and must be a whole scratchpad word, so
        # both of the strided extents are whole array tiles. Rows are not.
        for name, val in (("K", depth), ("N", cols)):
            if val % TPU_N:
                raise SystemExit(f"{name} = {val} must be a multiple of "
                                 f"{TPU_N} — a row stride of {val // 2} bytes "
                                 f"is not a whole scratchpad word")
        self.rows, self.k, self.n = rows, depth, cols
        self.transpose, self.acc = transpose, acc
        self.super_rows = fit_super_rows(rows, depth)

        self.dram = AddressMap(align=64, limit=DRAM_BYTES, what="tiled_simple")
        self.dram.alloc("DR_A", rows * i4_row(self.k))
        self.dram.alloc("DR_W", (self.n * i4_row(self.k) if transpose
                                 else self.k * i4_row(self.n)))
        self.dram.alloc("DR_C", rows * i4_row(self.n))

        # The same product either way: --transpose swaps W's storage, not the
        # index function, which is what the kernel's TPU_MM_T says.
        self.acc32 = [[sum(a_val(i, t) * w_val(t, j) for t in range(self.k))
                       for j in range(self.n)] for i in range(rows)]
        self.rq_c = fit_rq((v for row in self.acc32 for v in row), "RQ_C")

        self.defines = {"SPAD_SIZE": f"{TPU_SPAD_BYTES}u",
                        "M": rows, "K": self.k, "N": self.n,
                        "TRANSPOSE": int(transpose), "ACC": int(acc),
                        "RQ_C": f"{self.rq_c}u",
                        **self.dram.defines()}

    def result(self, i: int, j: int) -> int:
        v = narrow(self.acc32[i][j], self.rq_c)
        if self.acc:
            v = max(Q4_MIN, min(Q4_MAX, v + c_val(i, j)))
        return v

    def w_passes(self) -> int:
        """How many times the kernel reads the whole of W: once per row
        superblock. This is what the scratchpad buys."""
        return (self.rows + self.super_rows - 1) // self.super_rows

    def dma_bytes(self) -> tuple:
        """(fill, spill) bytes the kernel moves."""
        c_bytes = self.rows * i4_row(self.n)
        fill = (self.rows * i4_row(self.k)
                + self.w_passes() * self.k * i4_row(self.n)
                + (c_bytes if self.acc else 0))
        return fill, c_bytes

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.dram["DR_A"], self.rows, self.k, a_val)
        if self.transpose:
            put_rowmajor_i4(img, self.dram["DR_W"], self.n, self.k,
                            lambda j, t: w_val(t, j))
        else:
            put_rowmajor_i4(img, self.dram["DR_W"], self.k, self.n, w_val)
        # --acc reads C before it writes it, so the image has to be dense over
        # it too; without it the kernel only writes.
        put_rowmajor_i4(img, self.dram["DR_C"], self.rows, self.n,
                        c_val if self.acc else (lambda i, j: 0))
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.dram["DR_C"], self.rows, self.n,
                        self.result)
        yield Case(name=f"{self.rows}x{self.k} @ {self.k}x{self.n}, tiled"
                        + (", transposed" if self.transpose else "")
                        + (", accumulating" if self.acc else ""),
                   golden=golden,
                   check_ranges=[(self.dram["DR_C"],
                                  self.rows * i4_row(self.n))])


def program(backend, rows: int = 8, depth: int = 32, cols: int = 16,
            transpose: bool = False, acc: bool = False):
    return TPUProgram(os.path.join(HERE, "tiled_simple.c"), backend,
                      TiledSimpleVectors(rows, depth, cols, transpose, acc))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-M", "--rows", type=int, default=8,
                    help="token rows; a partial last row block is fine")
    ap.add_argument("-K", "--depth", type=int, default=32,
                    help=f"the contraction, a multiple of {TPU_N}")
    ap.add_argument("-N", "--cols", type=int, default=16,
                    help=f"output columns, a multiple of {TPU_N}")
    ap.add_argument("--transpose", action="store_true",
                    help="store W as [N][K] and run the matmul with TPU_MM_T")
    ap.add_argument("--acc", action="store_true",
                    help="seed DR_C and run with TPU_MM_ACC, so the kernel "
                         "fills each C block before it computes into it")
    args = ap.parse_args()

    # The tb clock is 10 ns and a DMA byte is a clock (two on a spill), so the
    # watchdog has to follow the shape or a large one times out mid-run.
    gen = TiledSimpleVectors(args.rows, args.depth, args.cols, args.transpose,
                             args.acc)
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
