#!/usr/bin/env python3
"""matmul: C = requant(A @ W), the block grid walked in firmware.

The shape, the operand addresses and the requant word are all computed here and
handed to the compiler as -D, so the numbers the kernel ran are the numbers the
golden was computed from. The golden itself is a plain Python matmul over the
same int4 codes the image holds.

    python accel/test/tests/matmul/generate.py -b iss
    python accel/test/tests/matmul/generate.py -b rtl --ktiles 16 --ntiles 16
    python accel/test/tests/matmul/generate.py -b rtl -M 32 --ktiles 8
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (TPU_N, AddressMap, Case,       # noqa: E402
                              VectorGenerator, a_val, fit_rq, i4_row, narrow,
                              put_rowmajor_i4, w_val)


class MatmulVectors(VectorGenerator):
    def __init__(self, rows: int = 8, ktiles: int = 4, ntiles: int = 2):
        self.rows = rows
        self.k = ktiles * TPU_N
        self.n = ntiles * TPU_N

        self.map = AddressMap(what="matmul")
        self.map.alloc("A_ADDR", rows * i4_row(self.k))
        self.map.alloc("W_ADDR", self.k * i4_row(self.n))
        self.map.alloc("C_ADDR", rows * i4_row(self.n))

        self.acc = [[sum(a_val(i, t) * w_val(t, j) for t in range(self.k))
                     for j in range(self.n)] for i in range(rows)]
        self.rq_c = fit_rq((v for row in self.acc for v in row), "RQ_C")

        self.defines = {"M": rows, "KTILES": ktiles, "NTILES": ntiles,
                        "RQ_C": f"{self.rq_c}u", **self.map.defines()}

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.map["A_ADDR"], self.rows, self.k, a_val)
        put_rowmajor_i4(img, self.map["W_ADDR"], self.k, self.n, w_val)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.map["C_ADDR"], self.rows, self.n,
                        lambda i, j: narrow(self.acc[i][j], self.rq_c))
        yield Case(name=f"{self.rows}x{self.k} @ {self.k}x{self.n}",
                   golden=golden,
                   check_ranges=[(self.map["C_ADDR"], self.rows * i4_row(self.n))])


def program(backend, rows: int = 8, ktiles: int = 4, ntiles: int = 2):
    return TPUProgram(os.path.join(HERE, "matmul.c"), backend,
                      MatmulVectors(rows, ktiles, ntiles))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-M", "--rows", type=int, default=8, help="token rows")
    ap.add_argument("--ktiles", type=int, default=4,
                    help=f"array tiles along the contraction (x{TPU_N})")
    ap.add_argument("--ntiles", type=int, default=2,
                    help=f"array tiles across the output (x{TPU_N})")
    args = ap.parse_args()

    prog = program(backend_from_args(args), args.rows, args.ktiles, args.ntiles)
    prog.run_program(limit=args.cases)
    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
