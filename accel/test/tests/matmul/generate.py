#!/usr/bin/env python3
"""matmul: C = requant(A @ W), the block grid walked in firmware.

The golden is a plain Python matmul over the same int4 codes the image holds —
nothing the kernel, the ISS or the RTL had a hand in.

    python accel/test/tests/matmul/generate.py -b iss
    python accel/test/tests/matmul/generate.py -b rtl --ktiles 16 --ntiles 16
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (Case, VectorGenerator, a_val,  # noqa: E402
                              i4_row, narrow, put_rowmajor_i4, rq_word, w_val)

ARRAY_N = 8
A_ADDR, W_ADDR, C_ADDR = 0x0000, 0x2000, 0x4000
RQ_C = rq_word(1, 4)                  # matmul.c's RQ_C literal


class MatmulVectors(VectorGenerator):
    def __init__(self, rows: int = 8, ktiles: int = 4, ntiles: int = 2):
        self.rows = rows
        self.k = ktiles * ARRAY_N
        self.n = ntiles * ARRAY_N
        self.defines = {"M": rows, "KTILES": ktiles, "NTILES": ntiles}

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, A_ADDR, self.rows, self.k, a_val)
        put_rowmajor_i4(img, W_ADDR, self.k, self.n, w_val)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, C_ADDR, self.rows, self.n, self._c)
        yield Case(name=f"{self.rows}x{self.k} @ {self.k}x{self.n}",
                   golden=golden,
                   check_ranges=[(C_ADDR, self.rows * i4_row(self.n))])

    def _c(self, i: int, j: int) -> int:
        acc = sum(a_val(i, t) * w_val(t, j) for t in range(self.k))
        return narrow(acc, RQ_C)


def program(backend, rows: int = 8, ktiles: int = 4, ntiles: int = 2):
    return TPUProgram(os.path.join(HERE, "matmul.c"), backend,
                      MatmulVectors(rows, ktiles, ntiles))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-M", "--rows", type=int, default=8)
    ap.add_argument("--ktiles", type=int, default=4)
    ap.add_argument("--ntiles", type=int, default=2)
    args = ap.parse_args()

    prog = program(backend_from_args(args), args.rows, args.ktiles, args.ntiles)
    prog.run_program(limit=args.cases)
    return report(prog)


if __name__ == "__main__":
    raise SystemExit(main())
