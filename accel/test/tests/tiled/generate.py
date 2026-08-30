#!/usr/bin/env python3
"""tiled: tpulib.h's block loops past the point where the scratchpad helps.

Four passes, all DRAM to DRAM: a 1024-deep contraction that has to be split, a
relu, a 2500-long vector add that spans three VPU chunks, and a matmul whose
extents are neither a whole array tile. A mis-tiled matmul is something the ISS
would reproduce as faithfully as the RTL, which is why the golden here is a
plain Python matmul instead.

    python accel/test/tests/tiled/generate.py -b iss
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

DR_A1, DR_W1, DR_C1, DR_C2 = 0x00000, 0x02000, 0x04000, 0x04100
DR_V1, DR_V2, DR_C3 = 0x04200, 0x04700, 0x04C00
DR_A4, DR_B4, DR_C4 = 0x05200, 0x05400, 0x05700

M1, K1, N1 = 16, 1024, 16
VEC = 2500
M4, K4, N4 = 12, 64, 20

RQ_C1, RQ_C2, RQ_C3, RQ_C4 = (rq_word(1, 8), rq_word(1, 0),
                              rq_word(1, 1), rq_word(1, 4))


def a4_val(r: int, c: int) -> int:
    return ((r * 7 + c * 3) % 9) - 4


def b4_val(r: int, c: int) -> int:
    return ((r * 3 + c * 7) % 16) - 8


def v1_val(i: int) -> int:
    return ((i * 5) % 16) - 8


def v2_val(i: int) -> int:
    return ((i * 3) % 16) - 8


class TiledVectors(VectorGenerator):
    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, DR_A1, M1, K1, a_val)
        put_rowmajor_i4(img, DR_W1, K1, N1, w_val)
        put_rowmajor_i4(img, DR_V1, 1, VEC, lambda r, c: v1_val(c))
        put_rowmajor_i4(img, DR_V2, 1, VEC, lambda r, c: v2_val(c))
        put_rowmajor_i4(img, DR_A4, M4, K4, a4_val)
        put_rowmajor_i4(img, DR_B4, N4, K4, b4_val)      # [cols][depth]
        return img

    def cases(self):
        c1 = [[narrow(sum(a_val(i, t) * w_val(t, j) for t in range(K1)), RQ_C1)
               for j in range(N1)] for i in range(M1)]
        c2 = [[narrow(max(v, 0), RQ_C2) for v in row] for row in c1]
        c3 = [narrow(v1_val(i) + v2_val(i), RQ_C3) for i in range(VEC)]
        # B is stored [cols][depth], so the index function is swapped, not the
        # storage — the same thing the kernel's `transpose` flag says.
        c4 = [[narrow(sum(a4_val(i, t) * b4_val(j, t) for t in range(K4)), RQ_C4)
               for j in range(N4)] for i in range(M4)]

        golden: dict = {}
        put_rowmajor_i4(golden, DR_C1, M1, N1, lambda i, j: c1[i][j])
        put_rowmajor_i4(golden, DR_C2, M1, N1, lambda i, j: c2[i][j])
        put_rowmajor_i4(golden, DR_C3, 1, VEC, lambda i, j: c3[j])
        put_rowmajor_i4(golden, DR_C4, M4, N4, lambda i, j: c4[i][j])

        yield Case(name="four DRAM-to-DRAM passes", golden=golden,
                   check_ranges=[(DR_C1, M1 * i4_row(N1)),
                                 (DR_C2, M1 * i4_row(N1)),
                                 (DR_C3, i4_row(VEC)),
                                 (DR_C4, M4 * i4_row(N4))])


def program(backend):
    return TPUProgram(os.path.join(HERE, "tiled.c"), backend, TiledVectors())


def main() -> int:
    args = standard_parser(__doc__).parse_args()
    prog = program(backend_from_args(args))
    prog.run_program(limit=args.cases)
    return report(prog)


if __name__ == "__main__":
    raise SystemExit(main())
