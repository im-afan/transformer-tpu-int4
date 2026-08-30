#!/usr/bin/env python3
"""ffn: the transformer's feed-forward block — X @ W1, relu, @ W2.

The first kernel that drives two units, so a wrong fence between the MXU and
the VPU shows up here rather than in the model.

    python accel/test/tests/ffn/generate.py -b iss
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

T, D, DFF = 8, 8, 16
X_ADDR, W1_ADDR, W2_ADDR, Y_ADDR = 0x0000, 0x1000, 0x2000, 0x4000
RQ_H, RQ_H_RELU, RQ_Y = rq_word(1, 3), rq_word(1, 0), rq_word(1, 3)


def w2_val(r: int, c: int) -> int:
    return ((r * 3 + c * 7) % 16) - 8


class FfnVectors(VectorGenerator):
    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, X_ADDR, T, D, a_val)
        put_rowmajor_i4(img, W1_ADDR, D, DFF, w_val)
        put_rowmajor_i4(img, W2_ADDR, DFF, D, w2_val)
        return img

    def cases(self):
        h = [[narrow(sum(a_val(t, k) * w_val(k, j) for k in range(D)), RQ_H)
              for j in range(DFF)] for t in range(T)]
        hr = [[narrow(max(v, 0), RQ_H_RELU) for v in row] for row in h]
        y = [[narrow(sum(hr[t][k] * w2_val(k, j) for k in range(DFF)), RQ_Y)
              for j in range(D)] for t in range(T)]

        golden: dict = {}
        put_rowmajor_i4(golden, Y_ADDR, T, D, lambda t, j: y[t][j])
        yield Case(name=f"ffn {T}x{D} -> {DFF} -> {D}", golden=golden,
                   check_ranges=[(Y_ADDR, T * i4_row(D))])


def program(backend):
    return TPUProgram(os.path.join(HERE, "ffn.c"), backend, FfnVectors())


def main() -> int:
    args = standard_parser(__doc__).parse_args()
    prog = program(backend_from_args(args))
    prog.run_program(limit=args.cases)
    return report(prog)


if __name__ == "__main__":
    raise SystemExit(main())
