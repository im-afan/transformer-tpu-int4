#!/usr/bin/env python3
"""mha: one head of ReLU attention — the projections, Q @ K^T, relu, P @ V.

No causal mask: this is a datapath and ISA test, not the model. It is what
covers the MXU's transpose flag.

    python accel/test/tests/mha/generate.py -b iss
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

T, D, DH = 8, 8, 8
X_ADDR, WQ_ADDR, WK_ADDR, WV_ADDR = 0x0000, 0x1000, 0x1400, 0x1800
A_ADDR = 0x6000
RQ_QKV, RQ_S, RQ_P, RQ_A = rq_word(1, 3), rq_word(1, 4), rq_word(1, 0), rq_word(1, 3)


def wk_val(r: int, c: int) -> int:
    return ((r * 3 + c * 7) % 16) - 8


def wv_val(r: int, c: int) -> int:
    return ((r * 11 + c * 5) % 16) - 8


class MhaVectors(VectorGenerator):
    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, X_ADDR, T, D, a_val)
        put_rowmajor_i4(img, WQ_ADDR, D, DH, w_val)
        put_rowmajor_i4(img, WK_ADDR, D, DH, wk_val)
        put_rowmajor_i4(img, WV_ADDR, D, DH, wv_val)
        return img

    def cases(self):
        def proj(w):
            return [[narrow(sum(a_val(t, k) * w(k, j) for k in range(D)), RQ_QKV)
                     for j in range(DH)] for t in range(T)]

        q, k, v = proj(w_val), proj(wk_val), proj(wv_val)
        s = [[narrow(sum(q[t][d] * k[u][d] for d in range(DH)), RQ_S)
              for u in range(T)] for t in range(T)]
        p = [[narrow(max(x, 0), RQ_P) for x in row] for row in s]
        a = [[narrow(sum(p[t][u] * v[u][j] for u in range(T)), RQ_A)
              for j in range(DH)] for t in range(T)]

        golden: dict = {}
        put_rowmajor_i4(golden, A_ADDR, T, DH, lambda t, j: a[t][j])
        yield Case(name=f"one head, T={T} head_dim={DH}", golden=golden,
                   check_ranges=[(A_ADDR, T * i4_row(DH))])


def program(backend):
    return TPUProgram(os.path.join(HERE, "mha.c"), backend, MhaVectors())


def main() -> int:
    args = standard_parser(__doc__).parse_args()
    prog = program(backend_from_args(args))
    prog.run_program(limit=args.cases)
    return report(prog)


if __name__ == "__main__":
    raise SystemExit(main())
