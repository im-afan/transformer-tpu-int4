#!/usr/bin/env python3
"""mha: one head of ReLU attention — the projections, Q @ K^T, relu, P @ V.

No causal mask: this is a datapath and ISA test, not the model. It is what
covers the MXU's transpose flag.

Shape, address map and the four requant words are computed here and handed to
the compiler as -D. The requant words are **fitted to the accumulators the
golden produced**, so changing T or head_dim does not silently saturate the
scores.

    python accel/test/tests/mha/generate.py -b iss
    python accel/test/tests/mha/generate.py -b rtl -T 32 -d 32 --head-dim 16
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (RQ_ONE, TPU_N, AddressMap, Case,  # noqa: E402
                              VectorGenerator, a_val, fit_rq, i4_row, narrow,
                              put_rowmajor_i4, w_val)


def wk_val(r: int, c: int) -> int:
    return ((r * 3 + c * 7) % 16) - 8


def wv_val(r: int, c: int) -> int:
    return ((r * 11 + c * 5) % 16) - 8


class MhaVectors(VectorGenerator):
    def __init__(self, tokens: int = 8, d: int = 8, head_dim: int = 8):
        for name, val in (("T", tokens), ("D", d), ("HEAD_DIM", head_dim)):
            if val % TPU_N:
                raise SystemExit(f"{name} = {val} is not a whole number of "
                                 f"{TPU_N}-wide array tiles")
        self.T, self.D, self.DH = tokens, d, head_dim

        self.map = AddressMap(what="mha")
        self.map.alloc("X_ADDR", tokens * i4_row(d))
        for name in ("WQ_ADDR", "WK_ADDR", "WV_ADDR"):
            self.map.alloc(name, d * i4_row(head_dim))
        for name in ("Q_ADDR", "K_ADDR", "V_ADDR"):
            self.map.alloc(name, tokens * i4_row(head_dim))
        self.map.alloc("S_ADDR", tokens * i4_row(tokens))
        self.map.alloc("A_ADDR", tokens * i4_row(head_dim))

        # The three projections share one requant word, so it is fitted to all
        # three at once — the same thing the kernel's one RQ_QKV means.
        proj_acc = {}
        for key, w in (("q", w_val), ("k", wk_val), ("v", wv_val)):
            proj_acc[key] = [[sum(a_val(t, i) * w(i, j) for i in range(d))
                              for j in range(head_dim)] for t in range(tokens)]
        self.rq_qkv = fit_rq((v for acc in proj_acc.values()
                              for row in acc for v in row), "RQ_QKV")
        q, k, v = ({key: [[narrow(x, self.rq_qkv) for x in row] for row in acc]
                    for key, acc in proj_acc.items()}[key]
                   for key in ("q", "k", "v"))

        s_acc = [[sum(q[t][i] * k[u][i] for i in range(head_dim))
                  for u in range(tokens)] for t in range(tokens)]
        self.rq_s = fit_rq((x for row in s_acc for x in row), "RQ_S")
        s = [[narrow(x, self.rq_s) for x in row] for row in s_acc]
        p = [[max(x, 0) for x in row] for row in s]      # relu shares S's scale

        a_acc = [[sum(p[t][u] * v[u][j] for u in range(tokens))
                  for j in range(head_dim)] for t in range(tokens)]
        self.rq_a = fit_rq((x for row in a_acc for x in row), "RQ_A")
        self.a = [[narrow(x, self.rq_a) for x in row] for row in a_acc]

        self.defines = {"T": tokens, "D": d, "HEAD_DIM": head_dim,
                        "RQ_QKV": f"{self.rq_qkv}u", "RQ_S": f"{self.rq_s}u",
                        "RQ_P": f"{RQ_ONE}u", "RQ_A": f"{self.rq_a}u",
                        **self.map.defines()}

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.map["X_ADDR"], self.T, self.D, a_val)
        for name, w in (("WQ_ADDR", w_val), ("WK_ADDR", wk_val),
                        ("WV_ADDR", wv_val)):
            put_rowmajor_i4(img, self.map[name], self.D, self.DH, w)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.map["A_ADDR"], self.T, self.DH,
                        lambda t, j: self.a[t][j])
        yield Case(name=f"one head, T={self.T} head_dim={self.DH}",
                   golden=golden,
                   check_ranges=[(self.map["A_ADDR"], self.T * i4_row(self.DH))])


def program(backend, tokens: int = 8, d: int = 8, head_dim: int = 8):
    return TPUProgram(os.path.join(HERE, "mha.c"), backend,
                      MhaVectors(tokens, d, head_dim))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-T", "--tokens", type=int, default=8, help="tokens (= keys)")
    ap.add_argument("-d", type=int, default=8, help="model width")
    ap.add_argument("--head-dim", type=int, default=8)
    args = ap.parse_args()

    prog = program(backend_from_args(args), args.tokens, args.d, args.head_dim)
    prog.run_program(limit=args.cases)
    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
