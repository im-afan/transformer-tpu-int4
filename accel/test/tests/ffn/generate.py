#!/usr/bin/env python3
"""ffn: the transformer's feed-forward block — X @ W1, relu, @ W2.

The first kernel that drives two units, so a wrong fence between the MXU and
the VPU shows up here rather than in the model.

Shape, address map and the three requant words are computed here and handed to
the compiler as -D. The requant words are **fitted to the accumulators the
golden produced**, so a wider DFF does not silently saturate the hidden layer.

    python accel/test/tests/ffn/generate.py -b iss
    python accel/test/tests/ffn/generate.py -b rtl -T 32 -d 64 -f 256
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


def w2_val(r: int, c: int) -> int:
    return ((r * 3 + c * 7) % 16) - 8


class FfnVectors(VectorGenerator):
    def __init__(self, tokens: int = 8, d: int = 8, dff: int = 16):
        for name, val in (("T", tokens), ("D", d), ("DFF", dff)):
            if val % TPU_N:
                raise SystemExit(f"{name} = {val} is not a whole number of "
                                 f"{TPU_N}-wide array tiles")
        self.T, self.D, self.DFF = tokens, d, dff

        self.map = AddressMap(what="ffn")
        self.map.alloc("X_ADDR", tokens * i4_row(d))
        self.map.alloc("W1_ADDR", d * i4_row(dff))
        self.map.alloc("W2_ADDR", dff * i4_row(d))
        self.map.alloc("H_ADDR", tokens * i4_row(dff))
        self.map.alloc("Y_ADDR", tokens * i4_row(d))

        # Forward, one stage at a time: each requant word is fitted to the
        # accumulators feeding it, then applied before the next stage runs.
        h_acc = [[sum(a_val(t, k) * w_val(k, j) for k in range(d))
                  for j in range(dff)] for t in range(tokens)]
        self.rq_h = fit_rq((v for row in h_acc for v in row), "RQ_H")
        h = [[narrow(v, self.rq_h) for v in row] for row in h_acc]
        hr = [[max(v, 0) for v in row] for row in h]     # relu shares H's scale

        y_acc = [[sum(hr[t][k] * w2_val(k, j) for k in range(dff))
                  for j in range(d)] for t in range(tokens)]
        self.rq_y = fit_rq((v for row in y_acc for v in row), "RQ_Y")
        self.y = [[narrow(v, self.rq_y) for v in row] for row in y_acc]

        self.defines = {"T": tokens, "D": d, "DFF": dff,
                        "RQ_H": f"{self.rq_h}u", "RQ_H_RELU": f"{RQ_ONE}u",
                        "RQ_Y": f"{self.rq_y}u", **self.map.defines()}

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.map["X_ADDR"], self.T, self.D, a_val)
        put_rowmajor_i4(img, self.map["W1_ADDR"], self.D, self.DFF, w_val)
        put_rowmajor_i4(img, self.map["W2_ADDR"], self.DFF, self.D, w2_val)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.map["Y_ADDR"], self.T, self.D,
                        lambda t, j: self.y[t][j])
        yield Case(name=f"ffn {self.T}x{self.D} -> {self.DFF} -> {self.D}",
                   golden=golden,
                   check_ranges=[(self.map["Y_ADDR"], self.T * i4_row(self.D))])


def program(backend, tokens: int = 8, d: int = 8, dff: int = 16):
    return TPUProgram(os.path.join(HERE, "ffn.c"), backend,
                      FfnVectors(tokens, d, dff))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-T", "--tokens", type=int, default=8)
    ap.add_argument("-d", type=int, default=8, help="model width")
    ap.add_argument("-f", "--dff", type=int, default=16, help="hidden width")
    args = ap.parse_args()

    prog = program(backend_from_args(args), args.tokens, args.d, args.dff)
    prog.run_program(limit=args.cases)
    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
