#!/usr/bin/env python3
"""tiled: tpulib.h's block loops past the point where the scratchpad helps.

Four passes, all DRAM to DRAM: a contraction long enough that it has to be
split, a relu, a vector add spanning several VPU chunks, and a matmul whose
extents are not whole array tiles. A mis-tiled matmul is something the ISS would
reproduce as faithfully as the RTL, which is why the golden here is a plain
Python matmul instead.

The shapes are the point of the test, so they are flags — but keep the
properties: `--depth1` past one dispatch, `--vec` past one VPU chunk, and
`--rows4 / --cols4` not multiples of the array edge.

    python accel/test/tests/tiled/generate.py -b iss
    python accel/test/tests/tiled/generate.py -b rtl --depth1 2048 --vec 4000
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (RQ_ONE, TPU_BANK_BYTES, AddressMap,  # noqa: E402
                              Case, VectorGenerator, a_val, fit_rq, i4_row,
                              narrow, put_rowmajor_i4, w_val)

# DRAM to DRAM, so the map is bounded by the 512 KB part rather than by the
# scratchpad; the arena is the only scratchpad this kernel spends.
DRAM_BYTES = 1 << 19


def a4_val(r: int, c: int) -> int:
    return ((r * 7 + c * 3) % 9) - 4


def b4_val(r: int, c: int) -> int:
    return ((r * 3 + c * 7) % 16) - 8


def v1_val(i: int) -> int:
    return ((i * 5) % 16) - 8


def v2_val(i: int) -> int:
    return ((i * 3) % 16) - 8


class TiledVectors(VectorGenerator):
    def __init__(self, m1: int = 16, k1: int = 1024, n1: int = 16,
                 vec: int = 2500, m4: int = 12, k4: int = 64, n4: int = 20,
                 arena_banks: int = 3):
        for name, val in (("depth1", k1), ("cols1", n1), ("vec", vec),
                          ("depth4", k4), ("cols4", n4)):
            if val % 2:
                raise SystemExit(f"{name} = {val} must be even — an int4 row is "
                                 f"two elements per byte")
        self.m1, self.k1, self.n1 = m1, k1, n1
        self.vec = vec
        self.m4, self.k4, self.n4 = m4, k4, n4

        # 64-byte granularity, not a whole bank: these are DRAM tensors, and the
        # only scratchpad contention is inside the arena tpulib.h manages.
        self.map = AddressMap(align=64, limit=DRAM_BYTES, what="tiled")
        self.map.alloc("DR_A1", m1 * i4_row(k1))
        self.map.alloc("DR_W1", k1 * i4_row(n1))
        self.map.alloc("DR_C1", m1 * i4_row(n1))
        self.map.alloc("DR_C2", m1 * i4_row(n1))
        self.map.alloc("DR_V1", i4_row(vec))
        self.map.alloc("DR_V2", i4_row(vec))
        self.map.alloc("DR_C3", i4_row(vec))
        self.map.alloc("DR_A4", m4 * i4_row(k4))
        self.map.alloc("DR_B4", n4 * i4_row(k4))
        self.map.alloc("DR_C4", m4 * i4_row(n4))

        c1_acc = [[sum(a_val(i, t) * w_val(t, j) for t in range(k1))
                   for j in range(n1)] for i in range(m1)]
        self.rq_c1 = fit_rq((v for row in c1_acc for v in row), "RQ_C1")
        self.c1 = [[narrow(v, self.rq_c1) for v in row] for row in c1_acc]
        # relu is a clamp, not a rescale; the add of two int4 is bounded by 16.
        self.c2 = [[narrow(max(v, 0), RQ_ONE) for v in row] for row in self.c1]
        self.rq_c3 = fit_rq((v1_val(i) + v2_val(i) for i in range(vec)), "RQ_C3")
        self.c3 = [narrow(v1_val(i) + v2_val(i), self.rq_c3) for i in range(vec)]

        # B is stored [cols][depth], so the index function is swapped, not the
        # storage — the same thing the kernel's `transpose` flag says.
        c4_acc = [[sum(a4_val(i, t) * b4_val(j, t) for t in range(k4))
                   for j in range(n4)] for i in range(m4)]
        self.rq_c4 = fit_rq((v for row in c4_acc for v in row), "RQ_C4")
        self.c4 = [[narrow(v, self.rq_c4) for v in row] for row in c4_acc]

        self.defines = {
            "MM1_ROWS": m1, "MM1_DEPTH": k1, "MM1_COLS": n1, "VEC_LEN": vec,
            "MM4_ROWS": m4, "MM4_DEPTH": k4, "MM4_COLS": n4,
            "ARENA_BANKS": f"{arena_banks}u", "SP_ARENA": "0x0000u",
            "RQ_C1": f"{self.rq_c1}u", "RQ_C2": f"{RQ_ONE}u",
            "RQ_C3": f"{self.rq_c3}u", "RQ_C4": f"{self.rq_c4}u",
            **self.map.defines()}

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.map["DR_A1"], self.m1, self.k1, a_val)
        put_rowmajor_i4(img, self.map["DR_W1"], self.k1, self.n1, w_val)
        put_rowmajor_i4(img, self.map["DR_V1"], 1, self.vec,
                        lambda r, c: v1_val(c))
        put_rowmajor_i4(img, self.map["DR_V2"], 1, self.vec,
                        lambda r, c: v2_val(c))
        put_rowmajor_i4(img, self.map["DR_A4"], self.m4, self.k4, a4_val)
        put_rowmajor_i4(img, self.map["DR_B4"], self.n4, self.k4, b4_val)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.map["DR_C1"], self.m1, self.n1,
                        lambda i, j: self.c1[i][j])
        put_rowmajor_i4(golden, self.map["DR_C2"], self.m1, self.n1,
                        lambda i, j: self.c2[i][j])
        put_rowmajor_i4(golden, self.map["DR_C3"], 1, self.vec,
                        lambda i, j: self.c3[j])
        put_rowmajor_i4(golden, self.map["DR_C4"], self.m4, self.n4,
                        lambda i, j: self.c4[i][j])

        yield Case(name="four DRAM-to-DRAM passes", golden=golden,
                   check_ranges=[(self.map["DR_C1"], self.m1 * i4_row(self.n1)),
                                 (self.map["DR_C2"], self.m1 * i4_row(self.n1)),
                                 (self.map["DR_C3"], i4_row(self.vec)),
                                 (self.map["DR_C4"], self.m4 * i4_row(self.n4))])


def program(backend, **kw):
    return TPUProgram(os.path.join(HERE, "tiled.c"), backend, TiledVectors(**kw))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("--rows1", type=int, default=16)
    ap.add_argument("--depth1", type=int, default=1024,
                    help="longer than one dispatch, so the contraction splits")
    ap.add_argument("--cols1", type=int, default=16)
    ap.add_argument("--vec", type=int, default=2500,
                    help="longer than one VPU chunk, and not a multiple of it")
    ap.add_argument("--rows4", type=int, default=12,
                    help="deliberately not a whole array tile")
    ap.add_argument("--depth4", type=int, default=64)
    ap.add_argument("--cols4", type=int, default=20)
    ap.add_argument("--arena-banks", type=int, default=3,
                    help="scratchpad banks tpulib.h may spend: one each for A, "
                         "B and C is the floor")
    args = ap.parse_args()

    prog = program(backend_from_args(args), m1=args.rows1, k1=args.depth1,
                   n1=args.cols1, vec=args.vec, m4=args.rows4, k4=args.depth4,
                   n4=args.cols4, arena_banks=args.arena_banks)
    prog.run_program(limit=args.cases)
    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
