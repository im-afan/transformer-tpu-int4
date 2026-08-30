#!/usr/bin/env python3
"""spadwin: the CPU's scratchpad window (cpu_subsys.sv's 0x9xxx_xxxx), alone.

The CPU scans a vector it can only reach through that window, writes its answer
back through it, reads one word back, and derives a DMA address from what it
found. It is the only thing covering the window, and the window is what makes
the argmax and the embedding gather live on the device.

    python accel/test/tests/spadwin/generate.py -b iss
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (Case, VectorGenerator, put_i32,  # noqa: E402
                              put_rowmajor_i8)

DR_VEC, DR_TABLE, DR_OUT = 0x0000, 0x0100, 0x0200
N, ROW_BYTES, ROWS = 16, 16, 13


def vec_val(i: int) -> int:
    """A permutation of 0..15 scaled out of int8 range, so the max is unique."""
    return ((i * 5 + 3) % 16) * 137 - 900


def table_val(r: int, c: int) -> int:
    return ((r * 7 + c * 3) % 251) - 128


class SpadwinVectors(VectorGenerator):
    def static(self) -> dict:
        img: dict = {}
        put_i32(img, DR_VEC, [vec_val(i) for i in range(N)])
        put_rowmajor_i8(img, DR_TABLE, ROWS, ROW_BYTES, ROW_BYTES, table_val)
        return img

    def cases(self):
        vals = [vec_val(i) for i in range(N)]
        best = max(range(N), key=lambda i: (vals[i], -i))   # ties take the lowest

        golden: dict = {}
        put_i32(golden, DR_OUT, [best, vals[best], best])
        put_rowmajor_i8(golden, DR_OUT + 16, 1, ROW_BYTES, ROW_BYTES,
                        lambda r, c: table_val(best, c))
        yield Case(name=f"max at index {best}, row gathered", golden=golden,
                   check_ranges=[(DR_OUT, 12), (DR_OUT + 16, ROW_BYTES)])


def program(backend):
    return TPUProgram(os.path.join(HERE, "spadwin.c"), backend, SpadwinVectors())


def main() -> int:
    args = standard_parser(__doc__).parse_args()
    prog = program(backend_from_args(args))
    prog.run_program(limit=args.cases)
    return report(prog)


if __name__ == "__main__":
    raise SystemExit(main())
