#!/usr/bin/env python3
"""spadwin: the CPU's scratchpad window (cpu_subsys.sv's 0x9xxx_xxxx), alone.

The CPU scans a vector it can only reach through that window, writes its answer
back through it, reads one word back, and derives a DMA address from what it
found. It is the only thing covering the window, and the window is what makes
the argmax and the embedding gather live on the device.

Both address maps and the two lengths come from here as -D. The table is sized
to the vector, so the gathered row is always in range whichever index wins.

    python accel/test/tests/spadwin/generate.py -b iss
    python accel/test/tests/spadwin/generate.py -b rtl -w 64 --row-bytes 32
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (AddressMap, Case, VectorGenerator,  # noqa: E402
                              put_i32, put_rowmajor_i8)

OUT_WORDS = 3           # {index, value, readback}, then the gathered row


def vec_val(i: int, n: int) -> int:
    """A permutation of 0..n-1 scaled out of int8 range, so the max is unique
    and a byte-wide comparison cannot have found it by accident."""
    return ((i * 5 + 3) % n) * 137 - 900


def table_val(r: int, c: int) -> int:
    return ((r * 7 + c * 3) % 251) - 128


class SpadwinVectors(VectorGenerator):
    def __init__(self, words: int = 16, row_bytes: int = 16):
        if words < 2:
            raise SystemExit("the CPU needs at least two words to compare")
        if row_bytes % 4:
            raise SystemExit(f"row_bytes = {row_bytes} must be a multiple of 4: "
                             f"the scratchpad window has no byte strobes")
        self.n, self.row_bytes = words, row_bytes
        self.rows = words          # every index the scan can pick has a row

        # Two maps, one per memory, because this kernel is the one that does not
        # give a tensor the same address in both.
        self.dram = AddressMap(align=64, what="spadwin DRAM")
        self.dram.alloc("DR_VEC", words * 4)
        self.dram.alloc("DR_TABLE", self.rows * row_bytes)
        self.dram.alloc("DR_OUT", OUT_WORDS * 4 + row_bytes)

        self.spad = AddressMap(align=64, what="spadwin scratchpad")
        self.spad.alloc("SP_VEC", words * 4)
        self.spad.alloc("SP_OUT", OUT_WORDS * 4)
        self.spad.alloc("SP_ROW", row_bytes)

        self.defines = {"VEC_WORDS": words, "ROW_BYTES": row_bytes,
                        **self.dram.defines(), **self.spad.defines()}

    def static(self) -> dict:
        img: dict = {}
        put_i32(img, self.dram["DR_VEC"],
                [vec_val(i, self.n) for i in range(self.n)])
        put_rowmajor_i8(img, self.dram["DR_TABLE"], self.rows, self.row_bytes,
                        self.row_bytes, table_val)
        return img

    def cases(self):
        vals = [vec_val(i, self.n) for i in range(self.n)]
        best = max(range(self.n), key=lambda i: (vals[i], -i))   # ties: lowest

        out, row = self.dram["DR_OUT"], self.dram["DR_OUT"] + 16
        golden: dict = {}
        put_i32(golden, out, [best, vals[best], best])
        put_rowmajor_i8(golden, row, 1, self.row_bytes, self.row_bytes,
                        lambda r, c: table_val(best, c))
        yield Case(name=f"max of {self.n} at index {best}, row gathered",
                   golden=golden,
                   check_ranges=[(out, OUT_WORDS * 4), (row, self.row_bytes)])


def program(backend, words: int = 16, row_bytes: int = 16):
    return TPUProgram(os.path.join(HERE, "spadwin.c"), backend,
                      SpadwinVectors(words, row_bytes))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-w", "--words", type=int, default=16,
                    help="int32 words the CPU scans through the window")
    ap.add_argument("--row-bytes", type=int, default=16,
                    help="bytes in the row the DMA gathers")
    args = ap.parse_args()

    prog = program(backend_from_args(args), args.words, args.row_bytes)
    prog.run_program(limit=args.cases)
    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
