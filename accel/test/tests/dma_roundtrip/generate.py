#!/usr/bin/env python3
"""dma_roundtrip: a matrix out of DRAM, through the scratchpad and back.

Nothing computes, so the run is the DMA plus what it costs to keep it fed. The
copy has to come back byte-identical, which is what stops a benchmark that moved
the wrong bytes — or none — from reading as fast.

sram.sv is 1 clock/byte on a fill and 2 on a spill, so the floor is 3 bytes'
worth of clocks per byte of the matrix; `report` prints what it actually took. A
tile narrower than the matrix is the interesting knob: each of its rows is a
separate range, so `--tcols` is how you price a strided move.

    python accel/test/tests/dma_roundtrip/generate.py -b rtl
    python accel/test/tests/dma_roundtrip/generate.py -b rtl --tcols 128
    python accel/test/tests/dma_roundtrip/generate.py -b rtl -R 64 -C 512 --tcols 8
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (AddressMap, Case, VectorGenerator,  # noqa: E402
                              i4_row, put_rowmajor_i4)

DRAM_BYTES = 1 << 19
DMA_LEN_MAX = 0xFFFF            # the macro-op's `len` field
DMA_ROWS_MAX = 0xFFFF           # and its `rows` field


def m_val(r: int, c: int) -> int:
    return ((r * 11 + c * 7) % 16) - 8


class DmaRoundtripVectors(VectorGenerator):
    def __init__(self, rows: int = 128, cols: int = 128, trows: int = 32,
                 tcols: int = 32):
        for name, val in (("C", cols), ("tcols", tcols)):
            if val % 2:
                raise SystemExit(f"{name} = {val} must be even — an int4 row is "
                                 f"two elements per byte")
        if tcols > DMA_LEN_MAX or trows > DMA_ROWS_MAX:
            raise SystemExit(f"a tile of {trows} x {tcols} does not fit the "
                             f"DMA command's 16-bit rows and len fields")
        self.rows, self.cols = rows, cols
        self.trows, self.tcols = trows, tcols

        self.dram = AddressMap(align=64, limit=DRAM_BYTES, what="dma_roundtrip")
        self.dram.alloc("DR_SRC", rows * i4_row(cols))
        self.dram.alloc("DR_DST", rows * i4_row(cols))

        self.spad = AddressMap(what="dma_roundtrip staging")
        self.spad.alloc("SP_BUF", trows * i4_row(tcols))

        self.defines = {"ROWS": rows, "COLS": cols,
                        "TROWS": trows, "TCOLS": tcols,
                        **self.dram.defines(), **self.spad.defines()}

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.dram["DR_SRC"], self.rows, self.cols, m_val)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.dram["DR_DST"], self.rows, self.cols,
                        m_val)
        tiles = (-(-self.rows // self.trows)) * (-(-self.cols // self.tcols))
        yield Case(name=f"{self.rows}x{self.cols} int4 each way in {tiles} "
                        f"tile(s) of {self.trows}x{self.tcols}",
                   golden=golden,
                   check_ranges=[(self.dram["DR_DST"],
                                  self.rows * i4_row(self.cols))])


def program(backend, rows: int = 128, cols: int = 128, trows: int = 32,
            tcols: int = 32):
    return TPUProgram(os.path.join(HERE, "dma_roundtrip.c"), backend,
                      DmaRoundtripVectors(rows, cols, trows, tcols))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-R", "--rows", type=int, default=128)
    ap.add_argument("-C", "--cols", type=int, default=128)
    ap.add_argument("--trows", type=int, default=32,
                    help="rows one DMA command moves")
    ap.add_argument("--tcols", type=int, default=32,
                    help="columns one DMA command moves; below --cols every "
                         "row of the tile is its own range")
    args = ap.parse_args()

    prog = program(backend_from_args(args), args.rows, args.cols, args.trows,
                   args.tcols)
    prog.run_program(limit=args.cases)
    rc = report(prog, args.clk_mhz)

    bench = prog.benchmark(args.clk_mhz)
    if bench:
        moved = args.rows * i4_row(args.cols)
        floor = 3 * moved                     # 1 clock/byte in, 2 back out
        cmds = 2 * (-(-args.rows // args.trows)) * (-(-args.cols // args.tcols))
        dma_rows = 2 * args.rows * (-(-args.cols // args.tcols))
        over = bench["clocks"] - floor
        print(f"  {'bytes moved':<34} {2 * moved:>12}  "
              f"({moved} in, {moved} out)")
        print(f"  {'sram.sv floor':<34} {floor:>12} clocks  "
              f"{bench['clocks'] / floor:.2f}x")
        print(f"  {'over the floor':<34} {over:>12} clocks  "
              f"{over / cmds:.1f} per command over {cmds}, {over / dma_rows:.2f} "
              f"per row over {dma_rows}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
