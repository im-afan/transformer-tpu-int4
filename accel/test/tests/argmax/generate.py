#!/usr/bin/env python3
"""argmax: tpulib.h's tpu_argmax, the only caller of the VPU's VOP_ARGMAX.

The op reduces one dispatch and reports an index within it, so anything longer
than a chunk is folded by the CPU through the scratchpad window. That fold is
what this covers: six lengths against two arena sizes, so each problem runs
once with the chunk capped by the 10-bit `vlen` field and once with it capped
by the arena, and every data pattern is run at all six lengths.

Two of the lengths are odd, which only a reduction accepts — it writes an int32
scalar and no nibbles, so the half-byte tail that pins the elementwise ops does
not exist. `head_argmax` reduces over `VOCAB`, which is 13.

The patterns are chosen for what they force rather than what they compute — a
peak in the last ragged chunk, two equal maxima so the tie rule is visible, an
all-Q4_MIN vector, and a vector whose maximum is negative. A pattern whose
answer is 0 at every length would pass against a kernel that returns 0.

    python accel/test/tests/argmax/generate.py -b iss
    python accel/test/tests/argmax/generate.py -b rtl
    python accel/test/tests/argmax/generate.py -b rtl --multi 4000 --arena-bytes 64
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (Q4_MAX, Q4_MIN, TPU_BANK_BYTES,  # noqa: E402
                              TPU_N, TPU_WORD_BYTES, AddressMap, Case,
                              VectorGenerator, put_i32, put_rowmajor_i4)

NAMES = ("MULTI", "EXACT", "WORD", "PAIR", "ODD", "VOCAB")


def ramp(i: int, n: int) -> int:
    """The whole grid, both clips included. Its maximum is near the front."""
    return ((i * 5) % 16) - 8


def peak_last(i: int, n: int) -> int:
    return Q4_MAX if i == n - 1 else Q4_MIN


def peak_pair(i: int, n: int) -> int:
    """Two equal maxima two thirds of the way in: the lower index must win."""
    k = (n // 3) * 2
    return Q4_MAX if i in (k, k + 1) else Q4_MIN


def ties(i: int, n: int) -> int:
    return 3


def all_min(i: int, n: int) -> int:
    return Q4_MIN


def negative(i: int, n: int) -> int:
    """A unique maximum of -1, so a fold opening its accumulator at 0 loses it."""
    return -1 if i == n // 2 else -8 + (i % 3)


PATTERNS = (("ramp", ramp), ("peak in the last chunk", peak_last),
            ("two equal maxima", peak_pair), ("every element equal", ties),
            ("every element Q4_MIN", all_min),
            ("maximum is negative", negative))


class ArgmaxVectors(VectorGenerator):
    def __init__(self, multi: int = 2500, exact: int = 1016, word: int = TPU_N,
                 pair: int = 2, odd: int = 1013, vocab: int = 13,
                 arena_bytes: int = TPU_BANK_BYTES,
                 arena_small_bytes: int = 128):
        self.lens = (multi, exact, word, pair, odd, vocab)
        for name, n in zip(NAMES, self.lens):
            if n < 2:
                raise SystemExit(f"{name} = {n} is shorter than one lane pair")
        for name, n in (("--arena-bytes", arena_bytes),
                        ("--arena-small-bytes", arena_small_bytes)):
            if n < TPU_N + 2 * TPU_WORD_BYTES:
                raise SystemExit(f"{name} = {n} leaves no room for a chunk of "
                                 f"{TPU_N} and the int32 scalar beside it")

        self.dram = AddressMap(align=64, what="argmax DRAM")
        for name, n in zip(NAMES, self.lens):
            self.dram.alloc(f"DR_{name}", (n + 1) // 2)
        self.dram.alloc("DR_OUT", 2 * len(self.lens) * 4)

        self.defines = {f"LEN_{name}": n for name, n in zip(NAMES, self.lens)}
        self.defines.update({"ARENA_BYTES": f"{arena_bytes}u",
                             "ARENA_SMALL_BYTES": f"{arena_small_bytes}u",
                             "SP_ARENA": "0x0000u", "SP_OUT": "0x8000u",
                             **self.dram.defines()})

    def _fill(self, img: dict, fn) -> None:
        for name, n in zip(NAMES, self.lens):
            put_rowmajor_i4(img, self.dram[f"DR_{name}"], 1, n,
                            lambda r, c, n=n: fn(c, n))

    def static(self) -> dict:
        img: dict = {}
        self._fill(img, PATTERNS[0][1])
        return img

    def cases(self):
        out = self.dram["DR_OUT"]
        for label, fn in PATTERNS:
            patch: dict = {}
            self._fill(patch, fn)
            # Ties take the lowest index, the rule the RTL folds to and the one
            # torch.argmax follows.
            best = [max(range(n), key=lambda i, n=n: (fn(i, n), -i))
                    for n in self.lens]
            golden: dict = {}
            put_i32(golden, out, best + best)   # the same answers, both arenas
            yield Case(name=f"{label}: {', '.join(map(str, best))}",
                       patch=patch, golden=golden,
                       check_ranges=[(out, 2 * len(self.lens) * 4)])


def program(backend, **kwargs):
    return TPUProgram(os.path.join(HERE, "argmax.c"), backend,
                      ArgmaxVectors(**kwargs))


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("--multi", type=int, default=2500,
                    help="length spanning several chunks, last one ragged")
    ap.add_argument("--exact", type=int, default=1016,
                    help="length of exactly one full vlen-capped chunk")
    ap.add_argument("--word", type=int, default=TPU_N,
                    help="length of one scratchpad word")
    ap.add_argument("--pair", type=int, default=2, help="shortest legal length")
    ap.add_argument("--odd", type=int, default=1013,
                    help="odd length spanning several chunks")
    ap.add_argument("--vocab", type=int, default=13,
                    help="odd length in one chunk: what head_argmax reduces")
    ap.add_argument("--arena-bytes", type=int, default=TPU_BANK_BYTES,
                    help="the arena that leaves the chunk vlen-capped")
    ap.add_argument("--arena-small-bytes", type=int, default=128,
                    help="the arena that caps the chunk below the vlen field")
    args = ap.parse_args()

    prog = program(backend_from_args(args), multi=args.multi, exact=args.exact,
                   word=args.word, pair=args.pair, odd=args.odd,
                   vocab=args.vocab, arena_bytes=args.arena_bytes,
                   arena_small_bytes=args.arena_small_bytes)
    prog.run_program(limit=args.cases)
    return report(prog, args.clk_mhz)


if __name__ == "__main__":
    raise SystemExit(main())
