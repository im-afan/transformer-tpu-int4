#!/usr/bin/env python3
"""flash: one tpu_flashattention — causal ReLU attention, tiled in the scratchpad.

`out = relu(Q @ K' + mask) @ V` for one head, with S and P never leaving the
scratchpad. `--rows` queries at `--first-pos` against a `-T`-long key axis, so
the same image is a prefill pass (rows = T, first_pos = 0) or a decode step
(rows = 1, first_pos = T-1).

The golden is a plain Python attention followed by the same narrows the hardware
does, computed **with the same key blocking the primitive uses**, because the
MXU's accumulate is an int4 add: a contraction split across key blocks clips at
every step where an unsplit one clips once.

The arena is what picks the block size, so `--arena-banks` and `-T` together
reach the two interesting shapes:

  - one key block (B == T) — the tiling is a no-op and the result is bit-equal
    to the unsplit reference. this is what checks the mask, the transpose and
    the addressing
  - several key blocks, the last one ragged — this is what checks the
    accumulate across blocks, the skip of a block above the diagonal, and the
    mask staged only on the diagonal tile

`--unsplit-check` reports how far the split result drifts from the unsplit one.

    python accel/test/tests/flash/generate.py -b iss
    python accel/test/tests/flash/generate.py -b iss --sweep
    python accel/test/tests/flash/generate.py -b rtl -T 96 --arena-banks 5
    python accel/test/tests/flash/generate.py -b iss -T 64 --rows 1 --first-pos 63
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (RQ_ONE, TPU_BANK_BYTES, TPU_N,  # noqa: E402
                              AddressMap, Case, VectorGenerator, fit_rq,
                              flash_block, i4_row, narrow, put_rowmajor_i4,
                              w_hash)

DRAM_BYTES = 1 << 19
Q4_MIN, Q4_MAX = -8, 7


def q_val(t: int, i: int) -> int:
    return w_hash(t, i, 1)


def k_val(t: int, i: int) -> int:
    return w_hash(t, i, 2)


def v_val(t: int, i: int) -> int:
    return w_hash(t, i, 3)


class FlashVectors(VectorGenerator):
    def __init__(self, tokens: int = 32, head_dim: int = 32, rows: int = 0,
                 first_pos: int = 0, arena_banks: int = 5):
        for name, val in (("T", tokens), ("HEAD_DIM", head_dim)):
            if val % TPU_N:
                raise SystemExit(f"{name} = {val} is not a whole number of "
                                 f"{TPU_N}-wide array tiles")
        self.keys, self.DH = tokens, head_dim
        self.rows = rows or tokens
        self.first_pos = first_pos
        if self.first_pos + self.rows > self.keys:
            raise SystemExit(f"the queries ({self.first_pos}.."
                             f"{self.first_pos + self.rows}) run past the "
                             f"{self.keys}-long key axis")
        self.arena_banks = arena_banks
        self.block = flash_block(arena_banks * TPU_BANK_BYTES, tokens, head_dim)

        self.map = AddressMap(align=64, limit=DRAM_BYTES, what="flash")
        self.map.alloc("DR_Q", self.rows * i4_row(head_dim))
        for name in ("DR_K", "DR_V"):
            self.map.alloc(name, tokens * i4_row(head_dim))
        self.map.alloc("DR_MASK", tokens * i4_row(tokens))
        self.map.alloc("DR_OUT", self.rows * i4_row(head_dim))

        # Query r of the call sits at first_pos + r on the key axis, so it is
        # hashed off its absolute position and the mask follows it.
        Q = [[q_val(self.first_pos + r, i) for i in range(head_dim)]
             for r in range(self.rows)]
        K = [[k_val(u, i) for i in range(head_dim)] for u in range(tokens)]
        V = [[v_val(u, i) for i in range(head_dim)] for u in range(tokens)]

        s_acc = [[sum(Q[r][i] * K[u][i] for i in range(head_dim))
                  for u in range(tokens)] for r in range(self.rows)]
        self.rq_s = fit_rq((x for row in s_acc for x in row), "RQ_S")

        # The mask add and the relu are identity passes, the way infer.c's
        # RQ_ID and RQ_P are: -8 against an int4 score clips to -8 and the relu
        # takes it to exactly zero, which is what lets the kernel skip a whole
        # key block above the diagonal.
        self.rq_mask, self.rq_p = RQ_ONE, RQ_ONE
        P = [[max(narrow(narrow(s_acc[r][u], self.rq_s)
                         + (0 if u <= self.first_pos + r else -8),
                         self.rq_mask), 0)
              for u in range(tokens)] for r in range(self.rows)]

        # rq_a is fitted to the WHOLE contraction's accumulators, not to one
        # block's: a block's partial is a fraction of the total, so the word
        # that lands the total on the grid keeps every partial on it too.
        a_acc = [[sum(P[r][u] * V[u][j] for u in range(tokens))
                  for j in range(head_dim)] for r in range(self.rows)]
        self.rq_a = fit_rq((x for row in a_acc for x in row), "RQ_A")
        self.unsplit = [[narrow(x, self.rq_a) for x in row] for row in a_acc]

        self.out = self._tiled(P, V)
        self.drift = sum(1 for r in range(self.rows) for j in range(head_dim)
                         if self.out[r][j] != self.unsplit[r][j])

        self.defines = {"T": tokens, "ROWS": self.rows,
                        "FIRST_POS": self.first_pos, "HEAD_DIM": head_dim,
                        "ARENA_BANKS": f"{arena_banks}u", "SP_ARENA": "0x0000u",
                        "RQ_S": f"{self.rq_s}u", "RQ_MASK": f"{self.rq_mask}u",
                        "RQ_P": f"{self.rq_p}u", "RQ_A": f"{self.rq_a}u",
                        **self.map.defines()}

    def _tiled(self, P, V) -> list:
        """P @ V the way the kernel takes it: one key block per accumulate, and
        the MXU's accumulate is `clip4(requant(partial) + C_old)`. The panel
        loop is here too, because it is what bounds the key blocks visited."""
        B, DH = self.block, self.DH
        out = [[0] * DH for _ in range(self.rows)]

        for i in range(0, self.rows, B):
            panel = min(B, self.rows - i)
            last = self.first_pos + i + panel
            for r in range(i, i + panel):
                first = True
                for j in range(0, last, B):
                    cols = min(B, self.keys - j)
                    part = [sum(P[r][u] * V[u][c] for u in range(j, j + cols))
                            for c in range(DH)]
                    for c in range(DH):
                        n = narrow(part[c], self.rq_a)
                        out[r][c] = n if first else max(
                            Q4_MIN, min(Q4_MAX, n + out[r][c]))
                    first = False
        return out

    def dma_clocks(self) -> int:
        """A loose bound: every byte a clock on a fill, two on a spill, with
        every tile counted whether or not the causal skip drops it."""
        B, DH = self.block, self.DH
        panels = (self.rows + B - 1) // B
        tiles = panels * ((self.keys + B - 1) // B)
        fill = panels * B * i4_row(DH) \
            + tiles * (2 * B * i4_row(DH) + B * i4_row(B))
        return fill + 2 * self.rows * i4_row(DH)

    def static(self) -> dict:
        img: dict = {}
        put_rowmajor_i4(img, self.map["DR_Q"], self.rows, self.DH,
                        lambda r, i: q_val(self.first_pos + r, i))
        put_rowmajor_i4(img, self.map["DR_K"], self.keys, self.DH, k_val)
        put_rowmajor_i4(img, self.map["DR_V"], self.keys, self.DH, v_val)
        put_rowmajor_i4(img, self.map["DR_MASK"], self.keys, self.keys,
                        lambda t, u: 0 if u <= t else -8)
        put_rowmajor_i4(img, self.map["DR_OUT"], self.rows, self.DH,
                        lambda r, j: 0)
        return img

    def cases(self):
        golden: dict = {}
        put_rowmajor_i4(golden, self.map["DR_OUT"], self.rows, self.DH,
                        lambda r, j: self.out[r][j])
        blocks = (self.keys + self.block - 1) // self.block
        yield Case(name=f"T={self.keys} rows={self.rows}@{self.first_pos} "
                        f"head_dim={self.DH}, block={self.block} "
                        f"({blocks} key block(s)"
                        + (", ragged" if self.keys % self.block else "") + ")",
                   golden=golden,
                   check_ranges=[(self.map["DR_OUT"],
                                  self.rows * i4_row(self.DH))])


def program(backend, **kw):
    return TPUProgram(os.path.join(HERE, "flash.c"), backend, FlashVectors(**kw))


# (tokens, head_dim, rows, first_pos, arena_banks): one block, then several,
# then a ragged tail, then the decode shape — one query against a full cache.
SWEEP = [(32, 32, 0, 0, 5), (96, 32, 0, 0, 5), (128, 32, 0, 0, 6),
         (64, 64, 0, 0, 5), (96, 32, 1, 95, 5), (96, 32, 8, 40, 5)]


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("-T", "--tokens", type=int, default=32,
                    help=f"the key axis, a multiple of {TPU_N}")
    ap.add_argument("-d", "--head-dim", type=int, default=32,
                    help=f"the contraction of Q @ K', a multiple of {TPU_N}")
    ap.add_argument("--rows", type=int, default=0,
                    help="queries in the call (default: the whole key axis, "
                         "which is a prefill)")
    ap.add_argument("--first-pos", type=int, default=0,
                    help="query 0's position on the key axis")
    ap.add_argument("--arena-banks", type=int, default=5,
                    help="scratchpad banks tpulib.h may spend. five is the "
                         "minimum — Q, K, V, the output panel and the score "
                         "region are bank-disjoint. more banks buys a bigger "
                         "key block, which is what makes the tiling a no-op")
    ap.add_argument("--unsplit-check", action="store_true",
                    help="report how far the block-accumulated result drifts "
                         "from contracting every key in one dispatch")
    ap.add_argument("--sweep", action="store_true",
                    help="run every shape in SWEEP in turn")
    args = ap.parse_args()

    combos = SWEEP if args.sweep else [(args.tokens, args.head_dim, args.rows,
                                        args.first_pos, args.arena_banks)]
    failed = 0

    for tokens, head_dim, rows, first_pos, banks in combos:
        gen = FlashVectors(tokens=tokens, head_dim=head_dim, rows=rows,
                           first_pos=first_pos, arena_banks=banks)
        cells = gen.rows * head_dim
        print(f"-- T={tokens} head_dim={head_dim} rows={gen.rows}@{first_pos} "
              f"banks={banks} -> block={gen.block}")
        if args.unsplit_check:
            print(f"   split vs unsplit: {gen.drift}/{cells} elements differ "
                  f"({100 * gen.drift / cells:.2f}%)")
        watchdog = max(2_000_000, 400 * gen.dma_clocks())
        backend = backend_from_args(args, watchdog_ns=watchdog)
        try:
            prog = TPUProgram(os.path.join(HERE, "flash.c"), backend, gen)
            prog.run_program(limit=args.cases)
            failed += report(prog, args.clk_mhz)
        finally:
            backend.close()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
