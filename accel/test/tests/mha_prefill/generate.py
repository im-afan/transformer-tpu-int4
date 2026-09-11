#!/usr/bin/env python3
"""mha_prefill: the prefill on its own, for measuring what attention costs.

The kernel is infer.c's layer body with nothing around it — no decode steps, no
output head, no argmax. The counters reset at the launch and freeze at the halt,
so an image that runs only the prefill IS the measurement of the prefill, and
--part is the same trick one level down: attention alone, the FFN alone, or a
whole block, as three images whose clocks can be subtracted.

Everything is a knob: -d, -f/--dff, -L/--layers, --heads and --prompt.

    python accel/test/tests/mha_prefill/generate.py -b iss
    python accel/test/tests/mha_prefill/generate.py -b rtl -d 128 -f 512 -L 4 --prompt 32
    python accel/test/tests/mha_prefill/generate.py -b rtl --part split
    python accel/test/tests/mha_prefill/generate.py -b rtl --bench --part attn

The weights are mixed hashes, not a checkpoint: this measures a shape, and a
step's clock count is not data-dependent. The requant word for every site is
fitted to the accumulators the reference produced, so changing the shape cannot
silently collapse a tensor to zero or saturate it flat.

--bench is timing only: DRAM is staged zeroed instead of with weights and
nothing is checked, so the run is the kernel and its perf counters. It skips the
reference, so it is the fast way to time a big shape (and needs no numpy).
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
TESTROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
REPO = os.path.normpath(os.path.join(TESTROOT, "..", ".."))
for _p in (TESTROOT, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from export import DRAM_BYTES, RQ_IDX, RQ_N, RQ_NAMES    # noqa: E402
from program import (TPUProgram, backend_from_args, report,  # noqa: E402
                     standard_parser)
from vector_generator import (RQ_ONE, TPU_N, TPU_SPAD_BYTES,       # noqa: E402
                              Case, VectorGenerator, fit_rq, flash_block,
                              i4_row, put_i32, put_rowmajor_i4, w_hash,
                              zero_range)

BUILD = os.path.join(TESTROOT, "build", "mha_prefill")
ALIGN = 64

MM_MODES = {"base": 0, "dbuf": 1, "fused": 2}
PARTS = {"block": (1, 1), "attn": (1, 0), "ffn": (0, 1)}

# ID, P, XO and HR are never fitted: their word is an identity, because the add
# each one feeds takes two operands at one scale.
# What --bench builds with instead of a fitted table. Nothing is checked there,
# so these only have to be legal words. tests/infer/generate.py's table.
BENCH_RQ = {"Q": (1, 6), "K": (1, 6), "V": (1, 6), "S": (1, 4), "ID": (1, 0),
            "P": (1, 0), "A": (1, 6), "O": (1, 6), "XO": (1, 0), "X1": (1, 1),
            "H": (1, 6), "HR": (1, 0), "F": (1, 7), "X2": (1, 1)}


def emb_val(v: int, d: int) -> int:
    """A synthetic embedding row. Distinct per token id, or a wrong gather is
    invisible in the output."""
    return ((v * 7 + d * 3) % 9) - 4


# =============================================================================
# Shape and the DRAM map.
# =============================================================================
@dataclass(frozen=True)
class Shape:
    T: int = 32
    D: int = 64
    DFF: int = 256
    HEADS: int = 4
    LAYERS: int = 2
    VOCAB: int = 16
    PROMPT: int = 32
    BATCH: int = 1
    BLOCK: int = 32
    ATTN: int = 1
    FFN: int = 1

    @property
    def head_dim(self) -> int:
        return self.D // self.HEADS

    @property
    def block_rows(self) -> int:
        return min(self.PROMPT, self.BLOCK)

    @property
    def last_rows(self) -> int:
        """Rows of the pass that ran last — the rows X still holds at the halt."""
        return self.PROMPT % self.BLOCK or self.block_rows

    @property
    def rows_max(self) -> int:
        return self.BATCH * self.block_rows

    def check(self) -> None:
        for name, val in (("D", self.D), ("DFF", self.DFF), ("T", self.T),
                          ("head_dim", self.head_dim)):
            if val % TPU_N:
                raise SystemExit(f"{name} = {val} is not a whole number of "
                                 f"{TPU_N}-wide array tiles")
        if self.D % self.HEADS:
            raise SystemExit(f"D = {self.D} does not divide into "
                             f"{self.HEADS} heads")
        if self.PROMPT > self.T:
            raise SystemExit(f"PROMPT {self.PROMPT} runs past T = {self.T}")
        if not (self.ATTN or self.FFN):
            raise SystemExit("build at least one half of the layer")


def _align(addr: int) -> int:
    return (addr + ALIGN - 1) & ~(ALIGN - 1)


def dram_map(s: Shape) -> dict:
    """Every DRAM address the kernel uses. The caches and the activations are
    one contiguous span so the scratch the kernel may scribble on is one range.

    Attention's working set and the FFN's hidden layer never coexist, so they
    share one region — the same union infer.c's map has.
    """
    m: dict = {}
    m["DR_EMBED"] = 0
    m["DR_TOKENS"] = _align(m["DR_EMBED"] + s.VOCAB * i4_row(s.D))
    m["DR_MASK"] = _align(m["DR_TOKENS"] + s.BATCH * s.T * 4)

    m["DR_K_CACHE"] = _align(m["DR_MASK"] + s.T * i4_row(s.T))
    cache = s.BATCH * s.LAYERS * s.T * i4_row(s.D)
    m["DR_V_CACHE"] = _align(m["DR_K_CACHE"] + cache)
    m["DR_X"] = _align(m["DR_V_CACHE"] + cache)
    m["DR_TMP_A"] = _align(m["DR_X"] + s.rows_max * i4_row(s.D))
    m["DR_TMP_B"] = _align(m["DR_TMP_A"] + s.rows_max * i4_row(s.D))

    scratch = _align(m["DR_TMP_B"] + s.rows_max * i4_row(s.D))
    m["DR_Q"] = scratch
    m["DR_S"] = _align(scratch + s.rows_max * i4_row(s.D))
    m["DR_H"] = scratch
    attn_end = _align(m["DR_S"] + s.block_rows * i4_row(s.T))
    ffn_end = _align(m["DR_H"] + s.rows_max * i4_row(s.DFF))
    m["DR_ACT_END"] = max(attn_end, ffn_end)

    m["LW_WQ"] = 0
    m["LW_WK"] = _align(m["LW_WQ"] + s.D * i4_row(s.D))
    m["LW_WV"] = _align(m["LW_WK"] + s.D * i4_row(s.D))
    m["LW_WO"] = _align(m["LW_WV"] + s.D * i4_row(s.D))
    m["LW_FF1"] = _align(m["LW_WO"] + s.D * i4_row(s.D))
    m["LW_FF2"] = _align(m["LW_FF1"] + s.D * i4_row(s.DFF))
    m["DR_LAYER_STRIDE"] = _align(m["LW_FF2"] + s.DFF * i4_row(s.D))
    m["DR_LAYER0"] = _align(m["DR_ACT_END"])

    end = m["DR_LAYER0"] + s.LAYERS * m["DR_LAYER_STRIDE"]
    if end > DRAM_BYTES:
        raise SystemExit(f"the map needs {end} bytes of a {DRAM_BYTES}-byte "
                         f"SRAM. Lower --block, then -L, then --batch.")
    m["DR_END"] = end
    return m


def watchdog_ns(s: Shape, clk_mhz: float) -> int:
    """A ceiling on the simulated run, from the shape. The array's MACs at 64 a
    clock plus the weight stream at a byte a clock, times ten — a prefill runs
    at about twice that estimate and the margin is what keeps a slow shape from
    reading as a deadlock."""
    rows = s.PROMPT * s.BATCH
    per_layer = (rows * (6 * s.D * s.D + 2 * s.D * s.DFF) // 64
                 + rows * s.HEADS * s.head_dim * s.T * 2 // 64
                 + (4 * s.D * s.D + 2 * s.D * s.DFF) // 2)
    return int((10 * s.LAYERS * per_layer + 100_000) * 1000 / clk_mhz)


def layer_base(m: dict, layer: int) -> int:
    return m["DR_LAYER0"] + layer * m["DR_LAYER_STRIDE"]


def k_cache(s: Shape, m: dict, seq: int, layer: int) -> int:
    return m["DR_K_CACHE"] + (seq * s.LAYERS + layer) * s.T * i4_row(s.D)


def v_cache(s: Shape, m: dict, seq: int, layer: int) -> int:
    return m["DR_V_CACHE"] + (seq * s.LAYERS + layer) * s.T * i4_row(s.D)


def write_config(path: str, s: Shape, m: dict, rq_table: list) -> None:
    shape_lines = [("T", s.T), ("D", s.D), ("DFF", s.DFF), ("HEADS", s.HEADS),
                   ("HEAD_DIM", s.head_dim), ("LAYERS", s.LAYERS),
                   ("VOCAB", s.VOCAB), ("PROMPT", s.PROMPT),
                   ("BATCH", s.BATCH), ("BLOCK", s.BLOCK),
                   ("PART_ATTN", s.ATTN), ("PART_FFN", s.FFN)]
    order = ["DR_EMBED", "DR_TOKENS", "DR_MASK", "DR_K_CACHE", "DR_V_CACHE",
             "DR_X", "DR_TMP_A", "DR_TMP_B", "DR_Q", "DR_S", "DR_H",
             "DR_ACT_END", "DR_LAYER0", "DR_LAYER_STRIDE",
             "LW_WQ", "LW_WK", "LW_WV", "LW_WO", "LW_FF1", "LW_FF2"]
    rows = [f"    /* layer {L} */ {{ " + ", ".join(f"0x{w:04x}u" for w in row)
            + " }" for L, row in enumerate(rq_table)]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"""/* Generated by tests/mha_prefill/generate.py — do not edit.
 *
 * Shape, the whole DRAM map, and {RQ_N} requant words per layer in the order
 * mha_prefill.c's enum declares: {', '.join(RQ_NAMES)}.
 */
#ifndef MHA_PREFILL_CONFIG_H
#define MHA_PREFILL_CONFIG_H

""")
        for name, val in shape_lines:
            f.write(f"#define {name:<16} {val}\n")
        f.write("\n/* DRAM, byte addresses. */\n")
        for name in order:
            f.write(f"#define {name:<16} 0x{m[name]:05x}u\n")
        f.write(f"\n#define INFER_RQ_SITES  {RQ_N}\n")
        f.write("#define INFER_RQ_INIT { \\\n")
        f.write(", \\\n".join(rows))
        f.write(" \\\n}\n\n#endif\n")


# =============================================================================
# The integer reference. numpy, because a pure-Python one is minutes at d=128.
# =============================================================================
def _rq(acc, word, lo=-8):
    """clip((acc*m0 + 2**(n-1)) >> n) over a numpy array. numpy's >> on a signed
    integer floors, which is what Verilog's >>> does."""
    import numpy as np

    m0, n = word & 0xFFF, (word >> 12) & 0xF
    v = (acc.astype(np.int64) * m0 + ((1 << (n - 1)) if n else 0)) >> n
    return np.clip(v, lo, 7)


class Reference:
    """The prefill as integers: every row at once, causal.

    The kernel contracts over all T keys and the cache past the prompt is zero,
    so a key the prompt never wrote scores 0, the mask takes it to -8 and the
    relu to 0 — which is why a reference over the prompt alone is exact.
    """

    def __init__(self, s: Shape, emb, layers, pv_block: int = 0):
        self.s, self.emb, self.layers = s, emb, layers
        # Non-zero under --attn flash: P @ V is contracted one key block at a
        # time and the MXU's accumulate is an int4 add, so the golden has to
        # clip where the hardware does. A block as long as the key axis is one
        # dispatch and one clip, which is what the live shapes come out at.
        self.pv_block = pv_block

    def _pv(self, P, V, word):
        """One head's P @ V, taken the way the kernel takes it: a key block per
        accumulate, and the MXU's accumulate is clip4(requant(partial) + C_old).
        Blocks past a row's own position contribute an all-zero partial, so a
        row does not depend on how far the panel's block loop ran."""
        import numpy as np

        keys = P.shape[-1]
        if not self.pv_block or self.pv_block >= keys:
            return _rq(P @ V, word)
        acc = None
        for j in range(0, keys, self.pv_block):
            part = _rq(P[..., j:j + self.pv_block] @ V[:, j:j + self.pv_block],
                       word)
            acc = part if acc is None else np.clip(acc + part, -8, 7)
        return acc

    def run(self, prompts: list, table: list | None = None) -> tuple:
        """(table, K, V, X). With `table` None every site's word is fitted to
        the accumulators as they are produced, which is what makes the shape a
        knob."""
        import numpy as np

        s = self.s
        n = s.PROMPT
        fitting = table is None
        table = [[RQ_ONE] * RQ_N for _ in range(s.LAYERS)] if fitting else table
        mask = np.where(np.arange(n)[None, :] <= np.arange(n)[:, None], 0, -8)

        X = np.stack([self.emb[p] for p in prompts]).astype(np.int64)  # [B][n][D]
        K_out = np.zeros((s.BATCH, s.LAYERS, n, s.D), dtype=np.int64)
        V_out = np.zeros_like(K_out)

        for L in range(s.LAYERS):
            w, rq = self.layers[L], table[L]

            def fit(acc, site):
                """The word for `site`, fitted to these accumulators the first
                time round and read back from the frozen table after."""
                if fitting:
                    rq[RQ_IDX[site]] = fit_rq(acc.ravel().tolist(),
                                              f"L{L} RQ_{site}")
                return rq[RQ_IDX[site]]

            X1 = X
            if s.ATTN:
                q_acc, k_acc, v_acc = X @ w["q"], X @ w["k"], X @ w["v"]
                Q = _rq(q_acc, fit(q_acc, "Q"))
                K = _rq(k_acc, fit(k_acc, "K"))
                V = _rq(v_acc, fit(v_acc, "V"))
                K_out[:, L], V_out[:, L] = K, V

                heads = [(h * s.head_dim, (h + 1) * s.head_dim)
                         for h in range(s.HEADS)]
                s_acc = np.stack([Q[:, :, a:b] @ K[:, :, a:b].transpose(0, 2, 1)
                                  for a, b in heads], axis=1)   # [B][H][n][n]
                S = _rq(_rq(s_acc, fit(s_acc, "S")) + mask, RQ_ONE)
                P = _rq(np.maximum(S, 0), RQ_ONE)
                a_acc = np.stack([P[:, h] @ V[:, :, a:b]
                                  for h, (a, b) in enumerate(heads)], axis=1)
                rq_a = fit(a_acc, "A")
                A = np.stack([self._pv(P[:, h], V[:, :, a:b], rq_a)
                              for h, (a, b) in enumerate(heads)], axis=1)
                A = np.concatenate([A[:, h] for h in range(s.HEADS)], axis=2)

                o_acc = A @ w["o"]
                O = _rq(o_acc, fit(o_acc, "O"))
                XO = _rq(X + O, RQ_ONE)
                X1 = _rq(XO + X, fit(XO + X, "X1"), lo=-7)      # dyt

            if s.FFN:
                h_acc = X1 @ w["w1"]
                H = _rq(h_acc, fit(h_acc, "H"))
                f_acc = np.maximum(H, 0) @ w["w2"]
                F = _rq(f_acc, fit(f_acc, "F"))
                X = _rq(X1 + F, fit(X1 + F, "X2"), lo=-7)       # dyt
            else:
                X = X1

        return table, K_out, V_out, X


# =============================================================================
# The vectors.
# =============================================================================
class PrefillVectors(VectorGenerator):
    def __init__(self, shape: Shape, problems: int, seed: int = 0,
                 wide: bool = True, bench: bool = False, mm: str = "dbuf",
                 attn: str = "blocks"):
        self.s = shape
        self.wide, self.mm, self.bench = wide, mm, bench
        self.attn = attn
        # The kernel hands tpulib.h everything below the mailbox, and the key
        # block size follows from that.
        self.pv_block = (flash_block(TPU_SPAD_BYTES - shape.BATCH * shape.T * 4,
                                     shape.T, shape.head_dim)
                         if attn == "flash" else 0)
        self.problems, self.seed = problems, seed
        self.map = dram_map(shape)
        self._weights()

        if bench:
            self.rq_table = [[(n << 12) | m0 for m0, n in
                              (BENCH_RQ[name] for name in RQ_NAMES)]
                             for _ in range(shape.LAYERS)]
        else:
            self.reference = Reference(shape, self.emb, self.layers,
                                       self.pv_block)
            # Fitted on the first case's prompt and then frozen: the words are
            # compiled into the image, so every case has to run on one table.
            self.rq_table, _, _, _ = self.reference.run(self._prompt(0))

        write_config(os.path.join(BUILD, "mha_prefill_config.h"), self.s,
                     self.map, self.rq_table)

    # ---- weights ------------------------------------------------------------
    def _weights(self) -> None:
        import numpy as np

        s = self.s

        def block(rows, cols, salt):
            return np.array([[w_hash(r, c, salt) for c in range(cols)]
                             for r in range(rows)], dtype=np.int64)

        self.emb = np.array([[emb_val(v, d) for d in range(s.D)]
                             for v in range(s.VOCAB)], dtype=np.int64)
        self.layers = []
        for L in range(s.LAYERS):
            self.layers.append(
                {"q": block(s.D, s.D, 6 * L + 1), "k": block(s.D, s.D, 6 * L + 2),
                 "v": block(s.D, s.D, 6 * L + 3), "o": block(s.D, s.D, 6 * L + 4),
                 "w1": block(s.D, s.DFF, 6 * L + 5),
                 "w2": block(s.DFF, s.D, 6 * L + 6)})

    def _prompt(self, case: int) -> list:
        """One BATCH-wide prompt of token ids. Which ids they are only decides
        which embedding rows are gathered; a step costs the same clocks."""
        rng = random.Random(self.seed * 1000 + case)
        return [[rng.randrange(self.s.VOCAB) for _ in range(self.s.PROMPT)]
                for _ in range(self.s.BATCH)]

    # ---- the images ---------------------------------------------------------
    @property
    def defines(self) -> dict:
        d = {"INFER_MM_MODE": MM_MODES[self.mm]}
        if self.attn == "flash":
            d["INFER_ATTN_FLASH"] = 1
        return d if self.wide else {**d, "INFER_MM_WIDE": 0}

    def static(self) -> dict:
        """--bench stages a zeroed map instead of weights, embeddings and a
        mask: a prefill's clock count is not data-dependent, and the simulation
        needs every byte the kernel reads to be a byte rather than an x."""
        s, m = self.s, self.map
        if self.bench:
            img = {}
            zero_range(img, 0, m["DR_END"])
            return img
        img: dict = {}

        put_rowmajor_i4(img, m["DR_EMBED"], s.VOCAB, s.D,
                        lambda v, d: int(self.emb[v][d]))
        put_rowmajor_i4(img, m["DR_MASK"], s.T, s.T,
                        lambda t, k: 0 if k <= t else -8)
        for L in range(s.LAYERS):
            base = layer_base(m, L)
            w = self.layers[L]
            for key, off, rows, cols in (("q", "LW_WQ", s.D, s.D),
                                         ("k", "LW_WK", s.D, s.D),
                                         ("v", "LW_WV", s.D, s.D),
                                         ("o", "LW_WO", s.D, s.D),
                                         ("w1", "LW_FF1", s.D, s.DFF),
                                         ("w2", "LW_FF2", s.DFF, s.D)):
                put_rowmajor_i4(img, base + m[off], rows, cols,
                                lambda r, c, b=w[key]: int(b[r][c]))

        # The cache past the prompt is read before it is written; zeroing it is
        # what makes the ISS and a board that kept the last run's bytes agree.
        cache = s.BATCH * s.LAYERS * s.T * i4_row(s.D)
        zero_range(img, m["DR_K_CACHE"], cache)
        zero_range(img, m["DR_V_CACHE"], cache)
        return img

    def writable_ranges(self) -> list:
        m = self.map
        return [(m["DR_K_CACHE"], m["DR_ACT_END"] - m["DR_K_CACHE"])]

    def cases(self):
        s, m = self.s, self.map
        for i in range(self.problems):
            prompts = self._prompt(i)
            patch = {}
            for seq, ids in enumerate(prompts):
                put_i32(patch, m["DR_TOKENS"] + seq * s.T * 4, ids)
            if self.bench:
                yield Case(name=f"bench {i}", patch=patch, golden={},
                           check_ranges=[])
                continue

            _, K, V, X = self.reference.run(prompts, self.rq_table)
            golden, ranges = {}, []
            if s.ATTN:
                for seq in range(s.BATCH):
                    for L in range(s.LAYERS):
                        for base, ref in ((k_cache(s, m, seq, L), K),
                                          (v_cache(s, m, seq, L), V)):
                            put_rowmajor_i4(golden, base, s.PROMPT, s.D,
                                            lambda t, d, r=ref, q=seq, l=L:
                                            int(r[q][l][t][d]))
                            ranges.append((base, s.PROMPT * i4_row(s.D)))
            # X holds only the rows of the pass that ran last.
            first = s.PROMPT - s.last_rows
            for seq in range(s.BATCH):
                base = m["DR_X"] + seq * s.last_rows * i4_row(s.D)
                put_rowmajor_i4(golden, base, s.last_rows, s.D,
                                lambda t, d, q=seq: int(X[q][first + t][d]))
                ranges.append((base, s.last_rows * i4_row(s.D)))
            yield Case(name=f"prefill {i}: {s.PROMPT} tokens", patch=patch,
                       golden=golden, check_ranges=ranges)


# =============================================================================
def program(backend, shape: Shape | None = None, problems: int = 2,
            seed: int = 0, wide: bool = True, bench: bool = False,
            mm: str = "dbuf", attn: str = "blocks"):
    shape = shape or Shape()
    shape.check()
    gen = PrefillVectors(shape, problems, seed, wide, bench, mm, attn)
    return TPUProgram(os.path.join(HERE, "mha_prefill.c"), backend, gen,
                      include_dirs=[BUILD])


def shape_for(args, part: str) -> Shape:
    attn, ffn = PARTS[part]
    shape = Shape(T=args.tokens or args.prompt, D=args.d, DFF=args.dff,
                  HEADS=args.heads, LAYERS=args.layers, VOCAB=args.vocab,
                  PROMPT=args.prompt, BATCH=args.batch, BLOCK=args.block,
                  ATTN=attn, FFN=ffn)
    shape.check()
    return shape


def run_part(args, backend, part: str) -> TPUProgram:
    prog = program(backend, shape_for(args, part),
                   args.cases or (1 if args.bench else 2), args.seed,
                   not args.general, args.bench, args.mm, args.attn)
    prog.name = "mha_prefill" if part == "block" else f"mha_prefill {part}-only"
    prog.run_program()
    return prog


def split_summary(progs: dict, s: Shape, clk_mhz: float) -> str:
    """The three images side by side. They are separate runs, so attn + ffn is
    not exactly block — each image pays the prompt load once — but the gap is a
    few hundred clocks against a prefill."""
    bench = {name: p.benchmark(clk_mhz) for name, p in progs.items()}
    if any(b is None for b in bench.values()):
        cmds = {name: [r.n_cmds for r in p.results if r.n_cmds is not None]
                for name, p in progs.items()}
        if not all(cmds.values()):
            return ""
        lines = ["  (the ISS has no cycle model — commands only)"]
        for name in progs:
            mean = sum(cmds[name]) / len(cmds[name])
            lines.append(f"  {name:<8} {mean:>12.0f} commands   "
                         f"{mean / s.PROMPT:>10.1f} per prompt token")
        return "\n".join(lines)

    lines = [f"  {'part':<8} {'clocks':>12} {'ms':>10}   {'per token':>12}"
             f" {'ms':>9}   {'mxu':>6} {'dma':>6} {'idlec':>6}"]
    for name in progs:
        b = bench[name]
        run, per = b["run_mean"], b["run_mean"] / s.PROMPT
        share = b["share"]
        lines.append(
            f"  {name:<8} {run:>12.0f} {run / (clk_mhz * 1e3):>10.3f}   "
            f"{per:>12.0f} {per / (clk_mhz * 1e3):>9.3f}   "
            f"{100 * share.get('mxu', 0):>5.1f}% {100 * share.get('dma', 0):>5.1f}%"
            f" {100 * share.get('idlec', 0):>5.1f}%")
    if "attn" in bench and "block" in bench:
        attn, block = bench["attn"]["run_mean"], bench["block"]["run_mean"]
        lines.append(f"  attention is {100 * attn / block:.1f}% of the block")
    return "\n".join(lines)


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("--prompt", type=int, default=32, help="prefill size")
    ap.add_argument("-T", "--tokens", type=int, default=None,
                    help="sequence length the cache is sized for "
                         "(default: --prompt)")
    ap.add_argument("-d", type=int, default=64, help="model width")
    ap.add_argument("-f", "--dff", type=int, default=256,
                    help="feed-forward width")
    ap.add_argument("-L", "--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1,
                    help="sequences sharing one weight stream")
    ap.add_argument("--block", type=int, default=32,
                    help="prefill rows per sequence per pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true",
                    help="timing only: stage a zeroed DRAM instead of "
                         "weights and check nothing, just run and read the "
                         "perf counters")
    ap.add_argument("--mm", choices=tuple(MM_MODES), default="dbuf",
                    help="which rung of the matmul ladder the image is built "
                         "on, the same three infer has")
    ap.add_argument("--attn", choices=("blocks", "flash"), default="blocks",
                    help="how attention runs: blocks is the score matmul, the "
                         "mask add, the relu and P@V as four passes over DRAM; "
                         "flash is one tpu_flashattention with the scores in "
                         "the scratchpad. infer.c's --attn, kept in step")
    ap.add_argument("--general", action="store_true",
                    help="every matmul through tpu_matmul instead of "
                         "tpu_matmul_wide — the A/B")
    ap.add_argument("--part", choices=("block", "attn", "ffn", "split"),
                    default="block",
                    help="which half of the layer the image runs. split builds "
                         "all three in turn and prints them side by side")
    args = ap.parse_args()

    backend = backend_from_args(
        args, watchdog_ns=watchdog_ns(shape_for(args, "block"), args.clk_mhz),
        max_cmds=1 << 16)

    if args.part != "split":
        return report(run_part(args, backend, args.part), args.clk_mhz)

    progs, rc = {}, 0
    for part in ("attn", "ffn", "block"):
        progs[part] = run_part(args, backend, part)
        rc |= report(progs[part], args.clk_mhz)
    s = shape_for(args, "block")
    summary = split_summary(progs, s, args.clk_mhz)
    if summary:
        print(f"mha_prefill parts on {backend.name}, d={s.D} f={s.DFF} "
              f"L={s.LAYERS} heads={s.HEADS} PROMPT={s.PROMPT} "
              f"BATCH={s.BATCH}:")
        print(summary)
        print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
