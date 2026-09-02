#!/usr/bin/env python3
"""export.py — a checkpoint to the two things the device needs.

    the config header   shape, the whole DRAM map, and the requant table,
                        included by tests/infer/infer.c
    the static image    every int4 weight, the causal mask, the output head
                        and the embedding table, as {addr: byte}

Python owns the addresses. The kernel used to derive them itself from its own
shape defines, which meant the map existed in C and again in whatever staged
DRAM; here it is computed once and the C file is handed the answer. See
accel/test/README.md.

    python -m accel.test.export --model-path model/saved/int4_d128_f512_l4.pt --dump-rq
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vector_generator import (M0_W, N_W, Q4_MAX, Q4_MIN, RQ_ONE,  # noqa: E402
                              fixed_point, i4_row, put_rowmajor_i4, zero_range)

DRAM_BYTES = 1 << 19
ALIGN = 64


# =============================================================================
# Shape and the DRAM map.
# =============================================================================
@dataclass(frozen=True)
class Shape:
    """Everything the kernel's addresses and loop bounds depend on."""

    T: int = 64                 # sequence length
    D: int = 128                # model width
    DFF: int = 512              # feed-forward width
    HEADS: int = 4
    LAYERS: int = 4
    VOCAB: int = 13
    VOCAB_PAD: int = 16         # the logits padded to a whole array tile
    PROMPT: int = 32            # '=' sits at PROMPT-1; the answer starts here
    GEN: int = 32               # tokens to generate, the prefill's included
    BATCH: int = 1              # sequences sharing one weight stream
    BLOCK: int = 32             # prefill rows per sequence per pass
    PREFILL: int = 1
    DECODE: int = 1
    ARRAY_N: int = 8            # the MXU's tile edge

    @property
    def head_dim(self) -> int:
        return self.D // self.HEADS

    @property
    def max_seq_rows(self) -> int:
        return min(self.PROMPT, self.BLOCK) if self.PREFILL else 1

    @property
    def rows_max(self) -> int:
        return self.BATCH * self.max_seq_rows

    def check(self) -> None:
        n = self.ARRAY_N
        for name, val in (("D", self.D), ("DFF", self.DFF), ("T", self.T),
                          ("head_dim", self.head_dim), ("VOCAB_PAD", self.VOCAB_PAD)):
            if val % n:
                raise SystemExit(f"{name} = {val} is not a whole number of "
                                 f"{n}-wide array tiles")
        if self.D % self.HEADS:
            raise SystemExit(f"D = {self.D} does not divide into {self.HEADS} heads")
        if self.VOCAB > self.VOCAB_PAD:
            raise SystemExit(f"VOCAB {self.VOCAB} > VOCAB_PAD {self.VOCAB_PAD}")
        if self.PROMPT + self.GEN > self.T:
            raise SystemExit(f"PROMPT {self.PROMPT} + GEN {self.GEN} runs past "
                             f"T = {self.T}")
        if self.DECODE and self.GEN < 2:
            raise SystemExit("a decode step needs GEN >= 2 (GEN counts the "
                             "prefill's token too)")
        if not (self.PREFILL or self.DECODE):
            raise SystemExit("build at least one phase")


def _align(addr: int) -> int:
    return (addr + ALIGN - 1) & ~(ALIGN - 1)


def dram_map(s: Shape) -> dict:
    """Every DRAM address the kernel uses, computed off the shape.

    Attention's working set and the FFN's hidden layer never coexist, so they
    share one region. X, TMP_A and TMP_B are the only buffers live across a
    whole layer.
    """
    m: dict = {}
    m["DR_EMBED"] = 0
    m["DR_TOKENS"] = _align(m["DR_EMBED"] + s.VOCAB * i4_row(s.D))
    m["DR_LOGITS"] = _align(m["DR_TOKENS"] + s.BATCH * s.T * 4)
    m["DR_MASK"] = _align(m["DR_LOGITS"] + s.BATCH * s.T * i4_row(s.VOCAB_PAD))
    m["DR_HEAD_WGT"] = _align(m["DR_MASK"] + s.T * i4_row(s.T))
    m["DR_K_CACHE"] = _align(m["DR_HEAD_WGT"] + s.D * i4_row(s.VOCAB_PAD))
    cache = s.BATCH * s.LAYERS * s.T * i4_row(s.D)
    m["DR_V_CACHE"] = _align(m["DR_K_CACHE"] + cache)
    m["DR_X"] = _align(m["DR_V_CACHE"] + cache)
    m["DR_TMP_A"] = _align(m["DR_X"] + s.rows_max * i4_row(s.D))
    m["DR_TMP_B"] = _align(m["DR_TMP_A"] + s.rows_max * i4_row(s.D))

    scratch = _align(m["DR_TMP_B"] + s.rows_max * i4_row(s.D))
    m["DR_Q"] = scratch
    m["DR_S"] = _align(scratch + s.rows_max * i4_row(s.D))
    m["DR_H"] = scratch
    attn_end = _align(m["DR_S"] + s.max_seq_rows * i4_row(s.T))
    ffn_end = _align(m["DR_H"] + s.rows_max * i4_row(s.DFF))
    m["DR_ACT_END"] = max(attn_end, ffn_end)

    # Layer weights, densely packed above the activations.
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
        raise SystemExit(
            f"the map needs {end} bytes of a {DRAM_BYTES}-byte SRAM. Two levers, "
            f"in this order: lower BLOCK (scales X, TMP_A, TMP_B and the scratch "
            f"union, at one weight stream per extra pass), then lower BATCH.")
    m["DR_END"] = end
    print(f"{end} bytes of SRAM in used total (including weights and KV cache)")
    return m


def layer_base(m: dict, layer: int) -> int:
    return m["DR_LAYER0"] + layer * m["DR_LAYER_STRIDE"]


def k_cache(s: Shape, m: dict, seq: int, layer: int) -> int:
    return m["DR_K_CACHE"] + (seq * s.LAYERS + layer) * s.T * i4_row(s.D)


def v_cache(s: Shape, m: dict, seq: int, layer: int) -> int:
    return m["DR_V_CACHE"] + (seq * s.LAYERS + layer) * s.T * i4_row(s.D)


def token_addr(s: Shape, m: dict, seq: int, pos: int) -> int:
    return m["DR_TOKENS"] + (seq * s.T + pos) * 4


def logit_addr(s: Shape, m: dict, seq: int, pos: int) -> int:
    return m["DR_LOGITS"] + (seq * s.T + pos) * i4_row(s.VOCAB_PAD)


# =============================================================================
# Fixed point. `fixed_point` lives in vector_generator so the kernel tests and
# the exporter cannot disagree about what a requant word means.
# =============================================================================
def rq_for_scale(mult: float, what: str) -> int:
    """A requant word from a real multiplier `s_in * s_weight / s_out`.

    Distinct from `vector_generator.rq_word`, which takes the two fields
    directly — this one is the scale arithmetic that produces them.
    """
    m0, n = fixed_point(mult, what)
    return (n << M0_W) | m0

# The requant sites, in the order tests/infer/infer.c's enum declares them.
# Changing this list changes that enum; INFER_RQ_SITES is what keeps them honest.
RQ_NAMES = ["Q", "K", "V", "S", "ID", "P", "A", "O", "XO", "X1", "H", "HR",
            "F", "X2"]
RQ_N = len(RQ_NAMES)
RQ_IDX = {n: i for i, n in enumerate(RQ_NAMES)}


# =============================================================================
# The checkpoint.
# =============================================================================
def load_checkpoint(path: str, heads: int = 4):
    """A QAT checkpoint, with the model rebuilt at the shape it was saved at."""
    import torch

    import model.transformer as transformer

    full = path if os.path.isabs(path) else os.path.join(REPO, path)
    state = torch.load(full, map_location="cpu")

    vocab, d = state["embedding.weight"].shape
    f = state["layers.0.ff.0.w"].shape[1]
    layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("layers."))

    net = transformer.Model(vocab, d=d, f=f, layers=layers, q_heads=heads,
                            kv_heads=heads, use_int4=True, use_bias=False)
    net.load_state_dict(state)
    net.eval()
    return net


def shape_of(net, **overrides) -> Shape:
    """The `Shape` a loaded checkpoint implies, before the run-time knobs."""
    base = dict(D=net.d, DFF=net.f, HEADS=net.q_heads, LAYERS=len(net.layers),
                VOCAB=net.vocab_size)
    base.update(overrides)
    s = Shape(**base)
    s.check()
    return s


def int4_weight(w) -> tuple:
    """Int4Linear's own quantization: codes in [-8, 7] and one scale.

    `w` is [in_dim, out_dim] — the transpose of nn.Linear, and already the
    array's row-major orientation, so nothing is transposed on the way to DRAM.
    """
    import torch

    import model.transformer as transformer

    scale = float((w.detach().abs().max() / transformer.INT4_QMAX).clamp_min(1e-5))
    codes = (w.detach() / scale).round().clamp(Q4_MIN, Q4_MAX).to(torch.int64)
    return codes, scale


def act_scale(q) -> float:
    """The site's scale as ActQuant.forward uses it: magnitude, floored. LSQ can
    drive a learned scale negative and the forward pass takes its absolute."""
    return float(q.scale.detach().abs().clamp_min(1e-8))


def derive(net) -> tuple:
    """(rq_table, weights): RQ_N words per layer, and every int4 block.

    Raises rather than exporting something wrong when the checkpoint needs a
    scale this ISA cannot express — `requant` carries one {m0,n} per dispatch,
    so the two operands of a residual add have to already share a scale.
    """
    rq_table, weights = [], {}
    s_x = act_scale(net.q_embed)                # layer 0 enters on the embedding

    for L, lay in enumerate(net.layers):
        att = lay.attention
        s_q, s_k, s_v = act_scale(att.q_q), act_scale(att.q_k), act_scale(att.q_v)
        s_s, s_a = act_scale(att.q_s), act_scale(att.q_a)
        s_x1, s_x2 = act_scale(lay.q_x1), act_scale(lay.q_x2)
        s_h, s_f = act_scale(lay.q_h), act_scale(lay.q_f)
        alpha1, alpha2 = float(lay.norm1.alpha), float(lay.norm2.alpha)

        for site, pinned, name in ((att.q_o, s_x, "RQ_O"), (att.q_xo, s_x, "RQ_XO"),
                                   (lay.q_hr, s_h, "RQ_HR"), (att.q_p, s_s, "RQ_P")):
            got = act_scale(site)
            if abs(got - pinned) > 1e-9 * max(1.0, pinned):
                raise SystemExit(
                    f"layer {L}: {name}'s site is at {got:.6g} but the residual/"
                    f"identity it feeds needs {pinned:.6g}. A vector add takes "
                    f"two operands at ONE scale — this ISA cannot express it.")
        if abs(s_f - s_x1) > 1e-9 * max(1.0, s_x1):
            raise SystemExit(f"layer {L}: RQ_F is at {s_f:.6g} but the second "
                             f"residual add needs s_x1 = {s_x1:.6g}")

        blocks = {"q": att.Wq.w, "k": att.Wk.w, "v": att.Wv.w, "o": att.Wo.w,
                  "w1": lay.ff[0].w, "w2": lay.ff[2].w}
        scales = {}
        for key, tensor in blocks.items():
            weights[(L, key)], scales[key] = int4_weight(tensor)

        head_dim = net.d // net.q_heads
        row = [0] * RQ_N
        row[RQ_IDX["Q"]] = rq_for_scale(s_x * scales["q"] / s_q, f"L{L} RQ_Q")
        row[RQ_IDX["K"]] = rq_for_scale(s_x * scales["k"] / s_k, f"L{L} RQ_K")
        row[RQ_IDX["V"]] = rq_for_scale(s_x * scales["v"] / s_v, f"L{L} RQ_V")
        row[RQ_IDX["S"]] = rq_for_scale(s_q * s_k / (math.sqrt(head_dim) * s_s),
                                   f"L{L} RQ_S")
        row[RQ_IDX["ID"]] = row[RQ_IDX["P"]] = RQ_ONE
        row[RQ_IDX["A"]] = rq_for_scale(s_s * s_v / s_a, f"L{L} RQ_A")
        row[RQ_IDX["O"]] = rq_for_scale(s_a * scales["o"] / s_x, f"L{L} RQ_O")
        row[RQ_IDX["XO"]] = RQ_ONE
        row[RQ_IDX["X1"]] = rq_for_scale(alpha1 * s_x / s_x1, f"L{L} RQ_X1")
        row[RQ_IDX["H"]] = rq_for_scale(s_x1 * scales["w1"] / s_h, f"L{L} RQ_H")
        row[RQ_IDX["HR"]] = RQ_ONE
        row[RQ_IDX["F"]] = rq_for_scale(s_h * scales["w2"] / s_x1, f"L{L} RQ_F")
        row[RQ_IDX["X2"]] = rq_for_scale(alpha2 * s_x1 / s_x2, f"L{L} RQ_X2")
        rq_table.append(row)

        s_x = s_x2      # every later layer enters on the previous DyT

    weights["fc"], _ = int4_weight(net.fc.w)                   # [D][VOCAB]
    return rq_table, weights


def logit_rq_word(fc_codes) -> int:
    """The output head's {m0,n}, which is not per-layer.

    The head requantizes on store, so a logit is int4 and this shift is what
    decides whether an argmax over the vocabulary separates anything. The
    residual stream reaching the head is a DyT output, bounded by 7; the largest
    accumulator a column can reach is that times the column's absolute weight
    sum, and mapping it onto the top of the int4 grid is the multiplier that
    cannot clip.
    """
    largest = int(Q4_MAX * fc_codes.abs().sum(dim=0).max())
    return rq_for_scale(Q4_MAX / largest, "RQ_LOGIT")


# =============================================================================
# The config header.
# =============================================================================
def write_config(path: str, s: Shape, mapping: dict, rq_table: list,
                 logit_word: int, note: str = "") -> None:
    """The one file the kernel is configured by."""
    shape_lines = [
        ("T", s.T), ("D", s.D), ("DFF", s.DFF), ("HEADS", s.HEADS),
        ("HEAD_DIM", s.head_dim), ("LAYERS", s.LAYERS), ("VOCAB", s.VOCAB),
        ("VOCAB_PAD", s.VOCAB_PAD), ("PROMPT", s.PROMPT), ("INFER_GEN", s.GEN),
        ("BATCH", s.BATCH), ("BLOCK", s.BLOCK),
        ("INFER_PREFILL", s.PREFILL), ("INFER_DECODE", s.DECODE),
    ]
    order = ["DR_EMBED", "DR_TOKENS", "DR_LOGITS", "DR_MASK", "DR_HEAD_WGT",
             "DR_K_CACHE", "DR_V_CACHE", "DR_X", "DR_TMP_A", "DR_TMP_B",
             "DR_Q", "DR_S", "DR_H", "DR_ACT_END",
             "DR_LAYER0", "DR_LAYER_STRIDE",
             "LW_WQ", "LW_WK", "LW_WV", "LW_WO", "LW_FF1", "LW_FF2"]

    rows = []
    for L, row in enumerate(rq_table):
        cells = ", ".join(f"0x{w:04x}u" for w in row)
        rows.append(f"    /* layer {L} */ {{ {cells} }}")

    with open(path, "w") as f:
        f.write(f"""/* Generated by accel/test/export.py — do not edit.
 *
 * {note or 'synthetic'}
 *
 * Shape, the whole DRAM map, and {RQ_N} requant words per layer in the order
 * infer.c's enum declares: {', '.join(RQ_NAMES)}.
 */
#ifndef INFER_CONFIG_H
#define INFER_CONFIG_H

""")
        for name, val in shape_lines:
            f.write(f"#define {name:<16} {val}\n")
        f.write("\n/* DRAM, byte addresses. */\n")
        for name in order:
            f.write(f"#define {name:<16} 0x{mapping[name]:05x}u\n")
        f.write(f"\n#define INFER_RQ_SITES  {RQ_N}\n")
        f.write("#define INFER_RQ_INIT { \\\n")
        f.write(", \\\n".join(rows))
        f.write(" \\\n}\n\n")
        f.write(f"#define INFER_RQ_LOGIT  0x{logit_word:04x}u\n\n#endif\n")


# =============================================================================
# The static DRAM image.
# =============================================================================
def static_image(s: Shape, mapping: dict, weights: dict, embed_codes) -> dict:
    """Everything the device is handed once: weights, mask, head, embeddings.

    The KV cache is zeroed here. Attention contracts over all T keys to keep the
    block shape constant, so it reads cache rows no step has written yet; on the
    board those hold whatever the last run left, and this is what makes the ISS
    and the hardware agree about them. The mask is what makes them harmless.
    """
    img: dict = {}

    put_rowmajor_i4(img, mapping["DR_EMBED"], s.VOCAB, s.D,
                    lambda v, d: int(embed_codes[v][d]))
    put_rowmajor_i4(img, mapping["DR_MASK"], s.T, s.T,
                    lambda t, k: 0 if k <= t else -8)

    fc = weights["fc"]
    cols = fc.shape[1]
    put_rowmajor_i4(img, mapping["DR_HEAD_WGT"], s.D, s.VOCAB_PAD,
                    lambda r, c: int(fc[r][c]) if c < cols else 0)

    for L in range(s.LAYERS):
        base = layer_base(mapping, L)
        for key, off in (("q", "LW_WQ"), ("k", "LW_WK"), ("v", "LW_WV"),
                         ("o", "LW_WO")):
            blk = weights[(L, key)]
            put_rowmajor_i4(img, base + mapping[off], s.D, s.D,
                            lambda r, c, b=blk: int(b[r][c]))
        w1, w2 = weights[(L, "w1")], weights[(L, "w2")]
        put_rowmajor_i4(img, base + mapping["LW_FF1"], s.D, s.DFF,
                        lambda r, c: int(w1[r][c]))
        put_rowmajor_i4(img, base + mapping["LW_FF2"], s.DFF, s.D,
                        lambda r, c: int(w2[r][c]))

    cache_bytes = s.BATCH * s.LAYERS * s.T * i4_row(s.D)
    zero_range(img, mapping["DR_K_CACHE"], cache_bytes)
    zero_range(img, mapping["DR_V_CACHE"], cache_bytes)
    return img


def embedding_codes(net):
    """The embedding table quantized onto its learned site, all rows at once."""
    import torch

    with torch.no_grad():
        return (net.embedding.weight / act_scale(net.q_embed)).round().clamp(
            Q4_MIN, Q4_MAX).to(torch.int64)


def export(model_path: str, out_dir: str, heads: int = 4, note: str = "",
           **overrides) -> tuple:
    """Checkpoint -> (shape, map, static image, config header path)."""
    net = load_checkpoint(model_path, heads)
    s = shape_of(net, **overrides)
    mapping = dram_map(s)
    rq_table, weights = derive(net)

    os.makedirs(out_dir, exist_ok=True)
    header = os.path.join(out_dir, "infer_config.h")
    write_config(header, s, mapping, rq_table, logit_rq_word(weights["fc"]),
                 note or f"from {model_path}")
    image = static_image(s, mapping, weights, embedding_codes(net))
    return s, mapping, image, header, rq_table, weights, net


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="model/saved/int4_d128_f512_l4.pt")
    ap.add_argument("--out", default=os.path.join(HERE, "build", "export"))
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--tokens", "-T", type=int, default=64)
    ap.add_argument("--gen", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--dump-rq", action="store_true",
                    help="print the requant words per layer as m0/2^n")
    args = ap.parse_args()

    gen = args.gen if args.gen is not None else args.tokens - args.prompt
    s, mapping, image, header, rq_table, _, _ = export(
        args.model_path, args.out, args.heads,
        T=args.tokens, PROMPT=args.prompt, GEN=gen, BATCH=args.batch,
        BLOCK=args.block)

    if args.dump_rq:
        for L, row in enumerate(rq_table):
            cells = " ".join(f"{n}={w & 0xFFF}/2^{w >> M0_W}"
                             for n, w in zip(RQ_NAMES, row))
            print(f"layer {L}: {cells}")

    print(f"shape   : d={s.D} f={s.DFF} layers={s.LAYERS} heads={s.HEADS} "
          f"T={s.T} prompt={s.PROMPT} batch={s.BATCH}")
    print(f"weights : 0x{mapping['DR_LAYER0']:05x}, stride "
          f"0x{mapping['DR_LAYER_STRIDE']:05x}, ending 0x{mapping['DR_END']:05x} "
          f"of 0x{DRAM_BYTES:05x}")
    print(f"image   : {len(image)} bytes")
    print(f"header  : {header}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
