#!/usr/bin/env python3
"""infer: the model generating — a prefill, then decode against a KV cache.

Two modes, one kernel:

    --synthetic     mixed-hash weights and a hand-picked requant table. No
                    checkpoint, no torch; this is the regression.
    --model-path    a real QAT checkpoint through accel/test/export.py. The
                    pass/fail is still the integer reference; the addition
                    accuracy is reported next to it.

The golden is an integer recompute in numpy that keeps NO cache: it redoes the
whole prefix at every step. That is the point — the thing under test is the
cache, and a reference that kept one would agree with a broken kernel.

    python accel/test/tests/infer/generate.py -b iss --synthetic -n 2
    python accel/test/tests/infer/generate.py -b iss -n 8 --gen 4
    python accel/test/tests/infer/generate.py -b rtl --synthetic --gen 3 -n 1
"""
from __future__ import annotations

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
REPO = os.path.normpath(os.path.join(TESTROOT, "..", ".."))
for _p in (TESTROOT, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                             # noqa: E402

import export                                                  # noqa: E402
from export import (RQ_IDX, RQ_N, Shape, dram_map, logit_addr,  # noqa: E402
                    token_addr, write_config)
from program import (TPUProgram, backend_from_args, report,    # noqa: E402
                     standard_parser)
from vector_generator import (Case, VectorGenerator, i4_row,   # noqa: E402
                              put_i32, put_rowmajor_i4, w_hash)

BUILD = os.path.join(TESTROOT, "build", "infer")

# The synthetic requant table: one shift per site, chosen so no tensor collapses
# to all-zero or saturates flat. Layer-independent, unlike a derived one.
SYNTHETIC_RQ = {"Q": (1, 6), "K": (1, 6), "V": (1, 6), "S": (1, 4),
                "ID": (1, 0), "P": (1, 0), "A": (1, 6), "O": (1, 6),
                "XO": (1, 0), "X1": (1, 1), "H": (1, 6), "HR": (1, 0),
                "F": (1, 7), "X2": (1, 1)}
SYNTHETIC_LOGIT = (3 << 12) | 1        # {m0 = 1, n = 3}


def emb_val(v: int, d: int) -> int:
    """A synthetic embedding row. Distinct per token id, or a wrong gather is
    invisible in the output."""
    return ((v * 7 + d * 3) % 9) - 4


# =============================================================================
# The integer reference.
# =============================================================================
def _rq(acc, word, lo=-8):
    """clip((acc*m0 + 2**(n-1)) >> n) over a numpy array.

    numpy's >> on a signed integer is arithmetic, i.e. it floors, which is what
    Verilog's >>> and Python's own >> do.
    """
    m0, n = word & 0xFFF, (word >> 12) & 0xF
    v = (acc.astype(np.int64) * m0 + ((1 << (n - 1)) if n else 0)) >> n
    return np.clip(v, lo, 7)


class Reference:
    """The model as integers, with no KV cache."""

    def __init__(self, s: Shape, emb, layers, head, rq_table, logit_word):
        self.s, self.emb, self.layers, self.head = s, emb, layers, head
        self.rq_table, self.logit_word = rq_table, logit_word

    def generate(self, prompt_ids: list, n_gen: int) -> tuple:
        """(tokens, logits) for each generated step, in the kernel's order."""
        s = self.s
        toks = list(prompt_ids)
        out_tok, out_log = [], []

        for step in range(n_gen):
            pos = s.PROMPT - 1 + step
            X = self.emb[toks[:pos + 1]]                # [n][D] int4 codes
            n = X.shape[0]
            mask = np.where(np.arange(n)[None, :] <= np.arange(n)[:, None], 0, -8)

            for w in self.layers:
                rq = w["rq"]
                Q = _rq(X @ w["q"], rq[RQ_IDX["Q"]])
                K = _rq(X @ w["k"], rq[RQ_IDX["K"]])
                V = _rq(X @ w["v"], rq[RQ_IDX["V"]])
                A = np.zeros((n, s.D), dtype=np.int64)
                for h in range(s.HEADS):
                    sl = slice(h * s.head_dim, (h + 1) * s.head_dim)
                    S = _rq(Q[:, sl] @ K[:, sl].T, rq[RQ_IDX["S"]])
                    S = _rq(S + mask, rq[RQ_IDX["ID"]])
                    P = _rq(np.maximum(S, 0), rq[RQ_IDX["P"]])
                    A[:, sl] = _rq(P @ V[:, sl], rq[RQ_IDX["A"]])
                O = _rq(A @ w["o"], rq[RQ_IDX["O"]])
                XO = _rq(X + O, rq[RQ_IDX["XO"]])
                X1 = _rq(XO + X, rq[RQ_IDX["X1"]], lo=-7)          # dyt
                H = _rq(X1 @ w["w1"], rq[RQ_IDX["H"]])
                HR = _rq(np.maximum(H, 0), rq[RQ_IDX["HR"]])
                F = _rq(HR @ w["w2"], rq[RQ_IDX["F"]])
                X = _rq(X1 + F, rq[RQ_IDX["X2"]], lo=-7)           # dyt

            logits = _rq(X[pos] @ self.head, self.logit_word)      # [VOCAB_PAD]
            nxt = int(np.argmax(logits[:s.VOCAB]))                 # ties -> lowest
            out_log.append(logits)
            out_tok.append(nxt)
            toks.append(nxt)
        return out_tok, out_log


# =============================================================================
# The vectors.
# =============================================================================
class InferVectors(VectorGenerator):
    def __init__(self, shape: Shape, problems: int, model_path: str | None = None,
                 seed: int = 0):
        self.s = shape
        self.map = dram_map(shape)
        self.problems, self.seed = problems, seed
        self.model_path = model_path
        self.net = None
        os.makedirs(BUILD, exist_ok=True)

        if model_path:
            self._from_checkpoint(model_path)
        else:
            self._synthetic()

        write_config(os.path.join(BUILD, "infer_config.h"), self.s, self.map,
                     self.rq_table, self.logit_word,
                     model_path or "synthetic weights (tests/infer/generate.py)")
        self.reference = Reference(self.s, self.emb, self.layers, self.head,
                                   self.rq_table, self.logit_word)

    # ---- weights ------------------------------------------------------------
    def _synthetic(self) -> None:
        s = self.s
        row = [0] * RQ_N
        for name, (m0, n) in SYNTHETIC_RQ.items():
            row[RQ_IDX[name]] = (n << 12) | m0
        self.rq_table = [list(row) for _ in range(s.LAYERS)]
        self.logit_word = SYNTHETIC_LOGIT

        def block(rows, cols, salt):
            return np.array([[w_hash(r, c, salt) for c in range(cols)]
                             for r in range(rows)], dtype=np.int64)

        self.emb = np.array([[emb_val(v, d) for d in range(s.D)]
                             for v in range(s.VOCAB)], dtype=np.int64)
        self.head = block(s.D, s.VOCAB_PAD, 0)
        self.weights = {"fc": self.head[:, :s.VOCAB]}
        self.layers = []
        for L in range(s.LAYERS):
            w = {"q": block(s.D, s.D, 6 * L + 1), "k": block(s.D, s.D, 6 * L + 2),
                 "v": block(s.D, s.D, 6 * L + 3), "o": block(s.D, s.D, 6 * L + 4),
                 "w1": block(s.D, s.DFF, 6 * L + 5),
                 "w2": block(s.DFF, s.D, 6 * L + 6), "rq": self.rq_table[L]}
            self.layers.append(w)
            for key in ("q", "k", "v", "o", "w1", "w2"):
                self.weights[(L, key)] = w[key]

    def _from_checkpoint(self, path: str) -> None:
        s = self.s
        self.net = export.load_checkpoint(path, s.HEADS)
        self.rq_table, self.weights = export.derive(self.net)
        self.logit_word = export.logit_rq_word(self.weights["fc"])
        self.emb = export.embedding_codes(self.net).numpy()

        fc = self.weights["fc"].numpy()
        self.head = np.zeros((s.D, s.VOCAB_PAD), dtype=np.int64)
        self.head[:, :fc.shape[1]] = fc
        self.layers = [
            {**{k: self.weights[(L, k)].numpy()
                for k in ("q", "k", "v", "o", "w1", "w2")},
             "rq": self.rq_table[L]}
            for L in range(s.LAYERS)]

    # ---- the images ---------------------------------------------------------
    @property
    def defines(self) -> dict:
        return {}                       # everything is in the config header

    def static(self) -> dict:
        return export.static_image(self.s, self.map, self.weights, self.emb)

    def writable_ranges(self) -> list:
        """The caches and every activation buffer: scratch, by design."""
        base = self.map["DR_K_CACHE"]
        return [(base, self.map["DR_ACT_END"] - base)]

    def prompts(self):
        """One BATCH-wide problem per case, from the addition dataset."""
        import model.numbers_data as numbers_data

        random.seed(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
        except ImportError:
            pass
        n = self.problems * self.s.BATCH
        exprs, tokens, _ = numbers_data.create_addition_batch(
            n, self.s.T, max_digits=(self.s.PROMPT - 2) // 2,
            equals_pos=self.s.PROMPT)
        for i in range(self.problems):
            rows = tokens[i * self.s.BATCH:(i + 1) * self.s.BATCH]
            yield exprs[i * self.s.BATCH], [list(r) for r in rows]

    def cases(self):
        s, m = self.s, self.map
        self.targets = []
        for i, (expr, rows) in enumerate(self.prompts()):
            patch, golden, ranges = {}, {}, []
            targets = []
            for seq, toks in enumerate(rows):
                prompt = toks[:s.PROMPT]
                put_i32(patch, token_addr(s, m, seq, 0), prompt)

                gen_tok, gen_log = self.reference.generate(prompt, s.GEN)
                put_i32(golden, token_addr(s, m, seq, s.PROMPT), gen_tok)
                for step, logits in enumerate(gen_log):
                    put_rowmajor_i4(
                        golden, logit_addr(s, m, seq, s.PROMPT - 1 + step),
                        1, s.VOCAB_PAD, lambda r, c, v=logits: int(v[c]))
                ranges.append((token_addr(s, m, seq, s.PROMPT), s.GEN * 4))
                ranges.append((logit_addr(s, m, seq, s.PROMPT - 1),
                               s.GEN * i4_row(s.VOCAB_PAD)))
                targets.append((prompt, toks[s.PROMPT:s.PROMPT + s.GEN], gen_tok))
            self.targets.append(targets)
            yield Case(name=f"problem {i}: {expr}", patch=patch, golden=golden,
                       check_ranges=ranges)


# =============================================================================
def program(backend, shape: Shape, problems: int, model_path: str | None = None,
            seed: int = 0):
    gen = InferVectors(shape, problems, model_path, seed)
    return TPUProgram(os.path.join(HERE, "infer.c"), backend, gen,
                      include_dirs=[BUILD])


def score(gen: InferVectors) -> None:
    """What the device generated against what the dataset wanted. Not a
    pass/fail: an untrained checkpoint scores chance and that is not a bug."""
    import model.numbers_data as numbers_data

    exact = tok_ok = tok_n = 0
    for problem in gen.targets:
        for prompt, want, got in problem:
            exact += int(list(got) == list(want))
            tok_ok += sum(a == b for a, b in zip(got, want))
            tok_n += len(want)
    n = sum(len(p) for p in gen.targets)
    if not n:
        return
    said = numbers_data.unreverse_expression(
        numbers_data.detokenize(gen.targets[0][0][2]))
    print(f"addition: {100 * exact / n:.2f}% exact-sequence, "
          f"{100 * tok_ok / max(tok_n, 1):.2f}% token; first answer {said!r}")

    every = {t for problem in gen.targets for _, _, got in problem for t in got}
    if len(every) == 1:
        print(f"  WARNING: every generated token is {every.pop()} — the weights "
              f"have collapsed and this check is weak", file=sys.stderr)


def main() -> int:
    ap = standard_parser(__doc__)
    ap.add_argument("--model-path", default=None,
                    help="a QAT checkpoint; omit for --synthetic")
    ap.add_argument("--synthetic", action="store_true",
                    help="mixed-hash weights instead of a checkpoint")
    ap.add_argument("-T", "--tokens", type=int, default=64)
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--gen", type=int, default=None,
                    help="tokens to generate, the prefill's included")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("-d", type=int, default=64, help="model width, --synthetic only")
    ap.add_argument("-f", "--dff", type=int, default=256,
                    help="feed-forward width, --synthetic only")
    ap.add_argument("-L", "--layers", type=int, default=4,
                    help="layers, --synthetic only")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.synthetic and not args.model_path:
        args.model_path = "model/saved/int4_d128_f512_l4.pt"

    gen_n = args.gen if args.gen is not None else args.tokens - args.prompt
    knobs = dict(T=args.tokens, PROMPT=args.prompt, GEN=gen_n, BATCH=args.batch,
                 BLOCK=args.block, HEADS=args.heads)
    if args.model_path:
        net = export.load_checkpoint(args.model_path, args.heads)
        shape = export.shape_of(net, **knobs)
    else:
        shape = Shape(D=args.d, DFF=args.dff, LAYERS=args.layers, **knobs)
        shape.check()

    problems = args.cases if args.cases else 4
    # infer is DMA-bound at ~830 k clocks per generated token; the watchdog has
    # to cover a whole run of them.
    watchdog = 1_000_000 * 1000 * max(1, gen_n)
    prog = program(backend_from_args(args, watchdog_ns=watchdog), shape,
                   problems, args.model_path, args.seed)
    prog.run_program()
    score(prog.generator)
    return report(prog)


if __name__ == "__main__":
    raise SystemExit(main())
