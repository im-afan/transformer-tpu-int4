# `model/` — the golden reference

A character-level decoder-only transformer trained to do multi-digit **addition**.
Everything under [`accel/`](../accel) is checked against this code, so it defines
correctness — not the other way round.

```
model/
  transformer.py            architecture, quantizers, named configs
  numbers_data.py           synthetic addition dataset + tokenizer
  train.py                  training loop
  make_dummy_checkpoint.py  an UNTRAINED checkpoint, for exercising accel/ plumbing
  tests/test_inference.py   load a checkpoint, decode a batch, eyeball it
  saved/                    checkpoints (gitignored)
```

## Running

Run from the **repo root** with `-m` so `import model.transformer` resolves.

```bash
python -m model.train --arch int4_wide
python -m model.tests.test_inference --arch int4_wide
python -m model.make_dummy_checkpoint
```

Deps: `torch` (+ `jupyter` for the notebook). The venv is checked in at `.venv/`
(Python 3.12); there is no `requirements.txt`.

---

## 1. The data (`numbers_data.py`)

### Vocabulary

13 tokens, one per character, no BOS/EOS.

| token | `0`–`9` | `+` | `=` | `N` (pad) |
| --- | --- | --- | --- | --- |
| **id** | 0–9 | 10 | 11 | 12 |

### Digits are reversed

`REVERSE_DIGITS = True`: every number is written least-significant digit first, so
`123+45=168` is emitted as `321+54=861`.

- Carries propagate from the ones digit up, which is the direction an autoregressive
  model can actually see. Answer digit *k* then depends only on operand digits `0..k`
  and the carry out of `k-1`.
- It also fixes place value to position: `EQUALS_POS + k` is always the 10^k place.

### Fixed answer position

`EQUALS_POS = 64` is the index of the **first answer digit**; `=` sits at
`EQUALS_POS - 1`. Operands are padded *between* the operands and the `=`, so the answer
always starts at the same offset.

```
index   0  1  2  3  4  5  ...  62 63 64 65 66 ... 127
token   3  2  1  +  5  4  N..N  N  =  8  6  1  N.. N
        |-operands-|  |- pad -|     |answer| |tail pad|
```

- `MAX_TOKENS = 128` is the sequence length.
- `max_digits <= (EQUALS_POS - 2) // 2` = **31**: both operands and the `+` have to fit
  before the `=`. `generate_addition_expression` raises if they do not.
- `train.py` and `test_inference.py` slice with the constant rather than searching for
  `=`. Logits at `EQUALS_POS - 1` predict the digit at `EQUALS_POS`.
- `equals_pos` is a per-call argument. `accel/test/export.py` pins it to the kernel's
  `PROMPT`,
  because `fw/infer.c` is compiled at `T=64` with a 32-token prompt.
- `MAX_INTEGER = 99999` is vestigial — `_sample_number` bounds by digit count.

### Sampling

`_sample_number` picks a **digit length** uniformly in `1..max_digits`, then a value
uniformly inside that length. Single-digit operands are as common as 31-digit ones.

### The attention mask

`tokenize(expr, max_tokens)` returns `(token_ids, mask)`. The mask is an additive
`[T, T]` float tensor: `-1e9` on any pair touching a pad.

**It is threaded through the model and never applied.** Only the causal mask is used.

---

## 2. Architecture (`transformer.py`)

```
tokens [B, T]
   | nn.Embedding(vocab, d) -> dropout(0.1) -> ActQuant     <- no positional encoding
   v
 +------------------- Transformer x layers -------------------+
 | X  <- q_x1( DyT( X + dropout(attention(X)) ) )             |
 | H  <- q_h(ff[0](X));  HR <- q_hr(relu(H));  F <- q_f(ff[2](HR)) |
 | X  <- q_x2( DyT( X + dropout(F) ) )                        |
 +------------------------------------------------------------+
   v
 Int4Linear(d, vocab) -> logits [B, T, vocab]   <- never requantized; argmax ignores scale
```

### The 5-D attention layout — the accelerator contract

Q/K/V are not in the usual flat head layout. GQA is expressed by giving Q an extra
`heads_per_q` axis that K/V broadcast over:

```
Q : [batch, tokens, kv_heads, heads_per_q, head_dim]   "btkgh"
K : [batch, tokens, kv_heads,              head_dim]   "bskh"
V : [batch, tokens, kv_heads,              head_dim]   "bskh"
```

The math is the einsum pair, inlined in `MultiHeadAttention.forward`:

```python
S = einsum("btkgh,bskh->btskg", Q, K) / sqrt(head_dim)
P = relu(S + causal_mask)                     # NOT softmax, no normalization
A = einsum("btskg,bskh->btkgh", P, V).reshape(B, T, q_heads * head_dim)
```

Load-bearing details:

- **ReLU attention, not softmax.** There is no normalization over the source axis at all.
- The causal mask is `triu(ones([T,T]) * -1e9, diagonal=1)`, strictly upper-triangular.
  It needs no quantization scale — a masked entry is negative whatever the scale is, so
  ReLU takes it to exactly zero. Exact, not a tolerance.
- The output reshape flattens `(kv_heads, heads_per_q, head_dim)` in that order, which is
  what `Wo` expects.
- `use_custom_attention` still threads down from `Model.forward` but is **accepted and
  ignored**. There is no CUDA path from the model.

### No positional encoding

`Model.forward` embeds and nothing else. The only position signal is the causal mask:
token *t* attends over *t+1* keys, and ReLU attention has no source-axis normalization,
so the **magnitude** of `P @ V` carries the count. The same removal under softmax would
leave the model position-blind.

### No LayerNorm — DyT instead

`norm1` / `norm2` are `DyT` (dynamic tanh, arxiv 2503.10622): `hardtanh(x * alpha)` with
one learned scalar, no gamma/beta. The VPU already does that shape.

### The double residual

`MultiHeadAttention.forward` ends in `return O + X`, and `Transformer.forward` adds `X`
again — so the attention residual is `2X + O`. The checkpoints were fitted to it; a
reimplementation must reproduce it.

### int4 weights (`Int4Linear`)

- Quantized to `[-8, 7]`, scaled by `absmax / 7`, with `RoundClip` as the STE backward.
- The scale is **detached**, so an absmax gradient lands entirely on one element. This
  differs from `TernaryLinear`, whose absmean averages over the tensor.
- The weight is `[in_dim, out_dim]` with `x @ w` — the **transpose** of `nn.Linear`, so
  int4 and float checkpoints are not interchangeable.
- `make_linear(..., use_int4)` selects it, **including `Model.fc`**. A float config gets
  an `nn.Linear` head, which is the only thing `quantize_head` / `dynamic_fake_quant`
  still reach.
- `TernaryLinear` is retained but **no config builds one**, and nothing in `accel/`
  reads one.

### int4 activations are QAT, not post-hoc

- Every requant site is an `ActQuant` holding one per-tensor scale, learned by LSQ
  (`FakeQuant` gives the STE input gradient and the analytic scale gradient).
- Sites the hardware **pins** to another tensor's scale share one `ActQuant`
  *instance* — `q_o`/`q_xo` are the residual stream's `x_quant`, `q_p is q_s`,
  `q_hr is q_h`. The sharing in `__init__` is load-bearing, not shorthand.
- Default grid: `INT4_QMIN, INT4_QMAX = -8, 7`.
- DyT outputs are pinned to `fixed_scale = 1/7` with `qmin = -7` — `hardtanh` bounds them
  analytically, and it is symmetric because hardtanh is odd.
- `set_quant_enabled(model, False)` turns them all off. **Needed for a real float
  baseline**, since they are live by default.
- `TernaryLinear.act_scale` is the old PTQ buffer and is vestigial; the deleted
  `calibrate.py` was the only thing that ever drove it.
- MoE is gone.

### Named configs

The factories at the bottom are the source of truth, wired to `train.py --arch`.

| factory | `--arch` | `d` | `f` | layers | `q_heads` | `kv_heads` | `head_dim` | int4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `adder_int4_wide` | `int4_wide` | 128 | 512 | 4 | 4 | 4 | 32 | yes |
| `adder_int4_vanilla` | `int4_vanilla` | 64 | 256 | 4 | 4 | 4 | 16 | yes |
| `adder_vanilla` | `vanilla` | 128 | 512 | 4 | 4 | 4 | 32 | no |
| `adder_gqa` | `gqa` | 128 | 512 | 6 | 8 | 2 | 16 | no |

- **`adder_int4_wide` is the live config** and the default for `train.py`,
  `test_inference.py` and every exporter under `accel/`.
- Both int4 configs set `use_bias=False`: the TPU's VPU has no row-broadcast operand, so
  a `[N]` bias over `[T, N]` would cost a per-row loop at every linear.
- Nothing here depends on sequence length — there is no positional encoding, so `T` lives
  in `numbers_data` and in the kernels.

---

## 3. Training (`train.py`)

```bash
python -m model.train --arch int4_wide --mini_batch_size 256 --batch_size 512
```

- Adam, `lr=1e-3`, fixed. Data is generated fresh every step: no held-out split.
- **Gradient accumulation.** `steps_per_batch = batch_size // mini_batch_size`;
  `optim.step()` fires every `steps_per_batch` micro-steps. `--batch_size` is the
  effective batch, `--mini_batch_size` is what has to fit in memory.
- **Gradient clipping** (`--grad_clip`, default 1.0) applies to the *accumulated*
  gradient, immediately before the step. Reported `grad_norm` stats are pre-clip.
- **Loss** spans `EQUALS_POS-1 : -1` against `EQUALS_POS:`, so the trailing pad tokens are
  part of the objective on purpose — the model is trained to emit `N` after the last
  answer digit.
- Defaults for `--max_tokens` and `--max_digits` come from `numbers_data`.
- Checkpoints go to `model/saved/test_model_*.pt`, keeping the last 3.

## 4. Checkpoints (`saved/`)

- All checkpoints are **gitignored**. There is no committed `.pt`.
- `model/saved/int4_d128_f512_l4.pt` is what every `accel/` path defaults to.
- **There is no trained wide checkpoint yet.** `python -m model.make_dummy_checkpoint`
  writes an untrained one at that path so the export/staging/RTL paths can run. Any
  accuracy it scores is chance.
- The dummy is not just `torch.save(adder_int4_wide().state_dict())`: learned `ActQuant`
  scales are seeded lazily by the first tensor through them, so it runs a few forward
  passes with the quantizers live and then checks every learned site came up.
- Older `colab_*.pt` and `ternary_mha.pt` no longer load — they predate ternary K/V, the
  removal of the positional encoding, the int4 head and `layers=4`, and there is no
  ternary config left to rebuild them with.

## 5. Why the model is QAT

`quant.py` (the hardware-exact int8 benchmark) and `calibrate.py` (the fake-quant PTQ
path) are **deleted**. They still described the ternary weight / int8 activation model and
failed on the missing `adder_ternary_vanilla` factory. What they established is worth
keeping:

**The model is not int8-quantizable post-hoc, and QAT is what fixed it.**

- Quantization error is *absolute*, and a per-tensor scale is pinned by the maximum, so
  what matters is `median/max`. With no normalization that collapses with depth — 1.3%
  of layer 0's residual stream rounded to exactly zero, 91.6% of layer 3's `A` did.
- The usual outlier check misses it: the *top* of the distribution is well behaved
  (`max/p99.9` was 1.1–2.5).
- DyT fixed that mechanism and still scored 0%, because what it exposed was the residual
  *addend* — a vector add puts `X` and `O = Wo(A)` on one scale while they differed by up
  to 7259x.
- Per-channel scales would fix it but the ISA cannot express them: `requant` takes one
  `{m0,n}` per dispatch.
- QAT needs no hardware change: instead of finding scales the trained weights tolerate,
  it trains weights that tolerate the scales.

Consequences worth knowing:

- **A "float" row is no longer a ceiling.** Removing the quantizers from a QAT checkpoint
  gives a network the weights were never trained for.
- **Saturation rates stopped being a health check.** A layer can learn to use the requant
  as a `sign()` and clip 99.88% of a tensor deliberately.
- **Scales come from the checkpoint, not from calibration.** Re-deriving by absmax moves
  every rounding grid the weights were fitted against. `accel/test/export.py` reads them
  off the checkpoint's `ActQuant` sites for exactly this reason.

## 6. Caveats

- **The padding mask is never applied** (above). Wiring it in would change the numerics
  and invalidate checkpoints.
- **`MultiHeadAttention` builds its causal mask on the module-level `device`**, not the
  model's. A CUDA-available machine with the model on CPU raises a device mismatch.
- **`notebook.ipynb` uses flat imports** from before the package layout, and one cell
  unpacks `create_addition_batch` into two values when it returns three.
