# model/ design notes

Rationale relocated out of code comments during the comment-style cleanup.
Kept here instead of deleted; the code now carries only short, load-bearing
one-liners.

## transformer.py

**`RoundClip`** — `round(x).clip(qmin, qmax)` with a straight-through
estimator backward. The STE passes gradient only where the input was inside
the clip, so a weight driven off the grid stops receiving it. `(-1, 1)` is
`TernaryLinear`'s grid, `(-8, 7)` is `Int4Linear`'s.

**`FakeQuant`** — symmetric per-tensor fake quantization:
`round(x/s).clip(qmin,qmax) * s`. Returns the dequantized value so the
surrounding math stays in real units; only the representable grid changes.
`qmin` is -8 for a `requant` site and -7 for a `dyt` site (the hardtanh clip
is symmetric because hardtanh is odd).

Backward is LSQ (arxiv 1902.08153): straight-through for the input, and the
analytic derivative w.r.t. the scalar step size, summed over the tensor
because one `s` broadcasts over every element. `grad_scale` is the LSQ
gradient-magnitude correction `1/sqrt(numel * qmax)`.

**`dynamic_fake_quant`** — per-tensor int8 fake-quant with a scale taken from
the current absmax. For weights the hardware stores as plain int8, where the
host derives the scale at export time rather than learning it.
`adder_int4_vanilla` has none: the output head is an `Int4Linear` too, so
every weight in it is int4, and this path is reached only by the float
`adder_vanilla` / `adder_gqa` configs.

**`ActQuant`** — one activation quantization site, one entry in the requant
table. A site owns a per-tensor scale. Sites the hardware pins to another
tensor's scale (the residual adds, the identity requants) are expressed by
sharing the same `ActQuant` instance, or by pinning `fixed_scale`.

`fixed_scale=None` learns the scale (LSQ), initialized from the absmax of the
first tensor that flows through. DyT outputs are pinned to `1/7` instead:
`hardtanh` bounds them to `[-1, 1]` analytically, so no calibration set is
involved (`adder_kernel.md` §4).

The default grid is symmetric int4 (`[-8, 7]`). `init="absmean"` seeds from
the absmean instead of the absmax, which is what a grid with very few levels
wants — an absmax seed rounds most of the tensor to zero and leaves LSQ
climbing out of a dead gradient.

`init_scale_from` is exposed separately so a site can be seeded from a
tensor other than the one it quantizes — `RQ_S` is seeded from the
post-mask, post-ReLU scores, because a large negative score clips to -8 and
ReLU takes it to exactly zero, so spending range on it is pure waste.

**`TernaryLinear`** — linear layer with 1.58-bit (ternary {-1, 0, 1})
weights, BitNet-style. No config builds one any more (`adder_int4_vanilla`
uses `Int4Linear`); it is kept because the `accel/`
export path still model a ternary weight.

Weights are scaled by their absmean and rounded/clipped to {-1, 0, 1}.
Activations feeding the matmul can optionally be quantized to symmetric int8
with a single per-tensor scalar, populated offline by
`calibrate_activations` (absmax over a calibration set, `scale = absmax /
127`) and stored in `act_scale`. An uncalibrated layer keeps `act_scale` at
NaN and passes activations through untouched, so existing checkpoints load
and run exactly as before.

**`Int4Linear`** — linear layer with symmetric per-tensor int4 weights. The
int4 analogue of `TernaryLinear`, with two differences: the scale is
`absmax / 7`, not the absmean (absmean is BitNet's rule for a grid with one
nonzero level per sign; on 15 levels it puts most of the grid past the end
of the weight distribution); and the scale is detached (`absmax` has a
gradient that lands entirely on one element, which is a spike, not a
signal — `TernaryLinear` gets away with a live scale because an absmean
averages over the whole tensor).

The weight is `[in_dim, out_dim]` with `x @ w` — the transpose of
`nn.Linear`, same as `TernaryLinear`, so checkpoints are not interchangeable
with a float config.

**Activation quantization placement** — in both `TernaryLinear.forward` and
`Int4Linear.forward`, activations are quantized by the `ActQuant` site that
*produced* x, not inside the linear: on the TPU every input to a matmul is
already an int4/int8 tensor left behind by the previous op's requant.

**`MultiHeadAttention`** quantization sites, in `adder_kernel.md` §4 block
order: `RQ_Q`/`RQ_K`/`RQ_V` (separate — each projection has its own scale),
`RQ_S` (with `1/sqrt(head_dim)` folded in, so quantize after the divide),
`RQ_P` and `RQ_XO` are the `{1,0}` identities and so share their producer's
site, and `RQ_A`. `RQ_O` is pinned to `s_x` because `vecadd` takes both
residual operands at one scale — that is the site the clipping shows up at,
so it deliberately keeps the tight `s_x` rather than widening to cover `O`.

Every tensor here is int4, K and V included. The old `RQ_KT`/`RQ_VT` pair —
a second narrow rounding K and V onto `{-1, 0, 1}` — is gone: it existed
only because the MXU could not multiply an int8 activation by an int8
weight, so the operand landing on the weight side had to be a trit. At equal
width there is nothing left for it to do but round twice.

`RQ_S` is calibrated on the scores that *survive* the mask+ReLU, which is
what makes `RQ_P` the `{1,0}` identity. Learning the scale finds that on its
own: clipped-away negatives contribute nothing to the loss.

**`Transformer`** — `RQ_X1`/`RQ_X2` are `dyt` ops: hardtanh bounds them to
`[-1, 1]`, so their scale is analytic (1/7) rather than calibrated, and the
clip is symmetric at -7 because hardtanh is odd. `RQ_F` is pinned to `s_x1`
by the second residual add, but it is a `requant`, so it clips at -8 rather
than -7.

`Transformer.forward`: the attention residual is `2X + O` —
`MultiHeadAttention` already returned `O + X` (at `s_x`), and `X` is added
again here before dyt.

**`Model`** — the `s_x` chain: layer 0 enters from the embedding and needs a
calibrated scale; every later layer enters on the previous layer's DyT
output, which is analytically 1/7, so its `s_x` *is* that site. Sharing the
module is what makes `RQ_O` / `RQ_XO` land on `s_x`.

There is no positional encoding. The model's only position signal is the
causal mask: token t attends over t+1 keys, and attention here is ReLU with
no normalization over the source axis, so the *magnitude* of the attention
output carries the count. Removing the sinusoidal PE also removes the one
unbounded contribution to layer 0's input, which is why `s_x0` is now the
embedding's scale alone.

`Model.forward`: the head's output scale is irrelevant to an argmax, so the
logits are never requantized — the device spills the raw int32 accumulator.
An int4 head carries its own weight quantization (`Int4Linear`'s absmax +
`RoundClip`), exactly as the projections do, so there is nothing for
`quantize_head` to switch; it and `dynamic_fake_quant` now only reach the
int8 `nn.Linear` head that a float config still gets.

**`adder_int4_wide`** — the live shape:
d=128, f=512, layers=4, q_heads=kv_heads=4, so head_dim=32. Same QAT scheme
as `adder_int4_vanilla` in every other respect — int4 weights and int4
activations, LSQ-learned ActQuant at every requant site, shared instances
where the hardware pins one site to another's scale, DyT pinned to 1/7, no
bias. Nothing here depends on the sequence length: there is no positional
encoding, so T lives in `numbers_data` (`EQUALS_POS = 64`, `MAX_TOKENS =
128`) and in the kernel, not in the model.

**`adder_int4_vanilla`** — int4 weights and int4 activations throughout,
same QAT scheme as the ternary config it replaces: every requant site is an
LSQ-learned ActQuant, the residual/identity sites share instances, the DyT
sites are pinned to 1/7. `head_dim = d // q_heads = 16`.

`use_bias=False`: the TPU's VPU has no row-broadcast operand, so a `[N]`
bias over `[T, N]` costs a T-iteration vecadd/requant loop per linear (~5
per layer). Dropping the biases keeps a hand-written kernel layer
straight-line.

## numbers_data.py

**`REVERSE_DIGITS`** — every number in a generated expression is written
least-significant digit first: `123+45=168` is emitted as `321+54=861`.

This is for learnability, and the answer is the half that matters. Addition
carries propagate from the ones digit upward, which is the direction an
autoregressive model cannot look: emitting the answer most-significant first
asks the model to know every carry before it writes the first digit.
Reversed, answer digit k depends only on operand digits 0..k and the carry
out of digit k-1 — the token it just emitted.

The second, smaller win is positional. The answer is left-aligned at
`EQUALS_POS` and padded on the right, so with the digits reversed, position
`EQUALS_POS+k` is always the 10^k place. Unreversed it is a different place
value for every answer length, so the model has to learn the alignment
separately at each magnitude.

Operands are reversed for the same alignment reason (position 0 is now
always the left operand's ones digit). The right operand still begins at a
length-dependent offset after '+'; fixing that would mean padding the
operands to a fixed width, which moves '+' and is a larger change.

## make_dummy_checkpoint.py

There is no trained checkpoint at the wide shape (d=128, f=512, layers=4,
q_heads=kv_heads=4) yet, and every host-side path below `accel/` needs one
before it can run at all: `accel/test/export.py` starts by loading a `.pt`, deriving the 14
requant `{m0,n}` words per layer from its scales, and compiling a kernel
against them. This writes a checkpoint that satisfies every one of those
steps and computes addition no better than chance.

It measures the plumbing, not the model. A run against it exercises the
staging, the DRAM map, the kernel's command stream and the ISS/RTL
agreement; the accuracy number it produces is noise, and the sequence it
generates is whatever random int4 weights happen to argmax to.

Why it is not just `torch.save(adder_int4_wide().state_dict())`: every
learned `ActQuant` scale is seeded lazily, by the first tensor that flows
through it (`ActQuant.init_scale_from`), and a freshly constructed model has
`initialized` clear and `scale` at exactly 1.0 everywhere. Exporting that
gives a requant table derived from a scale nothing chose, and
`quant.QATCalibration` would fall back to a calibration set rather than
reading the checkpoint. So this runs one forward pass over real batches with
the quantizers live, which is what seeds them, and then checks that every
site actually came up.

The sites the hardware pins to one another — `q_o`/`q_xo` to the residual
stream, `q_hr` to `q_h`, `q_p` to `q_s` — share an `ActQuant` instance, so
they are seeded together and `export.derive`'s equality checks hold by
construction. `q_x1`, `q_f` and `q_x2` are pinned to `1/7` analytically and
are never seeded at all.

Nothing here depends on the sequence length: the model has no positional
encoding, so `--max-tokens` and `--equals-pos` only shape the batches the
scales are seeded from. The defaults are `numbers_data`'s, i.e. what
`model/train.py` would train against.

## Historical: quant.py

**Purpose**: benchmark a checkpoint under int8 activations, hardware-exact.
`TernaryLinear.quantize_activations` fake-quantizes in float: it rounds an
activation onto an int8 grid and immediately multiplies the scale back in,
so everything downstream still runs in float32. That answers "how much does
int8 rounding cost", which is not the same question as "what would the
accelerator produce", because the accelerator has no floats at all. Between
the host's embedding lookup and the host's final argmax, every value in
`accel/tpu` is an int8 in a scratchpad or an int32 in an accumulator, and
the only thing that ever narrows one to the other is:

    requant(acc) = clip_int8((acc * m0 + 2**(n-1)) >> n)   m0 < 4096, n <= 15

with `>>` an arithmetic (flooring) shift. This module runs the model that
way: integers end to end, the same requant word per site the kernel would
load from DRAM, the same clipping, the same flooring. The reported accuracy
is what the hardware would score, not an estimate of it.

Four stages, each usable on its own from Python: `instrumented_forward` (the
model rewritten inline so every intermediate is visible, checked against
the real forward by `check_instrumented`); `calibrate` (absmax at every
site the integer pipeline needs a scale, over a disjoint calibration set);
`prepare` (weights to trits/int8, and the requant word per site);
`int_forward` (the integer pipeline, batched).

Relationship to `accel/test/export.py`: that module does the same job for one
frozen checkpoint against the live kernel, and its integer forward is verified
bit-exact against both the ISS and the RTL.

What is host-side, and why (unchanged from `adder_kernel.md` §1): the token
embedding, because the ISA has no gather; and the final argmax, because
`redmax` returns a maximum and not its index. Everything between them is
the device's. There is no longer a positional encoding to add.

**Exact integer linear algebra** (`ieinsum`/`imatmul`/`_guard_exact`): torch
has no fast integer matmul, and the obvious int64 broadcast-and-sum
materializes `[M, K, N]`, which at model scale is gigabytes. Everything here
is therefore evaluated in float64 and converted back — which is exact, not
approximate: float64 represents every integer below 2**53 exactly, and the
sum or product of two exactly-represented integers is itself exact whenever
the result is also below 2**53. Every partial sum in these contractions is
an integer bounded by `max|a| * max|b| * K`, so if that bound clears 2**53
the result is exact whatever order BLAS accumulates in. The bounds here are
not close: the widest is a ternary projection at `128 * 127 * 1 ~= 1.6e4`,
twelve orders of magnitude below the limit.

**`choose_word`** — best `{m0, n}` with `m0/2**n ~= m`, `1 <= m0 <= 4095`,
`0 <= n <= 15`. A larger shift gives finer relative resolution, so the
largest feasible `n` wins. A multiplier outside the representable band
(2**-15 to 4095) saturates rather than silently wrapping;
`report_words` flags how far each word missed by. Negative multipliers are
unrepresentable (`m0` is an unsigned field); the only way one arises here is
a negative DyT alpha.

**`dyt8`** — `DyT(x) = hardtanh(alpha*x, -1, 1)`, and with the output scale
pinned to 1/127 the saturating narrow *is* the hardtanh: the word carries
`alpha * s_in * 127` and every value the clip catches is one hardtanh would
have flattened. The clip has to be symmetric because hardtanh is odd —
int8's -128 would put the saturated end at -1.0079 instead of -1, which is
why this is a separate device op (`VOP_DYT`) and not `requant` with a chosen
scale.

**`tquant8`** — same fixed point as `requant8`, clipped to a single level
per sign. The device writes the result 2 bits wide (00=0, 01=+1, 11=-1)
straight into the column-major layout `matmul_t` reads its weights from, so
this op is both the quantization and the packing.

**`clip_rate`** — fraction of a narrow's inputs that saturate. For a DyT
site this is not a loss rate — saturating is the nonlinearity doing its job
— and for a ternary site it is not one either, since clipping is most of
what ternarizing is; but it is still the statistic that says how much of
the tensor is living in the flat region.

**`QLinear`** — a linear layer as the device holds it. `kind` decides which
unit runs it: ternary weights go to the MXU's `matmul_t` (2 bits each,
`scale` is the checkpoint's absmean and is never a tensor — it is just a
factor in the following requant's multiplier), and int8 weights go to the
VPU's `vecmatmul`. Either way `w` is `[in, out]`, which is the orientation
`TernaryLinear` already stores and the transpose of `nn.Linear`'s. The
ternary configs have no int8 weight left — the output head became a
`TernaryLinear` too, so `vecmatmul` has no caller in the shipped kernel and
every matmul in the model runs on the array. The kind still exists because
the float `adder_vanilla` / `adder_gqa` configs keep an `nn.Linear` head.

**`trits_and_absmean`** — the `+ eps` inside the division but not in the
returned absmean is `TernaryLinear.forward`'s own asymmetry, and reproducing
it here is the whole point.

**`dyt_alpha`** — a hardtanh is exactly what a saturating narrow already
does, so DyT costs the device nothing. It fuses into the narrow that was
going to follow the residual add anyway: pick the output scale 1/127 and
the clip saturates at exactly ±1. That narrow is `dyt` (`VOP_DYT`) rather
than `requant` only because hardtanh is odd and int8's clip is not.

**`Calibration`** — running absmax at every site the integer pipeline needs
a scale. It also keeps a running `[sum|t|, count]` beside the absmax,
because a ternary site is scaled by the absmean, so both statistics have to
be collected in the one pass.

**`instrumented_forward`** mirrors `model/transformer.py` as it stands,
including two things a reader of the class hierarchy would not predict and
that the integer pipeline has to reproduce exactly: `MultiHeadAttention`
returns `O + X`, and `Transformer` then adds `X` again — so the attention
residual is `2X + O`, not `X + O` (that extra copy is why there is an `XO`
requant slot: the device can only add two int8 operands at a time); and the
padding mask is ignored — `MultiHeadAttention` builds its own causal mask
and never reads the `attn_mask` it is handed, so the only mask in play is
the causal one.

`check_instrumented`: if the max abs difference is not ~0, the calibration
is describing a different network from the one being benchmarked, and every
number downstream is meaningless.

**`QATCalibration`** — the scales the model was trained with, not ones
measured after the fact. A QAT checkpoint carries an `ActQuant` per requant
site whose scale was learned by LSQ, and the weights were fitted against
exactly those scales — so re-deriving them by absmax is not a neutral
substitution, it moves every rounding grid the model was trained on. This
reads the learned scale out instead and presents it through the same
`scale(layer, site)` interface. `.abs()` mirrors `ActQuant.forward`: the
scale is a free parameter that LSQ can drive negative (layer 1's `q_v` in
`ternary_mha.pt` is -0.00405), and the forward pass uses its magnitude.

A site whose `initialized` flag is clear never saw a tensor, so its `scale`
is still the `nn.Parameter`'s 1.0 and means nothing — that is what a
checkpoint predating a site looks like after a `strict=False` load (the
ternary `q_kt`/`q_vt` against any pre-`tquant` checkpoint), and taking the
1.0 would silently quantize with a garbage scale instead of falling back to
the calibration set.

**`residual_scales`** — the residual-stream scale at each layer's input.
Single source of truth: it is used three times and every use has to agree
(to quantize X0 before layer 0, as `s_x` inside layer `li`'s words, and as
the target of layer `li-1`'s `RQ_X2`). With DyT in place the choice is
nearly free: `norm2` bounds its output to `[-1, 1]` analytically, so
`s = 1/127` needs no calibration and makes the `dyt` op's ±127 clip coincide
with the hardtanh exactly. Layer 0 is the exception: its input is the raw
embedding, which nothing bounds, so it stays calibrated.

**`derive_layer`** — the score scale is set by what survives, not by the
widest score: `S8` feeds `relu(S8 + mask)`, so a large negative score
contributes nothing (it clips to -128 and the ReLU takes it to exactly 0).
Calibrating on `|S|` would spend most of the int8 range representing values
that are then discarded; pinning `s_s` to the post-ReLU range is also what
collapses `RQ_P` to an exact identity (the multiplier comes out at 1.0).

`tquant` re-rounds the int8 K and V onto `{-1, 0, 1}` so the MXU can take
them as weight operands, which is what moves both attention matmuls off
`vecmatmul`. From there the attention scales are `s_kt`/`s_vt`, not
`s_k`/`s_v`.

The three pinned words: `vecadd` adds two int8 operands, so a residual is
only meaningful if both sides already share a scale. That pins `RQ_O` and
`RQ_XO` to `s_x` and `RQ_F` to `s_x1` — those outputs do not get to choose
their own scale, so they are where clipping is expected and where it is
measured.

DyT is fused into the narrow that follows the residual add: `X1 =
hardtanh(alpha1 * (X + XO) * s_x)`. The multiplier carries alpha; `dyt`'s
±127 clip is the hardtanh when `s_x1 = 1/127`.

Biases are added in int8 after the requant, not in int32 before it —
`vecadd` with a row stride of 0 is the broadcast the device has. Bias-free
configs (`adder_ternary_vanilla`) skip this entirely.

**`causal_mask_i8`** — 0 where key <= query, -128 above the diagonal. Exact,
not a tolerance: `S8` is int8, so `S8 - 128 <= -1` for every value it can
hold, and the ReLU after it takes any negative to exactly zero.

**`int_forward`** — every line is one device dispatch, in order. Nothing
here is float and nothing here is approximate: the only lossy steps are the
requants, and they lose exactly what the hardware's requant loses. K and V
become the weight operands (via `tquant`) while Q and P stay int8 as the
activation operands, which is the whole reason both attention matmuls can
run on the array. A layer with DyT narrows its residuals with the device's
`dyt` (symmetric clip); one without falls back to a plain `requant` (int8's
asymmetric clip). `MultiHeadAttention` returns `O + X`, and `Transformer`
adds `X` again; the device adds two int8 operands at a time, so `2X + O` is
two vecadds, and doing X+O first (rather than X+X) keeps the intermediate
in range. The head's output is never narrowed — an argmax does not care
about scale, so the device spills the int32 accumulator and the host reads
it.

**`fake_quant_forward`** — the same quantization scheme as `int_forward`, in
float32. This is the check that separates "the scheme loses the model" from
"the integer arithmetic has a bug": it clips at the same places and to the
same scales but never leaves float. If this scores like `int_forward`, the
fixed-point implementation is faithful and the loss is the quantization
itself; if the two diverge, the integer path is the suspect.

**`predictions`** — the answer region, scored the way `train.py` trains it.
The logit at position t predicts the token at t+1, and the answer occupies
`[ANS, T)`, so the slice is `[ANS-1, T-1)`. Note this is one position wider
than `accel/test/export.py`'s path scores — that path compares
`logits[ANS:-1]` against `tokens[ANS+1:]`, which is internally consistent
but drops the answer's leading digit from the metric.

**`benchmark`** — the calibration set is drawn from a different seed than
the evaluation set, so the scales are never fitted to the problems they are
scored on. The float baseline has to be the unquantized network: on a QAT
checkpoint the ActQuant sites are live by default, so calling the model
directly would score the fake-quantized model and call it "float",
flattering the comparison by moving the baseline, not the result.

**`report_clips`** — a high saturation rate at O/XO/X1/F means the residual
path is losing data, which is the failure mode a per-tensor scale is prone
to; a high rate at Q/K/V/H means the projection scale is too tight.

**`report_dynamic_range`** — a per-tensor scale is pinned by the tensor's
maximum, so the resolution every other element gets is set by how far the
bulk sits below it. The statistic that matters is therefore `median/max`,
not the usual outlier check against a high percentile: a tensor can have a
perfectly well-behaved top (`max/p99.9 ~ 1`) and still lose most of its
mass to zero.

**Status**: `quant.py` and `calibrate.py` are **deleted**. They targeted the
retired ternary/int8 model (`adder_ternary_vanilla`), which no longer exists in
`transformer.py`, and failed on the lookup rather than exporting something wrong.
The section above is the record of what they measured; `accel/test/export.py` is
the live path.

## train.py

**Gradient-norm bookkeeping** (reported alongside the loss): `clip_grad_norm_`
returns the norm before clipping, which is the diagnostic that matters when
chasing a blow-up — the mean says whether the run is drifting, the max says
whether a single batch spiked.

**Gradient clipping timing**: the fully accumulated gradient is clipped
immediately before the optimizer step, not per micro-batch. Clipping each
micro-batch separately would bound the partial sums independently and give
a different (smaller) effective bound.
