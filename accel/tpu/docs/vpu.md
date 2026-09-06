# VPU — Vector Unit

SIMD unit for every pointwise / reduction operation that is not a matmul.

Dispatch arrives from `cmd_vpu.sv`'s 128-bit command queue. The VPU has exactly **one**
command type (`0x01`); `0x02` (`VPU_GEOM`) is a retired hole.

## Operations — six, and nothing else

| `vpu_op` | Name | Kind | Operands | Result |
| --- | --- | --- | --- | --- |
| 0 | `DOT` | reduction | `src0`, `src1` | scalar `sum(src0[i]*src1[i])` -> `dst`, int32 |
| 1 | `ADD` | elementwise | `src0`, `src1`, `rq_word` | `dst[i] = clip[-8,7](rq(src0[i] + src1[i]))` |
| 3 | `RELU` | elementwise | `src0`, `rq_word` | `dst[i] = clip[-8,7](rq(max(src0[i], 0)))` |
| 10 | `REQUANT` | rescale | `src0`, `rq_word` | `dst[i] = clip[-8,7](rq(src0[i]))` |
| 16 | `DYT` | rescale | `src0`, `rq_word` | `dst[i] = clip[-7,7](rq(src0[i]))` |
| 18 | `ARGMAX` | reduction | `src0` | scalar `index of max(src0[i])` -> `dst`, int32 |

- The gaps (2, 4–9, 11–15, 17) are **retired codes, not free encoding space**. They are
  left vacant so a stale binary decodes to an unknown op rather than a different one. 17
  was `QUANT4`; see below. `ARGMAX` is 18 for the same reason — 2 was `SCALAR_MUL`.
- `DOT` and `ARGMAX` are the reductions; every other op produces a same-length vector.
- **No shipped kernel issues `DOT`.** It is kept because it *is* the reduction datapath —
  accumulator, lane fold, scalar store — and the only reduction the ISA has. It costs one
  decode arm over hardware that has to be there anyway.

### `ARGMAX`

`dst` gets the int32 **index**, not the value. Ties go to the lowest index, which is
`torch.argmax`'s rule. `infer.c`'s `head_argmax` is the caller, through
[`tpu_argmax`](fw.md).

- `rq_word` is ignored: an index has no scale.
- Each chunk folds to `(value, index)` in a binary tree, `log2(LANES)` comparator levels
  rather than `LANES` in series. Level 0 pads up to a power of two with `-8` and every tie
  goes left; the active lanes are a prefix of the chunk, so a padding lane can never
  outrank a real one.
- Across chunks it is strictly greater, so the earliest chunk holding the maximum keeps
  it. The running best value is 5 bits, one wider than an int4, so an all-`-8` vector
  still answers 0.
- The MXU requantizes to int4 on store, so the head's logits are already int4 and this is
  an argmax over exactly what it wrote. That is the caller.
- **`vlen` may be odd here.** The even-`vlen` rule below is about the packed int4
  destination; a reduction writes an int32 scalar and no nibbles. The vocabulary is 13.

### What the model uses

- **Activations:** `RELU`. The FFN is ReLU, not GELU.
- **Elementwise:** `ADD`, for the residuals `X + O` and `X1 + F`.
- **Rescaling:** `REQUANT`, `DYT`.
- **The attention mask:** `relu(S + mask)` is an `ADD` against an int4 mask and a `RELU`.
  There is no softmax.
- `Q@K^T` and `P@V` are **not** here — they are on the [MXU](mxu.md), whose result is
  already int4 and directly usable as the next matmul's operand.

## Datatypes and requant

**Everything is int4, packed two per byte, low nibble first** — every source, every
elementwise destination. The one exception is `DOT`'s scalar, which is int32 and takes a
whole word. That is the same layout the MXU reads and writes, so a value moves between
the two units with no pass in between.

```
dst[i] = clip( (v[i] * m0 + (1 << (n-1))) >> n )
```

where `v[i]` is the op's pre-narrow value: `src0[i] + src1[i]` for `ADD`, `max(src0[i], 0)`
for `RELU`, `src0[i]` for the two rescales.

- **The narrow is fused into every op**, not a separate instruction. Nothing holds an
  int32 vector in the scratchpad: an `ADD` of two int4 operands is at most ±16 and its
  requant lands back on the grid in the same pass. This is what deletes the int32 staging
  temp a kernel used to allocate for every pointwise pair.
- `{m0, n}` is a **literal in the command** (`vpu_rq_word`), latched at start.
- `{m0, n} = {1, 0}` is the identity, for an op that only wants the arithmetic.
- The result scale is chosen entirely by `m0`/`n`, so a producer emits its result already
  in the consumer's scale. That is the only mechanism for reconciling a residual add whose
  operands differ in scale — `ADD` takes two operands at one scale, so the producer's
  narrow has to land on it. This is what pins `RQ_O` and `RQ_F`.

### `DYT`

The same fixed point with the clip moved: `[-7, 7]` instead of `[-8, 7]`.

DyT ([arxiv 2503.10622](https://arxiv.org/abs/2503.10622)) is `hardtanh(alpha*x, -1, 1)`
with one learned scalar. Pin the output scale to `1/7` and fold `alpha` into the
multiplier, and the **saturation of a narrow is the hardtanh**: every value the clip
catches is exactly one hardtanh would have flattened to ±1.

- The clip has to be symmetric because hardtanh is odd. A floor of `-8/7 = -1.143` would
  be off-spec at one end only.
- So DyT costs **zero extra passes**: a normalization always follows a residual add, and
  that add's result was going to be narrowed anyway.
- Nothing checks that the output scale really is `1/7`. A `DYT` targeting some other scale
  is a rescale with an odd clip, not a hardtanh — the scale contract is the compiler's.

## Interface

| Signal | Dir | Width | Meaning |
| --- | --- | --- | --- |
| `vpu_start` | in | 1 | latch operands and begin |
| `vpu_op` | in | 5 | operation selector |
| `vpu_src0` / `vpu_src1` | in | `ADDR_W` | source byte addresses |
| `vpu_dst` | in | `ADDR_W` | destination byte address |
| `vpu_rq_word` | in | `M0_W+N_W` | `{n, m0}` literal |
| `vpu_vlen` | in | 10 | vector length in elements, <= 1023 |
| `vpu_busy` / `vpu_done` | out | 1 | busy from start until retire; one-cycle done pulse |

There is no geometry on this interface and no `vpu_mm_busy` — both existed only for
`VECMATMUL`. `tpu_top.sv`'s counter 6 (`vmm`) is tied low and kept as a retired slot so
the UART `'T'` reply does not renumber.

`src1` and `rq_word` occupy separate command fields even though no op uses both, so decode
does not depend on the opcode. Older hardware (`scalar_unit.sv`) passed the `{m0,n}` table
*address* through the `src1` slot instead of the literal itself; that overlap is gone now
that the word is a literal.

### The `V_rw` scratchpad port

`LANES = SCRATCHPAD_W * 2` int4 elements per access, where `SCRATCHPAD_W` is the port
width in bytes.

- **The port is one scratchpad bank word.** `tpu_top` passes `SCRATCHPAD_W = N/2`, so the
  Cmod A7's 8x8 array gives 4 bytes and `LANES = 8`. A vector longer than `LANES` is
  streamed, so the array size only changes the chunk count.
- **All three pointers step by one word per chunk**, because sources and destination are
  the same width. That is what keeps every access word-aligned: `scratchpad.sv` addresses
  whole words and ignores the low bits, so a sub-word stride would re-read one word
  forever. The int8 datapath this replaced had exactly that bug at `LANES = 1`.

Two constraints a kernel must respect, both from the byte-granular write strobe:

- **`vpu_vlen` must be even for the elementwise ops.** Two elements share a byte; a tail
  that half-fills its byte writes nibble 0 into the other half rather than preserving what
  was there. `DOT` and `ARGMAX` write no nibbles and take any length.
- **`src0`, `src1` and `dst` must be word-aligned** (multiples of `N/2` bytes).

Signals: `V_re`/`V_raddr`/`V_rdata` (valid the cycle **after** `V_re`),
`V_we`/`V_waddr`/`V_wdata`/`V_wstrb` (per-byte strobe for partial tails and the scalar).

### Streaming

```
 idle --> RD0 --> RD1 --> EXEC --> ... --> idle
          (per LANES-element chunk, ceil(vlen/LANES) times)
```

The FSM reads a chunk of each operand, computes all lanes in one cycle, and either writes
the chunk back or folds it into a running int32 accumulator. `ARGMAX` reads `src0` only,
so it skips `RD1` like the rescales do. A partial final chunk is
masked by `V_wstrb` and by a lane-active predicate, so out-of-range lanes never contribute
to a reduction. `{m0,n}` is latched once at start.

**The VPU can be denied the scratchpad.** Arbitration is real since the queues broke the
old exclusivity invariant; the VPU freezes its FSM for a clock when denied.

## Removed ops

These were implemented and are gone. They served a softmax-attention, LayerNorm, GELU-FFN
model; the current model is ReLU attention, DyT and a ReLU feed-forward.

| Removed | Was for | Went with it |
| --- | --- | --- |
| `GELU` (4), `EXP` (6) | GELU FFN; softmax's `exp` | both 256-entry int8 ROMs, `rtl/luts/`, `accel/tpulang/luts.py` (deleted with the directory), the `GELU_INIT`/`EXP_INIT` parameters |
| `SQUARE` (5) | LayerNorm variance | — |
| `ELEMENT_MUL` (9), `SCALAR_MUL` (2), `SCALAR_ADD` (11) | LayerNorm/softmax broadcasts | — |
| `SCALAR_DIV` (12) | softmax's `sum(exp)`, LayerNorm variance | the restoring divider and its state |
| `REDUCEMAX` (7), `REDUCESUM` (8) | softmax/LayerNorm statistics | the **max fold** — restored by `ARGMAX` (18), which keeps the index rather than the value |
| `SOFTMAX` (14), `SM_EXP` (15) | the fused row-wise softmax macro op | its four-pass sequencer and `cfg vscalar` (index 9) |
| `VECMATMUL` (13) | attention's `QK^T` and `PV`, before `QUANT4` | the pair counter and address steppers, the five geometry inputs (`cfg` 10–14), the `VPU_GEOM` command, and `vpu_mm_busy`. **Not** `DOT` |
| `QUANT4` (17) | narrowing an int8 activation into the MXU's packed int4 weight layout | the `dst_is4` strobe special case and the last place a nibble stride differed from a byte stride |

`QUANT4` existed because activations were int8 and weights int4, so making an activation
into a weight operand took a pass. The MXU requantizes to int4 on store now, which is
exactly what that pass produced, and the VPU's own operands are int4 — so the op has no
work left to do. It went with the int8 datapath around it: `src0_is32`, `dst_is8`, the
per-op source and destination strides, and the int32 elementwise result.

Two ISA-level consequences:

- **`cfg vscalar`'s index is retired, not reused.** Renumbering 10–14 would silently
  repoint the DMA transpose geometry at 15–17.
- **`vpu_scalar` is unconditionally the third register operand.** `SOFTMAX` was the one op
  whose three registers were `dst`/`src`/`tmp`, leaving no slot for the requant word, and
  it was the only reason that routing had a special case.

### Measured area

**Stale — re-measure before quoting.** These numbers are from the int8 datapath, before
the int4 rewrite narrowed every lane from 32 bits to 4 and dropped the per-op strides.

Out-of-context synthesis of `vpu` alone (`build.tcl mode=ooc module=vpu`, xc7a35t,
ROWS=COLS=8):

| | LUTs | FFs | DSPs |
| --- | --- | --- | --- |
| softmax/LayerNorm era | 10 012 | 897 | 90 |
| int8, six ops | 5 162 | 667 | 32 |

The DSP collapse there was not mysterious: `SCALAR_MUL`, `ELEMENT_MUL`, `SQUARE` and
`SCALAR_DIV`'s reciprocal multiply each wanted a per-lane multiplier. What remained was
`DOT`'s product and the narrowing ops' `acc * m0`. The int4 rewrite adds `LANES` back
(1 -> 8 at `N=8`) while shrinking each lane, so the multiplier count is what to watch.

### What restoring something would cost

- **`softmax`** is the one to think about before deleting anything further —
  `model/transformer.py` still carries the comment *"revert to softmax if training bad"*
  next to its ReLU attention. Restoring it means restoring `EXP` (and so the ROM,
  `luts.py` and the `$readmemh` plumbing), `REDUCEMAX`, `REDUCESUM`, `SCALAR_DIV` with its
  divider, `SM_EXP`, the four-pass sequencer and `cfg vscalar` — essentially the whole
  table. Recoverable from git history, but not a one-line revert.
- **LayerNorm** would additionally want `SQUARE`.
- **`vecmatmul`** is cheap by comparison: `DOT` is still here, so it is the wrapper, not
  the datapath. Only worth doing for a model whose attention operand cannot be narrowed to
  int4 — on this one the array does the same work far faster.
