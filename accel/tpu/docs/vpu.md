# VPU — Vector Unit

SIMD unit for every pointwise / reduction operation that is not a matmul.

Dispatch arrives from `cmd_vpu.sv`'s 128-bit command queue. The VPU has exactly **one**
command type (`0x01`); `0x02` (`VPU_GEOM`) is a retired hole.

## Operations — six, and nothing else

| `vpu_op` | Name | Kind | Operands | Result |
| --- | --- | --- | --- | --- |
| 0 | `DOT` | reduction | `src0`, `src1` | scalar `sum(src0[i]*src1[i])` -> `dst` |
| 1 | `ADD` | elementwise | `src0`, `src1` | `dst[i] = src0[i] + src1[i]` |
| 3 | `RELU` | elementwise | `src0` | `dst[i] = max(src0[i], 0)` |
| 10 | `REQUANT` | narrow | `src0` (int32), `rq_word` | `clip[-8,7]((src0[i]*m0 + rnd) >> n)` |
| 16 | `DYT` | narrow | `src0` (int32), `rq_word` | `clip[-7,7]((src0[i]*m0 + rnd) >> n)` |
| 17 | `QUANT4` | narrow + pack | `src0` (**int8**), `rq_word` | `clip[-8,7](...)`, written **4 bits wide** |

- The gaps (2, 4–9, 11–15) are **retired codes, not free encoding space**. They are left
  vacant so a stale binary decodes to an unknown op rather than a different one.
- `DOT` is the only reduction; every other op produces a same-length vector.
- **No shipped kernel issues `DOT`.** It is kept because it *is* the reduction datapath —
  accumulator, lane fold, scalar store — and the only int8 x int8 reduction the ISA has.
  It costs one decode arm over hardware that has to be there anyway.

### What the model uses

- **Activations:** `RELU`. The FFN is ReLU, not GELU.
- **Elementwise:** `ADD`, for the residuals `X + O` and `X1 + F`.
- **Narrowing:** `REQUANT`, `DYT`, `QUANT4`.
- **The attention mask:** `relu(S + mask)` is an `ADD` against an int8 mask and a `RELU`.
  There is no softmax.
- `Q@K^T` and `P@V` are **not** here — they are on the [MXU](mxu.md), because `QUANT4`
  packs K and V into the array's weight layout.

## Datatypes and requant

Compute ops read int8, accumulate in int32, and write int32 straight to the scratchpad —
the VPU does **not** narrow on the writeback path. Narrowing is an explicit instruction.

```
dst[i] = clip( (src0[i] * m0 + (1 << (n-1))) >> n )
```

- `{m0, n}` is a **literal in the command** (`vpu_rq_word`), latched at start. It used to
  be a scratchpad address; making it a literal deleted a two-state fetch here and in
  `mxu.sv`. Same 16 bits either way.
- The narrow **lands int4 in an int8 container**: `REQUANT` and `DYT` still write one byte
  per element, only the value range narrows. Clipping is what enforces the MXU's operand
  bound, so it is not optional.
- Keeping requant separate lets a kernel hold a value at int32 and narrow **only where a
  narrow activation is actually consumed**, instead of clipping after every pointwise op.
- The result scale is chosen entirely by `m0`/`n`, so a producer emits its result already
  in the consumer's scale. That is the only mechanism for reconciling a residual add whose
  operands differ in scale — `ADD` takes two operands at one scale, so the producer's
  narrow has to land on it. This is what pins `RQ_O` and `RQ_F`.

### `DYT`

The same instruction with the clip moved: `[-7, 7]` instead of `[-8, 7]`.

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

### `QUANT4`

The third op on the same shifter, and the only one whose destination is not a byte.

Two differences from `REQUANT`, both the point:

- **The source is int8**, not int32. It narrows an activation a requant already produced,
  so it does not read the accumulator width. It is the one op where `src0_is32` and
  `dst_is8` are genuinely independent predicates.
- **The destination is 4 bits** — a bare two's-complement nibble in the MXU's packed
  weight layout, two elements per byte. So the result is not "an int4 tensor that then
  needs packing"; it *is* a packed weight block, addressable by `matmul_t` with no pass in
  between.

That is what lets an **activation be a weight operand**, which is the whole reason the op
exists, and therefore what put both attention matmuls on the array.

Since weights and activations are the same width now, the *value* is usually unchanged —
whatever requant produced `src0` already clipped it to `[-8, 7]`, so `{m0,n} = {1,0}` makes
this a pure repack. Under the old ternary encoding it was a genuine second narrow onto
`{-1, 0, 1}`, and that rounding was real accuracy loss.

Two constraints a kernel must respect:

- **`vlen` must be a multiple of 2.** The write strobe is per byte and a byte holds two
  nibbles; a tail that does not fill its byte writes nibble 0 into the remaining slot
  rather than preserving what was there.
- **The destination pointer advances half as fast as the source.** A `CHUNK`-long pass
  reads `CHUNK` bytes and writes `CHUNK/2`.

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

### The `V_rw` scratchpad port

`LANES = SCRATCHPAD_W / 4` (int32-limited), where `SCRATCHPAD_W` is the port width in
bytes.

- `tpu_top`'s default `VPU_BYTES = 64` (512-bit) gives `LANES = 16`.
- **The Cmod A7 build is half that**: `VPU_BYTES = 32`, so `LANES = 8`. Only the chunk
  count changes; a vector longer than `LANES` is streamed either way.

int32 operands use the full width; int8 operands occupy the low `LANES` bytes, so the byte
strides differ (int8 advances `LANES`, int32 advances `LANES*4`). The port is one logical
read/modify/write port, so reads and writes go on separate cycles.

Signals: `V_re`/`V_raddr`/`V_rdata` (valid the cycle **after** `V_re`),
`V_we`/`V_waddr`/`V_wdata`/`V_wstrb` (per-byte strobe for partial tails and scalar writes).

### Streaming

```
 idle --> RD0 --> RD1 --> EXEC --> ... --> idle
          (per LANES-element chunk, ceil(vlen/LANES) times)
```

The FSM reads a chunk of each operand, computes all lanes in one cycle, and either writes
the chunk back or folds it into a running int32 accumulator. A partial final chunk is
masked by `V_wstrb` and by a lane-active predicate, so out-of-range lanes never contribute
to a reduction. `{m0,n}` is latched once at start.

**The VPU can be denied the scratchpad now.** Arbitration is real since the queues broke
the old exclusivity invariant; the VPU freezes its FSM for a clock when denied.

## Removed ops

These were implemented and are gone. They served a softmax-attention, LayerNorm, GELU-FFN
model; the current model is ReLU attention, DyT and a ReLU feed-forward.

| Removed | Was for | Went with it |
| --- | --- | --- |
| `GELU` (4), `EXP` (6) | GELU FFN; softmax's `exp` | both 256-entry int8 ROMs, `rtl/luts/`, `accel/tpulang/luts.py`, the `GELU_INIT`/`EXP_INIT` parameters |
| `SQUARE` (5) | LayerNorm variance | — |
| `ELEMENT_MUL` (9), `SCALAR_MUL` (2), `SCALAR_ADD` (11) | LayerNorm/softmax broadcasts | — |
| `SCALAR_DIV` (12) | softmax's `sum(exp)`, LayerNorm variance | the restoring divider and its state |
| `REDUCEMAX` (7), `REDUCESUM` (8) | softmax/LayerNorm statistics | the **max fold** — `acc` now always opens at zero |
| `SOFTMAX` (14), `SM_EXP` (15) | the fused row-wise softmax macro op | its four-pass sequencer and `cfg vscalar` (index 9) |
| `VECMATMUL` (13) | attention's `QK^T` and `PV`, before `QUANT4` | the pair counter and address steppers, the five geometry inputs (`cfg` 10–14), the `VPU_GEOM` command, and `vpu_mm_busy`. **Not** `DOT` |

Two ISA-level consequences:

- **`cfg vscalar`'s index is retired, not reused.** Renumbering 10–14 would silently
  repoint the DMA transpose geometry at 15–17.
- **`vpu_scalar` is unconditionally the third register operand.** `SOFTMAX` was the one op
  whose three registers were `dst`/`src`/`tmp`, leaving no slot for the requant word, and
  it was the only reason that routing had a special case.

### Measured area

Out-of-context synthesis of `vpu` alone (`build.tcl mode=ooc module=vpu`, xc7a35t,
ROWS=COLS=8):

| | LUTs | FFs | DSPs |
| --- | --- | --- | --- |
| before | 10 012 | 897 | 90 |
| after | **5 162** | **667** | **32** |
| | −48.4% | −25.6% | −64.4% |

The DSP collapse is not mysterious: `SCALAR_MUL`, `ELEMENT_MUL`, `SQUARE` and
`SCALAR_DIV`'s reciprocal multiply each wanted a per-lane multiplier. What remains is
`DOT`'s 8x8 product and the two narrowing ops' `acc32 * m0`.

In the full Cmod A7 build the VPU is **4021 LUTs and 24 DSPs** post-route.

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
