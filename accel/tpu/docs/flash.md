# tpu_flashattention

## what it is
- one head of `out = relu(Q @ K' + mask) @ V`, causal, done as a two-level tile
  loop with the score matrix living in the scratchpad the whole time
- `rows` queries starting at `first_pos` on the key axis, against `keys` keys.
  so one call is a prefill pass (`rows = keys`, `first_pos = 0`) or a decode
  step (`rows = 1`, `first_pos = pos`) with no second code path
- `Q` and `out` are `[rows][head_dim]`, `K` and `V` are `[keys][head_dim]`,
  `mask` is the additive causal mask indexed by absolute position — the same
  `DR_MASK` `infer.c` already keeps
- takes a `tpu_flash` and an arena, like every other primitive. self-fencing

## why the tiling is exact here
- classic FlashAttention tiles a softmax, so it carries a running maximum and a
  running denominator and rescales every time the max moves
- this model has ReLU attention and *no* normalization over the source axis
  (`model/transformer.py`: `P = relu(S + causal_mask)`, nothing after it). a key
  block's contribution to the output is a plain partial sum
- so there is nothing to rescale and none of that state exists. the whole
  algorithm is: tile the keys, accumulate

## what it saves over the four-pass sequence
- unfused, attention is `INFER_MM(rows, HEAD_DIM, T)` spilling `S` to DRAM, an
  add pass and a relu pass filling and spilling it again, then
  `INFER_MM(rows, T, HEAD_DIM)` filling it back. `S` is `[rows][T]` and crosses
  DRAM four times
- here `S` is `[B][B]` in the scratchpad and crosses nothing. the mask add and
  the ReLU run on the block where the MXU left it
- a key block starting at or past the panel's last query is skipped whole. `-8`
  against an int4 score and then a ReLU is *exactly* zero, so its contribution
  to `P@V` is zero and dropping it is not an approximation — the panel's `j`
  loop just stops. at a decode step that is most of the cache
- the mask is only staged for the block straddling the diagonal. one fill per
  row panel instead of one per tile

## what it costs
- the contraction over keys is split, one block per `j` step, and the MXU's
  accumulate is `clip4(requant(A @ B) + C_old)` — an int4 add, not an int32
  partial (`docs/mxu.md`). so a partial sum is requantized and clipped to
  `[-8, 7]` at every one of the `keys/B` steps, where `INFER_MM(rows, T,
  HEAD_DIM)` contracts every key in one dispatch and clips once
- **when `B == keys` that is a no-op** and the result is bit-identical to the
  unsplit contraction. `tests/flash` checks exactly that: every one-block shape
  matches the unsplit reference to the element
- when it is not, the drift is real but small — `tests/flash --sweep
  --unsplit-check` measures 7.4% of elements off by a grid step at
  `T=96, B=64` and 8.3% at `T=128, B=88`. it is a rounding-per-block effect,
  not a saturation one, because `RQ_A` is fitted to the *whole* contraction's
  accumulators and a block's partial is a fraction of that
- so `RQ_A` needs no retuning to move a site onto this primitive. an int32
  accumulate in the MXU's store path would remove the drift entirely; nothing
  else will

## layout
- `tpu_flash_fit` walks the block size down from the whole key axis in `TPU_N`
  steps until it fits, so `B` is the largest whole array word the arena holds
- five bank-disjoint regions: `Q`, `K`, `V`, `out` at
  `ALIGN_UP(B * head_dim/2, bank)` each, and one score region at
  `ALIGN_UP(B * B, bank)` holding `P` and the mask block back to back
- bank-disjoint because the MXU reads A, B and C on the same clock: `Q/K/P` for
  the scores, `P/V/res` for the output. the mask block shares `P`'s region
  because only the DMA and the VPU touch it, and the VPU reads its two sources
  in separate states — the same argument `tpu_matmul_wide_fused` makes for its
  add block
- `res` accumulates in the scratchpad across the whole `j` loop and is spilled
  once per row panel. nothing is zeroed first: the first live `j` block writes
  rather than accumulates, and the MXU stores whole `N x N` blocks, which covers
  `panel x head_dim` exactly
- five banks is the minimum whatever the shape, because every region rounds up
  to one. more banks buys a bigger `B`

## shape rules
- `keys % TPU_N == 0` and `head_dim % TPU_N == 0`, both build-time asserts.
  `rows` is free — a decode step passes 1
- `first_pos + rows <= keys`
- a short key block leaves the score columns past `cols` stale. the two VPU
  passes run over the whole staged width anyway and the `P@V` contraction is
  `len = cols`, so the stale columns are written and never read
- no double buffer. `K` and `V` for block *j+1* could stream under block *j*'s
  matmuls the way `tpu_matmul_wide` stages its weights, at the cost of two more
  banks. not done

## order of the two VPU passes
- the pseudocode this was written from has `relu(P_s)` before `P_s = P_s + mask`
- that is the wrong way round: a masked score would come out of the ReLU at
  `>= 0`, get `-8` added, and go into `P@V` as a nonzero negative. the model is
  `relu(S + mask)` and `infer.c` adds then relus
- implemented as add-then-relu

## in infer.c
- `-DINFER_ATTN_FLASH=1`, or `tests/infer/generate.py --attn flash`. the
  default is `blocks`, the four-pass sequence
- it replaces the whole per-head body: the score matmul, the mask add, the relu
  and `P@V`. `S` stops being a DRAM tensor, and in the fused build the fused
  score site goes with it
- orthogonal to `INFER_MM_MODE` and to `INFER_MM_WIDE` — those pick the
  primitive every *other* site uses
- **at the live shape the key axis fits one block**, so the golden is unchanged
  and the tokens are bit-identical. `T=32, head_dim=32` against a 65 408-byte
  arena gives `B = 32`; `T=128, head_dim=32` gives `B = 128`. it takes
  `T = 256` before `B` (208) is shorter than the axis
- `generate.py` computes the same `B` (`vector_generator.flash_block`) and, when
  it is shorter than the key axis, tiles the reference's `P @ V` the same way.
  if the two ever disagree about `B` every case fails, which is the point
- image size, synthetic `d=128 / f=512`, text+data:

```
            blocks   flash
  base       11364   11008
  dbuf       14420   12748
  fused      12468   11480
```

  smaller on every rung: it deletes two `INFER_MM` instantiations and the
  elementwise chunk loops at the mask and relu sites
- on the RTL, synthetic `d=64 / f=256, --gen 3, --mm dbuf`:

```
            clocks       cmds     mxu     vpu     dma   idlec   ovlap
  blocks   1385468       6744   239766   64480  763216  452925  134919
  flash    1137223       5528   239766   64480  604768  354948  126739
```

  **-17.9% clocks and -18.0% commands**, with `mxu` and `vpu` identical to the
  clock. the array does the same arithmetic and the VPU narrows the same number
  of elements; the whole difference is 158 448 clocks of DMA that were `S`
  round-tripping to DRAM, and the issue overhead that went with it

## tests/flash
- one `tpu_flashattention`, DRAM to DRAM, golden in plain Python
- `--sweep` covers one key block, two blocks with a ragged tail, a bigger
  `head_dim`, a decode shape (`rows=1` at the last position) and a mid-sequence
  panel. `--unsplit-check` prints the split-vs-unsplit drift
- `--arena-banks` is the knob that picks `B`, because the arena is what the fit
  reads. `-T`, `-d`, `--rows`, `--first-pos` are the rest
- in `run_suite.py`'s default set
