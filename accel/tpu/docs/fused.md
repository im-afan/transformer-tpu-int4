# tpu_matmul_wide_fused

## what it is
- one primitive for the shape every block of the model actually has: a matmul,
  then a residual add, then an activation
- unfused, each of those is its own pass over DRAM. the matmul spills its
  output, `tpu_elementwise` fills it back a chunk at a time, spills it again,
  and the activation does that a third time
- fused, an output block is matmulled, added to and activated where the MXU
  left it, and spilled once. the intermediate never lands in DRAM
- same flow, same layout arithmetic and same weight prefetch as
  `tpu_matmul_wide`. it is that function with the C region split in two

## the ops it fuses
- `add_op` is `TPU_V_ADD` or `TPU_ACT_NONE`
- `activation` is `TPU_V_RELU`, `TPU_V_DYT`, `TPU_V_REQUANT` or `TPU_ACT_NONE`
- `add` is one tensor shaped like C, and it is the second operand of *both*
  fusable steps. a DYT activation reads the same staged block again, which is
  what makes the double residual (`dyt(requant(O + X) + X)`) a single call
- three requant words, one per step: `rq_word` on the matmul's store, `rq_add`
  on the add, `rq_act` on the activation. an unfused sequence carries three
  different `{m0,n}` at these sites and a single word would be the wrong number
  at two of them
- no accumulate. C is written, never read — the accumulate path's staged C is
  where the add block goes instead

## layout
- `tpu_gemm_fit` is called with a C row of `2 * TPU_WORD_BYTES`, so a panel row
  costs `depth/2 + 2 * TPU_WORD_BYTES` against `tpu_matmul_wide`'s
  `depth/2 + TPU_WORD_BYTES`. that is the whole cost: a slightly shallower row
  panel, so a wide problem reads its weight stream a little more often
- the add block sits at `c_slot + ALIGN_UP(panel_rows, TPU_N) * TPU_WORD_BYTES`.
  the round-up is load-bearing: `panel_rows` is capped to `rows`, which can
  leave it short of a whole N-row block, and the MXU still stores whole blocks.
  without it a 20-row panel's last store walks 16 bytes into the add block —
  `tests/fused/generate.py --single-buffer` is the case that catches it
- C and the add block share a bank. the VPU reads its two sources in separate
  states so it costs nothing, and the MXU never touches the add block
- both VPU passes are in place on the output block. safe because the VPU runs
  one word at a time: it reads both sources before it writes

## infer.c's three builds
- `INFER_MM_MODE`, a ladder, one rung per thing being measured:
  - `0 base` — `tpu_matmul_wide`, one weight buffer (`TPU_WGT_PREFETCH=0`)
  - `1 dbuf` — the weight double buffer on top
  - `2 fused` — the fused primitive on top of that
- `tests/infer/generate.py --mm base|dbuf|fused`. same golden three ways
- `-DINFER_MM_WIDE=0` (`--general`) is a separate A/B and puts every site on
  `tpu_matmul`; it has no fused primitive, and a static assert says so
- the four fused sites, per layer:
  - `Q@K^T` + mask + relu — `rq[RQ_S]`, `rq[RQ_ID]`, `rq[RQ_P]`
  - `A@Wo` + X + dyt(+X) — `rq[RQ_O]`, `rq[RQ_XO]`, `rq[RQ_X1]`
  - `X1@W1` + relu, no add — `rq[RQ_H]`, `rq[RQ_HR]`
  - `HR@W2` + dyt(+X1), no add — `rq[RQ_F]`, `rq[RQ_X2]`
- O, X+O and F stop existing as DRAM tensors. so does H's pre-relu value

## measured
- on the RTL, synthetic `d=64 / f=256`, `--gen 3 -n 1`, one image per rung:

```
mode    clocks      mxu       dma      idlec   ovlap   commands
base   1544793   239766    789400     29.2%    0.0%       6792
dbuf   1421291   239766    789400     32.7%    9.7%       6792
fused  1245012   239766    588416     39.1%   10.9%       7876
```

- fused is **-12.4% against dbuf and -19.4% against base**, and the whole
  difference is DMA: 789 400 -> 588 416 clocks (-25.5%), which is the
  intermediates no longer round-tripping through DRAM. `mxu` is identical to
  the clock on all three — the array does the same work either way
- it costs +16.0% commands (the per-block VPU passes) and pushes `idlec` up,
  because what is left is more CPU per byte moved
- firmware image at `d=128 / f=512`, synthetic: base 11 680 bytes, dbuf 14 632,
  fused 12 824, of a 16 KB RAM. fused is *smaller* than dbuf — it deletes three
  `INFER_MM` instantiations and the elementwise chunk loops at four sites,
  which pays for the primitive twice over

## the test
- `tests/fused/` is the regression, DRAM to DRAM, and the golden is a plain
  Python matmul followed by the same narrows the VPU does
- `--sweep` runs every add/activation combination; `--no-add --act relu` is the
  FFN's shape, `--act dyt` the residual's
- default shape is wide.c's: two row panels, an odd number of column blocks and
  a ragged last one
