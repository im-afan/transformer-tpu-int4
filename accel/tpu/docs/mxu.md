# MXU — Matrix Unit

Weight-stationary systolic array. It computes **every matmul in the model**.

Dispatch arrives from `cmd_mxu.sv`'s 128-bit command queue, which carries the operands and
geometry — nothing is read out of a global config register at `start`.

## 1. What runs on it

At `d=128`, `f=512`, `head_dim=32`:

| Op | Shape per token tile | Notes |
| --- | --- | --- |
| Wq, Wk, Wv, Wo | `T x 128 @ 128 x 128` | attention projections |
| `Q @ K^T` | `T x 32 @ 32 x T` | per head; K is an int4 *activation* |
| `P @ V` | `T x T @ T x 32` | per head; V likewise |
| FFN `W1` | `T x 128 @ 128 x 512` | tiled over output columns |
| FFN `W2` | `T x 512 @ 512 x 128` | tiled over the contraction |
| output head | `T x 128 @ 128 x 16` | 13 logits padded to a whole tile |

- The head is padded because the array stores a whole `COLS`-wide output tile — a
  13-column result would have its second tile overrun into the next token's row.
- **Both attention matmuls are here now.** They are activation x activation, which this
  array cannot do, so they used to run on the VPU as `vecmatmul` — one serial dot product
  per output element, and 36.4% of the whole run. The fix was `quant4`: pack whichever
  operand lands on the *weight* side (K in `Q@K^T`, V in `P@V`) into the array's 4-bit
  layout. `vecmatmul` has since been removed.
- There is no softmax anywhere. Attention is ReLU, so the only thing between the two
  matmuls is a `vecadd` against an int8 mask and a `relu`, both on the VPU.

## 2. The PE is a real multiplier

This used to say the opposite, and the change is the main cost of int4.

- With **ternary** weights the multiply degenerated into a select + conditional negate, so
  a PE was an add/sub plus a 2-bit register — no DSP, no multiplier. That was the whole
  reason for ternary.
- With **int4** weights there are 16 levels, so `weight_product` is a signed multiply:
  `acc = psum_in + a * w`, `a` an int4 value in an int8 container.

What softens it: the activation narrowed at the same time. The product bound went from
`127 x 1` to `8 x 8`, so the **accumulator got cheaper**, not dearer — `PSUM_W = 16`
carries `ROWS*64` where the ternary array needed `ROWS*127`. The cost is confined to the
multiplier array.

The int4 range is **enforced, not assumed**: every `requant` / `dyt` / `quant4` clips to
`[-8, 7]` (or `[-7, 7]`), so no chain of matmuls can present an operand that overflows.
Genuine int8 activations would need `ROWS <= 32`.

## 3. Geometry

Parameterized `ROWS x COLS`. The `tpu_top.sv` default is 128x128; **the Cmod A7 board
builds 8x8** (`boards/cmod_a7/board.tcl`), which is what every kernel and golden vector is
written against.

- **Rows** = contraction dimension. Each row streams one activation element per clock.
- **Cols** = output features. Each column holds one stationary weight column.
- Dataflow: activations left→right, partial sums top→bottom.

```
          a_in (int8) --------------+
                                    v
weight reg (int4) --> [  x  ] -->( + )--> acc --> psum_out (down)
                                    ^
          psum_in (from PE above)
                        a_out --> next PE (right)
```

Each PE registers `a_in` for one cycle before passing it right, creating the systolic skew.

## 4. Phases

1. **Weight load.** Stream the `ROWS x COLS` int4 tile from the scratchpad into the PE
   registers. Weights are **row-major**, so one `W_rd` is one array *row* (`COLS` nibbles)
   and the loop runs `ROWS` times. Overlaps the previous tile's drain.
2. **Feed.** Read one activation column per clock and inject it **staggered** — row *i*
   delayed *i* cycles, so all contributions to an output element line up. This is why the
   scratchpad must deliver a full column per clock.
3. **Drain.** Column sums fall out after `ROWS + COLS + T` cycles into `result_buf`, which
   absorbs the output skew (each column result is placed by its propagated token id, so
   there is no separate de-skew network).

Latency for one tile is about `ROWS + COLS + T` cycles; throughput is one output column
per clock once full.

## 5. Numerics

- Accumulate in int32 (`PSUM_W` internally, widened on store).
- **Requantize on store**, gated by the `requant` input: assert it to narrow to int8, leave
  it clear to write int32 — the mode intermediate contraction tiles use so `accumulate` can
  keep running int32 partials in the result bank.
- The rescale is fixed-point `clip((acc*m0 + 2**(n-1)) >> n)`, applied by the store-path
  requantizer, not inside the PEs.
- **`{m0,n}` is a literal in the command** (`rq_word`), not a scratchpad address. Same 16
  bits either way, and it deleted a two-state fetch here and in `vpu.sv`.
- The store uses a per-byte `C_wstrb`, so an int8 requant row writes only its lanes and
  does not clobber neighbouring int32 result bytes.
- No bias add: the model is `use_bias=False`.

## 6. Tiling

- `COLS_array < N_out`: iterate output tiles, reloading weights each tile.
- `ROWS_array < K`: split the contraction, keep a running **int32 partial** in the result
  bank via `accumulate`, and requantize only on the last tile.
- `matmul_t` walks that grid **in hardware** — `k_tiles` and `n_tiles` are 8-bit fields in
  the command, and the whole walk costs the CPU ~380 clocks at any shape. Firmware can also
  walk it itself (`matmul_loop.c`), which costs ~85 clocks per dispatch.

## 7. Command fields

| Field | Meaning |
| --- | --- |
| `act_addr`, `weight_addr`, `out_addr` | scratchpad bases |
| `rq_word` | `{n, m0}` literal, used when `requant` |
| `t_len` | token rows, `<= 32` (`MAX_TOKENS`, the result buffer's depth) |
| `accumulate` | add into the existing int32 result |
| `requant` | narrow the store int32 -> int8 |
| `tiled` | use the strides below and the tile counts |
| `a_row`, `c_row`, `w_row` | row strides in bytes: `K`, `N*4`, `N*4/8`. `w_row` is the **output** width now, where under column-major weights it was the column stride |
| `k_tiles`, `n_tiles` | `K / ROWS`, `N / COLS`, 8 bits each |

Scratchpad ports: `A_rd` (ROWS int8), `W_rd` (COLS int4), `C_rw` (COLS int32, with
per-byte strobes). `load_active` drives the `mload` perf counter.

## 8. Sizing

Post-route on the Cmod A7 at 8x8: **7588 LUTs**, 5842 FFs, 18 DSPs — the largest block in
the design (43% of its LUTs). `TOK_W` (the result buffer depth) is the knob if area gets
tight.
