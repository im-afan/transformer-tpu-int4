# MXU — Matrix Unit

## Overview
- `N x N` output-stationary systolic array, int4 in and int4 out. One accumulator per PE.
- One dispatch computes `C[N][N] = A[N][len] @ B`, or `@ B'` when `transpose`.
- `len` is the contraction length in int4 elements, and also the number of streaming
  clocks. It is 16 bits, so a contraction never has to be split.
- Every operand word is `N*4` bits — one scratchpad bank word. A, B and C all use it.
- Requant to int4 is unconditional; there is no int32 store path and no hardware tiling.

## Ports
- Basic: `clk`, `rst_n`
- Dispatch (in): `start`, `transpose`, `accumulate`, `len`,
  `a_base`/`a_stride`, `b_base`/`b_stride`, `c_base`/`c_stride`, `rq_word`
- Dispatch (out): `busy`, `done`
- Scratchpad: `A_re`/`A_addr`/`A_rdata`/`A_gnt`, `B_re`/`B_addr`/`B_rdata`/`B_gnt`,
  `C_en`/`C_we`/`C_addr`/`C_wdata`/`C_rdata`/`C_gnt`

`start` is a one-clock pulse; the producer holds the operands until `done`.

## Operand layouts

All three are row-major, 4-bit packed, two elements per byte, low nibble first.

| Operand | Shape | Row stride | Read as |
| --- | --- | --- | --- |
| A | `[N][len]` | `a_stride` (default `len/2`) | one contiguous chunk of N elements per array row |
| B, `transpose=0` | `[len][N]` | `b_stride` (default `N/2`) | one whole B row per contraction step |
| B, `transpose=1` | `[N][len]` | `b_stride` (default `len/2`) | one contiguous chunk of N elements per array column |
| C | `[N][N]` | `c_stride` (default `N/2`) | one row per store |

- A zero stride means the densely packed default. There is no `tiled` flag: a stride is
  either given or defaulted, and nothing survives between dispatches.
- `transpose=0` is what the projections want (weights stored `[K][N]`).
  `transpose=1` is what `Q@K'` and `P@V` want (the KV cache stored `[T][head_dim]`),
  which is why the DMA no longer needs a transposing mode.

## Input constraints
- A, B and C must not share a scratchpad bank. A and B are read on the same clock and
  a bank serves one read per clock, so an overlap loses B's word silently. `scratchpad.sv`
  prints a simulation warning the first time it sees it.
- All three bases and strides must be multiples of `N/2` bytes — the banked scratchpad
  addresses whole words, with no unaligned window.
- `N` must be a power of two, at least 2.

## Output constraints
- Always writes `N` rows of `N` int4 to `c_base`, row by row. There is no row count in the
  dispatch, so a matmul with fewer than `N` live rows still reads `N` rows of A and writes
  `N` rows of C — the caller allocates the whole block.
- `C = requant(A @ B)`, and with `accumulate`, `C = clip4(requant(A @ B) + C_old)`.
  Because the store is int4, `accumulate` is a fused int4 add, not an int32 partial sum.

## Datapath
- A enters the column-0 edge and flows toward column `N-1`; B enters the row-0 edge and
  flows down. PE(i,j) accumulates `a_in * b_in` when the valid riding with A is set.
- **A is never read column-by-column.** A column of A is a strided gather out of a
  row-major matrix. Instead one `N*4`-bit chunk — N contraction elements of one array
  row — is loaded into that row's edge register, round-robin: row 0 at clock 0, row 1 at
  clock 1, and back to row 0 at clock N.
- **The round-robin is the skew.** Row *i* must start injecting at clock *i*, which is
  exactly when its chunk lands, and it holds N elements, which is exactly how long until
  its next chunk. No input skew registers on the A side at all.
- `transpose=1` feeds B the same way, one chunk per array column.
- `transpose=0` reads one whole B row per clock instead, so the top edge needs a
  triangular skew chain: column *j* taps stage *j*.
- Element *k* of row *i* is injected at step `i + k + 1`; element *k* of column *j* at
  step `j + k + 1`. Both reach PE(i,j) at step `i + j + k + 1`.
- Streaming ends at step `len + 2N - 2`, the last accumulation at PE(N-1,N-1).

## Stalls
- The operand fetch runs one step ahead of the array. A denied `A_gnt`/`B_gnt` re-presents
  the same address and the whole array freezes for that clock — the pipeline holds, no word
  is lost or repeated.
- The C port is top of the scratchpad's write chain and is only used while A and B are
  idle, so a store is never denied.

## Writeback
- `N` clocks without `accumulate`; the requant is `clip((acc*m0 + round) >> n)` to
  `[-8, 7]`, one nibble per column, packed into one word per row.
- With `accumulate`, each row costs a read then a write.
- `{m0, n}` is the `rq_word` literal in the command, not a scratchpad address.

## Command encoding

Two 128-bit commands. A self-contained matmul would need three addresses, three strides,
`len` and the requant word, which does not fit, so the geometry rides its own command and
is sticky inside this unit's queue. The producer can skip a `GEOM` whose values have not
changed; nothing outside the queue can write it.

`MXU_GEOM` (`0x01`) — latched, retires in one clock, starts nothing:

| bits | field |
| --- | --- |
| `w0[7:0]` | op |
| `w0[31:16]` | `a_stride` |
| `w1[15:0]` | `b_stride` |
| `w1[31:16]` | `c_stride` |
| `w2[15:0]` | `len` |

`MXU_MM` (`0x02`) — one matmul:

| bits | field |
| --- | --- |
| `w0[7:0]` | op |
| `w0[8]` | `accumulate` |
| `w0[9]` | `transpose` |
| `w0[31:16]` | `c_base` |
| `w1[15:0]` | `a_base` |
| `w1[31:16]` | `b_base` |
| `w2[15:0]` | `rq_word` = `{n, m0}` |

## Notable changes
- Weight-stationary became output-stationary, so there is no weight-load phase. The
  `mload` perf counter (index 2) is tied low rather than renumbered.
- Hardware tiling is gone — `tiled`, `k_tiles`, `n_tiles` and the tile-walk loop with it.
  Output-stationary already tiles the contraction, and the CPU walks the other two axes.
- `t_len` and the `MAX_TOKENS` result buffer are gone. The output block is always `N x N`.
- The int32 store path and its `requant` flag are gone. int4 is forced, so an MXU result
  is directly usable as the A or B operand of the next matmul, with no `quant4` pass.
