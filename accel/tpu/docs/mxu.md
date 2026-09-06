# MXU — Matrix Unit

## Overview
- `N x N` output-stationary systolic array, int4 in and int4 out. One accumulator per PE.
- One dispatch computes `C[N][N] = A[N][len] @ B`, or `@ B'` when `transpose`.
  - requants result using requant scalar {M0, N}
- `len` is the contraction length in int4 elements, and also the number of streaming
  clocks. It is 16 bits, so a contraction never has to be split.
- Streams a `4 * N`-bit word from scratchpad from A and B, and writes output to C.
  - A and B must be on different scratchpad banks to allow for simultaneous loading.

## Ports
- Basic: `clk`, `rst_n`
- Dispatch (in): `start`, `transpose`, `accumulate`, `len`,
  `a_base`/`a_stride`, `b_base`/`b_stride`, `c_base`/`c_stride`, `rq_word`
- Dispatch (out): `busy`, `done`
- Scratchpad: `A_re`/`A_addr`/`A_rdata`/`A_gnt`, `B_re`/`B_addr`/`B_rdata`/`B_gnt`,
  `C_en`/`C_we`/`C_addr`/`C_wdata`/`C_rdata`/`C_gnt`

`start` is a one-clock pulse; the producer holds the operands until `done`.

## Operand layouts

All are row-major, 4-bit packed (2 elements / byte).

- `A[N][len]` `a_stride` (default `len/2`)  1 contiguous chunk, N elements / row 
- `transpose=0` `B[len][N]` `b_stride` (default `N/2`) | one whole B row per contraction step |
- `transpose=1` `B[N][len]` `b_stride` (default `len/2`) | one contiguous chunk of N elements per array column |
| C | `[N][N]` | `c_stride` (default `N/2`) | one row per store |

- A 0 stride = densely packed default.
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
- An `N*4`-bit chunk is loaded into that row's edge register, round-robin: row 0 at clock 0, row 1 at
  clock 1, and back to row 0 at clock N. Row i gets its data at clock i; 
  exactly when its first input to the MAC array is needed.
- `transpose=1` feeds B the same way, one chunk / array column.
- `transpose=0` reads one whole B row per clock instead; the data arrives and is skewed before being read by the MAC array.
- Element *k* of row *i* is injected at step `i + k + 1`; element *k* of column *j* at
  step `j + k + 1`. Both reach PE(i,j) at step `i + j + k + 1`.
- Streaming ends at step `len + 2N - 2`, the last accumulation at PE(N-1,N-1).

## Writeback
- `N` clocks without `accumulate`; the requant is `clip((acc*m0 + round) >> n)` to
  `[-8, 7]`, a single N*4-bit word is written back to scratchpad, N times 
- With `accumulate`, each row is a read + a write.

## Command encoding

`MXU_GEOM` (`0x01`):

| bits | field |
| --- | --- |
| `w0[7:0]` | op |
| `w0[31:16]` | `a_stride` |
| `w1[15:0]` | `b_stride` |
| `w1[31:16]` | `c_stride` |
| `w2[15:0]` | `len` |

`MXU_MM` (`0x02`) one matmul:

| bits | field |
| --- | --- |
| `w0[7:0]` | op |
| `w0[8]` | `accumulate` |
| `w0[9]` | `transpose` |
| `w0[31:16]` | `c_base` |
| `w1[15:0]` | `a_base` |
| `w1[31:16]` | `b_base` |
| `w2[15:0]` | `rq_word` = `{n, m0}` |