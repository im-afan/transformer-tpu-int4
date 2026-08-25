# DMA Engine

Moves bytes between **DRAM** (the Cmod A7's external SRAM, driven by
[`sram.sv`](../rtl/sram.sv)) and the on-chip [scratchpad](scratchpad.md).

**Status: built.** `rtl/dma.sv` is the engine, `tb/dma_tb.sv` the bench (16 737 checks,
0 errors). Post-route: 493 LUTs, 132 FFs.

## 1. Where it sits

```
  cmd_dma  -- 128-bit command queue -->  +---------+
                                         |   DMA   |
  scratchpad <-- dma_* port, granted --> |  engine |
                                         |  (FSM)  |
  sram_ctrl  <-- range requests + byte streams -->  +
      |
      +-- FPGA pins --> external SRAM chip
```

The DMA is a **master on two buses**: a slave to the command queue, a master to both the
scratchpad and the SRAM controller.

- **Command side** — `cmd_dma.sv` pops a 128-bit macro-op carrying the addresses, the byte
  count, the direction and the transpose geometry. Nothing comes from a global config
  register, so a transfer carries its own length and the stale-`len` pitfall is gone by
  construction.
- **Scratchpad side** — a dedicated `dma_*` port, byte-strobed. The DMA takes
  `dma_rgnt`/`dma_wgnt` back from the arbiter and holds when denied.
- **DRAM side** — one `start`/`busy`/`done` request per **range** (`addr` + `len` +
  `stride`), `we` selecting read or write, 19-bit address, 8-bit data, with a `dout_valid`
  strobe per byte read and a ready/valid stream for bytes written. `tpu_top` arbitrates the
  controller between the DMA (while a program runs) and the UART host (while idle).

## 2. The command

| Field | Meaning |
| --- | --- |
| `dma_write` | **1 = spill** scratchpad -> DRAM, **0 = fill** DRAM -> scratchpad |
| `dma_scratch_addr` | scratchpad byte base |
| `dma_dram_addr` | DRAM byte base, **19 bits** |
| `dma_len` | bytes to move, 16 bits (so one transfer is under 64 KB) |
| `dma_transpose` | transposed addressing (§4) |
| `dma_tcols` | source row length, elements |
| `dma_tsrow` | source row stride, bytes |
| `dma_tdrow` | destination row stride, bytes |
| `dma_busy` / `dma_done` | in progress / complete |

## 3. The two flows

The engine walks a byte counter from 0 to `len-1`. The DRAM side is a **range**, issued once
per source row, whose bytes arrive or are asked for one per handshake.

**Fill (DRAM -> scratchpad).** Issue the range with `we=0`; then per byte, `sram.dout_valid`
pulses with the byte, and in that same clock the scratchpad write goes out with a one-hot
strobe. The scratchpad port takes one every clock, so nothing buffers and nothing needs to
be aligned.

**Spill (scratchpad -> DRAM).** Issue the range with `we=1`; then two clocks per byte,
pipelined against the controller's two-clock write beat:

1. Scratchpad read — `dma_rdata` is valid the *next* cycle.
2. Offer it — the controller takes it on `sram.din_ready`. A byte not taken immediately is
   held in a one-byte register rather than re-read, because the scratchpad only promises its
   read data for the one cycle after `re`.

The SRAM is the bottleneck, but it is a stream now rather than a sequence of transactions.

The FSM's whole memory is one byte counter, the 2-D walk (`col`/`row` and their running
offsets), and the spill holding register. The states exist because the two memories answer
on different clocks — the scratchpad one cycle after `re`, the SRAM one byte per beat.

## 4. Transpose mode

The byte count and the FSM are unchanged; only the two address generators differ.

```
for r = 0, 1, 2, ...          # rows, until dma_len bytes have moved
  for c = 0 .. tcols-1
     read   source      at  src + r*tsrow + c     # row-major
     write  destination at  dst + c*tdrow + r     # transposed
```

**"Source" is direction-relative, and that is the whole trick.** On a fill the source is
DRAM; on a spill it is the scratchpad. One convention covers both, because transposing out
of an `[R][C]` tensor and transposing into a `[C][R]` one are the same permutation. One rule:
**the source is read row-major, the destination is written transposed.**

### Why the strides are free, not just `C` and `R`

- `tsrow` lets the source be a **column slice of a wider tensor** — one head's share of a
  fused projection, read in place instead of copied out first.
- `tdrow` lets the result land inside a wider destination — one head's `K^T` written into a
  `[H*d][T]` block, or one token's column appended to a `[D][T]` KV cache.

The row count `R` is never a parameter: `dma_len` fixes it. A `dma_len` that is not a whole
number of rows stops part-way through the last one rather than faulting, and writes nothing
past it.

### Zero means "not set"

Each parameter falls back to the value that makes the mode degenerate gracefully rather than
collapse every address onto 0 — the same convention `mxu.sv` uses for its strides.

| Register | 0 means | Effect |
| --- | --- | --- |
| `tcols` | `dma_len` | one row: a **strided scatter/gather**, the same generator seen end-on |
| `tsrow` | `tcols` | dense source rows |
| `tdrow` | `1` | dense destination |

With all three unset, a transposed transfer is bit-for-bit a plain one.

### The mode is a command bit, the geometry is data

It has to be in the command either way: geometry that persisted between programs would let
one program's leftover strides silently rearrange the next program's plain fill. That is
exactly the bug the MXU hit under the old config-register model. A plain fill/spill ignores
all three registers outright.

### Cost

Two 16-bit counters, two running-sum offset registers, and a mux on each address — no
multipliers, since both offsets accumulate. Clocks per byte are unchanged: **2.03 measured**
for a transposed spill at `tcols = 128`, against 2.00 for a dense one.

It survived the move to ranges only because the controller takes a `stride`. A transposed
spill is one range per source row, `tcols` bytes at `tdrow` apart; without a stride it would
be one range per byte and the mode would have gained nothing.

**Source and destination must be distinct regions** — a transpose is not safe in place, and
nothing checks that. On-chip, a `.t` spill plus a plain fill turns a scratchpad tensor into
its transpose via DRAM in two dispatches; `tpulib.h` wraps that as `tpu_transpose_int8` and
`tpu_transpose_dram_int8`.

What it replaced: a byte-move loop with `len = 1`, **two dispatches per byte**. One `.t`
transfer is one dispatch.

## 5. Facts to respect

- **Single clock domain.** The SRAM is asynchronous but its controller runs on the same
  `clk` as the DMA and scratchpad — no CDC, no async FIFOs.
- **Arbitration is real.** The engine holds when denied the scratchpad; fill bytes wait in a
  four-entry skid buffer and, if that fills, `sram_dout_ready` stops the DRAM read stream
  rather than dropping a byte.
- **Byte granularity handles any alignment or length.** Every byte lands at its own absolute
  address on both sides, so unaligned bases and odd lengths just work.
- **A range is contiguous; a transposed transfer is not.** Ranges are issued one per *source
  row*. In linear mode `tcols` defaults to `dma_len`, so there is exactly one range.
- **19-bit DRAM addressing.** This used to be 16, which silently confined every program to
  the low 64 KB of a 512 KB part. **When something addresses DRAM, check it is not using
  `ADDR_W`** — the testbenches' DRAM byte maps were sized off it too, so high expectations
  were dropped by `$readmemh` without a word.

## 6. Throughput: ranges, not line buffers

The original plan was to buffer a 64-byte scratchpad line and drain it as 64 back-to-back
byte transactions. What was done inverts that: the **controller** learned to take a whole
range, and the DMA became a stream client. Same goal, less machinery, none of the alignment
bookkeeping.

The line buffer looked necessary because of a mis-attribution. The scratchpad was never the
problem — its port already takes one byte per clock at any alignment. The cost was
`sram_controller` sequencing IDLE/ACCESS/WAIT per **byte**, plus four DMA states around each:
8 clocks per byte on a fill and 9 on a spill, to move a byte the chip answers in 10 ns.

| | before | after |
| --- | --- | --- |
| fill (DRAM -> scratchpad) | ~8 clocks/byte | **1** |
| spill (scratchpad -> DRAM) | ~8 clocks/byte | **2** |
| transposed spill | ~8 clocks/byte | **2.03** measured at `tcols=128` |

End to end on the whole adder model, same program, byte-identical results:
**1 226 722 -> 541 590 clocks (2.27x)**. DMA fell from 66% of the run to 24%.

### Two things to know before touching `sram.sv`

- **Writes are two clocks because WE# is generated on the falling edge**, so its rising edge
  lands half a clock clear of the address/data change. The chip's address hold is 0 ns —
  met by skew alone if you drive it from the rising edge — and the failure mode is a byte
  written to its **neighbour**. Both `sram_tb` and `dma_tb` check for it.
- **`CLOCKS_PER_ACCESS` is now *extra* clocks per beat** and is 0 on every board.

### Still open

- The scratchpad's 64-byte port width is unused. A line-buffered fill would cut its access
  count 64x, but that access is free today.
- **Overlap** is the real next step, and most of it is now firmware's:
  `tpulib.h`'s weight prefetch double-buffers a staged block so a fill streams under a
  matmul. Doing it in hardware is [scheduler_plan.md](scheduler_plan.md).
