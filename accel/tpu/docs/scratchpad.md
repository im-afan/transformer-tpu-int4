# Scratchpad

## Overview
- On-chip working memory. Every unit names scratchpad byte addresses; DRAM is only ever
  reached through the DMA.
- `NBANK` independent BRAMs, each `N*4` bits wide, where `N` is the systolic array size.
- Banks are **contiguous address regions**, not interleaved bytes: bank 0 holds the first
  `BANK_WORDS` words, bank 1 the next, and so on. Two requesters naming different banks
  both go through on the same clock, which is what gives the machine as many ports as it
  needs without replicating storage.
- One access is one bank word. That is the whole datapath width now: A, B, C and the VPU
  are all `N*4` bits.

## Ports
- Basic: `clk`, `rst_n`
- `A_re`/`A_addr`/`A_rdata`/`A_gnt` — MXU A operand, read
- `B_re`/`B_addr`/`B_rdata`/`B_gnt` — MXU B operand, read
- `C_en`/`C_we`/`C_addr`/`C_wdata`/`C_rdata`/`C_gnt` — MXU result, one address, read or write
- `V_re`/`V_raddr`/`V_rdata`/`V_rgnt` and `V_we`/`V_waddr`/`V_wdata`/`V_wstrb`/`V_wgnt` — VPU
- `dma_re`/`dma_raddr`/`dma_rdata`/`dma_rgnt` and `dma_we`/`dma_waddr`/`dma_wdata`/`dma_wgnt`
  — DMA, **one byte** per access at any byte address
- `s_re`/`s_we`/`s_addr`/`s_wdata`/`s_rdata`/`s_rgnt`/`s_wgnt` — the CPU's window, one
  `S_BYTES` word

## Geometry

| parameter | meaning | Cmod A7 |
| --- | --- | --- |
| `N` | array size; a word is `N*4` bits = `N/2` bytes | 8 -> 32-bit words |
| `ADDR_W` | byte address width | 16 -> 64 KB |
| `BANK_WORDS` | words per bank | 1024 (a RAMB36 at 32 bits wide) |
| `NBANK` | `2**ADDR_W / (BANK_WORDS * N/2)` | 16 banks of 4 KB |
| `S_BYTES` | CPU word | 4 |

Address split: `addr[WOFF_W-1:0]` picks the byte in a word, the next `log2(BANK_WORDS)`
bits the row inside a bank, the top bits the bank.

## Arbitration
- Each bank is one write port and one read port, so a read and a write to the same bank
  both proceed. Two reads, or two writes, do not.
- Priority within a bank, highest first: **MXU (A > B > C), VPU, DMA, CPU.**
- Every requester gets a grant back, and a denied requester must treat it as a stall. The
  MXU freezes its whole array pipeline; the VPU freezes its FSM; the DMA holds its stream;
  the CPU port re-presents.
- **A and B in one bank is a software error, not something the arbiter can fix.** They are
  read on the same clock for the whole of a matmul, so B would be denied on every one.
  A simulation-only warning fires the first time it happens.

## Timing
- **Synchronous read**: address and enable on one clock, `*_rdata` valid the next. The
  bank output register is the port register — the bank-select mux is combinational after
  it, so the latency stays one clock.
- Read data is only guaranteed for that one cycle: the bank's output register is
  overwritten by whatever reads the same bank next. Every consumer already samples it in
  the following cycle.
- Read-during-write on the same address returns the old value (read-first).
- Writes take one clock under a per-byte strobe.
- `rst_n` does not clear the storage or the read data — a BRAM output register has no
  reset, and asking for one costs either a cycle of latency or the block RAM itself. It
  only clears the registered bank selects.

## Alignment
- A, B, C and the VPU address whole words: their bases and strides must be multiples of
  `N/2` bytes. The low address bits are ignored, not honoured.
- The DMA is byte-granular at any address (one-hot strobe into the word's lanes).
- The CPU port is `S_BYTES`-aligned, and `S_BYTES` must not exceed a word.

## Storage
- `MEM_STYLE = "BRAM"` gives `ram_style="block"`; `"REG"` gives flip-flops, which is only
  sensible at the depths unit testbenches use.
- Each bank is the canonical byte-write-enable simple-dual-port template: one clocked
  process, per-lane write, unconditional registered read.
- `INIT_FILE` preloads from a `$readmemh` image of one byte per line; each bank keeps its
  own contiguous slice. It elaborates away entirely when the string is empty.
- `bd_peek` / `bd_poke` are the simulation backdoor for a single byte at an absolute
  address — testbenches cannot reach into a bank by a runtime index.

## Notable changes
- Byte-lane banking and the two barrel-rotate networks are gone, and unaligned access with
  them. That is what made the storage a clean BRAM inference and what lets a bank serve one
  requester per clock without a 64Ki-to-1 byte mux behind every port.
- Ports no longer have their own widths (`A_BYTES`, `W_BYTES`, `C_BYTES`, `V_BYTES`,
  `DMA_BYTES`). Everything is one word, except the DMA's byte and the CPU's word.
- The VPU's port narrowed from 32 bytes to one word, so it processes far fewer lanes per
  access than it used to. It has not otherwise been touched.
- `W_rd` is `B` now: the second operand is an activation as often as it is a weight.
