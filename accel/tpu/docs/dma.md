# DMA engine

## Overview
- Moves `rows` rows of `len` int4 elements between DRAM (the board's async SRAM) and the
  scratchpad, with an independent row stride on each side.
- **It drives the SRAM chip pins itself.** `sram.sv` is not in the core any more; the
  controller's range interface and its handshake were the layer this rewrite removed.
- One byte per clock on a fill, one per two clocks on a spill.
- No transpose mode. The MXU's `transpose` flag and software cover what it did.

## Ports
- Basic: `clk`, `rst_n`
- SRAM pins: `sram_addr`, `sram_data` (inout), `sram_we`, `sram_ce`, `sram_oen`
- Dispatch (in): `dma_start`, `dma_op` (0 = DRAM -> scratchpad, 1 = scratchpad -> DRAM),
  `dma_len` (int4 elements per row), `dma_rows`, `dma_dram_stride`, `dma_spad_stride`,
  `dma_dram_base`, `dma_spad_base`
- Dispatch (out): `dma_busy`, `dma_done`
- Scratchpad byte port: `spad_re`/`spad_raddr`/`spad_rdata`/`spad_rgnt`,
  `spad_we`/`spad_waddr`/`spad_wdata`/`spad_wgnt`
- UART host byte port: `host_start`, `host_we`, `host_addr`, `host_din`,
  `host_dout`, `host_busy`, `host_done`

## Geometry
- A row is `(len + 1) / 2` bytes; `len` is elements, so it should be even.
- Row *r* is at `dram_base + r*dram_stride` and `spad_base + r*spad_stride`.
- A zero stride means densely packed rows (`= row bytes`), the same "zero is not set"
  convention the MXU strides use.
- `rows = 0` or `len = 0` is an empty transfer: it completes, moves nothing.

## Fill (DRAM -> scratchpad)
- The address is registered onto the pins; the byte is written into the scratchpad on the
  next clock edge, giving the async part a full clock of access time.
- **Backpressure is free.** If the scratchpad denies the write, the DMA holds the address:
  the SRAM keeps driving the same byte and it is written when the grant arrives. That is
  what the old skid buffer plus `dout_ready` existed to do.

## Spill (scratchpad -> DRAM)
- The scratchpad read runs one byte ahead of the write beat, so a beat starts every two
  clocks with a single holding register between them.
- **A write beat is two clocks and WE# is driven from the falling edge**, so both of its
  edges land half a clock clear of every address and data change. Driving WE# from the
  rising edge leaves the chip's address-hold requirement to output-path skew alone; the
  failure mode is a byte written to its neighbour, and `dma_tb` watches the pins across
  every pulse for it.
- A denied scratchpad read stalls the fetch, which stalls the beat. Nothing is lost.

## The UART host port
- The DMA is the only owner of the SRAM pins, so the host's `R`/`W` byte accesses go
  through it. `host_*` carries exactly the signals `uart_interface` already drove at the
  old controller, so that module is unchanged.
- Serviced only while no command is running. `uart_interface` NAKs anything arriving while
  the core is busy, so the two never contend.
- A host read is one clock; a host write is the same two-clock beat as a spill.

## Command encoding

`DMA_MOVE` (`0x01`), one 128-bit command, no sticky state at all:

| bits | field |
| --- | --- |
| `w0[7:0]` | op |
| `w0[8]` | `dma_op` |
| `w0[31:16]` | `dma_spad_base` |
| `w1[18:0]` | `dma_dram_base` |
| `w2[15:0]` | `dma_len` |
| `w2[31:16]` | `dma_rows` |
| `w3[15:0]` | `dma_dram_stride` |
| `w3[31:16]` | `dma_spad_stride` |

DRAM addressing is 19 bits — the whole 512 KB part, not the low 64 KB.

## Notable changes
- `sram.sv` left the core. It is still in the tree for `uart_memory.sv` / `uart_bram.sv`
  (the bring-up board images) and its own testbench.
- The range interface, the `dout_ready` backpressure path, the fill skid buffer and the
  one-range-per-row loop are all gone: owning the pins makes holding the beat a matter of
  not advancing a register.
- Transpose mode and its three geometry registers (`tcols`, `tsrow`, `tdrow`) are gone.
- A transfer is 2-D now (`rows` x `len` with two strides) where it used to be a flat byte
  count, so the row loop a caller used to write is one command.
