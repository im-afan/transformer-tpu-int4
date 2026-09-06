# DMA engine

## Overview
- Moves `rows` rows of `len` int4 elements between DRAM (the board's async SRAM) and the
  scratchpad, with an independent row stride on each side.
- Drives the Cmod A7 SRAM ports 
- 1 byte / clock on a fill (DRAM -> spad), 1 byte / 2 clock on a spill (spad -> DRAM).

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
- Row r is at `dram_base + r*dram_stride` and `spad_base + r*spad_stride`.
- A zero stride means densely packed rows (`= row bytes`).

## Fill (DRAM -> scratchpad)
- The address is registered onto the pins; the byte is written into the scratchpad on the
  next clock edge, giving the async part a full clock of access time.
- If the scratchpad denies the write, the DMA holds the address; it is written when the grant arrives.

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
- The DMA is the only driver of the SRAM pins, so the host's `R`/`W` byte accesses go
  through it.
- Serviced only while no command is running. `uart_interface` returns NAK if commands arrive while DMA is busy.
- A host read is one clock; a host write is the same two-clock beat as a spill.

## Command encoding

`DMA_MOVE` (`0x01`):

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