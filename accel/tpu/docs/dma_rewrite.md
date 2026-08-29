# DMA engine (updated)

## Ports
- clk, rst_n
- external SRAM interface (sram.sv not used):
  - out sram_addr, inout sram_data, out sram_we, out sram_ce, out sram_oen
- dispatch interface
  - start, op (0 = external -> spad, 1 = spad -> external), len (# of int4 per row),
  - rows (# of rows), dram_stride, scratchpad_stride, dram_base_addr, scratchpad_base_addr
  - busy, done

- no transpose mode--that is handled by the mxu and software

