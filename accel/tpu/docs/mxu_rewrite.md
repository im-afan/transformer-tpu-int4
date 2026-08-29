# MXU - Matrix Unit

## Overview
- NxN output-stationary systolic array, performs int4 matmuls. 
- Notation: data arrives from right and top
- Streams N*4 bits every clock to load into right and top
- Shift registers to create skew
- Performs NxROWSxN matmul (NxROWS) * (ROWSxN)

## Ports
- Basic: 
  - clk, rst_n
- Dispatch:
  - in:
  - transpose
  - accumulate
  - start (high for 1 clock)
  - len (number of tokens)
  - A_base_addr, A_stride (start addr & row stride for matrix A)
  - B_base_addr, B_stride (start addr & row stride for matrix B)
  - C_base_addr, C_stride
  - requant ({M0, N} determining requantization)
  - out:
  - busy, done
- Scratchpad: 
  - out A_re, out A_raddr, in A_rdata
  - B_re, B_raddr, B_rdata
  - out C_en, out C_we, C_addr, out C_wdata, in C_rdata

## Input constraints
- The range of (A_base_addr, end of A), (B_base_addr, end of B), (C_base_addr, end of C) must not overlap scratchpad banks, otherwise they cannot be accessed at the same time.
- Matrices A, B, C are stored row-major in scratchpad memory.

## Output constraints
- Streams A @ B to C_base_addr, row by row.
  - if transpose, B is transposed in the matmul
  - if accumulate, C is added to the result

## Datapath
- A is loaded from the right (every column of A is fed into the leftmost column of PEs) while B is loaded from the top.
- Instead of the array A being loaded column-by-column, we load a N*4-bit chunk of A into a row of registers corresponding with that PE row.
  - starts with row 1 at clock 1, then loads row 2 at clock 2, etc. then loops back to row 1 after all rows, until len clocks.
  - still hides latency because of the skew; row i receives its chunk at clock i
- same idea for array B
- when transpose is true, B is instead loaded traditionally, with the whole row of registers receiving its data in a single clock.

## Notable changes
- there is no longer hardware-supported tiling; the output-stationary tiles in the last dimension aready, amortizing the cpu overhead
- int4 requant is forced