# Scratchpad (rewrite)

## Ports
- same as before

## Design
- consists of multiple banks of BRAM synthesized as RAMB36, each with read/write width N*4 bits, where N is the systolic array size. 
- instead of bank 1 = addr 1, bank 2 = addr 2, etc, it is ordered as bank 1 = addr 1.....n, bank 2 = n+1....2n, etc. This allows for basically as many ports as we want, given that 2 ports don't address to the same bank
- arbitration within a bank: MXU highest priority, then VPU, then DMA, then UART