# Firmware rewrite for new architecture

- Goal: rewrite the firmware to match the new mxu/dma/spad architecture, while keeping the code simple and clutter-free

## tpu.h
- rewrite all the new command dispatches to match the new hardware; functionality should stay the same

## tpulib.h

- same overall idea. a library of primitives: tiled matmul, elementwise ops, dma.
- for cleanliness, we avoid manually allocating scratchpad memory in code. 
- functions:
    - matmul(a_addr, b_addr, c_addr, transpose, accumulate, len, requant word)
        - for now, do not do any prefetch optimizations, just do blocked DMA -> mxu
    - elementwise ops(in_addr, out_addr, len) 

## infer.c
- rewrite using revised tpulib.h primitives; should be much simpler than before.