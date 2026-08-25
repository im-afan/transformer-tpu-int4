# Plan: overlap DMA and compute with out-of-order dispatch in hardware

> **Not built.** Overlap today is firmware's: `fw/tpulib.h`'s weight prefetch
> double-buffers a staged weight block. This is the sketch for doing it generally.

- First, we synthesize the scratchpad memory as true dual port BRAM instead of simple dual port to not have to deal with arbitration when both the MXU/VPU and DMA are running. 
- The indidivdual command queues for each module is replaced with a global command queue and a scheduler. 
- The scheduler is an FSM with 4 states: IDLE, FIND, SWAP, DISPATCH.
    - IDLE: switches to FIND when it dma_busy or (mxu_busy || vpu_busy) switches low
    - FIND: iterates through the command queue to find the earliest instruction such that the start and end scratchpad addresses of the instruction do not overlap with the current running instructions or any instruction before it in the queue. If no such instruction is found, it goes back to IDLE
    - SWAP: moves the instruction to the top of the command queue by iterating towards the top and swapping it with the previous one
    - DISPATCH: pops from the command queue, dispatches to DMA/VPU/MXU
- Concerns: 
    - how to implement the finding. There are 2 ways
        - store the start & end addresses of each instruction it's already iterated through, and use combinational logic to detect overlap with previous instructions. Allows for O(n) clocks for scheduling
        - have a nested loop; iterate through every previous instruction to check for overlap. O(n^2) scheduling overhead
    - scheduling overhead: I believe it shouldn't be too much because most of the time an instruction will be found relatively early in the queue? 
    - how much LUT/FF will this add to the design?