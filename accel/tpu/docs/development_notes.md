# 8/21: decided to keep hardware tiling due to cpu overhead
- considering switching to int4 for both weight & activation (W4A4) for simplicity of shapes (I don't like having to either use vpu or requant to ternary for attention)
    - also switch to both A and W matrices row-major for simplicity, transposing W in scratchpad is unneccessary for a square mxu (row by row vs col by col)
    - mxu shouldn't take that much more space? it seemed like most LUT it came from resbuf and not the compute
    - enough memory - at d=64, f=256, L=4, T=32, B=8, weights are 98 KB and KV cache is 131 KB
    - need retrain & modify transformer.py
    - challenge: SRAM read is 8 bit, how to do transposed DMA without throwing away bandwidth?
- todo: look closer at how dma works for einsum
- todo: overlapping compute & dma

# 8/22: 
- successfully switched to W4A4 and picorv32 in sim, but still need to hardware test
- made library of firmware primitives; still need to go over and read the code
- optimizations:
- prefetch to overlap DMA with compute
    - synthesize spad as true dual port bram
    - protect against conflicts in software
- need to test full KV cached inference

# 8/25:
- wide model (d=128, f=512, T=64/128) is the live shape; `adder_int4_wide` in transformer.py
- digits reversed in numbers_data — carries go the direction the model can see
- infer.c rewritten: every tensor in DRAM, scratchpad is an arena + mailbox, BATCH/BLOCK/PHASE
- weight prefetch landed in tpulib.h (contraction split only, opt-in per call site)
- still to do:
    - adder.c not migrated to the wide shape; fw_vectors/adder_export already moved
    - infer.c is building at LAYERS=2, put it back to 4
    - no trained wide checkpoint — make_dummy_checkpoint is plumbing only
    - infer.hex is 15772 of 16384 bytes; watch the firmware RAM
    - prefill measures 1.77M clocks against a ~800k roofline
