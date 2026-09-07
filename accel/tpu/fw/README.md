# Firmware library (PicoRV32 command producer)

The firmware the PicoRV32 runs; produces 128-bit commands that are pushed through AXI-based MMIO to the MXU/VPU/DMA command queues.

This folder contains the library; kernels are defined in `../../test/tests/`.

## `tpu.h`
Basic functions that dispatch to MMIO; each function can be mapped to a single MXU/VPU/DMA instruction. Also includes tpu_wait, which blocks the program until the specified module is finished with all the commands in its command queue.
### functions
- tpu_push(unit, word0, word1, word2, word3): pushes words to command queue (128 bits)
- tpu_wait(unit): polls # of finished commands to unit vs. number issued, until issued - finished = 0.
- tpu_spad_ld(addr) / tpu_spad_st(addr, value): read/write value from TPU scratchpad from PicoRV32, using AXI address offset
- tpu_mxu_geom(a_stride, b_stride, c_stride, len): defines matmul scratchpad geometry; 128-bit instruction is not enough for everything, so we set the matmul geometry beforehand and reuse for minimal overhead
- tpu_mxu_mm(c_addr, a_addr, b_addr, flags): C = A @ B (in scratchpad). Flags: toggle B transpose, toggle accumulate at C
- tpu_vpu: one vpu command

## `tpulib.h`
- NN primitives: tiled matmul, elementwise ops (addition, relu, dyt), and DRAM copy.
- Primitives calls can only read/write to DRAM; scratchpad is controlled entirely inside each primitive; scratchpad is fully de-allocated after a single primitive run
### programming model
- tpu_buf: a tensor in DRAM
- tpu_arena: a memory "arena", denoting how much total scratchpad memory a primitive can use. 
- tpu_gemm: config struct for gemm; (rows x depth) * (depth x cols); a, b, c buffers; transpose/acc flag, requant word {M0, N} in a single int32
- tpu_gemm_fused: config struct for the fused matmul + add + activation; same as tpu_gemm plus the `add` tensor, an `add_op` (TPU_V_ADD or TPU_ACT_NONE), an `activation` (TPU_V_RELU / TPU_V_DYT / TPU_V_REQUANT / TPU_ACT_NONE) and a requant word per step (rq_word for the matmul, rq_add for the add, rq_act for the activation). No accumulate flag.
- tpu_gemm_layout: scratchpad layout for gemm, auto-fitted based on matrix sizes for optimal (minimum) memory transfers in a single matmul. 
    - a, b, c addr, b_half when b is double-buffered
### functions
- tpu_gemm_fit: fits a tpu_gemm to a tpu_gemm_layout
- tpu_matmul: fully general matmul, including when tiling across the k (contraction) dimension is needed. Does not use layout autofitting.
- tpu_matmul_wide: matmul using layout autofitting. 
    - double-buffers inner loop to overlap mxu with dma.
- tpu_matmul_wide_fused: C = act(A @ B + add), same flow and same autofitting as tpu_matmul_wide.
    - stages the matching block of `add` where the accumulate path would have staged C, so a panel row costs two output blocks of the C region instead of one.
    - each output block is matmulled, added to and activated in place in the scratchpad, then spilled once; the intermediate never goes back to DRAM.
    - `add` is the second operand of both steps: the add pass uses it, and a TPU_V_DYT activation reads it again, which is what makes the double residual one call.
    - every step requants, same as the unfused sequence in infer.c.
    - see ../docs/fused.md.
- tpu_vpu_tile: one VPU pass over a tile already in the scratchpad, chunked at the vlen field's limit; does not fence.
- tpu_elementwise: elementwise ops, loads chunks into scratchpad arena, does ops, loads back to DMA, repeat
- tpu_copy: copy from one address in DRAM to another by copying to scratchpad and copying back


## other files
- `memops.c`: `memcpy`/`memset`, which gcc emits calls to whatever the flags say. `--gc-sections` drops it from kernels that make none. 
- `mock/tpu_trace.c'`: The host-side `tpu_push`/`tpu_wait`/`tpu_spad_ld`/`tpu_spad_st`, so `-DTPU_TRACE` turns any kernel into its own trace producer
- `start.S`, `link.ld`: Reset entry (`gp`/`sp`, zero `.bss`, `main`, raise `done`) and the 16 KB firmware RAM at address 0
- `bin2hex.py`: `.bin` -> one 32-bit word per line, for `'I'` and for `$readmemh`
- `Makefile`: Builds any `.c` from anywhere: `PROG=`, `SRC=`, `BUILD=`, `EXTRA_CFLAGS=` |

