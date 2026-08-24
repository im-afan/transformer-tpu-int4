/* ffn.c — the transformer's feed-forward block, in firmware.
 *
 *   H      = requant( X  @ W1 )      [T][D] @ [D][DFF] -> [T][DFF]
 *   H_RELU = requant( relu(H) )      identity requant: it shares H's scale
 *   Y      = requant( H_RELU @ W2 )  [T][DFF] @ [DFF][D] -> [T][D]
 *
 * The first kernel to issue a VPU command. matmul.c proved the CPU could drive
 * the array; this proves it can drive the vector unit too, and that the two
 * queues interleave correctly — the `relu` reads what the matmul wrote, so the
 * MXU must be waited for before the VPU is pushed.
 *
 * Shapes are one array tile wide except DFF, which is two, so the hardware tile
 * loop runs in both directions across the pair (column tiles, then depth
 * tiles).
 *
 *   X  : [T][D]     int8, row-major
 *   W1 : [D][DFF]   int4, row-major
 *   W2 : [DFF][D]   int4, row-major
 *
 * Operands and the golden result come from accel/tpulang/fw_vectors.py, which
 * runs this kernel's own command trace through iss.py.
 */
#include "tpu.h"

#define ARRAY_ROWS 8            /* MXU geometry — fixed by the bitstream */
#define ARRAY_COLS 8

#define T   8                   /* tokens */
#define D   8                   /* model width   (one tile)  */
#define DFF 16                  /* hidden width  (two tiles) */

/* Row strides, bytes. An int4 weight row is two nibbles per byte; an int32
 * output row is 4 bytes per element. */
#define ACT_ROW_X      D
#define ACT_ROW_H_RELU DFF
#define WGT_ROW_W1     (DFF / 2)
#define WGT_ROW_W2     (D / 2)
#define OUT_ROW_H      (DFF * 4)
#define OUT_ROW_Y      (D * 4)

/* {m0, n} literals: m0 in the low 12 bits, n above. The shifts keep results
 * inside the int4 grid rather than pinned at the clip. Tuned, not guessed: the
 * accumulators here only reach ~56, and a larger shift collapses the whole
 * result to zero — which would pass against any datapath at all. These give 13
 * distinct values in H and 13 in Y, spanning [-8, 7]. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_H      RQ(1u, 3u)    /* X @ W1  -> H */
#define RQ_H_RELU RQ(1u, 0u)    /* relu(H): identity, it shares H's scale */
#define RQ_Y      RQ(1u, 3u)    /* H_RELU @ W2 -> Y */

/* Same address in DRAM and scratchpad, as in every kernel here. */
#define X_ADDR          0x0000u
#define W1_ADDR         0x0400u
#define W2_ADDR         0x0800u
#define H_ADDR          0x1000u /* [T][DFF] int8  — the matmul's narrowed store */
#define H_RELU32_ADDR   0x1400u /* [T][DFF] int32 — relu output (the VPU widens)*/
#define H_RELU_ADDR     0x1C00u /* [T][DFF] int8  — requantized back down       */
#define Y_ADDR          0x2000u /* [T][D]   int8  — the result                  */

#define X_BYTES  (T * D)
#define W1_BYTES (D * WGT_ROW_W1)
#define W2_BYTES (DFF * WGT_ROW_W2)
#define Y_BYTES  (T * D)

int main(void)
{
    /* operands in */
    tpu_dma(X_ADDR,  X_ADDR,  X_BYTES,  TPU_DMA_FILL);
    tpu_dma(W1_ADDR, W1_ADDR, W1_BYTES, TPU_DMA_FILL);
    tpu_dma(W2_ADDR, W2_ADDR, W2_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* H = requant(X @ W1). Two column tiles: DFF is two array widths. */
    tpu_mxu_geom(ACT_ROW_X, OUT_ROW_H, WGT_ROW_W1, 1u, DFF / ARRAY_COLS, T);
    tpu_mxu_mm(H_ADDR, X_ADDR, W1_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_H);
    tpu_wait(TPU_U_MXU);        /* the VPU queue is not ordered against the MXU's */

    /* H_RELU = requant(relu(H)). Two ops because relu widens int8 -> int32 and
     * only an explicit requant narrows back (docs/vpu.md) — which is why
     * RQ_H_RELU is the {1,0} identity. */
    tpu_vpu(TPU_V_RELU,    H_RELU32_ADDR, H_ADDR,        0u, T * DFF, 0u);
    tpu_vpu(TPU_V_REQUANT, H_RELU_ADDR,   H_RELU32_ADDR, 0u, T * DFF, RQ_H_RELU);
    tpu_wait(TPU_U_VPU);

    /* Y = requant(H_RELU @ W2). Two depth tiles now: the contraction is the
     * wide axis. */
    tpu_mxu_geom(ACT_ROW_H_RELU, OUT_ROW_Y, WGT_ROW_W2, DFF / ARRAY_ROWS, 1u, T);
    tpu_mxu_mm(Y_ADDR, H_RELU_ADDR, W2_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_Y);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(Y_ADDR, Y_ADDR, Y_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
