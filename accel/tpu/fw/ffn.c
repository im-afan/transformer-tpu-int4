/* ffn.c — the transformer's feed-forward block, in firmware. Drives both units
 * at once: the relu reads what the matmul wrote, so the MXU is waited for
 * before the VPU is pushed. See docs/fw.md. */
#include "tpu.h"

#define T   8                   /* tokens */
#define D   8                   /* model width   (one tile)  */
#define DFF 16                  /* hidden width  (two tiles) */

#define X_ROW  (D / 2)
#define W1_ROW (DFF / 2)
#define W2_ROW (D / 2)
#define H_ROW  (DFF / 2)
#define Y_ROW  (D / 2)

/* {m0, n} literals: m0 in the low 12 bits, n above. Tuned against the measured
 * accumulator range (~56), not guessed — see docs/fw.md. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_H      RQ(1u, 3u)    /* X @ W1  -> H */
#define RQ_H_RELU RQ(1u, 0u)    /* relu(H): identity, it shares H's scale */
#define RQ_Y      RQ(1u, 3u)    /* H @ W2  -> Y */

/* Same address in DRAM and scratchpad, one bank apart so no matmul's A, B and
 * C share one. */
#define X_ADDR  0x0000u
#define W1_ADDR 0x1000u
#define W2_ADDR 0x2000u
#define H_ADDR  0x3000u
#define Y_ADDR  0x4000u

int main(void)
{
    tpu_dma(X_ADDR,  X_ADDR,  D,   T,   0u, 0u, TPU_DMA_FILL);
    tpu_dma(W1_ADDR, W1_ADDR, DFF, D,   0u, 0u, TPU_DMA_FILL);
    tpu_dma(W2_ADDR, W2_ADDR, D,   DFF, 0u, 0u, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* H = requant(X @ W1). Two column blocks: DFF is two array widths. */
    tpu_mxu_geom(X_ROW, W1_ROW, H_ROW, D);
    for (unsigned j = 0; j < DFF; j += TPU_N)
        tpu_mxu_mm(H_ADDR + j / 2, X_ADDR, W1_ADDR + j / 2, 0u, RQ_H);
    tpu_wait(TPU_U_MXU);        /* the VPU queue is not ordered against the MXU's */

    /* One pass: the VPU's narrow is fused into the op, so relu and its requant
     * are one command and H can be rewritten in place. */
    tpu_vpu(TPU_V_RELU, H_ADDR, H_ADDR, 0u, T * DFF, RQ_H_RELU);
    tpu_wait(TPU_U_VPU);

    /* Y = requant(H @ W2). One block: the contraction is the wide axis and the
     * dispatch takes all of it. */
    tpu_mxu_geom(H_ROW, W2_ROW, Y_ROW, DFF);
    tpu_mxu_mm(Y_ADDR, H_ADDR, W2_ADDR, 0u, RQ_Y);
    tpu_wait(TPU_U_MXU);

    tpu_dma(Y_ADDR, Y_ADDR, D, T, 0u, 0u, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
