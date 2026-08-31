/* ffn.c — the transformer's feed-forward block, in firmware. Drives both units
 * at once: the relu reads what the matmul wrote, so the MXU is waited for
 * before the VPU is pushed. See docs/fw.md. */
#include "tpu.h"

/* Shape, the address map and the three requant words all come from
 * generate.py as -D. The defaults below are only for a bare `make`; to run a
 * different shape use the flags, which move the golden with it:
 *   python accel/test/tests/ffn/generate.py -b rtl -T 16 -d 16 -f 32 */
#ifndef T
#define T   8                   /* tokens */
#endif
#ifndef D
#define D   8                   /* model width   (one tile)  */
#endif
#ifndef DFF
#define DFF 16                  /* hidden width  (two tiles) */
#endif

#define X_ROW  (D / 2)
#define W1_ROW (DFF / 2)
#define W2_ROW (D / 2)
#define H_ROW  (DFF / 2)
#define Y_ROW  (D / 2)

/* {m0, n} literals: m0 in the low 12 bits, n above. generate.py fits them to
 * the accumulators the golden actually produced, so they follow the shape. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#ifndef RQ_H
#define RQ_H      RQ(1u, 3u)    /* X @ W1  -> H */
#endif
#ifndef RQ_H_RELU
#define RQ_H_RELU RQ(1u, 0u)    /* relu(H): identity, it shares H's scale */
#endif
#ifndef RQ_Y
#define RQ_Y      RQ(1u, 3u)    /* H @ W2  -> Y */
#endif

/* Same address in DRAM and scratchpad, one bank apart so no matmul's A, B and
 * C share one. */
#ifndef X_ADDR
#define X_ADDR  0x0000u
#endif
#ifndef W1_ADDR
#define W1_ADDR 0x1000u
#endif
#ifndef W2_ADDR
#define W2_ADDR 0x2000u
#endif
#ifndef H_ADDR
#define H_ADDR  0x3000u
#endif
#ifndef Y_ADDR
#define Y_ADDR  0x4000u
#endif

int main(void)
{
    tpu_dma(X_ADDR,  X_ADDR,  D,   T,   0u, 0u, TPU_DMA_FILL);
    tpu_dma(W1_ADDR, W1_ADDR, DFF, D,   0u, 0u, TPU_DMA_FILL);
    tpu_dma(W2_ADDR, W2_ADDR, D,   DFF, 0u, 0u, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* H = requant(X @ W1). The array's output block is TPU_N x TPU_N whatever
     * the shape, so both extents are walked; the contraction is not, a dispatch
     * takes all of it. */
    tpu_mxu_geom(X_ROW, W1_ROW, H_ROW, D);
    for (unsigned i = 0; i < T; i += TPU_N)
        for (unsigned j = 0; j < DFF; j += TPU_N)
            tpu_mxu_mm(H_ADDR + i * H_ROW + j / 2, X_ADDR + i * X_ROW,
                       W1_ADDR + j / 2, 0u, RQ_H);
    tpu_wait(TPU_U_MXU);        /* the VPU queue is not ordered against the MXU's */

    /* The VPU's narrow is fused into the op, so relu and its requant are one
     * command and H is rewritten in place. `vlen` is 10 bits, so a pass longer
     * than TPU_VCHUNK_MAX is several commands — a raw-command kernel has to
     * chunk exactly as tpulib.h's primitives do, and a vlen past the field
     * truncates silently. */
    for (unsigned off = 0; off < T * DFF; off += TPU_VCHUNK_MAX) {
        unsigned len = T * DFF - off;

        if (len > TPU_VCHUNK_MAX)
            len = TPU_VCHUNK_MAX;
        tpu_vpu(TPU_V_RELU, H_ADDR + off / 2, H_ADDR + off / 2, 0u, len,
                RQ_H_RELU);
    }
    tpu_wait(TPU_U_VPU);

    /* Y = requant(H @ W2). The contraction is the wide axis here. */
    tpu_mxu_geom(H_ROW, W2_ROW, Y_ROW, DFF);
    for (unsigned i = 0; i < T; i += TPU_N)
        for (unsigned j = 0; j < D; j += TPU_N)
            tpu_mxu_mm(Y_ADDR + i * Y_ROW + j / 2, H_ADDR + i * H_ROW,
                       W2_ADDR + j / 2, 0u, RQ_Y);
    tpu_wait(TPU_U_MXU);

    tpu_dma(Y_ADDR, Y_ADDR, D, T, 0u, 0u, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
