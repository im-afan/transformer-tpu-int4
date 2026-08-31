/* mha.c — one head of ReLU attention, in firmware. Exercises both units, both
 * DMA directions, and the MXU's transpose flag. No causal mask — a datapath
 * and ISA test, not the model. See docs/fw.md for why K needs `transpose` and
 * V does not. */
#include "tpu.h"

/* Shape, the address map and the four requant words all come from generate.py
 * as -D. The defaults below are only for a bare `make`. */
#ifndef T
#define T        8              /* tokens (= keys) */
#endif
#ifndef D
#define D        8              /* model width */
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 8
#endif

/* {m0, n}: m0 in the low 12 bits, n above. generate.py fits them to the
 * accumulators the golden actually produced, so they follow the shape. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#ifndef RQ_QKV
#define RQ_QKV RQ(1u, 3u)       /* X @ Wq|Wk|Wv -> Q/K/V */
#endif
#ifndef RQ_S
#define RQ_S   RQ(1u, 4u)       /* Q @ K^T -> S */
#endif
#ifndef RQ_P
#define RQ_P   RQ(1u, 0u)       /* relu(S) -> P : identity, P shares S's scale */
#endif
#ifndef RQ_A
#define RQ_A   RQ(1u, 3u)       /* P @ V   -> A */
#endif

/* Same address in DRAM and scratchpad, one bank apart so no matmul's A, B and
 * C share one. */
#ifndef X_ADDR
#define X_ADDR  0x0000u         /* [T][D]        int4 */
#endif
#ifndef WQ_ADDR
#define WQ_ADDR 0x1000u         /* [D][HEAD_DIM] int4 */
#endif
#ifndef WK_ADDR
#define WK_ADDR 0x1400u
#endif
#ifndef WV_ADDR
#define WV_ADDR 0x1800u
#endif
#ifndef Q_ADDR
#define Q_ADDR  0x2000u         /* [T][HEAD_DIM] int4 */
#endif
#ifndef K_ADDR
#define K_ADDR  0x3000u
#endif
#ifndef V_ADDR
#define V_ADDR  0x4000u
#endif
#ifndef S_ADDR
#define S_ADDR  0x5000u         /* [T][T]        int4 */
#endif
#ifndef A_ADDR
#define A_ADDR  0x6000u         /* [T][HEAD_DIM] int4 — the result */
#endif

#define X_ROW    (D / 2)
#define PROJ_ROW (HEAD_DIM / 2)
#define S_ROW    (T / 2)

int main(void)
{
    tpu_dma(X_ADDR,  X_ADDR,  D,        T, 0u, 0u, TPU_DMA_FILL);
    tpu_dma(WQ_ADDR, WQ_ADDR, HEAD_DIM, D, 0u, 0u, TPU_DMA_FILL);
    tpu_dma(WK_ADDR, WK_ADDR, HEAD_DIM, D, 0u, 0u, TPU_DMA_FILL);
    tpu_dma(WV_ADDR, WV_ADDR, HEAD_DIM, D, 0u, 0u, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* ---- projections: all three share one geometry. The array's output block
     * is TPU_N x TPU_N whatever the shape, so both extents are walked. ---- */
    tpu_mxu_geom(X_ROW, PROJ_ROW, PROJ_ROW, D);
    for (unsigned i = 0; i < T; i += TPU_N)
        for (unsigned j = 0; j < HEAD_DIM; j += TPU_N) {
            const unsigned a = i * X_ROW, c = i * PROJ_ROW + j / 2;

            tpu_mxu_mm(Q_ADDR + c, X_ADDR + a, WQ_ADDR + j / 2, 0u, RQ_QKV);
            tpu_mxu_mm(K_ADDR + c, X_ADDR + a, WK_ADDR + j / 2, 0u, RQ_QKV);
            tpu_mxu_mm(V_ADDR + c, X_ADDR + a, WV_ADDR + j / 2, 0u, RQ_QKV);
        }
    tpu_wait(TPU_U_MXU);

    /* ---- S = requant(Q @ K^T). K is [T][HEAD_DIM], so its stride is a row of
     * the CONTRACTION under the transpose, not a row of the output. ---- */
    tpu_mxu_geom(PROJ_ROW, PROJ_ROW, S_ROW, HEAD_DIM);
    for (unsigned i = 0; i < T; i += TPU_N)
        for (unsigned j = 0; j < T; j += TPU_N)
            tpu_mxu_mm(S_ADDR + i * S_ROW + j / 2, Q_ADDR + i * PROJ_ROW,
                       K_ADDR + j * PROJ_ROW, TPU_MM_T, RQ_S);
    tpu_wait(TPU_U_MXU);        /* the VPU queue is not ordered against the MXU's */

    /* ---- P = relu(S), in place: the narrow is fused into the op. `vlen` is
     * 10 bits, so a longer pass is several commands. ---- */
    for (unsigned off = 0; off < T * T; off += TPU_VCHUNK_MAX) {
        unsigned len = T * T - off;

        if (len > TPU_VCHUNK_MAX)
            len = TPU_VCHUNK_MAX;
        tpu_vpu(TPU_V_RELU, S_ADDR + off / 2, S_ADDR + off / 2, 0u, len, RQ_P);
    }
    tpu_wait(TPU_U_VPU);

    /* ---- A = requant(P @ V) ---- */
    tpu_mxu_geom(S_ROW, PROJ_ROW, PROJ_ROW, T);
    for (unsigned i = 0; i < T; i += TPU_N)
        for (unsigned j = 0; j < HEAD_DIM; j += TPU_N)
            tpu_mxu_mm(A_ADDR + i * PROJ_ROW + j / 2, S_ADDR + i * S_ROW,
                       V_ADDR + j / 2, 0u, RQ_A);
    tpu_wait(TPU_U_MXU);

    tpu_dma(A_ADDR, A_ADDR, HEAD_DIM, T, 0u, 0u, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
