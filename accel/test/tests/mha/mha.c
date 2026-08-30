/* mha.c — one head of ReLU attention, in firmware. Exercises both units, both
 * DMA directions, and the MXU's transpose flag. No causal mask — a datapath
 * and ISA test, not the model. See docs/fw.md for why K needs `transpose` and
 * V does not. */
#include "tpu.h"

#define T        8              /* tokens (= keys) */
#define D        8              /* model width */
#define HEAD_DIM 8

/* {m0, n}: m0 in the low 12 bits, n above. Tuned against the measured
 * accumulator range (~56), not guessed — see docs/fw.md. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_QKV RQ(1u, 3u)       /* X @ Wq|Wk|Wv -> Q/K/V */
#define RQ_S   RQ(1u, 4u)       /* Q @ K^T -> S */
#define RQ_P   RQ(1u, 0u)       /* relu(S) -> P : identity, P shares S's scale */
#define RQ_A   RQ(1u, 3u)       /* P @ V   -> A */

/* Same address in DRAM and scratchpad, one bank apart so no matmul's A, B and
 * C share one. */
#define X_ADDR  0x0000u         /* [T][D]        int4 */
#define WQ_ADDR 0x1000u         /* [D][HEAD_DIM] int4 */
#define WK_ADDR 0x1400u
#define WV_ADDR 0x1800u
#define Q_ADDR  0x2000u         /* [T][HEAD_DIM] int4 */
#define K_ADDR  0x3000u
#define V_ADDR  0x4000u
#define S_ADDR  0x5000u         /* [T][T]        int4 */
#define A_ADDR  0x6000u         /* [T][HEAD_DIM] int4 — the result */

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

    /* ---- projections: all three share one geometry ---- */
    tpu_mxu_geom(X_ROW, PROJ_ROW, PROJ_ROW, D);
    tpu_mxu_mm(Q_ADDR, X_ADDR, WQ_ADDR, 0u, RQ_QKV);
    tpu_mxu_mm(K_ADDR, X_ADDR, WK_ADDR, 0u, RQ_QKV);
    tpu_mxu_mm(V_ADDR, X_ADDR, WV_ADDR, 0u, RQ_QKV);
    tpu_wait(TPU_U_MXU);

    /* ---- S = requant(Q @ K^T). K is [T][HEAD_DIM], so its stride is a row of
     * the CONTRACTION under the transpose, not a row of the output. ---- */
    tpu_mxu_geom(PROJ_ROW, PROJ_ROW, S_ROW, HEAD_DIM);
    tpu_mxu_mm(S_ADDR, Q_ADDR, K_ADDR, TPU_MM_T, RQ_S);
    tpu_wait(TPU_U_MXU);        /* the VPU queue is not ordered against the MXU's */

    /* ---- P = relu(S), in place: the narrow is fused into the op ---- */
    tpu_vpu(TPU_V_RELU, S_ADDR, S_ADDR, 0u, T * T, RQ_P);
    tpu_wait(TPU_U_VPU);

    /* ---- A = requant(P @ V) ---- */
    tpu_mxu_geom(S_ROW, PROJ_ROW, PROJ_ROW, T);
    tpu_mxu_mm(A_ADDR, S_ADDR, V_ADDR, 0u, RQ_A);
    tpu_wait(TPU_U_MXU);

    tpu_dma(A_ADDR, A_ADDR, HEAD_DIM, T, 0u, 0u, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
