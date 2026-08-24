/* mha.c — one head of ReLU attention, in firmware.
 *
 *   Q = requant(X @ Wq)    K = requant(X @ Wk)    V = requant(X @ Wv)
 *   S = requant(Q @ K^T)
 *   P = requant(relu(S))                     identity: P shares S's scale
 *   A = requant(P @ V)
 *
 * The kernel that exercises everything at once: both units, all three DMA modes
 * (fill, spill, TRANSPOSING spill), and the `quant4` narrow that turns an
 * activation into a weight operand. No causal mask — this is a datapath and ISA
 * test, not the model.
 *
 * WHY K IS TRANSPOSED AND V IS NOT. The array reads weights row-major, so for
 * out = act @ wgt the operand wgt[k][n] has row k contiguous over n.
 *
 *   Q @ K^T contracts over the head dim, so its weight is K^T[h][s] — K
 *           COLUMN-major, while K comes out of its projection row-major. Hence
 *           the transpose.
 *   P @ V   contracts over keys, so its weight is V[s][h] — exactly how V left
 *           its projection. Free.
 *
 * The transpose runs on int8 because the DMA moves whole bytes and a packed
 * int4 nibble is half of one, which is why there is a KT_INT8 buffer between
 * the transpose and the pack.
 */
#include "tpu.h"

#define ARRAY_ROWS 8            /* MXU geometry — fixed by the bitstream */
#define ARRAY_COLS 8

#define T        8              /* tokens (= keys) */
#define D        8              /* model width */
#define HEAD_DIM 8

/* {m0, n}: m0 in the low 12 bits, n above. The shifts are TUNED, not guessed —
 * the projection accumulators only reach ~56 here, and the first draft used
 * n=5/6 which drove every score to zero. A test whose golden answer is all
 * zeros passes against any datapath, so these were chosen to spread the result
 * across the int4 grid: 12 / 7 / 12 distinct values in Q / S / A. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_QKV RQ(1u, 3u)       /* X @ Wq|Wk|Wv -> Q/K/V */
#define RQ_PACK RQ(1u, 0u)      /* both packs: the values are already int4 */
#define RQ_S   RQ(1u, 4u)       /* Q @ K^T -> S */
#define RQ_P   RQ(1u, 0u)       /* relu(S) -> P : identity, P shares S's scale */
#define RQ_A   RQ(1u, 3u)       /* P @ V   -> A */

/* Same address in DRAM and scratchpad. */
#define X_ADDR       0x0000u    /* [T][D]        int8 */
#define WQ_ADDR      0x0400u    /* [D][HEAD_DIM] int4, row-major */
#define WK_ADDR      0x0500u
#define WV_ADDR      0x0600u
#define Q_ADDR       0x1000u    /* [T][HEAD_DIM] int8 */
#define K_ADDR       0x1100u
#define V_ADDR       0x1200u
#define KT_INT8_ADDR 0x1300u    /* [HEAD_DIM][T] int8 — K transposed        */
#define KT_INT4_ADDR 0x1400u    /* [HEAD_DIM][T] int4 — the Q@K^T weight    */
#define V_INT4_ADDR  0x1500u    /* [T][HEAD_DIM] int4 — the P@V weight      */
#define S_ADDR       0x1600u    /* [T][T]  int8  */
#define P_WIDE_ADDR  0x1800u    /* [T][T]  int32 — relu output */
#define P_ADDR       0x1C00u    /* [T][T]  int8  */
#define A_ADDR       0x2000u    /* [T][HEAD_DIM] int8 — the result */

/* Row strides, bytes: an int4 weight row is two nibbles per byte. */
#define WGT_ROW_PROJ (HEAD_DIM / 2)   /* a projection weight row */
#define WGT_ROW_KT   (T / 2)          /* a row of K^T            */
#define X_BYTES (T * D)
#define W_BYTES (D * WGT_ROW_PROJ)
#define A_BYTES (T * HEAD_DIM)

int main(void)
{
    /* operands in */
    tpu_dma(X_ADDR,  X_ADDR,  X_BYTES, TPU_DMA_FILL);
    tpu_dma(WQ_ADDR, WQ_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_dma(WK_ADDR, WK_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_dma(WV_ADDR, WV_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* ---- projections: all three share one geometry ---- */
    tpu_mxu_geom(D, HEAD_DIM * 4, WGT_ROW_PROJ, 1u, 1u, T);
    tpu_mxu_mm(Q_ADDR, X_ADDR, WQ_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_QKV);
    tpu_mxu_mm(K_ADDR, X_ADDR, WK_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_QKV);
    tpu_mxu_mm(V_ADDR, X_ADDR, WV_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_QKV);
    tpu_wait(TPU_U_MXU);

    /* ---- K -> K^T, through DRAM, as bytes ----
     * dst[col*dst_row + row] = src[row*src_row + col], so a [T][HEAD_DIM] block
     * wants src_cols = HEAD_DIM, src_row = HEAD_DIM, dst_row = T. */
    tpu_dma_transpose(K_ADDR, KT_INT8_ADDR, T * HEAD_DIM, TPU_DMA_SPILL,
                      HEAD_DIM, HEAD_DIM, T);
    tpu_wait(TPU_U_DMA);
    tpu_dma(KT_INT8_ADDR, KT_INT8_ADDR, T * HEAD_DIM, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* ---- pack both weight operands ----
     * quant4 writes 4 bits per element, so each destination advances half as
     * fast as its source and the count must be even (docs/vpu.md). */
    tpu_vpu(TPU_V_QUANT4, KT_INT4_ADDR, KT_INT8_ADDR, 0u, HEAD_DIM * T, RQ_PACK);
    tpu_vpu(TPU_V_QUANT4, V_INT4_ADDR,  V_ADDR,       0u, T * HEAD_DIM, RQ_PACK);
    tpu_wait(TPU_U_VPU);

    /* ---- S = requant(Q @ K^T) ---- */
    tpu_mxu_geom(HEAD_DIM, T * 4, WGT_ROW_KT, 1u, 1u, T);
    tpu_mxu_mm(S_ADDR, Q_ADDR, KT_INT4_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_S);
    tpu_wait(TPU_U_MXU);

    /* ---- P = requant(relu(S)) ---- */
    tpu_vpu(TPU_V_RELU,    P_WIDE_ADDR, S_ADDR,      0u, T * T, 0u);
    tpu_vpu(TPU_V_REQUANT, P_ADDR,      P_WIDE_ADDR, 0u, T * T, RQ_P);
    tpu_wait(TPU_U_VPU);

    /* ---- A = requant(P @ V) ---- */
    tpu_mxu_geom(T, HEAD_DIM * 4, WGT_ROW_PROJ, 1u, 1u, T);
    tpu_mxu_mm(A_ADDR, P_ADDR, V_INT4_ADDR, TPU_MM_TILED | TPU_MM_RQ, RQ_A);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(A_ADDR, A_ADDR, A_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
