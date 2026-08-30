/* matmul.c — C = requant(A @ W), the block grid walked in firmware. The shape
 * comes from generate.py, which builds the operands and the golden against the
 * same three numbers. See accel/tpu/docs/fw.md. */
#include "tpu.h"

/* Shape, supplied as -D by generate.py's VectorGenerator.defines. There is no
 * hardware row limit any more. */
#ifndef M
#define M      8                /* token rows */
#endif
#ifndef KTILES
#define KTILES 4                /* array tiles along the contraction  */
#endif
#ifndef NTILES
#define NTILES 2                /* array tiles across the output      */
#endif

#define K (KTILES * TPU_N)
#define N (NTILES * TPU_N)

#define A_ROW (K / 2)
#define W_ROW (N / 2)
#define C_ROW (N / 2)

#define RQ_C ((uint32_t)(4u << 12) | 1u)   /* {m0,n}: the store is int4 */

/* Same address in DRAM and in the scratchpad, and one bank apart so A, B and C
 * never contend (scratchpad.sv: a bank serves one reader per clock). */
#define A_ADDR 0x0000u
#define W_ADDR 0x2000u
#define C_ADDR 0x4000u

int main(void)
{
    tpu_dma(A_ADDR, A_ADDR, K, M, 0u, 0u, TPU_DMA_FILL);
    tpu_dma(W_ADDR, W_ADDR, N, K, 0u, 0u, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);        /* the MXU queue is not ordered against the DMA's */

    tpu_mxu_geom(A_ROW, W_ROW, C_ROW, K);
    for (unsigned i = 0; i < M; i += TPU_N)
        for (unsigned j = 0; j < N; j += TPU_N)
            tpu_mxu_mm(C_ADDR + i * C_ROW + j / 2,
                       A_ADDR + i * A_ROW,
                       W_ADDR + j / 2, 0u, RQ_C);
    tpu_wait(TPU_U_MXU);

    tpu_dma(C_ADDR, C_ADDR, N, M, 0u, 0u, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
