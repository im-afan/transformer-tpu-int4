/* tiled_simple.c — C = requant(A @ W) with every operand in DRAM, tiled by hand
 * out of raw tpu.h commands. tiled.c is the same job through tpulib.h; this is
 * the floor to measure it against. See accel/tpu/docs/fw.md. */
#include "tpu.h"

/* Shape, both address maps and the requant word all come from generate.py as
 * -D. The defaults below are only for a bare `make`; the numbers that ran are
 * the generator's, which is also what computed the golden. */
#ifndef M
#define M 8                     /* token rows; a partial last block is fine */
#endif
#ifndef K
#define K 32                    /* the contraction, a multiple of TPU_N */
#endif
#ifndef N
#define N 16                    /* output columns, a multiple of TPU_N  */
#endif
#ifndef SUPER_ROWS              /* rows of A staged at once, a multiple of TPU_N */
#define SUPER_ROWS TPU_N
#endif
#ifndef TILED_OPTIMIZED         /* which of the two kernels main() runs */
#define TILED_OPTIMIZED 0
#endif

#define A_ROW    (K / 2)        /* DRAM row strides, packed int4 */
#define W_ROW    (N / 2)
#define C_ROW    (N / 2)
#define TILE_ROW (TPU_N / 2)    /* a staged tile is TPU_N wide */

#ifndef RQ_C                    /* {m0,n}: the store is int4 */
#define RQ_C ((uint32_t)(4u << 12) | 1u)
#endif

#ifndef DR_A
#define DR_A 0x00000u           /* [M][K] */
#endif
#ifndef DR_W
#define DR_W 0x02000u           /* [K][N] */
#endif
#ifndef DR_C
#define DR_C 0x04000u           /* [M][N] */
#endif

/* One staging region each, a bank apart: the MXU reads A, B and C on the same
 * clock and a scratchpad bank serves one requester per clock. A column block of
 * W is [K][TPU_N]. SP_A and SP_C are sized by generate.py for a whole row
 * superblock -- [SUPER_ROWS][K] and [SUPER_ROWS][TPU_N] -- so the simple kernel
 * uses the first TPU_N rows of each and the optimized one uses all of them. */
#ifndef SP_A
#define SP_A 0x00000u
#endif
#ifndef SP_W
#define SP_W 0x01000u
#endif
#ifndef SP_C
#define SP_C 0x02000u
#endif

void tiled_matmul_simple() {
    tpu_mxu_geom(A_ROW, TILE_ROW, TILE_ROW, K);

    for (unsigned i = 0; i < M; i += TPU_N) {
        unsigned rows = (M - i) < TPU_N ? (M - i) : TPU_N; // max(M-i, TPU_N)

        tpu_dma(SP_A, DR_A + i * A_ROW, K, rows, A_ROW, 0u, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);    /* the MXU queue is not ordered against the DMA's */

        for (unsigned j = 0; j < N; j += TPU_N) {
            /* A column block is TPU_N of every row of W, so the fill is K rows
             * of TILE_ROW bytes rather than one range. */
            tpu_dma(SP_W, DR_W + j / 2, TPU_N, K, W_ROW, 0u, TPU_DMA_FILL);
            tpu_wait(TPU_U_DMA);

            tpu_mxu_mm(SP_C, SP_A, SP_W, 0u, RQ_C);
            tpu_wait(TPU_U_MXU);

            /* The block is always TPU_N rows wide on chip; only the rows this
             * block actually has go back to DRAM. */
            tpu_dma(SP_C, DR_C + i * C_ROW + j / 2, TPU_N, rows, C_ROW, 0u,
                    TPU_DMA_SPILL);
            tpu_wait(TPU_U_DMA);
        }
    }
}

void tiled_matmul_optimized(void) {
    tpu_mxu_geom(A_ROW, TILE_ROW, TILE_ROW, K);

    /* SUPER_ROWS rows of A at a time instead of TPU_N, so a column block of W
     * is filled once per superblock rather than once per row block: the W
     * stream shrinks by SUPER_ROWS / TPU_N. The C blocks of one superblock
     * column stack contiguously, so they leave as one strided spill. */
    for (unsigned i = 0; i < M; i += SUPER_ROWS) {
        unsigned super_rows = (M - i) < SUPER_ROWS ? (M - i) : SUPER_ROWS;

        tpu_dma(SP_A, DR_A + i * A_ROW, K, super_rows, A_ROW, 0u,
                TPU_DMA_FILL);

        for (unsigned j = 0; j < N; j += TPU_N) {
            tpu_dma(SP_W, DR_W + j / 2, TPU_N, K, W_ROW, 0u, TPU_DMA_FILL);
            tpu_wait(TPU_U_DMA);        /* also retires the previous spill */

            for (unsigned r = 0; r < super_rows; r += TPU_N)
                tpu_mxu_mm(SP_C + r * TILE_ROW, SP_A + r * A_ROW, SP_W, 0u,
                           RQ_C);
            tpu_wait(TPU_U_MXU);

            tpu_dma(SP_C, DR_C + i * C_ROW + j / 2, TPU_N, super_rows, C_ROW,
                    0u, TPU_DMA_SPILL);
        }
    }

    tpu_wait(TPU_U_DMA);
}

int main(void)
{
#if TILED_OPTIMIZED
    tiled_matmul_optimized();
#else
    tiled_matmul_simple();
#endif

    return 0;                   /* start.S raises `done` from here */
}
