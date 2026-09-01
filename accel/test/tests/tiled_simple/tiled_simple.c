/* tiled_simple.c — C = requant(A @ W) with every operand in DRAM, tiled by hand
 * out of raw tpu.h commands. tiled.c is the same job through tpulib.h; this is
 * the floor to measure it against. See accel/tpu/docs/fw.md. */
#include "tpu.h"

/* Shape, the DRAM map, the scratchpad size and the requant word all come from
 * generate.py as -D. The defaults below are only for a bare `make`; the numbers
 * that ran are the generator's, which is also what computed the golden. */
#ifndef SPAD_SIZE
#define SPAD_SIZE 65536u        /* bytes of scratchpad the kernel owns */
#endif

#ifndef M
#define M 8                     /* token rows; a partial last block is fine */
#endif
#ifndef K
#define K 32                    /* the contraction, a multiple of TPU_N */
#endif
#ifndef N
#define N 16                    /* output columns, a multiple of TPU_N */
#endif
#ifndef TRANSPOSE
#define TRANSPOSE 0             /* W is stored [N][K] rather than [K][N] */
#endif
#ifndef ACC
#define ACC 0                   /* C = clip4(requant(A @ W) + C_old) */
#endif

#define A_ROW    (K / 2)        /* DRAM row strides, packed int4 */
#define W_ROW    (N / 2)
#define WT_ROW   (K / 2)        /* ...W's, when it is stored transposed */
#define C_ROW    (N / 2)
#define TILE_ROW (TPU_N / 2)    /* a staged tile is TPU_N wide */

#ifndef RQ_C                    /* {m0,n}: the store is int4 */
#define RQ_C ((uint32_t)(4u << 12) | 1u)
#endif

#ifndef DR_A
#define DR_A 0x00000u           /* [M][K] */
#endif
#ifndef DR_W
#define DR_W 0x02000u           /* [K][N], or [N][K] under TRANSPOSE */
#endif
#ifndef DR_C
#define DR_C 0x04000u           /* [M][N] */
#endif

/* One staging region each, a bank apart: the MXU reads A, B and C on the same
 * clock and a scratchpad bank serves one requester per clock. W is one column
 * block, [K][TPU_N] (or [TPU_N][K] transposed, the same bytes). A and C take
 * the rest of the scratchpad as one row superblock, so a column block of W is
 * filled once per superblock rather than once per row block. */
#define BANK_BYTES    (1024u * TPU_WORD_BYTES)
#define BANK_ROUND(x) (((x) + BANK_BYTES - 1u) / BANK_BYTES * BANK_BYTES)
#define TILE_UP(x)    (((x) + TPU_N - 1u) / TPU_N * TPU_N)
#define TILE_DOWN(x)  ((x) / TPU_N * TPU_N)

#define SP_W      0u
#define SP_W_SIZE (K * TILE_ROW)
#define SP_A      BANK_ROUND(SP_W + SP_W_SIZE)

/* A superblock row costs a row of A and a row of C; one bank of the remainder
 * pays for rounding SP_C up to a bank of its own. */
#define ROWS_FIT   TILE_DOWN((SPAD_SIZE - SP_A - BANK_BYTES) / (A_ROW + TILE_ROW))
#define ROWS_NEED  TILE_UP(M)
#define SUPER_ROWS (ROWS_FIT < ROWS_NEED ? ROWS_FIT : ROWS_NEED)

#define SP_A_SIZE (SUPER_ROWS * A_ROW)
#define SP_C      BANK_ROUND(SP_A + SP_A_SIZE)
#define SP_C_SIZE (SUPER_ROWS * TILE_ROW)

_Static_assert(SP_A + BANK_BYTES + TPU_N * (A_ROW + TILE_ROW) <= SPAD_SIZE,
               "one column block of W and one row block of A and C do not fit "
               "the scratchpad; the contraction would have to be tiled too");
_Static_assert(SP_C + SP_C_SIZE <= SPAD_SIZE, "the row superblock overruns "
                                              "the scratchpad");

void tiled_matmul_optimized(int transpose, int acc)
{
    const uint32_t flags = (transpose ? TPU_MM_T : 0u)
                         | (acc ? TPU_MM_ACC : 0u);

    tpu_mxu_geom(A_ROW, transpose ? WT_ROW : TILE_ROW, TILE_ROW, K);

    for (unsigned i = 0; i < M; i += SUPER_ROWS) {
        unsigned rows = (M - i) < SUPER_ROWS ? (M - i) : SUPER_ROWS;

        tpu_dma(SP_A, DR_A + i * A_ROW, K, rows, A_ROW, 0u, TPU_DMA_FILL);

        for (unsigned j = 0; j < N; j += TPU_N) {
            if (transpose)
                tpu_dma(SP_W, DR_W + j * WT_ROW, K, TPU_N, WT_ROW, 0u,
                        TPU_DMA_FILL);
            else
                tpu_dma(SP_W, DR_W + j / 2, TPU_N, K, W_ROW, 0u, TPU_DMA_FILL);

            if (acc)
                tpu_dma(SP_C, DR_C + i * C_ROW + j / 2, TPU_N, rows, C_ROW, 0u,
                        TPU_DMA_FILL);

            tpu_wait(TPU_U_DMA);        /* also retires the previous spill */

            for (unsigned r = 0; r < rows; r += TPU_N)
                tpu_mxu_mm(SP_C + r * TILE_ROW, SP_A + r * A_ROW, SP_W, flags,
                           RQ_C);
            tpu_wait(TPU_U_MXU);

            /* The C blocks of one superblock column stack contiguously, so
             * they leave as one strided spill. */
            tpu_dma(SP_C, DR_C + i * C_ROW + j / 2, TPU_N, rows, C_ROW, 0u,
                    TPU_DMA_SPILL);
        }
    }

    tpu_wait(TPU_U_DMA);
}

int main(void)
{
    tiled_matmul_optimized(TRANSPOSE, ACC);

    return 0;                   /* start.S raises `done` from here */
}
