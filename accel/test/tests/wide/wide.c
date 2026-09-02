/* wide.c — one tpu_matmul_wide, DRAM to DRAM. The default shape is what the
 * double buffer was written against: more rows than one panel of the arena, an
 * odd number of column blocks so the parity ends odd, and a last block that is
 * not TPU_N wide. See accel/tpu/docs/fw.md. */
#include "tpulib.h"

/* Shape, the DRAM map, the requant word and the arena all come from
 * generate.py as -D. The defaults below are only for a bare `make`; the
 * numbers that ran are the generator's, which is also what computed the
 * golden. */
#ifndef M
#define M 20                    /* more rows than one panel of this arena */
#endif
#ifndef K
#define K 512                   /* the contraction, a multiple of TPU_N */
#endif
#ifndef N
#define N 52                    /* 7 column blocks, the last one ragged */
#endif
#ifndef TRANSPOSE
#define TRANSPOSE 0             /* B is stored [N][K] rather than [K][N] */
#endif
#ifndef ACC
#define ACC 0                   /* C = clip4(requant(A @ B) + C_old) */
#endif

#ifndef RQ_C
#define RQ_C ((uint32_t)(7u << 12) | 1u)
#endif

#define I4(cols) ((cols) / 2)

#ifndef DR_A
#define DR_A 0x00000u           /* [M][K] */
#endif
#ifndef DR_B
#define DR_B 0x03000u           /* [K][N], or [N][K] under TRANSPOSE */
#endif
#ifndef DR_C
#define DR_C 0x07000u           /* [M][N] */
#endif

/* Four banks: one each for A and C and two for B, which is what the prefetch
 * costs. Three is the floor and single-buffers on its own. */
#ifndef SP_ARENA
#define SP_ARENA  0x0000u
#endif
#ifndef ARENA_BANKS
#define ARENA_BANKS 4u
#endif

int main(void)
{
    tpu_arena arena;

    const tpu_gemm mm = {
        .rows = M, .depth = K, .cols = N,
        .a = TPU_ROWS(DR_A, I4(K)),
        .b = TPU_ROWS(DR_B, TRANSPOSE ? I4(K) : I4(N)),
        .c = TPU_ROWS(DR_C, I4(N)),
        .transpose = TRANSPOSE,
        .accumulate = ACC,
        .rq_word = RQ_C,
    };

    tpu_arena_init(&arena, SP_ARENA, ARENA_BANKS * TPU_BANK_BYTES);
    tpu_matmul_wide(&mm, &arena);

    return 0;                   /* start.S raises `done` from here */
}
