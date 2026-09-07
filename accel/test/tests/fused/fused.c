/* fused.c — one tpu_matmul_wide_fused, DRAM to DRAM. Same shape as wide.c —
 * more rows than one panel of the arena, an odd number of column blocks, a
 * ragged last one — with the add tensor and the activation the fused primitive
 * folds into the same visit. See accel/tpu/fw/README.md. */
#include "tpulib.h"

/* Shape, the DRAM map, the three requant words and the arena all come from
 * generate.py as -D. The defaults below are only for a bare `make`. */
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

/* TPU_V_ADD or TPU_ACT_NONE. */
#ifndef ADD_OP
#define ADD_OP TPU_V_ADD
#endif
/* TPU_V_RELU, TPU_V_DYT, TPU_V_REQUANT or TPU_ACT_NONE. DYT reads the add
 * tensor a second time, which is the double residual. */
#ifndef ACT_OP
#define ACT_OP TPU_V_DYT
#endif

#ifndef RQ_C
#define RQ_C ((uint32_t)(7u << 12) | 1u)
#endif
#ifndef RQ_ADD
#define RQ_ADD 1u
#endif
#ifndef RQ_ACT
#define RQ_ACT 1u
#endif

#define I4(cols) ((cols) / 2)

#ifndef DR_A
#define DR_A 0x00000u           /* [M][K] */
#endif
#ifndef DR_B
#define DR_B 0x03000u           /* [K][N], or [N][K] under TRANSPOSE */
#endif
#ifndef DR_ADD
#define DR_ADD 0x07000u         /* [M][N] */
#endif
#ifndef DR_C
#define DR_C 0x08000u           /* [M][N] */
#endif

/* Four banks: one each for A and the C region and two for B. The C region
 * holds the output block and the add block, so it is the same bank either
 * way. */
#ifndef SP_ARENA
#define SP_ARENA  0x0000u
#endif
#ifndef ARENA_BANKS
#define ARENA_BANKS 4u
#endif

int main(void)
{
    tpu_arena arena;

    const tpu_gemm_fused mm = {
        .rows = M, .depth = K, .cols = N,
        .a = TPU_ROWS(DR_A, I4(K)),
        .b = TPU_ROWS(DR_B, TRANSPOSE ? I4(K) : I4(N)),
        .c = TPU_ROWS(DR_C, I4(N)),
        .add = TPU_ROWS(DR_ADD, I4(N)),
        .add_op = ADD_OP,
        .activation = ACT_OP,
        .transpose = TRANSPOSE,
        .rq_word = RQ_C,
        .rq_add = RQ_ADD,
        .rq_act = RQ_ACT,
    };

    tpu_arena_init(&arena, SP_ARENA, ARENA_BANKS * TPU_BANK_BYTES);
    tpu_matmul_wide_fused(&mm, &arena);

    return 0;                   /* start.S raises `done` from here */
}
