/* flash.c — one tpu_flashattention, DRAM to DRAM. One head of causal ReLU
 * attention with the score matrix living in the scratchpad. ROWS queries at
 * FIRST_POS against a T-long key axis, so one image covers a prefill pass and
 * a decode step both. See accel/tpu/docs/flash.md. */
#include "tpulib.h"

/* Shape, the DRAM map, the four requant words and the arena all come from
 * generate.py as -D. The defaults below are only for a bare `make`. */
#ifndef T
#define T 32                    /* the key axis */
#endif
#ifndef ROWS
#define ROWS T                  /* queries in the call */
#endif
#ifndef FIRST_POS
#define FIRST_POS 0             /* query 0's position on the key axis */
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 32
#endif

#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#ifndef RQ_S
#define RQ_S    RQ(1u, 4u)      /* Q @ K' -> S                      */
#endif
#ifndef RQ_MASK
#define RQ_MASK RQ(1u, 0u)      /* S + mask : identity, -8 then relu is 0 */
#endif
#ifndef RQ_P
#define RQ_P    RQ(1u, 0u)      /* relu(S) -> P : P shares S's scale */
#endif
#ifndef RQ_A
#define RQ_A    RQ(1u, 3u)      /* P @ V  -> out                    */
#endif

#define I4(cols) ((cols) / 2)

#ifndef DR_Q
#define DR_Q    0x00000u        /* [ROWS][HEAD_DIM] */
#endif
#ifndef DR_K
#define DR_K    0x01000u        /* [T][HEAD_DIM] */
#endif
#ifndef DR_V
#define DR_V    0x02000u
#endif
#ifndef DR_MASK
#define DR_MASK 0x03000u        /* [T][T], 0 or -8 */
#endif
#ifndef DR_OUT
#define DR_OUT  0x05000u        /* [ROWS][HEAD_DIM] — the result */
#endif

/* Five bank-disjoint slots: Q, K, V, the output panel, and the score region
 * that holds P and the mask block. The block size follows from what is left. */
#ifndef SP_ARENA
#define SP_ARENA 0x0000u
#endif
#ifndef ARENA_BANKS
#define ARENA_BANKS 5u
#endif

int main(void)
{
    tpu_arena arena;

    const tpu_flash fa = {
        .rows = ROWS, .first_pos = FIRST_POS, .keys = T, .head_dim = HEAD_DIM,
        .q    = TPU_ROWS(DR_Q,    I4(HEAD_DIM)),
        .k    = TPU_ROWS(DR_K,    I4(HEAD_DIM)),
        .v    = TPU_ROWS(DR_V,    I4(HEAD_DIM)),
        .mask = TPU_ROWS(DR_MASK, I4(T)),
        .out  = TPU_ROWS(DR_OUT,  I4(HEAD_DIM)),
        .rq_s = RQ_S, .rq_mask = RQ_MASK, .rq_p = RQ_P, .rq_a = RQ_A,
    };

    tpu_arena_init(&arena, SP_ARENA, ARENA_BANKS * TPU_BANK_BYTES);
    tpu_flashattention(&fa, &arena);

    return 0;                   /* start.S raises `done` from here */
}
