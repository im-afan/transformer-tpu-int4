/* tiled.c — tpulib.h past the point where the scratchpad stops helping. The
 * arena is exactly three banks, the smallest a matmul can run in, so every
 * loop in tpu_matmul and tpu_elementwise runs more than once. See docs/fw.md
 * for the four shapes this forces and why fw_vectors.py double-checks it
 * against a plain Python matmul. */
#include "tpulib.h"

#define MM1_ROWS  16
#define MM1_DEPTH 1024
#define MM1_COLS  16

#define VEC_LEN   2500          /* two whole VPU chunks and a partial one */

#define MM4_ROWS  12            /* neither extent is a whole array tile */
#define MM4_DEPTH 64
#define MM4_COLS  20

/* {m0, n} literals: m0 in the low 12 bits, n above. Tuned, not guessed. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_C1 RQ(1u, 8u)        /* a 1024-long contraction reaches ~4000 */
#define RQ_C2 RQ(1u, 0u)        /* relu is a clamp, not a rescale: identity */
#define RQ_C3 RQ(1u, 1u)        /* the sum of two int4 is bounded by 16 */
#define RQ_C4 RQ(1u, 4u)

#define I4(cols) ((cols) / 2)

/* ---- DRAM ---------------------------------------------------------------- */
#define DR_A1 0x00000u          /* [16][1024] int4 */
#define DR_W1 0x02000u          /* [1024][16] int4 */
#define DR_C1 0x04000u          /* [16][16]   int4 */
#define DR_C2 0x04100u          /* [16][16]   int4 */
#define DR_V1 0x04200u          /* [2500]     int4 */
#define DR_V2 0x04700u
#define DR_C3 0x04C00u
#define DR_A4 0x05200u          /* [12][64]   int4 */
#define DR_B4 0x05400u          /* [20][64]   int4 */
#define DR_C4 0x05700u          /* [12][20]   int4 */

/* ---- scratchpad ---------------------------------------------------------- */
/* Three banks: one each for A, B and C, and nothing spare. The arena is how a
 * kernel says how much scratchpad a primitive may spend, and this is the floor
 * rather than a budget chosen to be fast. */
#define SP_ARENA  0x0000u
#define ARENA_MM  (3u * TPU_BANK_BYTES)

int main(void)
{
    tpu_arena arena;

    const tpu_gemm mm1 = {
        .rows = MM1_ROWS, .depth = MM1_DEPTH, .cols = MM1_COLS,
        .a = TPU_ROWS(DR_A1, I4(MM1_DEPTH)),
        .b = TPU_ROWS(DR_W1, I4(MM1_COLS)),
        .c = TPU_ROWS(DR_C1, I4(MM1_COLS)),
        .rq_word = RQ_C1,
    };
    tpu_arena_init(&arena, SP_ARENA, ARENA_MM);
    tpu_matmul(&mm1, &arena);

    tpu_relu(TPU_AT(DR_C2), TPU_AT(DR_C1), MM1_ROWS * MM1_COLS, RQ_C2, &arena);
    tpu_add(TPU_AT(DR_C3), TPU_AT(DR_V1), TPU_AT(DR_V2), VEC_LEN, RQ_C3, &arena);

    const tpu_gemm mm4 = {
        .rows = MM4_ROWS, .depth = MM4_DEPTH, .cols = MM4_COLS,
        .a = TPU_ROWS(DR_A4, I4(MM4_DEPTH)),
        .b = TPU_ROWS(DR_B4, I4(MM4_DEPTH)),    /* [cols][depth], transposed */
        .c = TPU_ROWS(DR_C4, I4(MM4_COLS)),
        .transpose = 1,
        .rq_word = RQ_C4,
    };
    tpu_matmul(&mm4, &arena);

    return 0;                   /* start.S raises `done` from here */
}
