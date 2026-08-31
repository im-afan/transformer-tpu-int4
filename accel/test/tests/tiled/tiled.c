/* tiled.c — tpulib.h past the point where the scratchpad stops helping.
 * See docs/fw.md. */
#include "tpulib.h"

/* Shape, the DRAM map, the four requant words and the arena all come from
 * generate.py as -D. The defaults below are only for a bare `make`; they are
 * the shape the block loops were written against — a contraction that has to be
 * split, a vector spanning three VPU chunks, and extents that are not whole
 * array tiles. Keep those properties when changing them. */
#ifndef MM1_ROWS
#define MM1_ROWS  16
#endif
#ifndef MM1_DEPTH
#define MM1_DEPTH 1024          /* longer than one dispatch: split */
#endif
#ifndef MM1_COLS
#define MM1_COLS  16
#endif

#ifndef VEC_LEN
#define VEC_LEN   2500          /* two whole VPU chunks and a partial one */
#endif

#ifndef MM4_ROWS
#define MM4_ROWS  12            /* neither extent is a whole array tile */
#endif
#ifndef MM4_DEPTH
#define MM4_DEPTH 64
#endif
#ifndef MM4_COLS
#define MM4_COLS  20
#endif

/* {m0, n} literals: m0 in the low 12 bits, n above. generate.py fits them to
 * the accumulators the golden actually produced, so they follow the shape. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#ifndef RQ_C1
#define RQ_C1 RQ(1u, 8u)        /* a 1024-long contraction reaches ~4000 */
#endif
#ifndef RQ_C2
#define RQ_C2 RQ(1u, 0u)        /* relu is a clamp, not a rescale: identity */
#endif
#ifndef RQ_C3
#define RQ_C3 RQ(1u, 1u)        /* the sum of two int4 is bounded by 16 */
#endif
#ifndef RQ_C4
#define RQ_C4 RQ(1u, 4u)
#endif

#define I4(cols) ((cols) / 2)

/* ---- DRAM ---------------------------------------------------------------- */
#ifndef DR_A1
#define DR_A1 0x00000u          /* [MM1_ROWS][MM1_DEPTH] int4 */
#endif
#ifndef DR_W1
#define DR_W1 0x02000u          /* [MM1_DEPTH][MM1_COLS] int4 */
#endif
#ifndef DR_C1
#define DR_C1 0x04000u          /* [MM1_ROWS][MM1_COLS]  int4 */
#endif
#ifndef DR_C2
#define DR_C2 0x04100u
#endif
#ifndef DR_V1
#define DR_V1 0x04200u          /* [VEC_LEN]             int4 */
#endif
#ifndef DR_V2
#define DR_V2 0x04700u
#endif
#ifndef DR_C3
#define DR_C3 0x04C00u
#endif
#ifndef DR_A4
#define DR_A4 0x05200u          /* [MM4_ROWS][MM4_DEPTH] int4 */
#endif
#ifndef DR_B4
#define DR_B4 0x05400u          /* [MM4_COLS][MM4_DEPTH] int4 */
#endif
#ifndef DR_C4
#define DR_C4 0x05700u          /* [MM4_ROWS][MM4_COLS]  int4 */
#endif

/* ---- scratchpad ---------------------------------------------------------- */
/* Three banks: one each for A, B and C, and nothing spare. The arena is how a
 * kernel says how much scratchpad a primitive may spend, and this is the floor
 * rather than a budget chosen to be fast. */
#ifndef SP_ARENA
#define SP_ARENA  0x0000u
#endif
#ifndef ARENA_BANKS
#define ARENA_BANKS 3u
#endif
#define ARENA_MM  (ARENA_BANKS * TPU_BANK_BYTES)

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
