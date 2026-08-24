/* tiled.c — tpulib.h past the point where the scratchpad stops helping.
 *
 * adder.c exercises the library with every activation resident and only the
 * weights streaming, so no operand there is ever bigger than the arena and
 * every block loop runs exactly once. This kernel is the other half: nothing
 * lives in the scratchpad at all — every operand is a DRAM tensor and the arena
 * is deliberately too small — so every loop in `tpu_matmul` and
 * `tpu_elementwise` has to run more than once to produce a right answer.
 *
 * Three problems, chosen for what they force rather than for what they compute:
 *
 *   1. C1 = A1 @ W1, 40x32 @ 32x32, 1 KB arena
 *      Rows past the array's 32-row result buffer, so the row loop runs; and
 *      the arena is small enough that the columns block too, which makes the
 *      output spill STRIDED — one DMA range per row, the slow path in
 *      tpu_move2d. The weight block is hoisted out of the row loop, so it is
 *      fetched once per column block rather than once per dispatch.
 *
 *   2. C2 = requant(relu(C1)), 1280 elements, both operands in DRAM
 *      Streams through the arena a chunk at a time: fill, widen, narrow, spill.
 *      This is the case a scratchpad-resident elementwise cannot express.
 *
 *   3. C3 = A3 @ W3, 8x64 @ 64x8, 256-byte arena
 *      Small enough that the CONTRACTION has to split, which is the one path
 *      that cannot narrow on store: partial tiles accumulate int32 in the arena
 *      through TPU_MM_ACC and a VPU pass brings the finished block down to
 *      int4. It also drives the row block down to a single token.
 *
 * The golden DRAM image comes from iss.py via accel/tpulang/fw_vectors.py, as
 * for every kernel here — but this is the one kernel that alone would not test,
 * because a tiling bug is something the ISS reproduces as faithfully as the
 * RTL. So fw_vectors.py also carries an independent `reference_tiled` and
 * checks the ISS against a plain Python matmul before any vectors are written.
 */
#include "tpulib.h"

/* ---- problem 1: a matmul that blocks in rows and columns ----------------- */
#define MM1_ROWS  40
#define MM1_DEPTH 32
#define MM1_COLS  32

/* ---- problem 3: a matmul that blocks in the contraction ------------------ */
#define MM3_ROWS  8
#define MM3_DEPTH 64
#define MM3_COLS  8

/* {m0, n} literals: m0 in the low 12 bits, n above. Tuned, not guessed — the
 * accumulators reach ~120 here and n=5 collapses the result onto four values,
 * while a golden answer with no spread passes against any datapath at all.
 * These give 11 distinct values in C1 and 13 in C3, across [-8, 7]. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_C1 RQ(1u, 4u)
#define RQ_C2 RQ(1u, 0u)        /* relu is a clamp, not a rescale: identity */
#define RQ_C3 RQ(1u, 4u)

/* Bytes in one row of a row-major int4 weight, i.e. two nibbles per byte. */
#define WGT_ROW(cols) ((cols) / 2)

/* ---- DRAM ---------------------------------------------------------------- */
#define DR_A1 0x00000u          /* [40][32] int8  */
#define DR_W1 0x00600u          /* [32][32] int4  */
#define DR_C1 0x00800u          /* [40][32] int8  */
#define DR_C2 0x00E00u          /* [40][32] int8  */
#define DR_A3 0x01400u          /* [8][64]  int8  */
#define DR_W3 0x01600u          /* [64][8]  int4  */
#define DR_C3 0x01700u          /* [8][8]   int8  */

/* ---- scratchpad ---------------------------------------------------------- */
/* One region, three different budgets. The arena is how a kernel says how much
 * scratchpad a primitive may spend, and these three are sized to force the
 * block loops rather than to be fast. */
#define SP_ARENA          0x0000u
#define ARENA_MM1         1024u
#define ARENA_ELEMENTWISE 4096u /* two chunk buffers + the int32 temp */
#define ARENA_MM3         256u

int main(void)
{
    tpu_arena arena;

    const tpu_gemm mm1 = {
        .rows = MM1_ROWS, .depth = MM1_DEPTH, .cols = MM1_COLS,
        .act  = TPU_DRAM_ROWS(DR_A1, MM1_DEPTH),
        .wgt  = TPU_DRAM_ROWS(DR_W1, WGT_ROW(MM1_COLS)),
        .out  = TPU_DRAM_ROWS(DR_C1, MM1_COLS),
        .rq_word = RQ_C1,
    };
    tpu_arena_init(&arena, SP_ARENA, ARENA_MM1);
    tpu_matmul(&mm1, &arena);

    tpu_arena_init(&arena, SP_ARENA, ARENA_ELEMENTWISE);
    tpu_relu_narrow(TPU_DRAM_AT(DR_C2), TPU_DRAM_AT(DR_C1),
                    MM1_ROWS * MM1_COLS, RQ_C2, &arena);

    const tpu_gemm mm3 = {
        .rows = MM3_ROWS, .depth = MM3_DEPTH, .cols = MM3_COLS,
        .act  = TPU_DRAM_ROWS(DR_A3, MM3_DEPTH),
        .wgt  = TPU_DRAM_ROWS(DR_W3, WGT_ROW(MM3_COLS)),
        .out  = TPU_DRAM_ROWS(DR_C3, MM3_COLS),
        .rq_word = RQ_C3,
    };
    tpu_arena_init(&arena, SP_ARENA, ARENA_MM3);
    tpu_matmul(&mm3, &arena);

    return 0;                   /* start.S raises `done` from here */
}
