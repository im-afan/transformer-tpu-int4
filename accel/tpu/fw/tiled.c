/* tiled.c — tpulib.h past the point where the scratchpad stops helping.
 *
 * adder.c exercises the library with every activation resident and only the
 * weights streaming, which is the *easy* half: no operand there is ever bigger
 * than the arena, so the block loops all run once. This kernel is the other
 * half. Nothing here lives in the scratchpad at all — A, W and C are DRAM
 * tensors and the arena is deliberately too small — so every loop in
 * `tpu_matmul` and `tpu_ew` has to run more than one iteration to produce a
 * right answer.
 *
 * Three problems, chosen for what they force rather than for what they compute:
 *
 *   1. C1[40][32] = A1[40][32] @ W1[32][32], 1 KB arena
 *      M past the array's 32-row result buffer (mxu.sv MAX_TOKENS), so the row
 *      loop runs; and the arena is small enough that N blocks too, which makes
 *      the C spill *strided* — one DMA range per row, the slow path in
 *      `tpu_move2d`. The weight block is hoisted out of the row loop, so W is
 *      fetched once per N block and not once per dispatch.
 *
 *   2. C2 = requant(relu(C1)), 1280 elements, both operands in DRAM
 *      Streams through the arena a chunk at a time: fill, widen, narrow, spill.
 *      This is the case a scratchpad-resident elementwise cannot express at all.
 *
 *   3. C3[8][8] = A3[8][64] @ W3[64][8], 256-byte arena
 *      Small enough that the *contraction* has to split, which is the one path
 *      that cannot narrow on store: the partial tiles accumulate int32 in the
 *      arena through `.acc`, and a VPU pass brings the finished block down to
 *      int4. It also drives the row block down to a single token.
 *
 * The golden DRAM image comes from iss.py by way of
 * ../../tpulang/fw_vectors.py, as for every kernel here — but this is the one
 * kernel whose answer that alone would not test, because a tiling bug is
 * something the ISS would reproduce as faithfully as the RTL. So fw_vectors.py
 * also carries an independent reference for it (`reference_tiled`) and checks
 * the ISS against a plain Python matmul before the vectors are ever written.
 */
#include "tpulib.h"

/* ---- problem 1: a matmul that blocks in M and N -------------------------- */
#define M1 40
#define K1 32
#define N1 32

/* ---- problem 3: a matmul that blocks in the contraction ------------------ */
#define M3 8
#define K3 64
#define N3 8

/* {m0, n} literals: m0 in the low 12 bits, n above. Tuned, not guessed — the
 * accumulators reach ~120 here and n=5 collapses the result onto four values,
 * while a golden answer with no spread passes against any datapath at all.
 * These give 11 distinct values in C1 and 13 in C3, across [-8, 7]. */
#define RQ(m0, n) ((uint32_t)((n) << 12) | (m0))
#define RQ_C1 RQ(1u, 4u)
#define RQ_C2 RQ(1u, 0u)        /* relu is a clamp, not a rescale: identity */
#define RQ_C3 RQ(1u, 4u)

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
 * scratchpad a primitive may spend, and the three here are sized to force the
 * block loops rather than to be fast. */
#define SP_ARENA 0x0000u
#define WS_MM1   1024u
#define WS_EW    4096u          /* two chunk buffers + the int32 temp */
#define WS_MM3   256u

int main(void)
{
    tpu_arena ws;

    const tpu_gemm mm1 = {
        .m = M1, .k = K1, .n = N1,
        .a  = TPU_DRR(DR_A1, K1),
        .w  = TPU_DRR(DR_W1, N1 * 4 / 8),
        .c  = TPU_DRR(DR_C1, N1),
        .rq = RQ_C1,
    };
    tpu_arena_init(&ws, SP_ARENA, WS_MM1);
    tpu_matmul(&mm1, &ws);

    tpu_arena_init(&ws, SP_ARENA, WS_EW);
    tpu_relu_narrow(TPU_DR(DR_C2), TPU_DR(DR_C1), M1 * N1, RQ_C2, &ws);

    const tpu_gemm mm3 = {
        .m = M3, .k = K3, .n = N3,
        .a  = TPU_DRR(DR_A3, K3),
        .w  = TPU_DRR(DR_W3, N3 * 4 / 8),
        .c  = TPU_DRR(DR_C3, N3),
        .rq = RQ_C3,
    };
    tpu_arena_init(&ws, SP_ARENA, WS_MM3);
    tpu_matmul(&mm3, &ws);

    return 0;                   /* start.S raises `done` from here */
}
