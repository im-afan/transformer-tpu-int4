/* mha_prefill.c — infer.c's prefill and nothing else: embed the prompt, then
 * run every layer over all PROMPT rows at once, appending to the KV cache.
 *
 * The layer body is infer.c's infer_block verbatim. What is gone is everything
 * around it: no decode steps, no output head, no argmax, no generated ids. The
 * counters reset at the launch and freeze at the halt, so an image that runs
 * only the prefill IS the measurement of the prefill.
 *
 * PART_ATTN / PART_FFN pick which half of the layer runs, the same way infer.c
 * splits prefill from decode. Attention alone is what this test is for; the FFN
 * half is here so the two add up to a whole block.
 *
 * Shape, the whole DRAM map and the requant table come from the generated
 * mha_prefill_config.h. See accel/tpu/docs/mha_prefill.md. */
#define INFER_MM_BASE  0
#define INFER_MM_DBUF  1
#define INFER_MM_FUSED 2

#ifndef INFER_MM_MODE
#define INFER_MM_MODE INFER_MM_DBUF
#endif

#if INFER_MM_MODE == INFER_MM_BASE
#define TPU_WGT_PREFETCH 0
#endif

#include "tpulib.h"
#include "mha_prefill_config.h"

#define INFER_FUSED (INFER_MM_MODE == INFER_MM_FUSED)

#define PREFILL_PASSES  (PROMPT / BLOCK)
#define PREFILL_TAIL    (PROMPT % BLOCK)
#define LAST_PASS_ROWS  (PREFILL_TAIL ? PREFILL_TAIL : BLOCK)

_Static_assert(PART_ATTN || PART_FFN, "build at least one half of the layer");
_Static_assert(PROMPT <= T, "the prompt runs past the sequence");
_Static_assert(D % TPU_N == 0 && DFF % TPU_N == 0 && T % TPU_N == 0 &&
                   HEAD_DIM % TPU_N == 0,
               "every contraction must be a whole number of array tiles");

/* Requant sites, in block order — the same list, in the same order, as
 * infer.c's enum and export.py's RQ_NAMES. */
enum {
    RQ_Q, RQ_K, RQ_V,
    RQ_S,
    RQ_ID,
    RQ_P,
    RQ_A,
    RQ_O,
    RQ_XO,
    RQ_X1,
    RQ_H,
    RQ_HR,
    RQ_F,
    RQ_X2,
    RQ_N
};
_Static_assert(RQ_N == INFER_RQ_SITES,
               "mha_prefill_config.h was generated against a different site list");

static const uint16_t rq_table[LAYERS][RQ_N] = INFER_RQ_INIT;

#define I4(cols) ((cols) / 2)           /* bytes in a row-major int4 row */

#define K_CACHE(seq, layer) \
    (DR_K_CACHE + ((seq) * LAYERS + (layer)) * (T * I4(D)))
#define V_CACHE(seq, layer) \
    (DR_V_CACHE + ((seq) * LAYERS + (layer)) * (T * I4(D)))

/* ---- scratchpad: a fixed mailbox for the CPU, the rest is arena ---------- */
#define SP_BYTES    TPU_SPAD_BYTES
#define SP_MAILBOX  (SP_BYTES - BATCH * T * 4u)
#define SP_TOKENS   SP_MAILBOX                       /* [BATCH][T] int32 */

#define SP_TOKEN_AT(seq, pos) (SP_TOKENS + ((seq) * T + (pos)) * 4u)

static tpu_arena arena;

#define X_BUF   TPU_ROWS(DR_X,     I4(D))
#define TMP_A   TPU_ROWS(DR_TMP_A, I4(D))
#define TMP_B   TPU_ROWS(DR_TMP_B, I4(D))
#define Q_BUF   TPU_ROWS(DR_Q,     I4(D))
#define S_BUF   TPU_ROWS(DR_S,     I4(T))
#define H_BUF   TPU_ROWS(DR_H,     I4(DFF))

/* The residual stream between the two halves. Whichever half runs last leaves
 * the layer's output in X_BUF, so a half-built image feeds itself. Writing a
 * buffer that is also an operand is safe in both matmul paths: a block's `add`
 * is filled before that block's C is spilled, and tpu_elementwise fences per
 * chunk. */
#if PART_ATTN && PART_FFN
#define X1_BUF  TMP_B
#else
#define X1_BUF  X_BUF
#endif

#ifndef INFER_MM_WIDE
#define INFER_MM_WIDE 1
#endif

_Static_assert(INFER_MM_WIDE || !INFER_FUSED,
               "the fused build is tpu_matmul_wide's layout; --general has no "
               "fused primitive to call");

/* Attention on tpu_flashattention instead of the score matmul, the mask add,
 * the relu and P@V — infer.c's INFER_ATTN_FLASH, kept in step. */
#ifndef INFER_ATTN_FLASH
#define INFER_ATTN_FLASH 0
#endif

#define INFER_MM(rows_, depth_, cols_, ...)                                   \
    do {                                                                      \
        const tpu_gemm mm = { .rows = (rows_), .depth = (depth_),             \
                              .cols = (cols_), __VA_ARGS__ };                 \
                                                                              \
        if (INFER_MM_WIDE)                                                    \
            tpu_matmul_wide(&mm, &arena);                                     \
        else                                                                  \
            tpu_matmul(&mm, &arena);                                          \
    } while (0)

#define INFER_MM_F(rows_, depth_, cols_, ...)                                 \
    do {                                                                      \
        const tpu_gemm_fused mm = { .rows = (rows_), .depth = (depth_),       \
                                    .cols = (cols_), __VA_ARGS__ };           \
                                                                              \
        tpu_matmul_wide_fused(&mm, &arena);                                   \
    } while (0)

__attribute__((always_inline))
static inline void infer_block(unsigned rows, unsigned first_pos)
{
    const unsigned rows_all = BATCH * rows;

    for (unsigned layer = 0; layer < LAYERS; layer++) {
        const uint16_t *rq = rq_table[layer];
        const uint32_t layer_wgt = DR_LAYER0 + layer * DR_LAYER_STRIDE;

#if PART_ATTN
        /* Three [D][D] projections off the same X, separate because each weight
         * has its own scale and so its own {m0,n}. */
        INFER_MM(rows_all, D, D,
                 .a = X_BUF,
                 .b = TPU_ROWS(layer_wgt + LW_WQ, I4(D)),
                 .c = Q_BUF,
                 .rq_word = rq[RQ_Q]);
        /* K and V spill straight into the cache: a sequence's rows at
         * first_pos are exactly the layout and encoding their projection
         * produces, so there is no append. One call per sequence, because a
         * sequence's cache is its own region — the weight stream is read BATCH
         * times where Q reads it once. */
        for (unsigned seq = 0; seq < BATCH; seq++) {
            const uint32_t seq_off = seq * rows * I4(D);

            INFER_MM(rows, D, D,
                     .a = tpu_off(X_BUF, seq_off),
                     .b = TPU_ROWS(layer_wgt + LW_WK, I4(D)),
                     .c = TPU_ROWS(K_CACHE(seq, layer) + first_pos * I4(D), I4(D)),
                     .rq_word = rq[RQ_K]);
            INFER_MM(rows, D, D,
                     .a = tpu_off(X_BUF, seq_off),
                     .b = TPU_ROWS(layer_wgt + LW_WV, I4(D)),
                     .c = TPU_ROWS(V_CACHE(seq, layer) + first_pos * I4(D), I4(D)),
                     .rq_word = rq[RQ_V]);
        }

        /* Per sequence: attend over its cache. */
        for (unsigned seq = 0; seq < BATCH; seq++) {
            const uint32_t k_cache = K_CACHE(seq, layer);
            const uint32_t v_cache = V_CACHE(seq, layer);
            const uint32_t seq_off = seq * rows * I4(D);

            for (unsigned head = 0; head < HEADS; head++) {
#if INFER_ATTN_FLASH
                const tpu_flash fa = {
                    .rows = rows, .first_pos = first_pos,
                    .keys = T, .head_dim = HEAD_DIM,
                    .q = tpu_off(Q_BUF, seq_off + head * I4(HEAD_DIM)),
                    .k = TPU_ROWS(k_cache + head * I4(HEAD_DIM), I4(D)),
                    .v = TPU_ROWS(v_cache + head * I4(HEAD_DIM), I4(D)),
                    .mask = TPU_ROWS(DR_MASK, I4(T)),
                    .out = tpu_off(TMP_A, seq_off + head * I4(HEAD_DIM)),
                    .rq_s = rq[RQ_S], .rq_mask = rq[RQ_ID],
                    .rq_p = rq[RQ_P], .rq_a = rq[RQ_A],
                };

                tpu_flashattention(&fa, &arena);
#else
                /* P = relu(S + mask), over all T keys — the tail of the cache
                 * is zero and the mask is what makes it harmless. */
                if (INFER_FUSED) {
                    INFER_MM_F(rows, HEAD_DIM, T,
                               .a = tpu_off(Q_BUF, seq_off + head * I4(HEAD_DIM)),
                               .b = TPU_ROWS(k_cache + head * I4(HEAD_DIM), I4(D)),
                               .c = S_BUF,
                               .add = TPU_ROWS(DR_MASK + first_pos * I4(T), I4(T)),
                               .add_op = TPU_V_ADD,
                               .activation = TPU_V_RELU,
                               .transpose = 1,
                               .rq_word = rq[RQ_S],
                               .rq_add = rq[RQ_ID],
                               .rq_act = rq[RQ_P]);
                } else {
                    INFER_MM(rows, HEAD_DIM, T,
                             .a = tpu_off(Q_BUF, seq_off + head * I4(HEAD_DIM)),
                             .b = TPU_ROWS(k_cache + head * I4(HEAD_DIM), I4(D)),
                             .c = S_BUF,
                             .transpose = 1,
                             .rq_word = rq[RQ_S]);

                    tpu_add(S_BUF, S_BUF,
                            TPU_ROWS(DR_MASK + first_pos * I4(T), I4(T)),
                            rows * T, rq[RQ_ID], &arena);
                    tpu_relu(S_BUF, S_BUF, rows * T, rq[RQ_P], &arena);
                }

                INFER_MM(rows, T, HEAD_DIM,
                         .a = S_BUF,
                         .b = TPU_ROWS(v_cache + head * I4(HEAD_DIM), I4(D)),
                         .c = tpu_off(TMP_A, seq_off + head * I4(HEAD_DIM)),
                         .rq_word = rq[RQ_A]);              /* A */
#endif
            }
        }

        /* The residual is 2X + O in two adds; the second is the DyT. */
        if (INFER_FUSED) {
            INFER_MM_F(rows_all, D, D,
                       .a = TMP_A,              /* A */
                       .b = TPU_ROWS(layer_wgt + LW_WO, I4(D)),
                       .c = X1_BUF,             /* O and X+O never land */
                       .add = X_BUF,
                       .add_op = TPU_V_ADD,
                       .activation = TPU_V_DYT,
                       .rq_word = rq[RQ_O],
                       .rq_add = rq[RQ_XO],
                       .rq_act = rq[RQ_X1]);
        } else {
            INFER_MM(rows_all, D, D,
                     .a = TMP_A,                /* A */
                     .b = TPU_ROWS(layer_wgt + LW_WO, I4(D)),
                     .c = TMP_B,                /* O; V_new is dead */
                     .rq_word = rq[RQ_O]);

            tpu_add(TMP_A, X_BUF, TMP_B, rows_all * D, rq[RQ_XO], &arena);
            tpu_dyt(X1_BUF, TMP_A, X_BUF, rows_all * D, rq[RQ_X1], &arena);
        }
#endif  /* PART_ATTN */

#if PART_FFN
        if (INFER_FUSED) {
            INFER_MM_F(rows_all, D, DFF,
                       .a = X1_BUF,
                       .b = TPU_ROWS(layer_wgt + LW_FF1, I4(DFF)),
                       .c = H_BUF,
                       .add_op = TPU_ACT_NONE,
                       .activation = TPU_V_RELU,
                       .rq_word = rq[RQ_H],
                       .rq_act = rq[RQ_HR]);

            INFER_MM_F(rows_all, DFF, D,
                       .a = H_BUF,
                       .b = TPU_ROWS(layer_wgt + LW_FF2, I4(D)),
                       .c = X_BUF,              /* X2; F never lands */
                       .add = X1_BUF,
                       .add_op = TPU_ACT_NONE,
                       .activation = TPU_V_DYT,
                       .rq_word = rq[RQ_F],
                       .rq_act = rq[RQ_X2]);
        } else {
            INFER_MM(rows_all, D, DFF,
                     .a = X1_BUF,
                     .b = TPU_ROWS(layer_wgt + LW_FF1, I4(DFF)),
                     .c = H_BUF,
                     .rq_word = rq[RQ_H]);
            tpu_relu(H_BUF, H_BUF, rows_all * DFF, rq[RQ_HR], &arena);
            INFER_MM(rows_all, DFF, D,
                     .a = H_BUF,
                     .b = TPU_ROWS(layer_wgt + LW_FF2, I4(D)),
                     .c = TMP_A,                /* F; X+O is dead */
                     .rq_word = rq[RQ_F]);

            tpu_dyt(X_BUF, X1_BUF, TMP_A, rows_all * D, rq[RQ_X2], &arena);
        }
#endif  /* PART_FFN */
    }
}

/* The ISA has no gather; a DMA whose DRAM address the CPU computed is one. */
static void embed(unsigned dst_row, unsigned token)
{
    tpu_copy(tpu_off(X_BUF, dst_row * I4(D)),
             TPU_ROWS(DR_EMBED + token * I4(D), I4(D)), 1u, D, &arena);
}

int main(void)
{
    unsigned base = 0;
    unsigned seq;

    tpu_arena_init(&arena, 0u, SP_MAILBOX);

    /* The CPU has no path to DRAM, so a host-written prompt reaches it as a DMA
     * into the mailbox and a load through the window. */
    for (seq = 0; seq < BATCH; seq++)
        tpu_move_bytes(SP_TOKEN_AT(seq, 0), DR_TOKENS + seq * T * 4u,
                       PROMPT * 4u, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* Chunking is safe because a pass runs every layer before the next starts,
     * so the cache a later pass attends over is complete at every depth. */
    for (unsigned pass = 0; pass < PREFILL_PASSES; pass++, base += BLOCK) {
        for (seq = 0; seq < BATCH; seq++)
            for (unsigned row = 0; row < BLOCK; row++)
                embed(seq * BLOCK + row,
                      tpu_spad_ld(SP_TOKEN_AT(seq, base + row)) & 0xFFu);
        infer_block(BLOCK, base);
    }
#if PREFILL_TAIL
    for (seq = 0; seq < BATCH; seq++)
        for (unsigned row = 0; row < PREFILL_TAIL; row++)
            embed(seq * PREFILL_TAIL + row,
                  tpu_spad_ld(SP_TOKEN_AT(seq, base + row)) & 0xFFu);
    infer_block(PREFILL_TAIL, base);
#endif

    /* Every tensor's home is DRAM, so there is nothing to spill: X and both
     * caches are already where the host reads them. */
    return 0;                       /* start.S raises `done` from here */
}
