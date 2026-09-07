/* infer.c — the int4 adder model as inference: prefill, then decode against a
 * KV cache. Token ids in, token ids out, one run.
 *
 * Every shape and every DRAM address comes from infer_config.h, which
 * accel/test/export.py generates from the checkpoint. Nothing here is tuned to
 * one model: M is the only difference between the training shape and the
 * generation shape, and infer_block serves both. See accel/tpu/docs/fw.md. */
/* Three builds, a ladder: base is tpu_matmul_wide with one weight buffer,
 * +dbuf double-buffers it so the next column block's weights fill under this
 * one's matmuls, +fused folds each matmul's residual add and activation onto
 * the block while it is still in the scratchpad. generate.py's --mm picks one;
 * -DINFER_MM_WIDE=0 is the separate A/B that puts every site on tpu_matmul. */
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
#include "infer_config.h"

#define INFER_FUSED (INFER_MM_MODE == INFER_MM_FUSED)

#define PREFILL_PASSES  (PROMPT / BLOCK)
#define PREFILL_TAIL    (PROMPT % BLOCK)
#define LAST_PASS_ROWS  (PREFILL_TAIL ? PREFILL_TAIL : BLOCK)

/* The prefill produces the token at PROMPT, so a generation of INFER_GEN tokens
 * is that one plus INFER_GEN-1 decode steps. */
#define DECODE_STEPS (INFER_GEN - 1u)

/* Where the generated ids land, and how many there are. Decode-only never
 * writes the token at PROMPT (that is the prefill's), so it starts one later. */
#if INFER_PREFILL
#define TOK_FIRST PROMPT
#define TOK_COUNT (1u + (INFER_DECODE ? DECODE_STEPS : 0u))
#else
#define TOK_FIRST (PROMPT + 1u)
#define TOK_COUNT DECODE_STEPS
#endif

_Static_assert(INFER_PREFILL || INFER_DECODE, "build at least one phase");
_Static_assert(PROMPT + INFER_GEN <= T, "generation runs past the sequence");
_Static_assert(!INFER_DECODE || INFER_GEN >= 2,
               "a decode step needs INFER_GEN >= 2; GEN counts the prefill's "
               "token too");
_Static_assert(D % TPU_N == 0 && DFF % TPU_N == 0 && T % TPU_N == 0 &&
                   HEAD_DIM % TPU_N == 0 && VOCAB_PAD % TPU_N == 0,
               "every contraction must be a whole number of array tiles");

/* Requant sites, in block order. This enum and export.py's RQ_NAMES are one
 * list written twice; INFER_RQ_SITES is what catches them drifting. */
enum {
    RQ_Q, RQ_K, RQ_V,           /* the three projections            */
    RQ_S,                       /* Q@K^T, with 1/sqrt(head_dim)     */
    RQ_ID,                      /* mask add — {1,0}                 */
    RQ_P,                       /* relu(S) -> P — {1,0}, s_p = s_s  */
    RQ_A,                       /* P@V                              */
    RQ_O,                       /* A@Wo -> O, pinned to s_x         */
    RQ_XO,                      /* X + O — {1,0}, stays on s_x      */
    RQ_X1,                      /* dyt(XO + X)                      */
    RQ_H,                       /* X1@W1                            */
    RQ_HR,                      /* relu(H) — {1,0}, s_hr = s_h      */
    RQ_F,                       /* HR@W2 -> F, pinned to s_x1       */
    RQ_X2,                      /* dyt(X1 + F)                      */
    RQ_N
};
_Static_assert(RQ_N == INFER_RQ_SITES,
               "infer_config.h was generated against a different site list");

static const uint16_t rq_table[LAYERS][RQ_N] = INFER_RQ_INIT;

#define I4(cols) ((cols) / 2)           /* bytes in a row-major int4 row */
_Static_assert(I4(VOCAB_PAD) % TPU_WORD_BYTES == 0,
               "a logits row must start on a scratchpad word for tpu_argmax");

/* The cache is one region per sequence per layer, and each half is stored in
 * the orientation its matmul wants. */
#define K_CACHE(seq, layer) \
    (DR_K_CACHE + ((seq) * LAYERS + (layer)) * (T * I4(D)))
#define V_CACHE(seq, layer) \
    (DR_V_CACHE + ((seq) * LAYERS + (layer)) * (T * I4(D)))

/* ---- scratchpad (64 KB): a fixed mailbox for the CPU, the rest is arena --- */
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

/* -DINFER_MM_WIDE=0 puts every site back on tpu_matmul, which is the A/B. */
#ifndef INFER_MM_WIDE
#define INFER_MM_WIDE 1
#endif

_Static_assert(INFER_MM_WIDE || !INFER_FUSED,
               "the fused build is tpu_matmul_wide's layout; --general has no "
               "fused primitive to call");

/* Every matmul in this kernel goes through this. tpu_matmul_wide stages one
 * column block of C instead of a whole C row, which spends the arena on
 * row-panel depth, and it double-buffers B so the next column block's weights
 * fill under this one's matmuls. It pays one spill per column block instead of
 * one per row panel. always_inline plus a constant shape at the call site folds
 * the whole thing; see docs/fw.md. */
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

/* The same, with the add and the activation that follow the matmul folded into
 * its visit to each output block: the intermediate never goes back to DRAM.
 * Only the fused build calls it, and INFER_FUSED is a constant, so the other
 * two builds drop it whole. */
#define INFER_MM_F(rows_, depth_, cols_, ...)                                 \
    do {                                                                      \
        const tpu_gemm_fused mm = { .rows = (rows_), .depth = (depth_),       \
                                    .cols = (cols_), __VA_ARGS__ };           \
                                                                              \
        tpu_matmul_wide_fused(&mm, &arena);                                   \
    } while (0)

/* X holds embeddings for positions first_pos..first_pos+rows-1 of each of the
 * BATCH sequences, sequence-major; on return, the residual stream after all
 * four layers. `rows` is per sequence. See docs/fw.md. */
__attribute__((always_inline))
static inline void infer_block(unsigned rows, unsigned first_pos)
{
    const unsigned rows_all = BATCH * rows;

    for (unsigned layer = 0; layer < LAYERS; layer++) {
        const uint16_t *rq = rq_table[layer];
        const uint32_t layer_wgt = DR_LAYER0 + layer * DR_LAYER_STRIDE;

        /* Three [D][D] projections off the same X, separate because each weight
         * has its own scale and so its own {m0,n}. */
        INFER_MM(rows_all, D, D,
                 .a = X_BUF,
                 .b = TPU_ROWS(layer_wgt + LW_WQ, I4(D)),
                 .c = Q_BUF,
                 .rq_word = rq[RQ_Q]);
        INFER_MM(rows_all, D, D,
                 .a = X_BUF,
                 .b = TPU_ROWS(layer_wgt + LW_WK, I4(D)),
                 .c = TMP_A,                    /* K_new */
                 .rq_word = rq[RQ_K]);
        INFER_MM(rows_all, D, D,
                 .a = X_BUF,
                 .b = TPU_ROWS(layer_wgt + LW_WV, I4(D)),
                 .c = TMP_B,                    /* V_new */
                 .rq_word = rq[RQ_V]);

        /* Per sequence: append to its cache, then attend over it. Sequence
         * `seq`'s scores overwrite its own rows of K_new with A, which is safe
         * because a later sequence's K_new lives in the rows below. */
        for (unsigned seq = 0; seq < BATCH; seq++) {
            const uint32_t k_cache = K_CACHE(seq, layer);
            const uint32_t v_cache = V_CACHE(seq, layer);
            const uint32_t seq_off = seq * rows * I4(D);

            /* The append is a copy: both halves are already in the orientation
             * and the encoding their matmul wants. */
            tpu_copy(TPU_ROWS(k_cache + first_pos * I4(D), I4(D)),
                     tpu_off(TMP_A, seq_off), rows, D, &arena);
            tpu_copy(TPU_ROWS(v_cache + first_pos * I4(D), I4(D)),
                     tpu_off(TMP_B, seq_off), rows, D, &arena);

            for (unsigned head = 0; head < HEADS; head++) {
                /* Contracting over all T keys rather than first_pos+rows of
                 * them keeps the shape constant; a runtime column count would
                 * unfold the block loop, which costs more than the extra
                 * tiles. */
                /* P = relu(S + mask). Fused, the mask block is staged
                 * beside the score block and both passes run there. */
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
            }
        }

        /* MultiHeadAttention.forward ends in `O + X` and Transformer.forward
         * adds X again, so the residual is 2X + O in two adds. The second is
         * the DyT, which is the same add with the odd clip — so fused, X is
         * the add block and the block is read twice, once per pass. */
        if (INFER_FUSED) {
            INFER_MM_F(rows_all, D, D,
                       .a = TMP_A,              /* A */
                       .b = TPU_ROWS(layer_wgt + LW_WO, I4(D)),
                       .c = TMP_B,              /* X1; O and X+O never land */
                       .add = X_BUF,
                       .add_op = TPU_V_ADD,
                       .activation = TPU_V_DYT,
                       .rq_word = rq[RQ_O],
                       .rq_add = rq[RQ_XO],
                       .rq_act = rq[RQ_X1]);

            INFER_MM_F(rows_all, D, DFF,
                       .a = TMP_B,              /* X1 */
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
                       .add = TMP_B,            /* X1 */
                       .add_op = TPU_ACT_NONE,
                       .activation = TPU_V_DYT,
                       .rq_word = rq[RQ_F],
                       .rq_act = rq[RQ_X2]);
        } else {
            INFER_MM(rows_all, D, D,
                     .a = TMP_A,                /* A */
                     .b = TPU_ROWS(layer_wgt + LW_WO, I4(D)),
                     .c = TMP_B,                /* O; V_new is dead */
                     .rq_word = rq[RQ_O]);

            tpu_add(TMP_A, X_BUF, TMP_B, rows_all * D, rq[RQ_XO], &arena);
            tpu_dyt(TMP_B, TMP_A, X_BUF, rows_all * D, rq[RQ_X1], &arena);

            INFER_MM(rows_all, D, DFF,
                     .a = TMP_B,                /* X1 */
                     .b = TPU_ROWS(layer_wgt + LW_FF1, I4(DFF)),
                     .c = H_BUF,
                     .rq_word = rq[RQ_H]);
            tpu_relu(H_BUF, H_BUF, rows_all * DFF, rq[RQ_HR], &arena);
            INFER_MM(rows_all, DFF, D,
                     .a = H_BUF,
                     .b = TPU_ROWS(layer_wgt + LW_FF2, I4(D)),
                     .c = TMP_A,                /* F; X+O is dead */
                     .rq_word = rq[RQ_F]);

            tpu_dyt(X_BUF, TMP_B, TMP_A, rows_all * D, rq[RQ_X2], &arena);
        }
    }
}

/* The ISA has no gather; a DMA whose DRAM address the CPU computed is one. */
static void embed(unsigned dst_row, unsigned token)
{
    tpu_copy(tpu_off(X_BUF, dst_row * I4(D)),
             TPU_ROWS(DR_EMBED + token * I4(D), I4(D)), 1u, D, &arena);
}

/* Logits for the row of X at `x_row`; returns sequence `seq`'s token at pos+1.
 * They land in DRAM for the host to check against PyTorch, and tpu_argmax
 * reduces them on the array. The MXU requantizes on store, so a logit is int4
 * and ties are common — the head's {m0,n} is what spreads them over the grid,
 * and tpu_argmax breaks a tie toward the lowest id like torch.argmax.
 *
 * VOCAB, not VOCAB_PAD: the head's padding columns are zero, so a padded logit
 * is 0 and would outrank every real one that came out negative. */
static unsigned head_argmax(unsigned x_row, unsigned seq, unsigned pos)
{
    const uint32_t dram_logits = DR_LOGITS + (seq * T + pos) * I4(VOCAB_PAD);
    unsigned best_token;

    INFER_MM(1u, D, VOCAB_PAD,
             .a = tpu_off(X_BUF, x_row * I4(D)),
             .b = TPU_ROWS(DR_HEAD_WGT, I4(VOCAB_PAD)),
             .c = TPU_ROWS(dram_logits, I4(VOCAB_PAD)),
             .rq_word = INFER_RQ_LOGIT);

    best_token = tpu_argmax(TPU_AT(dram_logits), VOCAB, &arena);

    tpu_spad_st(SP_TOKEN_AT(seq, pos + 1), best_token);
    return best_token;
}

int main(void)
{
    unsigned next_token[BATCH];
    unsigned seq;

    tpu_arena_init(&arena, 0u, SP_MAILBOX);

    /* The CPU has no path to DRAM, so a host-written number reaches it as a DMA
     * into the mailbox and a load through the window. The prompt only: pulling
     * the answer field in would put the host's own bytes back in the output. */
    for (seq = 0; seq < BATCH; seq++)
        tpu_move_bytes(SP_TOKEN_AT(seq, 0), DR_TOKENS + seq * T * 4u,
                       PROMPT * 4u, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

#if INFER_PREFILL
    {
        unsigned base = 0;

        /* Chunking is safe because a pass runs every layer before the next
         * starts, so the cache a later pass attends over is complete at every
         * depth. */
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
        for (seq = 0; seq < BATCH; seq++)
            next_token[seq] =
                head_argmax(seq * LAST_PASS_ROWS + LAST_PASS_ROWS - 1u, seq,
                            PROMPT - 1);
    }
#else
    /* Decode-only: no prefill ran, so the step at PROMPT starts from the
     * prompt's last token instead of from the one the prefill would have
     * produced. A step's cost is not data-dependent, so this measures the same
     * clocks; the ids it emits are not scored. */
    for (seq = 0; seq < BATCH; seq++)
        next_token[seq] = tpu_spad_ld(SP_TOKEN_AT(seq, PROMPT - 1)) & 0xFFu;
#endif

#if INFER_DECODE
    /* The prefill already produced the token at PROMPT, so this runs one time
     * fewer than there are tokens to generate. */
    for (unsigned pos = PROMPT; pos < PROMPT + DECODE_STEPS; pos++) {
        for (seq = 0; seq < BATCH; seq++)
            embed(seq, next_token[seq]);
        infer_block(1, pos);
        for (seq = 0; seq < BATCH; seq++)
            next_token[seq] = head_argmax(seq, seq, pos);
    }
#else
    (void)next_token;
#endif

    /* Only the ids this image actually generated: spilling a position it never
     * wrote would put uninitialized scratchpad in the output image. */
    for (seq = 0; seq < BATCH; seq++)
        tpu_move_bytes(SP_TOKEN_AT(seq, TOK_FIRST),
                       DR_TOKENS + (seq * T + TOK_FIRST) * 4u,
                       TOK_COUNT * 4u, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                       /* start.S raises `done` from here */
}
