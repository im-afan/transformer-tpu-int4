/* infer.c — the int4 adder model as inference: prefill, then decode against a
 * KV cache. Token ids in, token ids out, one run. See docs/fw.md. */
#include "tpulib.h"

#ifdef ADDER_RQ_H
#include ADDER_RQ_H
#else
#include "infer_rq.h"
#endif

#define T         64
#define D         64
#define DFF       256
#define HEADS     4
#define HEAD_DIM  (D / HEADS)
#define LAYERS    4
#define VOCAB     13
#define VOCAB_PAD 16            /* the 13 logits, padded to a whole array tile */

/* '=' is at 31 and operands are padded so that holds for every problem. */
#ifndef PROMPT
#define PROMPT 32
#endif

#ifndef INFER_GEN
#define INFER_GEN (T - PROMPT)
#endif

/* Independent sequences sharing one weight stream. */
#ifndef BATCH
#define BATCH 1
#endif

/* Which half of a generation this image runs. Both on is the whole thing. */
#ifndef INFER_PREFILL
#define INFER_PREFILL 1
#endif
#ifndef INFER_DECODE
#define INFER_DECODE 1
#endif

/* A prompt longer than one BLOCK costs a weight stream per BLOCK rows. There is
 * no hardware row limit any more — the array's output block is N x N whatever
 * the caller's live extent — so this is bounded by DRAM, not by a dispatch. */
#ifndef BLOCK
#define BLOCK 32u
#endif

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

/* Rows per sequence in the widest pass this image runs, and the row count a
 * batched matmul sees. A decode-only image never runs a prefill block, so its
 * activation buffers are BATCH rows rather than BATCH*BLOCK. */
#define PREFILL_ROWS ((PROMPT < BLOCK) ? PROMPT : BLOCK)
#if INFER_PREFILL
#define MAX_SEQ_ROWS PREFILL_ROWS
#else
#define MAX_SEQ_ROWS 1u
#endif
#define ROWS_MAX (BATCH * MAX_SEQ_ROWS)

_Static_assert(INFER_PREFILL || INFER_DECODE, "build at least one phase");
_Static_assert(PROMPT + INFER_GEN <= T, "generation runs past the sequence");
_Static_assert(PROMPT >= 1, "the prefill needs at least one token");
_Static_assert(BATCH >= 1, "BATCH is a sequence count");
_Static_assert(!INFER_DECODE || INFER_GEN >= 2,
               "a decode step needs INFER_GEN >= 2; GEN counts the prefill's "
               "token too");
_Static_assert(D % TPU_N == 0 && DFF % TPU_N == 0 && T % TPU_N == 0 &&
                   HEAD_DIM % TPU_N == 0 && VOCAB_PAD % TPU_N == 0,
               "every contraction must be a whole number of array tiles");

/* Requant sites, in block order. RQ_KP and RQ_VP are retired holes: they fed
 * the `quant4` passes that turned an int8 activation into a weight operand, and
 * the MXU stores int4 itself now. The indices are kept so a table emitted by
 * accel/tpulang does not renumber. */
enum {
    RQ_Q, RQ_K, RQ_V,           /* the three projections            */
    RQ_KP, RQ_VP,               /* retired                          */
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

static const uint16_t rq_table[LAYERS][RQ_N] = ADDER_RQ_INIT;

#define I4(cols) ((cols) / 2)           /* bytes in a row-major int4 row */

/* ---- DRAM (512 KB); layout notes in docs/fw.md ---------------------------- */
#define DR_ALIGN(a) (((a) + 63u) & ~63u)

#define DR_EMBED    0x00000u                                       /* host      */
#define DR_TOKENS   DR_ALIGN(DR_EMBED   + VOCAB * I4(D))           /* host, out */
#define DR_LOGITS   DR_ALIGN(DR_TOKENS  + BATCH * T * 4u)          /* out       */
#define DR_MASK     DR_ALIGN(DR_LOGITS  + BATCH * T * I4(VOCAB_PAD)) /* host    */
#define DR_HEAD_WGT DR_ALIGN(DR_MASK    + T * I4(T))               /* host      */
#define DR_K_CACHE  DR_ALIGN(DR_HEAD_WGT + D * I4(VOCAB_PAD))
#define DR_V_CACHE  DR_ALIGN(DR_K_CACHE + BATCH * LAYERS * T * I4(D))
#define DR_X        DR_ALIGN(DR_V_CACHE + BATCH * LAYERS * T * I4(D))
#define DR_TMP_A    DR_ALIGN(DR_X       + ROWS_MAX * I4(D))
#define DR_TMP_B    DR_ALIGN(DR_TMP_A   + ROWS_MAX * I4(D))

/* Attention's working set and the FFN's hidden layer never coexist, so they
 * share one region. The only aliasing in the map: X, TMP_A, TMP_B are live
 * across the whole layer. */
#define DR_SCRATCH  DR_ALIGN(DR_TMP_B   + ROWS_MAX * I4(D))
#define DR_Q        DR_SCRATCH
#define DR_S        DR_ALIGN(DR_Q       + ROWS_MAX * I4(D))
#define DR_ATTN_END DR_ALIGN(DR_S       + MAX_SEQ_ROWS * I4(T))
#define DR_H        DR_SCRATCH
#define DR_FFN_END  DR_ALIGN(DR_H       + ROWS_MAX * I4(DFF))
#define DR_END      ((DR_ATTN_END > DR_FFN_END) ? DR_ATTN_END : DR_FFN_END)

#define DR_LAYER0        0x20000u
#define DR_LAYER_STRIDE  0x18000u
#define LW_WQ  0x00000u                 /* [D][D]   int4 */
#define LW_WK  0x02000u
#define LW_WV  0x04000u
#define LW_WO  0x06000u
#define LW_FF1 0x08000u                 /* [D][DFF] int4 */
#define LW_FF2 0x10000u                 /* [DFF][D] int4 */

#define K_CACHE(seq, layer) \
    (DR_K_CACHE + ((seq) * LAYERS + (layer)) * (T * I4(D)))
#define V_CACHE(seq, layer) \
    (DR_V_CACHE + ((seq) * LAYERS + (layer)) * (T * I4(D)))

/* Two levers when this fires, in this order: lower BLOCK (the per-sequence row
 * count, which scales X, TMP_A, TMP_B and the scratch union and costs one
 * weight stream per extra pass), then lower BATCH. */
_Static_assert(DR_END <= DR_LAYER0,
               "the activation map runs into layer 0's weights: lower BLOCK, "
               "then BATCH");
_Static_assert(DR_LAYER0 + LAYERS * DR_LAYER_STRIDE <= 0x80000u,
               "the weights overrun the 512 KB SRAM");

/* ---- scratchpad (64 KB): a fixed mailbox for the CPU, the rest is arena --- */
#define SP_BYTES    TPU_SPAD_BYTES
#define SP_MAILBOX  (SP_BYTES - (I4(VOCAB_PAD) + BATCH * T * 4u))
#define SP_LOGITS   SP_MAILBOX                       /* [VOCAB_PAD] int4 */
#define SP_TOKENS   (SP_LOGITS + I4(VOCAB_PAD))      /* [BATCH][T] int32 */

#define SP_TOKEN_AT(seq, pos) (SP_TOKENS + ((seq) * T + (pos)) * 4u)

static tpu_arena arena;

#define X_BUF   TPU_ROWS(DR_X,     I4(D))
#define TMP_A   TPU_ROWS(DR_TMP_A, I4(D))
#define TMP_B   TPU_ROWS(DR_TMP_B, I4(D))
#define Q_BUF   TPU_ROWS(DR_Q,     I4(D))
#define S_BUF   TPU_ROWS(DR_S,     I4(T))
#define H_BUF   TPU_ROWS(DR_H,     I4(DFF))

/* X holds the embeddings for positions first_pos .. first_pos+rows-1 of each of
 * the BATCH sequences, sequence-major, and on return the residual stream after
 * all four layers.
 *
 * `rows` is rows PER SEQUENCE; the weight matmuls run over BATCH*rows of them
 * so one weight stream serves the whole batch.
 *
 * `always_inline` with `rows` a literal at both call sites folds tpu_matmul's
 * block chooser at every matmul site, specializing the prefill's shape and the
 * decode's. `first_pos` stays a runtime value. */
__attribute__((always_inline))
static inline void infer_block(unsigned rows, unsigned first_pos)
{
    const unsigned rows_all = BATCH * rows;

    for (unsigned layer = 0; layer < LAYERS; layer++) {
        const uint16_t *rq = rq_table[layer];
        const uint32_t layer_wgt = DR_LAYER0 + layer * DR_LAYER_STRIDE;

        /* Three [D][D] projections off the same X, separate because each weight
         * has its own scale and so its own {m0,n}. */
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = D,
            .a = X_BUF,
            .b = TPU_ROWS(layer_wgt + LW_WQ, I4(D)),
            .c = Q_BUF,
            .rq_word = rq[RQ_Q] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = D,
            .a = X_BUF,
            .b = TPU_ROWS(layer_wgt + LW_WK, I4(D)),
            .c = TMP_A,                         /* K_new */
            .rq_word = rq[RQ_K] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = D,
            .a = X_BUF,
            .b = TPU_ROWS(layer_wgt + LW_WV, I4(D)),
            .c = TMP_B,                         /* V_new */
            .rq_word = rq[RQ_V] }, &arena);

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
                tpu_matmul(&(const tpu_gemm){
                    .rows = rows, .depth = HEAD_DIM, .cols = T,
                    .a = tpu_off(Q_BUF, seq_off + head * I4(HEAD_DIM)),
                    .b = TPU_ROWS(k_cache + head * I4(HEAD_DIM), I4(D)),
                    .c = S_BUF,
                    .transpose = 1,
                    .rq_word = rq[RQ_S] }, &arena);

                /* P = relu(S + mask), both passes in place. */
                tpu_add(S_BUF, S_BUF, TPU_ROWS(DR_MASK + first_pos * I4(T), I4(T)),
                        rows * T, rq[RQ_ID], &arena);
                tpu_relu(S_BUF, S_BUF, rows * T, rq[RQ_P], &arena);

                tpu_matmul(&(const tpu_gemm){
                    .rows = rows, .depth = T, .cols = HEAD_DIM,
                    .a = S_BUF,
                    .b = TPU_ROWS(v_cache + head * I4(HEAD_DIM), I4(D)),
                    .c = tpu_off(TMP_A, seq_off + head * I4(HEAD_DIM)),
                    .rq_word = rq[RQ_A] }, &arena);        /* A */
            }
        }

        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = D,
            .a = TMP_A,                         /* A */
            .b = TPU_ROWS(layer_wgt + LW_WO, I4(D)),
            .c = TMP_B,                         /* O; V_new is dead */
            .rq_word = rq[RQ_O] }, &arena);

        /* MultiHeadAttention.forward ends in `O + X` and Transformer.forward
         * adds X again, so this is 2X + O in two adds. The second is the DyT,
         * which is the same add with the odd clip. */
        tpu_add(TMP_A, X_BUF, TMP_B, rows_all * D, rq[RQ_XO], &arena);
        tpu_dyt(TMP_B, TMP_A, X_BUF, rows_all * D, rq[RQ_X1], &arena);

        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = DFF,
            .a = TMP_B,                         /* X1 */
            .b = TPU_ROWS(layer_wgt + LW_FF1, I4(DFF)),
            .c = H_BUF,
            .rq_word = rq[RQ_H] }, &arena);
        tpu_relu(H_BUF, H_BUF, rows_all * DFF, rq[RQ_HR], &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = DFF, .cols = D,
            .a = H_BUF,
            .b = TPU_ROWS(layer_wgt + LW_FF2, I4(D)),
            .c = TMP_A,                         /* F; X+O is dead */
            .rq_word = rq[RQ_F] }, &arena);

        tpu_dyt(X_BUF, TMP_B, TMP_A, rows_all * D, rq[RQ_X2], &arena);
    }
}

/* The ISA has no gather; a DMA whose DRAM address the CPU computed is one. */
static void embed(unsigned dst_row, unsigned token)
{
    tpu_copy(tpu_off(X_BUF, dst_row * I4(D)),
             TPU_ROWS(DR_EMBED + token * I4(D), I4(D)), 1u, D, &arena);
}

/* Logits for the row of X at `x_row`; returns sequence `seq`'s token at pos+1.
 * They land in DRAM for the host to check against PyTorch and in the mailbox so
 * the CPU can read them. The MXU requantizes on store, so a logit is int4 and
 * ties are common — the head's {m0,n} is what spreads them over the grid. */
static unsigned head_argmax(unsigned x_row, unsigned seq, unsigned pos)
{
    const uint32_t dram_logits = DR_LOGITS + (seq * T + pos) * I4(VOCAB_PAD);
    unsigned best_token = 0;
    int best_logit;
    uint32_t packed[I4(VOCAB_PAD) / 4u];

    tpu_matmul(&(const tpu_gemm){
        .rows = 1, .depth = D, .cols = VOCAB_PAD,
        .a = tpu_off(X_BUF, x_row * I4(D)),
        .b = TPU_ROWS(DR_HEAD_WGT, I4(VOCAB_PAD)),
        .c = TPU_ROWS(dram_logits, I4(VOCAB_PAD)),
        .rq_word = INFER_RQ_LOGIT }, &arena);

    tpu_move_bytes(SP_LOGITS, dram_logits, I4(VOCAB_PAD), TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    for (unsigned w = 0; w < I4(VOCAB_PAD) / 4u; w++)
        packed[w] = tpu_spad_ld(SP_LOGITS + w * 4u);

    /* Strictly greater, so a tie takes the lowest id — torch.argmax's rule. */
    best_logit = -16;
    for (unsigned token = 0; token < VOCAB; token++) {
        const unsigned nib = (packed[token / 8u] >> (4u * (token % 8u))) & 0xFu;
        const int logit = (int)(nib >= 8u ? (int)nib - 16 : (int)nib);

        if (logit > best_logit) {
            best_logit = logit;
            best_token = token;
        }
    }

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
