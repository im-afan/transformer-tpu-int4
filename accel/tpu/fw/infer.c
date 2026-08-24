/* infer.c — the int4 adder model as INFERENCE: prefill, then decode, against a
 * statically allocated KV cache. Tokens in, tokens out, one run.
 *
 * `fw/adder.c` is the same four layers as one T=32 forward pass — the training
 * shape, every position computed at once from a sequence somebody already
 * knows. Generating an answer that way costs a whole T-token forward per token
 * emitted and re-derives K and V for the prefix every time. This kernel is the
 * generative shape instead:
 *
 *   prefill   the PROMPT tokens up to and including '=', in ONE block of 15
 *             rows. Their K and V land in the cache; the last row's logits give
 *             the first answer digit.
 *   decode    one token at a time. The new token's K and V are appended and
 *             attention contracts against the whole cache, so a step is one row
 *             everywhere — 4 KB of arithmetic instead of 130 KB.
 *
 * Both are the same code: `infer_block(rows, first_pos)` runs `rows` new
 * positions starting at `first_pos`. The cache, the mask and tpulib.h's block
 * loops do not care how many rows arrive at once, so the row count is the ONLY
 * difference between the training shape and the generation shape.
 *
 * ---- what the device owns that adder.c gave to the host ---------------------
 *
 * cpu_subsys.sv decodes 0x9xxx_xxxx onto the scratchpad's S port, so:
 *
 *   argmax    the head writes 13 int32 logits to the SCRATCHPAD instead of
 *             DRAM, and the CPU reads them back and compares (tpu_spad_ld).
 *   gather    the embedding table is a DRAM tensor and a DMA takes a COMPUTED
 *             address, so `DR_EMBED + token*D` is the gather. The token index
 *             never leaves the CPU.
 *
 * The whole autoregressive loop therefore closes on the device: the host stages
 * the weights, the mask, the embedding table and the prompt's token ids,
 * presses 'G' once, and reads a finished sequence back out of DR_TOKENS.
 *
 * ---- the KV cache -----------------------------------------------------------
 *
 * Statically allocated, scratchpad-resident, 3 KB per layer. Each half is
 * stored in the orientation its matmul wants, because the cost of a cache is
 * what you pay to APPEND to it:
 *
 *   V   [T][D] int4.  `P @ V` contracts over keys, so its weight is V[s][h] —
 *       exactly how V leaves its projection. Appending is one `quant4` writing
 *       one 32-byte row. Free.
 *   K   [D][T] int8.  `Q @ K^T`'s weight is K^T[h][s], so the cache is
 *       column-major and appending a token writes one byte into each of D rows.
 *       That scatter is the transposing DMA, which is one command for any
 *       number of rows.
 *
 * The K cache stays int8 and is re-packed to int4 whole (into SP_KT_INT4) once
 * per layer per step. That looks wasteful — 2048 elements packed to use at most
 * first_pos+1 columns — but it is 4 VPU commands against the ~24 000 DMA clocks
 * the same layer spends fetching weights, and the alternative cannot be done:
 * the nibble for (d, t) sits in the middle of the byte at row d, and no op
 * writes half a byte.
 *
 * NOTHING IS ZEROED, and nothing needs to be. Cache columns past the current
 * position hold whatever the last problem left there and reach S as garbage —
 * but S is int4 and the mask is -8, so a masked score is at most -1 whatever
 * the garbage was, and ReLU takes it to exactly zero. The mask that makes
 * attention causal is the same mask that makes an uninitialized cache safe.
 *
 * ---- what the host stages ---------------------------------------------------
 *
 *   DR_TOKENS[0 .. PROMPT-1]   the prompt, as int32 token ids
 *   DR_EMBED                   the embedding table, already quantized onto s_x0
 *   DR_MASK, DR_HEAD_WGT, the six weight blocks per layer   as adder.c
 *
 * and reads back DR_TOKENS[PROMPT .. T-1] (the generated ids) and, to check the
 * arithmetic rather than the answer, DR_LOGITS.
 *
 * The requant table is the same 16 {m0,n} words per layer as adder.c, from the
 * same header: this is the same arithmetic in a different order, so a
 * checkpoint exported for one runs the other unchanged.
 */
#include "tpulib.h"

#ifdef ADDER_RQ_H
#include ADDER_RQ_H
#else
#include "adder_rq.h"
#endif

#define T         32            /* tokens (train.py --max_tokens) */
#define D         64            /* model width */
#define DFF       256           /* feed-forward width */
#define HEADS     4             /* q_heads == kv_heads */
#define HEAD_DIM  (D / HEADS)   /* 16 */
#define LAYERS    4
#define VOCAB     13
#define VOCAB_PAD 16            /* the 13 logits, padded to a whole array tile */

/* The prompt length is a compile-time constant because the dataset makes it
 * one: numbers_data.EQUALS_POS = 15 is the index of the first ANSWER digit and
 * operands are padded so that holds for every problem, so '=' is at 14 and the
 * prompt is exactly 15 tokens. A model with varying prompts would pass this in
 * and lose the constant folding in the prefill block only. */
#ifndef PROMPT
#define PROMPT 15
#endif

/* Tokens to generate: one from the prefill, then a decode step each. T - PROMPT
 * fills the sequence; lower it for a quick regression, since an RTL decode step
 * costs ~150 k clocks. */
#ifndef INFER_GEN
#define INFER_GEN (T - PROMPT)
#endif

_Static_assert(PROMPT + INFER_GEN <= T, "generation runs past the sequence");
_Static_assert(PROMPT >= 1, "the prefill needs at least one token");

/* Requant sites, in block order — identical to adder.c's enum: same sites, same
 * order, same header. */
enum {
    RQ_Q, RQ_K, RQ_V,           /* the three projections            */
    RQ_KP, RQ_VP,               /* quant4 packs — {1,0}, both int4  */
    RQ_S,                       /* Q@K^T, with 1/sqrt(head_dim)     */
    RQ_ID,                      /* mask clamp — {1,0}               */
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

/* Bytes in one row of a row-major int4 weight, i.e. two nibbles per byte. */
#define WGT_ROW(cols) ((cols) / 2)

/* ---- DRAM ----------------------------------------------------------------
 * Everything this kernel shares with adder.c is at the same address, so one
 * host stages either. What used to be the embedded X0 is now the embedding
 * table plus the token sequence. */
#define DR_EMBED        0x00000u  /* [VOCAB][D] int8 — the embedding, host   */
#define DR_TOKENS       0x00400u  /* [T] int32 — prompt in, generated out    */
#define DR_MASK         0x00800u  /* [T][T]  int8  — causal mask, host       */
#define DR_KT_SCRATCH   0x00C00u  /* [D][T]  int8  — cache-append staging    */
#define DR_HEAD_WGT     0x01400u  /* [D][16] int4  — output head, host       */
#define DR_LOGITS       0x01800u  /* [T][16] int32 — per-position logits, out*/
#define DR_LAYER0       0x02000u  /* layer 0's weight block...               */
#define DR_LAYER_STRIDE 0x06000u  /* ...and the stride between layers        */

#define LW_WQ  0x0000u          /* [D][D]   int4, 2048 B */
#define LW_WK  0x0800u
#define LW_WV  0x1000u
#define LW_WO  0x1800u
#define LW_FF1 0x2000u          /* [D][DFF] int4, 8192 B */
#define LW_FF2 0x4000u          /* [DFF][D] int4, 8192 B */

/* ---- scratchpad (64 KB) --------------------------------------------------
 * The arena, then the KV cache (the only tensors that live across a step), then
 * one block's worth of everything else. BLOCK_ROWS is the widest any activation
 * has to be: the prefill's, since a decode step is a single row. */
#define BLOCK_ROWS PROMPT

#define SP_ARENA       0x0000u  /* 8192 — the largest weight block is 8 KB */
#define SP_ARENA_BYTES 0x2000u

#define SP_MASK      0x2000u    /* [T][T] int8                      1024 B  */
#define SP_K_CACHE   0x2400u    /* LAYERS x [D][T] int8   K^T       2048 B ea*/
#define SP_V_CACHE   0x4400u    /* LAYERS x [T][D] int4   V packed  1024 B ea*/
#define SP_KT_INT4   0x5400u    /* [D][T] int4 — the K cache, packed 1024 B  */

#define SP_X         0x5800u    /* [BLOCK_ROWS][D] int8 — residual stream   */
#define SP_Q         0x5C00u    /* [BLOCK_ROWS][D] int8                     */
#define SP_K_NEW     0x6000u    /* [BLOCK_ROWS][D] int8 — this block's K    */
#define SP_V_NEW     0x6400u    /* [BLOCK_ROWS][D] int8 — this block's V    */
#define SP_S         0x6800u    /* [BLOCK_ROWS][T] int8 — raw scores        */
#define SP_S_MASKED  0x6A00u    /* [BLOCK_ROWS][T] int8 — S + causal mask   */
#define SP_P         0x6C00u    /* [BLOCK_ROWS][T] int8 — relu of that      */
#define SP_A         0x6E00u    /* [BLOCK_ROWS][D] int8 — attention output  */
#define SP_O         0x7200u    /* [BLOCK_ROWS][D] int8 — A @ Wo            */
#define SP_X_PLUS_O  0x7600u    /* [BLOCK_ROWS][D] int8                     */
#define SP_X1        0x7A00u    /* [BLOCK_ROWS][D] int8 — after DyT         */
#define SP_H         0x7E00u    /* [BLOCK_ROWS][DFF] int8                   */
#define SP_H_RELU    0x8E00u    /* [BLOCK_ROWS][DFF] int8                   */
#define SP_FFN_OUT   0x9E00u    /* [BLOCK_ROWS][D] int8                     */
#define SP_LOGITS    0xA200u    /* [VOCAB_PAD] int32 — the CPU reads these  */
#define SP_TOKENS    0xA240u    /* [T] int32 — the sequence, prompt included*/
#define SP_END       0xA2C0u

/* The map above is hand-packed, so check it: a tensor that outgrew its slot
 * would otherwise show up as a wrong answer several blocks later. */
_Static_assert(SP_K_CACHE + LAYERS * (D * T) <= SP_V_CACHE,
               "K cache overruns the V cache");
_Static_assert(SP_V_CACHE + LAYERS * (T * D / 2) <= SP_KT_INT4,
               "V cache overruns SP_KT_INT4");
_Static_assert(SP_KT_INT4 + (D * T / 2) <= SP_X, "SP_KT_INT4 overruns SP_X");
_Static_assert(BLOCK_ROWS * D <= SP_Q - SP_X,
               "a [BLOCK_ROWS][D] activation does not fit");
_Static_assert(BLOCK_ROWS * T <= SP_S_MASKED - SP_S,
               "a [BLOCK_ROWS][T] activation does not fit");
_Static_assert(BLOCK_ROWS * DFF <= SP_H_RELU - SP_H,
               "a [BLOCK_ROWS][DFF] activation does not fit");
_Static_assert(SP_LOGITS + VOCAB_PAD * 4 <= SP_TOKENS,
               "the logits overrun SP_TOKENS");
_Static_assert(SP_TOKENS + T * 4 <= SP_END, "the token block overruns the map");
_Static_assert(SP_END <= 0x10000u, "the scratchpad is 64 KB");

#define K_CACHE(layer) (SP_K_CACHE + (layer) * (D * T))
#define V_CACHE(layer) (SP_V_CACHE + (layer) * (T * D / 2))

static tpu_arena arena;

/* ---- one block of new tokens ---------------------------------------------
 *
 * X at SP_X holds the input embeddings for positions first_pos ..
 * first_pos+rows-1; on return it holds the residual stream after all four
 * layers, and the KV cache has `rows` more columns/rows in it.
 *
 * `always_inline` with `rows` a literal at both call sites is what keeps this
 * affordable: it folds tpu_matmul's block chooser, block loops and staging
 * branches at all eight matmul sites, specializing the two calls to the
 * prefill's shape and the decode's. `first_pos` stays a runtime value — the
 * library folds on SHAPE, and an address costs nothing to compute. */
__attribute__((always_inline))
static inline void infer_block(unsigned rows, unsigned first_pos)
{
    for (unsigned layer = 0; layer < LAYERS; layer++) {
        const uint16_t *rq = rq_table[layer];
        const uint32_t layer_wgt = DR_LAYER0 + layer * DR_LAYER_STRIDE;
        const uint32_t k_cache = K_CACHE(layer);
        const uint32_t v_cache = V_CACHE(layer);

        /* ---- Q, K, V ----
         * Three [D][D] projections off the same X, separate because each weight
         * has its own scale and so its own {m0,n}. */
        tpu_matmul(&(const tpu_gemm){
            .rows = rows, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_X, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WQ, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_Q, D),
            .rq_word = rq[RQ_Q] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_X, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WK, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_K_NEW, D),
            .rq_word = rq[RQ_K] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_X, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WV, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_V_NEW, D),
            .rq_word = rq[RQ_V] }, &arena);

        /* ---- append to the K cache ----
         * The new [rows][D] K becomes columns first_pos.. of the [D][T] cache.
         * The DMA is the only unit that transposes and one of its sides is
         * always DRAM, so it is a linear spill followed by a transposing fill:
         * the fill reads DRAM row-major over D columns and writes
         * k_cache + d*T + row — one command for any number of rows. It moves
         * bytes, which is why the cache is int8 and the pack comes after. */
        tpu_move(SP_K_NEW, DR_KT_SCRATCH, rows * D, TPU_DMA_SPILL);
        tpu_dma_transpose(k_cache + first_pos, DR_KT_SCRATCH, rows * D,
                          TPU_DMA_FILL, D, D, T);
        tpu_wait(TPU_U_DMA);

        /* ---- append to the V cache ----
         * V leaves its projection in exactly the orientation `P @ V` wants, so
         * the append IS the pack: `rows` rows of D nibbles, starting at row
         * first_pos. Both packs here are the {1,0} identity — K and V are
         * already int4, whatever requant produced them clipped to [-8, 7]. */
        tpu_pack4(TPU_SPAD_AT(v_cache + first_pos * (D / 2)),
                  TPU_SPAD_AT(SP_V_NEW), rows * D, rq[RQ_VP], &arena);

        /* The Q@K^T weight operand: the whole cache, packed. Columns past the
         * current position are garbage and stay garbage — the mask below
         * deletes them. */
        tpu_pack4(TPU_SPAD_AT(SP_KT_INT4), TPU_SPAD_AT(k_cache), D * T,
                  rq[RQ_KP], &arena);

        for (unsigned head = 0; head < HEADS; head++) {
            /* S = requant(Q_head @ K_head^T) over the WHOLE cache width. A head
             * is a column slice of Q and a row slice of K^T, both addressed in
             * place. Contracting over all T keys rather than first_pos+rows of
             * them costs a few tiles and keeps the shape constant, which is
             * worth more than the tiles: a runtime column count would unfold
             * the block loop. */
            tpu_matmul(&(const tpu_gemm){
                .rows = rows, .depth = HEAD_DIM, .cols = T,
                .act = TPU_SPAD_ROWS(SP_Q + head * HEAD_DIM, D),
                .wgt = TPU_SPAD_ROWS(SP_KT_INT4 + head * HEAD_DIM * WGT_ROW(T),
                                     WGT_ROW(T)),
                .out = TPU_SPAD_ROWS(SP_S, T),
                .rq_word = rq[RQ_S] }, &arena);

            /* P = requant(relu(requant(S + mask))), over mask rows
             * first_pos.. . The mask is 0 or -8 and S is int4, so a masked
             * entry is at most -1 whatever s_s is and ReLU takes it to exactly
             * zero. That is what makes both causality and the uninitialized
             * tail of the cache exact rather than approximate. */
            tpu_add_narrow(TPU_V_REQUANT, TPU_SPAD_AT(SP_S_MASKED),
                           TPU_SPAD_AT(SP_S),
                           TPU_SPAD_AT(SP_MASK + first_pos * T), rows * T,
                           rq[RQ_ID], &arena);
            tpu_relu_narrow(TPU_SPAD_AT(SP_P), TPU_SPAD_AT(SP_S_MASKED),
                            rows * T, rq[RQ_P], &arena);

            /* A_head = requant(P @ V_head), into A's column block. The
             * contraction is the full T: cache rows past the current position
             * are multiplied by a P of exactly zero. */
            tpu_matmul(&(const tpu_gemm){
                .rows = rows, .depth = T, .cols = HEAD_DIM,
                .act = TPU_SPAD_ROWS(SP_P, T),
                .wgt = TPU_SPAD_ROWS(v_cache + head * HEAD_DIM / 2,
                                     WGT_ROW(D)),
                .out = TPU_SPAD_ROWS(SP_A + head * HEAD_DIM, D),
                .rq_word = rq[RQ_A] }, &arena);
        }

        /* ---- O = requant(A @ Wo), pinned to s_x by the residual ---- */
        tpu_matmul(&(const tpu_gemm){
            .rows = rows, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_A, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WO, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_O, D),
            .rq_word = rq[RQ_O] }, &arena);

        /* ---- the double residual, then DyT ----
         * MultiHeadAttention.forward ends in `O + X` and Transformer.forward
         * adds X again, so this is 2X + O in two adds. */
        tpu_add_narrow(TPU_V_REQUANT, TPU_SPAD_AT(SP_X_PLUS_O),
                       TPU_SPAD_AT(SP_X), TPU_SPAD_AT(SP_O), rows * D,
                       rq[RQ_XO], &arena);
        tpu_add_narrow(TPU_V_DYT, TPU_SPAD_AT(SP_X1),
                       TPU_SPAD_AT(SP_X_PLUS_O), TPU_SPAD_AT(SP_X), rows * D,
                       rq[RQ_X1], &arena);

        /* ---- the feed-forward block ---- */
        tpu_matmul(&(const tpu_gemm){
            .rows = rows, .depth = D, .cols = DFF,
            .act = TPU_SPAD_ROWS(SP_X1, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_FF1, WGT_ROW(DFF)),
            .out = TPU_SPAD_ROWS(SP_H, DFF),
            .rq_word = rq[RQ_H] }, &arena);
        tpu_relu_narrow(TPU_SPAD_AT(SP_H_RELU), TPU_SPAD_AT(SP_H), rows * DFF,
                        rq[RQ_HR], &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows, .depth = DFF, .cols = D,
            .act = TPU_SPAD_ROWS(SP_H_RELU, DFF),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_FF2, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_FFN_OUT, D),
            .rq_word = rq[RQ_F] }, &arena);

        tpu_add_narrow(TPU_V_DYT, TPU_SPAD_AT(SP_X), TPU_SPAD_AT(SP_X1),
                       TPU_SPAD_AT(SP_FFN_OUT), rows * D, rq[RQ_X2], &arena);
    }
}

/* Row `token` of the embedding table into row `dst_row` of X. The ISA has no
 * gather; a DMA whose DRAM address the CPU computed is one. */
static void embed(unsigned dst_row, unsigned token)
{
    tpu_move(SP_X + dst_row * D, DR_EMBED + token * D, D, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);
}

/* ---- the head, the argmax, and the next token -----------------------------
 *
 * `x_row_addr` is the row of X that position `pos` ended up in; the token this
 * returns is the one at pos+1.
 *
 * The logits are never requantized — an argmax does not care about scale — so
 * the raw int32 accumulator is what the array stores. It stores it to the
 * SCRATCHPAD rather than to DRAM, which is the whole difference between this
 * kernel and adder.c: the CPU can read the scratchpad, so the comparison
 * happens here instead of on the host. The DRAM copy is for the host to check
 * against PyTorch; nothing on the device reads it back. */
static unsigned head_argmax(uint32_t x_row_addr, unsigned pos)
{
    unsigned best_token = 0;
    int32_t best_logit;

    tpu_matmul(&(const tpu_gemm){
        .rows = 1, .depth = D, .cols = VOCAB_PAD,
        .act = TPU_SPAD_ROWS(x_row_addr, D),
        .wgt = TPU_DRAM_ROWS(DR_HEAD_WGT, WGT_ROW(VOCAB_PAD)),
        .out = TPU_SPAD_ROWS(SP_LOGITS, VOCAB_PAD * 4),
        .rq_word = 0u }, &arena);   /* 0 = store int32, do not narrow */

    tpu_move(SP_LOGITS, DR_LOGITS + pos * (VOCAB_PAD * 4), VOCAB_PAD * 4,
             TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    /* Strictly greater, so a tie takes the lowest id — torch.argmax's rule, and
     * the only place the two could disagree on a checkpoint whose logits tie. */
    best_logit = (int32_t)tpu_spad_ld(SP_LOGITS);
    for (unsigned token = 1; token < VOCAB; token++) {
        int32_t logit = (int32_t)tpu_spad_ld(SP_LOGITS + token * 4);

        if (logit > best_logit) {
            best_logit = logit;
            best_token = token;
        }
    }

    tpu_spad_st(SP_TOKENS + (pos + 1) * 4, best_token);
    return best_token;
}

int main(void)
{
    unsigned next_token;

    tpu_arena_init(&arena, SP_ARENA, SP_ARENA_BYTES);

    /* The mask is read by every layer of every step. The prompt is read once,
     * by the CPU — which has no path to DRAM at all, so a DMA into the
     * scratchpad and a load through the window is how a host-written NUMBER
     * reaches it. */
    tpu_move(SP_MASK, DR_MASK, T * T, TPU_DMA_FILL);
    tpu_move(SP_TOKENS, DR_TOKENS, PROMPT * 4, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    /* ---- prefill: PROMPT tokens in one block ---- */
    for (unsigned pos = 0; pos < PROMPT; pos++)
        embed(pos, tpu_spad_ld(SP_TOKENS + pos * 4) & 0xFFu);

    infer_block(PROMPT, 0);
    next_token = head_argmax(SP_X + (PROMPT - 1) * D, PROMPT - 1);

    /* ---- decode: one token per step ----
     * Step `pos` embeds the token the previous step chose — the one at position
     * pos — into row 0 of X, runs it through every layer against the cache, and
     * produces the token at pos+1. The prefill already produced the token at
     * PROMPT, so this runs INFER_GEN-1 times. */
    for (unsigned pos = PROMPT; pos < PROMPT + INFER_GEN - 1; pos++) {
        embed(0, next_token);
        infer_block(1, pos);
        next_token = head_argmax(SP_X, pos);
    }

    /* Only the generated ids: the prompt end of the block is what the host
     * wrote, and spilling it back would put the host's own bytes in the golden
     * output image. */
    tpu_move(SP_TOKENS + PROMPT * 4, DR_TOKENS + PROMPT * 4, INFER_GEN * 4,
             TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                       /* start.S raises `done` from here */
}
