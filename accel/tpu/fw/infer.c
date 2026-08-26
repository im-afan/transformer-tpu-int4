/* infer.c — the int4 adder model as inference: prefill, then decode against a
 * KV cache. Token ids in, token ids out, one run.
 *
 * `adder.c` is the training shape — every position at once over a sequence
 * somebody already knows. This is the generative one. Both are the same code:
 * `infer_block(rows, first_pos)` runs `rows` new positions per sequence, so the
 * row count is the only difference between them.
 *
 * Shape: `model/transformer.py::adder_int4_wide` — d=128, f=512, 4 layers, 4
 * heads of 32, int4 weights and activations, no bias, no LayerNorm, no
 * positional encoding, T=64 (a 32-token prompt and the answer after it).
 *
 * ---- phases, for benchmarking ----------------------------------------------
 *
 * INFER_PREFILL and INFER_DECODE select which half of a generation this image
 * runs; both default to 1, which is the whole thing and is what the accuracy
 * path builds. `make PROG=infer PHASE=prefill|decode|both` sets them.
 *
 * The counters cannot be read mid-run — they reset at `G` and freeze at the
 * halt — so an image that runs one half IS the measurement of that half, for
 * every counter rather than only the total. The decode-only image starts from
 * the prompt's last token instead of the one the prefill would have produced:
 * a step's cost is not data-dependent, so the clocks are the same and the
 * tokens it emits are not scored.
 *
 * ---- batching --------------------------------------------------------------
 *
 * BATCH independent sequences share every weight stream. X is [BATCH][rows][D]
 * — sequence-major — so the three projections, Wo and both FFN matmuls run once
 * over BATCH*rows rows and the weight is staged once for all of them. Attention
 * stays per sequence, because each one has its own KV cache.
 *
 * That is the whole point at decode: a step is one row of arithmetic against
 * ~390 KB of weights, so BATCH rows cost the same DMA as one. What it costs is
 * DRAM — the cache is 48 KB per sequence at this shape — and DR_END's assert is
 * what says whether a given BATCH still fits under the weights.
 *
 * ---- where tensors live ----------------------------------------------------
 *
 * Every tensor's home is DRAM. The scratchpad holds the staging arena and a
 * mailbox for the values the CPU has to touch. On top of that the kernel
 * promotes some tensors to a scratchpad copy — a performance choice with a DRAM
 * fallback, not a requirement. Drop every promotion and this computes the same
 * bytes, slower, which is why nothing here asserts a tensor fits.
 *
 * Rows block at TPU_TOKENS_MAX, the array's dispatch limit, so the prefill is
 * one pass per sequence at this prompt length and each weight streams once.
 *
 * ---- the KV cache ----------------------------------------------------------
 *
 * Each half is stored in the orientation its matmul wants, because a cache
 * costs what you pay to append to it. V is [T][D] int4 — exactly how V leaves
 * its projection, so the append IS the pack. K is [D][T] int8, column-major,
 * so appending writes one byte into each of D rows: that scatter is the
 * transposing DMA, one command for any number of rows. K stays int8 and is
 * re-packed to int4 whole each layer-step because the nibble for (d, t) sits in
 * the middle of a byte and no op writes half a byte.
 *
 * NOTHING IS ZEROED. Cache columns past the current position are garbage and
 * reach S as garbage, but S is int4 and the mask is -8, so a masked score is at
 * most -1 and ReLU takes it to exactly zero. The mask that makes attention
 * causal is what makes an uninitialized cache safe.
 *
 * ---- the host --------------------------------------------------------------
 *
 * Stages DR_TOKENS (the prompt ids), DR_EMBED, DR_MASK, DR_HEAD_WGT and the six
 * weight blocks per layer; presses 'G'; reads DR_TOKENS[seq][PROMPT..] back.
 * The argmax and the embedding gather are on the device — cpu_subsys.sv maps
 * the scratchpad at 0x9xxx_xxxx, and a DMA takes an address the CPU computed.
 *
 * The weight blocks are at adder.c's addresses so one staging pass feeds either
 * kernel; everything below them is computed off the shape AND off BATCH, so a
 * batched image moves the mask, the head and the caches — `infer_export.layout`
 * walks the same chain. The requant table is not shared — adder.c runs T=128
 * and its shifts are set by its own contractions.
 */
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

/* A prompt longer than one BLOCK costs a weight stream per BLOCK rows, which is
 * what keeping a pass's intermediates local is worth paying. */
#ifndef BLOCK
#define BLOCK TPU_TOKENS_MAX
#endif

#define PREFILL_PASSES  (PROMPT / BLOCK)
#define PREFILL_TAIL    (PROMPT % BLOCK)
#define LAST_PASS_ROWS  (PREFILL_TAIL ? PREFILL_TAIL : BLOCK)

/* The prefill produces the token at PROMPT, so a generation of INFER_GEN tokens
 * is that one plus INFER_GEN-1 decode steps. The decode-only image runs those
 * same INFER_GEN-1 steps, which is what makes its clocks the prefill-only
 * image's complement rather than something to be scaled. */
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
_Static_assert(BLOCK <= TPU_TOKENS_MAX,
               "a dispatch takes at most TPU_TOKENS_MAX rows");
_Static_assert(BLOCK * D <= TPU_DMA_BYTES_MAX,
               "the K append is one transposing DMA and would not fit in one");

/* Requant sites, in block order — adder.c's enum, same sites and same header. */
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

#define I4_ROW(cols) ((cols) / 2)       /* bytes in a row-major int4 row */

/* ---- DRAM (512 KB) --------------------------------------------------------
 * A computed chain, so a shape change re-lays it out and DR_END's assert says
 * whether the result still fits. Three rotating temporaries are the whole
 * activation working set — an intermediate dies as soon as its consumer has
 * read it, so the roles are:
 *
 *   DR_TMP_A  K_new -> A -> X+O -> F
 *   DR_TMP_B  V_new -> O -> X1
 *
 * and DR_SCRATCH holds whichever phase of the layer is running: Q, KT and S
 * while attention does, H once the FFN starts.
 *
 * Anything per-sequence is indexed [seq] and anything per-row is [seq][row],
 * sequence-major, so a matmul over BATCH*rows rows walks one dense tensor.
 */
#define DR_ALIGN(a) (((a) + 63u) & ~63u)

#define DR_EMBED    0x00000u                                          /* host      */
#define DR_TOKENS   DR_ALIGN(DR_EMBED   + VOCAB * D)                  /* host, out */
#define DR_LOGITS   DR_ALIGN(DR_TOKENS  + BATCH * T * 4u)             /* out       */
#define DR_MASK     DR_ALIGN(DR_LOGITS  + BATCH * T * VOCAB_PAD * 4u) /* host      */
#define DR_HEAD_WGT DR_ALIGN(DR_MASK    + T * T)                      /* host      */
#define DR_K_CACHE  DR_ALIGN(DR_HEAD_WGT + D * I4_ROW(VOCAB_PAD))
#define DR_V_CACHE  DR_ALIGN(DR_K_CACHE + BATCH * LAYERS * D * T)
#define DR_X        DR_ALIGN(DR_V_CACHE + BATCH * LAYERS * T * I4_ROW(D))
#define DR_TMP_A    DR_ALIGN(DR_X       + ROWS_MAX * D)
#define DR_TMP_B    DR_ALIGN(DR_TMP_A   + ROWS_MAX * D)

/* Attention's working set and the FFN's hidden layer never coexist, so they
 * are one region and the layer costs the larger of the two rather than the sum.
 *
 * Q, KT and S are all read for the last time by the per-head loop, and A@Wo is
 * the dispatch that ends it; H is not written until X1 exists, two residual
 * adds later, and is dead again at HR@W2 — before the next layer's Q
 * projection. Nothing else in the layer reads either group.
 *
 * This is the only aliasing in the map. X, TMP_A and TMP_B are live across the
 * whole layer (TMP_A is K_new, then A, then X+O, then F; TMP_B is V_new, then
 * O, then X1, which the last add still needs), so none of them can share. */
#define DR_SCRATCH  DR_ALIGN(DR_TMP_B   + ROWS_MAX * D)
#define DR_Q        DR_SCRATCH
#define DR_KT       DR_ALIGN(DR_Q       + ROWS_MAX * D)
#define DR_S        DR_ALIGN(DR_KT      + D * I4_ROW(T))
#define DR_ATTN_END DR_ALIGN(DR_S       + MAX_SEQ_ROWS * T)
#define DR_H        DR_SCRATCH
#define DR_FFN_END  DR_ALIGN(DR_H       + ROWS_MAX * DFF)
#define DR_END      ((DR_ATTN_END > DR_FFN_END) ? DR_ATTN_END : DR_FFN_END)

#define DR_LAYER0        0x20000u       /* adder.c's weight map, byte for byte */
#define DR_LAYER_STRIDE  0x18000u
#define LW_WQ  0x00000u                 /* [D][D]   int4 */
#define LW_WK  0x02000u
#define LW_WV  0x04000u
#define LW_WO  0x06000u
#define LW_FF1 0x08000u                 /* [D][DFF] int4 */
#define LW_FF2 0x10000u                 /* [DFF][D] int4 */

#define K_CACHE(seq, layer) \
    (DR_K_CACHE + ((seq) * LAYERS + (layer)) * (D * T))
#define V_CACHE(seq, layer) \
    (DR_V_CACHE + ((seq) * LAYERS + (layer)) * (T * I4_ROW(D)))

/* Two levers when this fires, in this order: lower BLOCK (the per-sequence row
 * count, which scales X, TMP_A, TMP_B and the scratch union and costs one
 * weight stream per extra pass), then lower BATCH. The KV cache is 48 KB per
 * sequence here and no aliasing reaches it, so BATCH is the hard wall. */
_Static_assert(DR_END <= DR_LAYER0,
               "the activation map runs into layer 0's weights: lower BLOCK, "
               "then BATCH");
_Static_assert(DR_LAYER0 + LAYERS * DR_LAYER_STRIDE <= 0x80000u,
               "the weights overrun the 512 KB SRAM");

/* ---- scratchpad (64 KB) ---------------------------------------------------
 * The CPU has no path to DRAM, so what it reads (the prompt ids) and writes
 * (the token it chose) lives in a fixed mailbox the scratchpad window reaches.
 * That is the only fixed allocation here; the rest is arena. */
#define SP_BYTES    0x10000u
#define SP_MAILBOX  (SP_BYTES - (VOCAB_PAD * 4u + BATCH * T * 4u))
#define SP_LOGITS   SP_MAILBOX                   /* [VOCAB_PAD] int32 */
#define SP_TOKENS   (SP_LOGITS + VOCAB_PAD * 4u) /* [BATCH][T] int32  */

#define SP_TOKEN_AT(seq, pos) (SP_TOKENS + ((seq) * T + (pos)) * 4u)

/* ---- residency ------------------------------------------------------------
 * Tensors the kernel also keeps on chip, in falling order of DMA saved per byte
 * of arena spent. The cascade stops when a promotion would leave less than
 * SP_STAGE_RESERVE for the primitives to stage into, so a wider model runs more
 * of itself out of DRAM.
 *
 * The reserve is the tuning knob. Measured on the ISS at this shape (prefill
 * plus one decode step), the default beats a reserve big enough to keep every
 * weight fill a single dense command: 2743 commands and ~976 k DMA clocks
 * against 7179 and ~1 447 k. H's three DRAM passes cost more than the 128
 * column-split fill commands W1 takes when H stays local. */
#ifndef SP_STAGE_RESERVE
#define SP_STAGE_RESERVE (D * I4_ROW(D) + TPU_CHUNK * 7u)
#endif

#define SZ_S     (MAX_SEQ_ROWS * T)
#define SZ_TMP_A (ROWS_MAX * D)
#define SZ_TMP_B (ROWS_MAX * D)
#define SZ_X     (ROWS_MAX * D)
#define SZ_H     (ROWS_MAX * DFF)
#define SZ_KT    (D * I4_ROW(T))
#define SZ_Q     (ROWS_MAX * D)
#define SZ_V4    (T * I4_ROW(D))        /* one sequence's V cache, one layer */

// #define PROMO_FITS(end) ((end) + SP_STAGE_RESERVE <= SP_MAILBOX)
#define PROMO_FITS(end) false

#define SP_S        0u
#define PROMO_S     PROMO_FITS(SP_S + SZ_S)
#define SP_TMP_A    (SP_S + (PROMO_S ? SZ_S : 0u))
#define PROMO_TMP_A PROMO_FITS(SP_TMP_A + SZ_TMP_A)
#define SP_TMP_B    (SP_TMP_A + (PROMO_TMP_A ? SZ_TMP_A : 0u))
#define PROMO_TMP_B PROMO_FITS(SP_TMP_B + SZ_TMP_B)
#define SP_X        (SP_TMP_B + (PROMO_TMP_B ? SZ_TMP_B : 0u))
#define PROMO_X     PROMO_FITS(SP_X + SZ_X)
#define SP_H        (SP_X + (PROMO_X ? SZ_X : 0u))
#define PROMO_H     PROMO_FITS(SP_H + SZ_H)
#define SP_KT       (SP_H + (PROMO_H ? SZ_H : 0u))
#define PROMO_KT    PROMO_FITS(SP_KT + SZ_KT)
#define SP_Q        (SP_KT + (PROMO_KT ? SZ_KT : 0u))
#define PROMO_Q     PROMO_FITS(SP_Q + SZ_Q)
#define SP_V4       (SP_Q + (PROMO_Q ? SZ_Q : 0u))
#define PROMO_V4    PROMO_FITS(SP_V4 + SZ_V4)
#define SP_ARENA    (SP_V4 + (PROMO_V4 ? SZ_V4 : 0u))

#define SP_ARENA_BYTES (SP_MAILBOX - SP_ARENA)

_Static_assert(SP_ARENA_BYTES >= SP_STAGE_RESERVE,
               "the cascade promoted past its own reserve");

/* The conditions are compile-time, so tpu_matmul still folds its staging
 * branches away at every call site (tpulib.h's INLINING note). */
#define B_S     (PROMO_S     ? TPU_SPAD_ROWS(SP_S, T)      : TPU_DRAM_ROWS(DR_S, T))
#define B_TMP_A (PROMO_TMP_A ? TPU_SPAD_ROWS(SP_TMP_A, D)  : TPU_DRAM_ROWS(DR_TMP_A, D))
#define B_TMP_B (PROMO_TMP_B ? TPU_SPAD_ROWS(SP_TMP_B, D)  : TPU_DRAM_ROWS(DR_TMP_B, D))
#define B_X     (PROMO_X     ? TPU_SPAD_ROWS(SP_X, D)      : TPU_DRAM_ROWS(DR_X, D))
#define B_H     (PROMO_H     ? TPU_SPAD_ROWS(SP_H, DFF)    : TPU_DRAM_ROWS(DR_H, DFF))
#define B_KT    (PROMO_KT    ? TPU_SPAD_ROWS(SP_KT, I4_ROW(T)) \
                             : TPU_DRAM_ROWS(DR_KT, I4_ROW(T)))
#define B_Q     (PROMO_Q     ? TPU_SPAD_ROWS(SP_Q, D)      : TPU_DRAM_ROWS(DR_Q, D))

static tpu_arena arena;

/* X holds the embeddings for positions first_pos .. first_pos+rows-1 of each of
 * the BATCH sequences, sequence-major, and on return the residual stream after
 * all four layers.
 *
 * `rows` is rows PER SEQUENCE; the weight matmuls run over BATCH*rows of them
 * so one weight stream serves the whole batch. Attention does not batch — each
 * sequence attends over its own cache — so that loop is per sequence.
 *
 * `always_inline` with `rows` a literal at both call sites folds tpu_matmul's
 * block chooser and staging branches at all eight matmul sites, specializing
 * the prefill's shape and the decode's. `first_pos` stays a runtime value. */
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
            .act = B_X,
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WQ, I4_ROW(D)),
            .out = B_Q,
            .rq_word = rq[RQ_Q] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = D,
            .act = B_X,
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WK, I4_ROW(D)),
            .out = B_TMP_A,                     /* K_new */
            .rq_word = rq[RQ_K] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = D,
            .act = B_X,
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WV, I4_ROW(D)),
            .out = B_TMP_B,                     /* V_new */
            .rq_word = rq[RQ_V] }, &arena);

        /* Per sequence: append to its cache, then attend over it. Sequence
         * `seq`'s scores overwrite its own rows of K_new with A, which is safe
         * because a later sequence's K_new lives in the rows below. */
        for (unsigned seq = 0; seq < BATCH; seq++) {
            const uint32_t k_cache = K_CACHE(seq, layer);
            const uint32_t v_cache = V_CACHE(seq, layer);
            const uint32_t seq_off = seq * rows * D;  /* into an X-shaped tensor */
            /* Not promoted, this IS the DRAM cache — nothing to copy in or out. */
            const tpu_buf b_v4 = PROMO_V4
                ? TPU_SPAD_ROWS(SP_V4, I4_ROW(D))
                : TPU_DRAM_ROWS(v_cache, I4_ROW(D));

            /* K_new becomes columns first_pos.. of the [D][T] cache. The DMA
             * always has DRAM on one side, so a DRAM K_new has to bounce
             * through the arena instead of spilling straight out. */
#if PROMO_TMP_A
            tpu_dma_transpose(SP_TMP_A + seq_off, k_cache + first_pos, rows * D,
                              TPU_DMA_SPILL, D, D, T);
            tpu_wait(TPU_U_DMA);
#else
            tpu_transpose_dram_int8(k_cache + first_pos, T, DR_TMP_A + seq_off,
                                    D, rows, D, rows, &arena);
#endif

            tpu_pack4(B_KT, TPU_DRAM_AT(k_cache), D * T, rq[RQ_KP], &arena);

            /* V's append is the pack. Both packs are the {1,0} identity: K and
             * V are already int4, whatever requant produced them clipped
             * them. */
#if PROMO_V4
            tpu_move(SP_V4, v_cache, T * I4_ROW(D), TPU_DMA_FILL);
            tpu_wait(TPU_U_DMA);
#endif
            tpu_pack4(tpu_buf_off(b_v4, first_pos * I4_ROW(D)),
                      tpu_buf_off(B_TMP_B, seq_off), rows * D, rq[RQ_VP],
                      &arena);
#if PROMO_V4
            tpu_move(SP_V4 + first_pos * I4_ROW(D),
                     v_cache + first_pos * I4_ROW(D), rows * I4_ROW(D),
                     TPU_DMA_SPILL);
            tpu_wait(TPU_U_DMA);
#endif

            for (unsigned head = 0; head < HEADS; head++) {
                /* Contracting over all T keys rather than first_pos+rows of
                 * them keeps the shape constant; a runtime column count would
                 * unfold the block loop, which costs more than the extra
                 * tiles. */
                tpu_matmul(&(const tpu_gemm){
                    .rows = rows, .depth = HEAD_DIM, .cols = T,
                    .act = tpu_buf_off(B_Q, seq_off + head * HEAD_DIM),
                    .wgt = tpu_buf_off(B_KT, head * HEAD_DIM * I4_ROW(T)),
                    .out = B_S,
                    .rq_word = rq[RQ_S] }, &arena);

                /* P = requant(relu(requant(S + mask))), both passes in place.
                 * Safe because a VPU pair reads its chunk into the arena's
                 * int32 temp before writing the same chunk of destination. */
                tpu_add_narrow(TPU_V_REQUANT, B_S, B_S,
                               TPU_DRAM_ROWS(DR_MASK + first_pos * T, T),
                               rows * T, rq[RQ_ID], &arena);
                tpu_relu_narrow(B_S, B_S, rows * T, rq[RQ_P], &arena);

                tpu_matmul(&(const tpu_gemm){
                    .rows = rows, .depth = T, .cols = HEAD_DIM,
                    .act = B_S,
                    .wgt = tpu_buf_off(b_v4, head * I4_ROW(HEAD_DIM)),
                    .out = tpu_buf_off(B_TMP_A, seq_off + head * HEAD_DIM),
                    .rq_word = rq[RQ_A] }, &arena);        /* A */
            }
        }

        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = D,
            .act = B_TMP_A,                     /* A */
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WO, I4_ROW(D)),
            .out = B_TMP_B,                     /* O; V_new is dead */
            .rq_word = rq[RQ_O] }, &arena);

        /* MultiHeadAttention.forward ends in `O + X` and Transformer.forward
         * adds X again, so this is 2X + O in two adds. */
        tpu_add_narrow(TPU_V_REQUANT, B_TMP_A, B_X, B_TMP_B, rows_all * D,
                       rq[RQ_XO], &arena);
        tpu_add_narrow(TPU_V_DYT, B_TMP_B, B_TMP_A, B_X, rows_all * D,
                       rq[RQ_X1], &arena);

        /* Both ask for the weight prefetch and only W2 can take it: the
         * prefetch splits the contraction, and a split contraction leaves
         * rows*cols int32 partials, which at DFF columns is more than the whole
         * scratchpad. W1 falls back to a column split. */
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = D, .cols = DFF,
            .act = B_TMP_B,                     /* X1 */
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_FF1, I4_ROW(DFF)),
            .out = B_H,
            .rq_word = rq[RQ_H], .prefetch = 1 }, &arena);
        tpu_relu_narrow(B_H, B_H, rows_all * DFF, rq[RQ_HR], &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = rows_all, .depth = DFF, .cols = D,
            .act = B_H,
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_FF2, I4_ROW(D)),
            .out = B_TMP_A,                     /* F; X+O is dead */
            .rq_word = rq[RQ_F], .prefetch = 1 }, &arena);

        tpu_add_narrow(TPU_V_DYT, B_X, B_TMP_B, B_TMP_A, rows_all * D,
                       rq[RQ_X2], &arena);
    }
}

/* The ISA has no gather; a DMA whose DRAM address the CPU computed is one. */
static void embed(unsigned dst_row, unsigned token)
{
    const uint32_t src = DR_EMBED + token * D;

#if PROMO_X
    tpu_move(SP_X + dst_row * D, src, D, TPU_DMA_FILL);
#else
    const uint32_t mark = arena.next_free;
    const uint32_t bounce = tpu_arena_alloc(&arena, D);

    /* No barrier between them: the DMA queue is in-order. */
    tpu_move(bounce, src, D, TPU_DMA_FILL);
    tpu_move(bounce, DR_X + dst_row * D, D, TPU_DMA_SPILL);
    arena.next_free = mark;
#endif
    tpu_wait(TPU_U_DMA);
}

/* Logits for the row of X at `x_row`; returns sequence `seq`'s token at pos+1.
 * They are never requantized — an argmax does not care about scale — and they
 * land in the mailbox so the CPU can read them. The DRAM copy is the host's, to
 * check against PyTorch. */
static unsigned head_argmax(unsigned x_row, unsigned seq, unsigned pos)
{
    unsigned best_token = 0;
    int32_t best_logit;

    tpu_matmul(&(const tpu_gemm){
        .rows = 1, .depth = D, .cols = VOCAB_PAD,
        .act = tpu_buf_off(B_X, x_row * D),
        .wgt = TPU_DRAM_ROWS(DR_HEAD_WGT, I4_ROW(VOCAB_PAD)),
        .out = TPU_SPAD_ROWS(SP_LOGITS, VOCAB_PAD * 4),
        .rq_word = 0u }, &arena);   /* 0 = store int32, do not narrow */

    tpu_move(SP_LOGITS, DR_LOGITS + (seq * T + pos) * (VOCAB_PAD * 4),
             VOCAB_PAD * 4, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    /* Strictly greater, so a tie takes the lowest id — torch.argmax's rule. */
    best_logit = (int32_t)tpu_spad_ld(SP_LOGITS);
    for (unsigned token = 1; token < VOCAB; token++) {
        int32_t logit = (int32_t)tpu_spad_ld(SP_LOGITS + token * 4);

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

    tpu_arena_init(&arena, SP_ARENA, SP_ARENA_BYTES);

    /* The CPU has no path to DRAM, so a host-written number reaches it as a DMA
     * into the mailbox and a load through the window. The prompt only: pulling
     * the answer field in would put the host's own bytes back in the output. */
    for (seq = 0; seq < BATCH; seq++)
        tpu_move(SP_TOKEN_AT(seq, 0), DR_TOKENS + seq * T * 4u, PROMPT * 4,
                 TPU_DMA_FILL);
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
        tpu_move(SP_TOKEN_AT(seq, TOK_FIRST),
                 DR_TOKENS + (seq * T + TOK_FIRST) * 4u, TOK_COUNT * 4u,
                 TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                       /* start.S raises `done` from here */
}
