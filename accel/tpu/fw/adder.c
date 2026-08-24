/* adder.c — the whole int4 adder model, in firmware, as one program.
 *
 * `model/transformer.py::adder_int4_vanilla` — d=64, f=256, layers=4,
 * q_heads=kv_heads=4, vocab=13, int4 weights AND int4 activations, no bias, no
 * LayerNorm, no positional encoding:
 *
 *   for layer in 0..3:
 *       Q, K, V = X@Wq, X@Wk, X@Wv
 *       S       = Q @ K^T / sqrt(head_dim)   per head, on the array
 *       P       = relu(S + causal_mask)      ReLU attention, NOT softmax
 *       A       = P @ V                      per head, on the array
 *       X       = dyt(X + (Wo(A) + X))       DOUBLE residual, then DyT
 *       X       = dyt(X + W2(relu(W1(X))))
 *   logits = X @ fc.w                        int32, never requantized
 *
 * The host owns the token embedding (the command ISA has no gather) and the
 * final argmax (nothing returns an index). X0 and the causal mask arrive in
 * DRAM; the logits leave in DRAM. `fw/infer.c` is the same model with both of
 * those moved onto the device, via the scratchpad window.
 *
 * RESIDENCY. Every matmul below is one `tpu_matmul` over a descriptor saying
 * where each operand lives; tpulib.h owns the block loops, the staging and the
 * barriers. Where a tensor lives is still this kernel's choice, because that is
 * the part that costs clocks:
 *
 *   activations  scratchpad-resident for the whole run (~42 KB), read and
 *                written several times per layer
 *   weights      DRAM, 24 KB per layer against an 8 KB arena, re-fetched every
 *                forward pass and staged block by block
 *   logits       DRAM, staged out by tpu_matmul itself — which is why this
 *                kernel needs no explicit spill anywhere
 *
 * WHY K IS TRANSPOSED AND V IS NOT. The array reads weights row-major, so for
 * out = act @ wgt the operand wgt[k][n] has row k contiguous over n.
 *
 *   Q @ K^T contracts over the head dim, so its weight is K^T[h][s] — K
 *           COLUMN-major. K leaves its projection row-major, hence the
 *           transpose (SP_KT_INT8), then the int4 pack (SP_KT_INT4).
 *   P @ V   contracts over keys, so its weight is V[s][h] — exactly how V
 *           leaves its projection. Free.
 *
 * The transpose runs on int8 because the DMA moves whole bytes and a packed
 * int4 nibble is half of one.
 *
 * THE REQUANT TABLE. Every tensor carries a compile-time scale: integer v means
 * real v*scale. The 16 {m0,n} words per layer are the only thing here that
 * depends on the checkpoint, and they are literals in the command, so there is
 * no path by which the device could read them out of memory — the table has to
 * be compiled in.
 *
 *   adder_rq.h            the checked-in default, tuned for the synthetic
 *                         operands accel/tpulang/fw_vectors.py stages, so
 *                         `make fw FWPROG=adder` needs no checkpoint
 *   -DADDER_RQ_H='"..."'  a table derived from a real checkpoint by
 *                         accel/tpulang/adder_export.py
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
/* The array stores a whole TPU_COLS-wide tile, so a 13-column head would be two
 * tiles with a 13-word row stride and the second tile would land on the next
 * token's row. Round the head's output up to a whole tile, stage the padding
 * columns as zero weights, and read 13 of every 16 words back. */
#define VOCAB_PAD 16

/* Requant sites, in block order — the index into one layer's ADDER_RQ row. Six
 * of the sixteen are the {1,0} identity by construction; they stay in the table
 * so it matches model/transformer.py's site list, and adder_export.py's
 * RQ_NAMES, one for one. */
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

/* ---- DRAM ---------------------------------------------------------------- */
#define DR_X            0x00000u  /* [T][D]  int8  — embedded input, host    */
#define DR_MASK         0x00800u  /* [T][T]  int8  — causal mask, host       */
#define DR_KT_SCRATCH   0x00C00u  /* [D][T]  int8  — transpose staging       */
#define DR_HEAD_WGT     0x01400u  /* [D][16] int4  — output head, host       */
#define DR_LOGITS       0x01800u  /* [T][16] int32 — the result, device      */
#define DR_LAYER0       0x02000u  /* layer 0's weight block...               */
#define DR_LAYER_STRIDE 0x06000u  /* ...and the stride between layers        */

/* Offsets within one layer's weight block. Wq/Wk/Wv are three separate dense
 * [D][D] blocks rather than one fused [D][3D]: tpu_matmul stages a weight block
 * itself, and a column slice of a fused block is strided, which would cost D
 * transfers instead of one. */
#define LW_WQ  0x0000u          /* [D][D]   int4, 2048 B */
#define LW_WK  0x0800u
#define LW_WV  0x1000u
#define LW_WO  0x1800u
#define LW_FF1 0x2000u          /* [D][DFF] int4, 8192 B */
#define LW_FF2 0x4000u          /* [DFF][D] int4, 8192 B */

/* ---- scratchpad (64 KB; the top byte used is 0xC7FF) ---------------------- */
/* The staging arena first, then every activation, all resident. The arena is
 * sized by the largest single weight block (W1/W2, 8 KB); the elementwise int32
 * temp (2 KB) and the logit staging (2.5 KB) both fit inside it, and every
 * primitive rewinds it on the way out. */
#define SP_ARENA       0x0000u
#define SP_ARENA_BYTES 0x2000u

#define SP_X         0x2000u    /* [T][D] int8 — the residual stream        */
#define SP_Q         0x2800u    /* [T][D] int8                              */
#define SP_K         0x3000u
#define SP_V         0x3800u
#define SP_KT_INT8   0x4000u    /* [D][T] int8 — K transposed               */
#define SP_KT_INT4   0x4800u    /* [D][T] int4 — the Q@K^T weight operand   */
#define SP_V_INT4    0x4C00u    /* [T][D] int4 — the P@V weight operand     */
#define SP_MASK      0x5000u    /* [T][T] int8                              */
#define SP_S         0x5400u    /* [T][T] int8 — raw scores                 */
#define SP_S_MASKED  0x5800u    /* [T][T] int8 — S + causal mask            */
#define SP_P         0x5C00u    /* [T][T] int8 — relu of that               */
#define SP_A         0x6000u    /* [T][D] int8 — attention output           */
#define SP_O         0x6800u    /* [T][D] int8 — A @ Wo                     */
#define SP_X_PLUS_O  0x7000u    /* [T][D] int8 — first half of the residual */
#define SP_X1        0x7800u    /* [T][D] int8 — after DyT                  */
#define SP_H         0x8000u    /* [T][DFF] int8 — X1 @ W1                  */
#define SP_H_RELU    0xA000u    /* [T][DFF] int8                            */
#define SP_FFN_OUT   0xC000u    /* [T][D] int8 — HR @ W2                    */

/* Every matmul below spells its shape out at the call site rather than going
 * through a helper, and that is a performance decision: with the shape constant
 * gcc folds tpu_matmul's block chooser, block loops and staging branches away
 * and the call becomes the two pushes it would have been by hand. Routing them
 * through a helper with runtime shapes costs ~2000 exposed clocks per matmul
 * (docs/picorv32_migration.md 9.10). */

int main(void)
{
    tpu_arena arena;

    tpu_arena_init(&arena, SP_ARENA, SP_ARENA_BYTES);

    tpu_move(SP_X,    DR_X,    T * D, TPU_DMA_FILL);
    tpu_move(SP_MASK, DR_MASK, T * T, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    for (unsigned layer = 0; layer < LAYERS; layer++) {
        const uint16_t *rq = rq_table[layer];
        const uint32_t layer_wgt = DR_LAYER0 + layer * DR_LAYER_STRIDE;

        /* ---- Q, K, V ----
         * Three [D][D] projections off the same X. They cannot share a dispatch
         * even though they share an activation: each weight has its own scale
         * and so its own {m0,n}. */
        tpu_matmul(&(const tpu_gemm){
            .rows = T, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_X, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WQ, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_Q, D),
            .rq_word = rq[RQ_Q] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = T, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_X, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WK, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_K, D),
            .rq_word = rq[RQ_K] }, &arena);
        tpu_matmul(&(const tpu_gemm){
            .rows = T, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_X, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WV, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_V, D),
            .rq_word = rq[RQ_V] }, &arena);

        /* K -> K^T, out through DRAM and back, as bytes. */
        tpu_transpose_int8(SP_KT_INT8, T, SP_K, D, T, D, DR_KT_SCRATCH);

        /* Both attention weight operands. K and V are already int4 — whatever
         * requant produced them clipped to [-8, 7] — so both packs are the
         * {1,0} identity and lose nothing. */
        tpu_pack4(TPU_SPAD_AT(SP_KT_INT4), TPU_SPAD_AT(SP_KT_INT8), D * T,
                  rq[RQ_KP], &arena);
        tpu_pack4(TPU_SPAD_AT(SP_V_INT4), TPU_SPAD_AT(SP_V), T * D,
                  rq[RQ_VP], &arena);

        for (unsigned head = 0; head < HEADS; head++) {
            /* S = requant(Q_head @ K_head^T). A head is a column slice of Q and
             * a row slice of K^T, both addressed in place — a column slice is
             * just the same row stride from a shifted base. */
            tpu_matmul(&(const tpu_gemm){
                .rows = T, .depth = HEAD_DIM, .cols = T,
                .act = TPU_SPAD_ROWS(SP_Q + head * HEAD_DIM, D),
                .wgt = TPU_SPAD_ROWS(SP_KT_INT4 + head * HEAD_DIM * WGT_ROW(T),
                                     WGT_ROW(T)),
                .out = TPU_SPAD_ROWS(SP_S, T),
                .rq_word = rq[RQ_S] }, &arena);

            /* P = requant(relu(requant(S + mask))).
             *
             * The mask is 0 or -8 and S is already int4, so a masked entry is
             * at most -1 whatever s_s is and ReLU takes it to exactly zero —
             * exact, not a tolerance. The RQ_ID narrow is there because `relu`
             * reads int8 while the add writes int32. */
            tpu_add_narrow(TPU_V_REQUANT, TPU_SPAD_AT(SP_S_MASKED),
                           TPU_SPAD_AT(SP_S), TPU_SPAD_AT(SP_MASK), T * T,
                           rq[RQ_ID], &arena);
            tpu_relu_narrow(TPU_SPAD_AT(SP_P), TPU_SPAD_AT(SP_S_MASKED), T * T,
                            rq[RQ_P], &arena);

            /* A_head = requant(P @ V_head), written straight into A's column
             * block. */
            tpu_matmul(&(const tpu_gemm){
                .rows = T, .depth = T, .cols = HEAD_DIM,
                .act = TPU_SPAD_ROWS(SP_P, T),
                .wgt = TPU_SPAD_ROWS(SP_V_INT4 + head * HEAD_DIM / 2,
                                     WGT_ROW(D)),
                .out = TPU_SPAD_ROWS(SP_A + head * HEAD_DIM, D),
                .rq_word = rq[RQ_A] }, &arena);
        }

        /* ---- O = requant(A @ Wo) ----
         * RQ_O has to land on s_x: the residual add takes two int8 operands at
         * ONE scale, so it pins O's output scale to the stream's. */
        tpu_matmul(&(const tpu_gemm){
            .rows = T, .depth = D, .cols = D,
            .act = TPU_SPAD_ROWS(SP_A, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_WO, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_O, D),
            .rq_word = rq[RQ_O] }, &arena);

        /* ---- the double residual, then DyT ----
         * MultiHeadAttention.forward ends in `O + X` and Transformer.forward
         * adds X again, so the attention residual is 2X + O. The VPU's add
         * takes two operands, hence two adds with the {1,0} identity between
         * them. */
        tpu_add_narrow(TPU_V_REQUANT, TPU_SPAD_AT(SP_X_PLUS_O),
                       TPU_SPAD_AT(SP_X), TPU_SPAD_AT(SP_O), T * D,
                       rq[RQ_XO], &arena);
        tpu_add_narrow(TPU_V_DYT, TPU_SPAD_AT(SP_X1),
                       TPU_SPAD_AT(SP_X_PLUS_O), TPU_SPAD_AT(SP_X), T * D,
                       rq[RQ_X1], &arena);

        /* ---- the feed-forward block ---- */
        tpu_matmul(&(const tpu_gemm){
            .rows = T, .depth = D, .cols = DFF,
            .act = TPU_SPAD_ROWS(SP_X1, D),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_FF1, WGT_ROW(DFF)),
            .out = TPU_SPAD_ROWS(SP_H, DFF),
            .rq_word = rq[RQ_H] }, &arena);
        tpu_relu_narrow(TPU_SPAD_AT(SP_H_RELU), TPU_SPAD_AT(SP_H), T * DFF,
                        rq[RQ_HR], &arena);
        /* F is pinned to s_x1 by the second residual, as O is to s_x. */
        tpu_matmul(&(const tpu_gemm){
            .rows = T, .depth = DFF, .cols = D,
            .act = TPU_SPAD_ROWS(SP_H_RELU, DFF),
            .wgt = TPU_DRAM_ROWS(layer_wgt + LW_FF2, WGT_ROW(D)),
            .out = TPU_SPAD_ROWS(SP_FFN_OUT, D),
            .rq_word = rq[RQ_F] }, &arena);

        /* The result lands back in X: this layer's input is dead by now. */
        tpu_add_narrow(TPU_V_DYT, TPU_SPAD_AT(SP_X), TPU_SPAD_AT(SP_X1),
                       TPU_SPAD_AT(SP_FFN_OUT), T * D, rq[RQ_X2], &arena);
    }

    /* ---- logits = X @ fc.w ----
     * Never requantized: an argmax does not care about scale, so the raw int32
     * accumulator is what leaves. Both the head's weights and its result are in
     * DRAM, so this one dispatch is the whole round trip. */
    tpu_matmul(&(const tpu_gemm){
        .rows = T, .depth = D, .cols = VOCAB_PAD,
        .act = TPU_SPAD_ROWS(SP_X, D),
        .wgt = TPU_DRAM_ROWS(DR_HEAD_WGT, WGT_ROW(VOCAB_PAD)),
        .out = TPU_DRAM_ROWS(DR_LOGITS, VOCAB_PAD * 4),
        .rq_word = 0u }, &arena);   /* 0 = store int32, do not narrow */

    return 0;                       /* start.S raises `done` from here */
}
