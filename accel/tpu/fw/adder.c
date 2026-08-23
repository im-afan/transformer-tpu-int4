/* adder.c — the whole int4 adder model, in firmware.
 *
 * `model/transformer.py::adder_int4_vanilla` — d=64, f=256, layers=4,
 * q_heads=kv_heads=4 (head_dim=16), vocab=13, int4 weights *and* int4
 * activations, no bias, no LayerNorm, no positional encoding — as one program,
 * one run:
 *
 *   for L in 0..3:
 *       Q, K, V = X@Wq, X@Wk, X@Wv          int4 weights, int4 activations
 *       S       = Q @ K^T / sqrt(head_dim)  per head, on the array
 *       P       = relu(S + causal_mask)     ReLU attention, NOT softmax
 *       A       = P @ V                     per head, on the array
 *       X       = dyt(X + (Wo(A) + X))      DOUBLE residual, then DyT
 *       X       = dyt(X + W2(relu(W1(X))))
 *   logits = X @ fc.w                       int32, never requantized
 *
 * The host owns exactly two things, both structural: the token embedding (the
 * ISA has no gather) and the final argmax (nothing returns an index). X0 and
 * the causal mask arrive in DRAM; the logits leave in DRAM.
 *
 * ---- what this kernel is written against ------------------------------------
 *
 * [`tpulib.h`](tpulib.h), not [`tpu.h`](tpu.h). Every matmul here is one
 * `tpu_matmul` over a descriptor that says where each operand lives; the
 * library owns the block loop, the weight staging and the barriers. That is
 * what makes this file a transcription of the model rather than a scratchpad
 * budget: the earlier version of it worked only because d=64/f=256/T=32 all fit
 * in 64 KB at once, and nothing in the arithmetic below now depends on that.
 *
 * The residency choices are still the kernel's, because they are the part that
 * costs clocks:
 *
 *   activations  scratchpad-resident for the whole run. They total ~42 KB, they
 *                are read and written several times per layer, and the MXU can
 *                address a head's column slice of one in place.
 *   weights      DRAM. 24 KB per layer against an 8 KB arena, re-fetched every
 *                forward. `tpu_matmul` stages each block as it needs it.
 *   logits       DRAM, staged out by `tpu_matmul` itself — the only tensor here
 *                whose *destination* is DRAM, and the reason the kernel needs
 *                no explicit spill at all.
 *
 * ---- why K is transposed and V is not ---------------------------------------
 *
 * The array reads weights row-major, so for C = A @ W the operand W[k][n] has
 * row k contiguous over n.
 *
 *   Q @ K^T contracts over the head dim: its weight is K^T[h][s], row h
 *           contiguous over s — K *column*-major. K leaves its projection
 *           row-major, so this one needs the transpose.
 *   P @ V   contracts over keys: its weight is V[s][h], row s contiguous over
 *           h, which is exactly how V left its projection. Free.
 *
 * That is the opposite of the ternary column-major kernel, where K was free and
 * V needed the transpose; row-major swapped the two. The transpose goes through
 * int8 because the DMA moves *bytes* and a packed int4 nibble is half of one,
 * so K is transposed first (SP_KT) and packed second (SP_KTP).
 *
 * ---- the requant table ------------------------------------------------------
 *
 * Every tensor carries a compile-time scale; integer v means real v*s. The 16
 * {m0,n} words per layer are the *only* thing in this kernel that depends on
 * the checkpoint, and they arrive as ADDER_RQ_INIT from a generated header:
 *
 *   adder_rq.h                the checked-in default, tuned for the synthetic
 *                             operands ../../tpulang/fw_vectors.py stages, so
 *                             `make fw FWPROG=adder` needs no checkpoint
 *   -DADDER_RQ_H='"..."'      a header derived from a real checkpoint by
 *                             ../../tpulang/adder_export.py
 *
 * They are literals in the command, so unlike the scalar unit's ISA there is no
 * path by which the device could read them out of memory — the table has to be
 * in the image. See adder_export.py for where each multiplier comes from.
 */
#include "tpulib.h"

#ifdef ADDER_RQ_H
#include ADDER_RQ_H
#else
#include "adder_rq.h"
#endif

#define T      32               /* tokens (train.py --max_tokens) */
#define D      64               /* model width */
#define DFF    256              /* feed-forward width */
#define NH     4                /* heads (q_heads == kv_heads) */
#define DH     (D / NH)         /* head dim, 16 */
#define LAYERS 4
#define VOCAB  13
/* The array stores a whole TPU_COLS-wide tile, so a 13-column head is two tiles
 * and a 13-word row stride would let the second tile land on the next token's
 * row. Round the head's output up to a whole tile and read 13 of every 16 words
 * back; the padding columns are staged as zero weights. */
#define VPAD   16

/* Requant sites, in block order — the index into one layer's ADDER_RQ row.
 * Six of the sixteen are the {1,0} identity by construction and are kept in the
 * table anyway so it matches model/transformer.py's site list one for one. */
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

static const uint16_t rq_tab[LAYERS][RQ_N] = ADDER_RQ_INIT;

/* ---- weight row strides, bytes (row-major int4: a row is N nibbles) ------- */
#define WROW_D   (D * 4 / 8)         /* [.][D]  — Wq/Wk/Wv, Wo, W2         */
#define WROW_F   (DFF * 4 / 8)       /* [D][F]  — W1                       */
#define WROW_KT  (T * 4 / 8)         /* [D][T]  — packed K^T               */
#define WROW_FC  (VPAD * 4 / 8)      /* [D][16] — the output head          */

/* ---- DRAM ---------------------------------------------------------------- */
#define DR_X     0x00000u       /* [T][D]  int8  — embedded input, host     */
#define DR_MASK  0x00800u       /* [T][T]  int8  — causal mask, host        */
#define DR_KT    0x00C00u       /* [D][T]  int8  — K^T scratch, device      */
#define DR_WFC   0x01400u       /* [D][16] int4  — output head, host        */
#define DR_LOG   0x01800u       /* [T][16] int32 — the result, device       */
#define DR_LAYER 0x02000u       /* layer 0's weight block                   */
#define DR_LSTEP 0x06000u       /* ...and the stride between layers         */
/* Four [.][D] projections and two feed-forward blocks, each dense row-major so
 * that staging one is a single DMA range. Wq/Wk/Wv were one fused [D][3D] block
 * while the kernel staged its own weights — fusing the *fill* was free then.
 * Under `tpu_matmul` it is the opposite: a column slice of a fused block is
 * strided, so it would cost D transfers instead of one. */
#define LW_Q     0x0000u        /*   [D][D]  int4, 2048 B                   */
#define LW_K     0x0800u
#define LW_V     0x1000u
#define LW_O     0x1800u
#define LW_1     0x2000u        /*   [D][F]  int4, 8192 B                   */
#define LW_2     0x4000u        /*   [F][D]  int4, 8192 B                   */

/* ---- scratchpad (64 KB; top byte used is 0xC7FF) -------------------------- */
/* One arena the library stages through, then every activation, resident. The
 * arena is sized by the largest single weight block (W1/W2, 8 KB); the elementwise
 * int32 temp (2 KB) and the logits staging (2.5 KB) both fit inside it, and
 * every primitive rewinds it on the way out. */
#define SP_ARENA 0x0000u        /* 8192                               */
#define SP_X     0x2000u        /* [T][D]  int8 — the residual stream */
#define SP_Q     0x2800u        /* [T][D]  int8                       */
#define SP_K     0x3000u
#define SP_V     0x3800u
#define SP_KT    0x4000u        /* [D][T]  int8  — K transposed       */
#define SP_KTP   0x4800u        /* [D][T]  int4  — the Q@K^T weight   */
#define SP_VP    0x4C00u        /* [T][D]  int4  — the P@V weight     */
#define SP_MASK  0x5000u        /* [T][T]  int8                       */
#define SP_S     0x5400u        /* [T][T]  int8                       */
#define SP_SM    0x5800u        /* [T][T]  int8  — S after the mask   */
#define SP_P     0x5C00u        /* [T][T]  int8                       */
#define SP_A     0x6000u        /* [T][D]  int8                       */
#define SP_O     0x6800u        /* [T][D]  int8                       */
#define SP_XO    0x7000u        /* [T][D]  int8                       */
#define SP_X1    0x7800u        /* [T][D]  int8                       */
#define SP_H     0x8000u        /* [T][F]  int8                       */
#define SP_HR    0xA000u        /* [T][F]  int8                       */
#define SP_FF    0xC000u        /* [T][D]  int8                       */

#define SP_ARENA_BYTES 0x2000u

/* Every matmul below is a descriptor written out at its call site rather than
 * built by a helper, and that is a performance decision, not a style one: with
 * the shape spelled out as constants gcc folds `tpu_matmul`'s block chooser,
 * block loops and staging branches away entirely and the call becomes the two
 * pushes it would have been by hand. Routing them through a helper with runtime
 * `m`/`k`/`n` costs ~2000 exposed clocks per matmul on the PicoRV32
 * (docs/picorv32_migration.md §9.10). */

int main(void)
{
    tpu_arena ws;
    tpu_arena_init(&ws, SP_ARENA, SP_ARENA_BYTES);

    tpu_move(SP_X,    DR_X,    T * D, TPU_DMA_FILL);
    tpu_move(SP_MASK, DR_MASK, T * T, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    for (unsigned l = 0; l < LAYERS; l++) {
        const uint16_t *rq = rq_tab[l];
        const uint32_t wb = DR_LAYER + l * DR_LSTEP;

        /* ---- Q, K, V ----
         * Three [D][D] projections off the same X. They cannot share a dispatch
         * even though they share an activation: each has its own weight scale
         * and so its own {m0,n}. */
        tpu_matmul(&(const tpu_gemm){
            .m = T, .k = D, .n = D, .a = TPU_SPR(SP_X, D),
            .w = TPU_DRR(wb + LW_Q, WROW_D), .c = TPU_SPR(SP_Q, D),
            .rq = rq[RQ_Q] }, &ws);
        tpu_matmul(&(const tpu_gemm){
            .m = T, .k = D, .n = D, .a = TPU_SPR(SP_X, D),
            .w = TPU_DRR(wb + LW_K, WROW_D), .c = TPU_SPR(SP_K, D),
            .rq = rq[RQ_K] }, &ws);
        tpu_matmul(&(const tpu_gemm){
            .m = T, .k = D, .n = D, .a = TPU_SPR(SP_X, D),
            .w = TPU_DRR(wb + LW_V, WROW_D), .c = TPU_SPR(SP_V, D),
            .rq = rq[RQ_V] }, &ws);

        /* K -> K^T, out through DRAM and back, as bytes. */
        tpu_transpose8(SP_KT, T, SP_K, D, T, D, DR_KT);

        /* Both attention weight operands. K and V are already int4 — whatever
         * requant produced them clipped to [-8, 7] — so both packs are the
         * {1,0} identity and lose nothing. */
        tpu_pack4(TPU_SP(SP_KTP), TPU_SP(SP_KT), D * T, rq[RQ_KP], &ws);
        tpu_pack4(TPU_SP(SP_VP),  TPU_SP(SP_V),  T * D, rq[RQ_VP], &ws);

        for (unsigned h = 0; h < NH; h++) {
            /* S = requant(Q_h @ K_h^T). Head h is a column slice of Q and a row
             * slice of K^T, and both are addressed in place — a column slice is
             * just the same row stride from a shifted base. */
            tpu_matmul(&(const tpu_gemm){
                .m = T, .k = DH, .n = T,
                .a = TPU_SPR(SP_Q + h * DH, D),
                .w = TPU_SPR(SP_KTP + h * DH * WROW_KT, WROW_KT),
                .c = TPU_SPR(SP_S, T), .rq = rq[RQ_S] }, &ws);

            /* P = requant(relu(requant(S + mask))).
             *
             * The mask is 0 or -8 and S is already int4, so a masked entry is
             * at most -1 whatever s_s is and ReLU takes it to exactly zero —
             * exact, not a tolerance. It costs the RQ_ID narrow because `relu`
             * reads int8 while `vecadd` writes int32. */
            tpu_add_narrow(TPU_V_REQUANT, TPU_SP(SP_SM), TPU_SP(SP_S),
                           TPU_SP(SP_MASK), T * T, rq[RQ_ID], &ws);
            tpu_relu_narrow(TPU_SP(SP_P), TPU_SP(SP_SM), T * T, rq[RQ_P], &ws);

            /* A_h = requant(P @ V_h), written straight into A's column block. */
            tpu_matmul(&(const tpu_gemm){
                .m = T, .k = T, .n = DH,
                .a = TPU_SPR(SP_P, T),
                .w = TPU_SPR(SP_VP + h * DH / 2, WROW_D),
                .c = TPU_SPR(SP_A + h * DH, D), .rq = rq[RQ_A] }, &ws);
        }

        /* ---- O = requant(A @ Wo) ----
         * RQ_O has to land on s_x: `vecadd` takes two int8 operands at one
         * scale, so the residual pins O's output scale to the stream's. */
        tpu_matmul(&(const tpu_gemm){
            .m = T, .k = D, .n = D, .a = TPU_SPR(SP_A, D),
            .w = TPU_DRR(wb + LW_O, WROW_D), .c = TPU_SPR(SP_O, D),
            .rq = rq[RQ_O] }, &ws);

        /* ---- the double residual, then DyT ----
         * MultiHeadAttention.forward ends in `O + X` and Transformer.forward
         * adds X again, so the attention residual is 2X + O. `vecadd` takes two
         * operands, hence two adds with the {1,0} identity between them. */
        tpu_add_narrow(TPU_V_REQUANT, TPU_SP(SP_XO), TPU_SP(SP_X), TPU_SP(SP_O),
                       T * D, rq[RQ_XO], &ws);
        tpu_add_narrow(TPU_V_DYT, TPU_SP(SP_X1), TPU_SP(SP_XO), TPU_SP(SP_X),
                       T * D, rq[RQ_X1], &ws);

        /* ---- the feed-forward block ---- */
        tpu_matmul(&(const tpu_gemm){
            .m = T, .k = D, .n = DFF, .a = TPU_SPR(SP_X1, D),
            .w = TPU_DRR(wb + LW_1, WROW_F), .c = TPU_SPR(SP_H, DFF),
            .rq = rq[RQ_H] }, &ws);
        tpu_relu_narrow(TPU_SP(SP_HR), TPU_SP(SP_H), T * DFF, rq[RQ_HR], &ws);
        /* F is pinned to s_x1 by the second residual, the same way O is to s_x. */
        tpu_matmul(&(const tpu_gemm){
            .m = T, .k = DFF, .n = D, .a = TPU_SPR(SP_HR, DFF),
            .w = TPU_DRR(wb + LW_2, WROW_D), .c = TPU_SPR(SP_FF, D),
            .rq = rq[RQ_F] }, &ws);

        /* X2 lands back in X: this layer's input is dead by now. */
        tpu_add_narrow(TPU_V_DYT, TPU_SP(SP_X), TPU_SP(SP_X1), TPU_SP(SP_FF),
                       T * D, rq[RQ_X2], &ws);
    }

    /* ---- logits = X @ fc.w ----
     * Never requantized: the head's output scale is irrelevant to an argmax, so
     * the raw int32 accumulator is what leaves. Both the head's weights and its
     * result are in DRAM, so this one dispatch is the whole round trip. */
    tpu_matmul(&(const tpu_gemm){
        .m = T, .k = D, .n = VPAD,
        .a  = TPU_SPR(SP_X, D),
        .w  = TPU_DRR(DR_WFC, WROW_FC),
        .c  = TPU_DRR(DR_LOG, VPAD * 4),
        .rq = 0u }, &ws);              /* rq 0 = store int32, do not narrow */

    return 0;                   /* start.S raises `done` from here */
}
