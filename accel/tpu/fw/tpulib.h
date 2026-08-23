/* tpulib.h — size-independent primitives, one level up from tpu.h.
 *
 * tpu.h packs single macro-ops: one matmul over operands that are already in
 * the scratchpad, one DMA of at most 64 KB, one VPU pass of at most 1023
 * elements. Every limit in that list is a *hardware* limit, and a kernel
 * written straight against it has to be shaped around all of them at once —
 * which is why the first version of adder.c only worked because the whole
 * model happened to fit in 64 KB of scratchpad.
 *
 * This file is the loop that hides them. A primitive here takes a problem of
 * any size, splits it into pieces the hardware can take, moves operands in and
 * results out, and fences between the units on the way. What it does not do is
 * hide *where a tensor lives*: that stays in the caller's hands, because it is
 * the one decision that actually costs clocks.
 *
 * ---- the four hardware limits the loops exist for ---------------------------
 *
 *   t_len <= 32          mxu.sv's result buffer is MAX_TOKENS = 1 << TOK_W deep,
 *                        so one dispatch covers at most 32 rows of A.
 *   k_tiles, n_tiles <= 255   both are 8-bit fields in MXU_GEOM.
 *   dma_len <= 65535     one DMA command moves at most 64 KB - 1.
 *   vpu_vlen <= 1023     10 bits, and QUANT4 additionally needs it even.
 *
 * The scratchpad is the fifth limit and the only one that is not a field width:
 * 64 KB shared by every resident tensor. The primitives here take an arena to
 * stage through, and the block sizes come out of how much of it is left.
 *
 * ---- residency --------------------------------------------------------------
 *
 * A `tpu_buf` is an address plus which memory it is in. A scratchpad-resident
 * operand is used in place — the MXU addresses a sub-block of a larger matrix
 * natively, through the three GEOM strides — so a kernel whose activations fit
 * pays nothing for going through this layer. A DRAM-resident operand is staged
 * a block at a time into the arena. That is the whole of the difference, and it
 * is why adder.c can keep every activation resident while its weights stream.
 *
 * ---- barriers ---------------------------------------------------------------
 *
 * Every primitive here is self-fencing: it returns only once the commands it
 * pushed have retired, so composing two of them is always safe. Within a
 * primitive, ordering between commands *on the same unit* is free — a unit's
 * queue is in-order (cmd_dma.sv is a two-state FSM over one FIFO), so a fill
 * that follows a spill cannot start early. Only cross-unit dependencies cost a
 * `tpu_wait`, and they are taken only when something was actually staged.
 *
 * ---- why three functions are `always_inline` --------------------------------
 *
 * Because on this machine the producer is the scarce resource. Everything a
 * primitive computes before its first push happens with no unit busy — the
 * caller has just fenced — so it is exposed clock for clock, and the PicoRV32
 * runs 5-9 clocks per instruction with no cache. Two hundred instructions of
 * block arithmetic per matmul is ~1400 exposed clocks against the ~800 the
 * array spends on the dispatch it produces.
 *
 * The fix is not to make the arithmetic cheaper, it is to make it *disappear*
 * at call sites whose shape is a compile-time constant — which is every kernel
 * here, because a transformer's dimensions are `#define`s. Given the shape,
 * `tpu_gemm_blocks` folds to a constant, the block loops fold to one iteration,
 * every staging branch resolves, and the descriptor never reaches memory. What
 * is left is the two stores of a GEOM and the two of a matmul.
 *
 * gcc will not do that on its own at -Os: `tpu_matmul` is a kilobyte of object
 * code before folding, and the inliner decides before it knows the folding is
 * available. `always_inline` on the three entry points that see the shape
 * (`tpu_matmul`, `tpu_gemm_blocks`, `tpu_gemm_need`) is what lets it, and it
 * makes adder.c *smaller* — 1992 bytes against 6420 without them, because the
 * general paths become dead code at every site. A kernel with genuinely runtime
 * shapes gets the loops instead, which is the same code doing the same thing.
 *
 * ---- what is NOT here -------------------------------------------------------
 *
 * There is no scratchpad-to-scratchpad move: the DMA has DRAM on one side by
 * construction. `tpu_transpose8` therefore goes out to DRAM and back, and the
 * matmul's split-contraction path narrows int32 partials with a VPU pass rather
 * than a copy.
 */
#ifndef TPULIB_H
#define TPULIB_H

#include "tpu.h"

/* Checked in the -DTPU_TRACE build only, which is where a kernel's tiling is
 * exercised first and where a bad one is cheapest to find: `make -C fw trace
 * PROG=<kernel>` runs the loops with the host compiler, no board and no
 * simulator. On the device it compiles to nothing — there is no output path to
 * report on and nothing useful to do about it. */
#ifdef TPU_TRACE
#include <stdio.h>
#include <stdlib.h>
#define TPU_ASSERT(cond, msg)                                                 \
    do {                                                                      \
        if (!(cond)) {                                                        \
            fprintf(stderr, "tpulib: %s (%s:%d)\n", (msg), __FILE__, __LINE__);\
            exit(2);                                                          \
        }                                                                     \
    } while (0)
#else
#define TPU_ASSERT(cond, msg) ((void)0)
#endif

/* ---- the machine --------------------------------------------------------- */
/* The array geometry is fixed by the bitstream, not by the program. Overridable
 * so a kernel built for a different array does not have to fork this file. */
#ifndef TPU_ROWS
#define TPU_ROWS 8
#endif
#ifndef TPU_COLS
#define TPU_COLS 8
#endif

#define TPU_TLEN_MAX  32u       /* mxu.sv MAX_TOKENS = 1 << TOK_W            */
#define TPU_TILES_MAX 255u      /* k_tiles / n_tiles are 8 bits              */
#define TPU_DMA_MAX   0xFF00u   /* dma_len is 16 bits; kept round            */

/* Elements per VPU pass. Bounded by the 10-bit `vlen`, kept even for QUANT4
 * (two nibbles share a byte) and a multiple of the 16 int32 lanes so a chunk
 * never straddles a partial access. Also sets the int32 temp at 4x this. */
#ifndef TPU_CHUNK
#define TPU_CHUNK 512u
#endif

/* ---- where a tensor lives ------------------------------------------------ */
#define TPU_SPAD 0u
#define TPU_DRAM 1u

/* `row` is the byte stride between rows of the tensor this buffer names, and is
 * read only by the 2-D primitives — the elementwise ops treat their operands as
 * flat. It is a *byte* stride whatever the element width is, so an int32 [m][n]
 * has row = 4*n and an int4 [m][n] has row = n/2. */
typedef struct {
    uint32_t addr;
    uint32_t row;
    uint32_t space;
} tpu_buf;

#define TPU_SP(a)     ((tpu_buf){ (uint32_t)(a), 0u, TPU_SPAD })
#define TPU_DR(a)     ((tpu_buf){ (uint32_t)(a), 0u, TPU_DRAM })
#define TPU_SPR(a, r) ((tpu_buf){ (uint32_t)(a), (uint32_t)(r), TPU_SPAD })
#define TPU_DRR(a, r) ((tpu_buf){ (uint32_t)(a), (uint32_t)(r), TPU_DRAM })

/* ---- the staging arena --------------------------------------------------- */
/* A bump allocator over one scratchpad region. Every primitive allocates what
 * it needs on the way in and rewinds on the way out, so the arena's high-water
 * mark is the largest single primitive rather than the sum of them. */
typedef struct {
    uint32_t next;
    uint32_t end;
} tpu_arena;

static inline void tpu_arena_init(tpu_arena *a, uint32_t base, uint32_t bytes)
{
    a->next = base;
    a->end  = base + bytes;
}

static inline uint32_t tpu_alloc(tpu_arena *a, uint32_t bytes)
{
    uint32_t p = a->next;
    bytes = (bytes + 3u) & ~3u;         /* int32 staging wants 4-byte starts */
    TPU_ASSERT(p + bytes <= a->end, "arena overflow — give the kernel a bigger one");
    a->next = p + bytes;
    return p;
}

/* ---- moves --------------------------------------------------------------- */

/* `bytes` between scratchpad and DRAM, any length. One command per 64 KB. */
static inline void tpu_move(uint32_t spad, uint32_t dram, uint32_t bytes,
                            unsigned dir)
{
    while (bytes) {
        uint32_t n = (bytes > TPU_DMA_MAX) ? TPU_DMA_MAX : bytes;
        tpu_dma(spad, dram, n, dir);
        spad  += n;
        dram  += n;
        bytes -= n;
    }
}

/* A `rows` x `cols` byte block, honouring both row strides. Dense on both sides
 * is one transfer; anything else is one per row, because a DMA range is
 * contiguous in address space and a strided block is not. Keeping a staged
 * block dense is therefore worth real commands, which is why every staging
 * buffer below is allocated at exactly the block's width. */
static inline void tpu_move2d(uint32_t spad, uint32_t spad_row,
                              uint32_t dram, uint32_t dram_row,
                              uint32_t rows, uint32_t cols, unsigned dir)
{
    if (spad_row == cols && dram_row == cols) {
        tpu_move(spad, dram, rows * cols, dir);
        return;
    }
    for (uint32_t r = 0; r < rows; r++)
        tpu_move(spad + r * spad_row, dram + r * dram_row, cols, dir);
}

/* dst[c][r] = src[r][c], int8, via `scratch` in DRAM (rows*cols bytes).
 *
 * The DMA is the only unit that can transpose and one of its two sides is
 * always DRAM, so this is a `.t` spill followed by a plain fill. No barrier
 * between them: both ride the DMA's own in-order queue.
 *
 * dma.sv writes dest[col*tdrow + row], so a transfer that covers rows
 * [r0, r0+rb) lands at `scratch + r0` with tdrow = the FULL row count — which
 * is what lets a transpose bigger than one DMA command be split by rows. */
static inline void tpu_transpose8(uint32_t dst, uint32_t dst_row,
                                  uint32_t src, uint32_t src_row,
                                  uint32_t rows, uint32_t cols, uint32_t scratch)
{
    TPU_ASSERT(cols <= TPU_DMA_MAX, "transpose: a source row exceeds one DMA");
    if (rows == 0u || cols == 0u)
        return;

    for (uint32_t r0 = 0; r0 < rows; ) {
        uint32_t rb = 0, len = 0;
        while (r0 + rb < rows && len + cols <= TPU_DMA_MAX) {
            rb++;
            len += cols;
        }
        tpu_dma_t(src + r0 * src_row, scratch + r0, len, TPU_DMA_SPILL,
                  cols, src_row, rows);
        r0 += rb;
    }
    /* The DRAM image is [cols][rows], dense. */
    tpu_move2d(dst, dst_row, scratch, rows, cols, rows, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);
}

/* ---- elementwise --------------------------------------------------------- */

/* dst = narrow(widen(src0, src1)) over `n` elements, chunked at TPU_CHUNK.
 *
 * The pair is fused rather than run as two full passes because the widening op
 * writes int32 and only an explicit narrow comes back down (vpu.md): a separate
 * pass would need an int32 temp the size of the whole tensor. QUANT4 is the one
 * op with no widening half — it reads int8 and writes 4 bits — and is selected
 * by passing it as `narrow_op`.
 *
 * Any operand may be in DRAM; it is then streamed through the arena one chunk
 * at a time, which is what makes this work on a tensor larger than the
 * scratchpad. All-resident operands stage nothing and take no barrier until the
 * single drain on the way out — the same command stream a hand-written loop
 * would push. */
static inline void tpu_ew(unsigned widen_op, unsigned narrow_op,
                          tpu_buf dst, tpu_buf src0, tpu_buf src1,
                          uint32_t n, uint32_t rq, tpu_arena *ws)
{
    const unsigned pack    = (narrow_op == TPU_V_QUANT4);
    const unsigned two_src = !pack && (widen_op == TPU_V_ADD);

    TPU_ASSERT(!pack || (n % 2u) == 0u, "quant4: n must be even (a nibble pair "
                                        "shares a byte)");
    if (n == 0u)
        return;

    const uint32_t mark = ws->next;

    /* Everything resident: no staging, no barrier until the drain, and the loop
     * body is the two pushes and nothing else. Its own path rather than a
     * degenerate case of the streaming one below, because a per-chunk residency
     * test costs the CPU more clocks than the VPU spends on the chunk — and
     * because it folds to the hand-written loop when the shapes are constant. */
    if (dst.space == TPU_SPAD && src0.space == TPU_SPAD &&
        (!two_src || src1.space == TPU_SPAD)) {
        const uint32_t tmp = pack ? 0u : tpu_alloc(ws, TPU_CHUNK * 4u);

        for (uint32_t i = 0; i < n; i += TPU_CHUNK) {
            const uint32_t len = (n - i < TPU_CHUNK) ? (n - i) : TPU_CHUNK;

            if (pack) {
                tpu_vpu(TPU_V_QUANT4, dst.addr + (i >> 1), src0.addr + i, 0u,
                        len, rq);
            } else {
                /* A one-source widening op ignores src1 (vpu.sv needs_src1),
                 * but it still rides in the command — send zero rather than a
                 * live address, so a trace of a `relu` reads as one. */
                tpu_vpu(widen_op, tmp, src0.addr + i,
                        two_src ? src1.addr + i : 0u, len, 0u);
                tpu_vpu(narrow_op, dst.addr + i, tmp, 0u, len, rq);
            }
        }
        tpu_wait(TPU_U_VPU);
        ws->next = mark;
        return;
    }

    const uint32_t buf0 = (src0.space == TPU_DRAM) ? tpu_alloc(ws, TPU_CHUNK) : 0u;
    const uint32_t buf1 = (two_src && src1.space == TPU_DRAM)
                              ? tpu_alloc(ws, TPU_CHUNK) : 0u;
    const uint32_t bufd = (dst.space == TPU_DRAM) ? tpu_alloc(ws, TPU_CHUNK) : 0u;
    const uint32_t tmp32 = pack ? 0u : tpu_alloc(ws, TPU_CHUNK * 4u);

    for (uint32_t i = 0; i < n; i += TPU_CHUNK) {
        const uint32_t len = (n - i < TPU_CHUNK) ? (n - i) : TPU_CHUNK;
        const uint32_t dst_off = pack ? (i >> 1) : i;
        uint32_t a0, a1 = 0u, ad;
        unsigned staged_in = 0;

        if (src0.space == TPU_DRAM) {
            tpu_move(buf0, src0.addr + i, len, TPU_DMA_FILL);
            a0 = buf0;
            staged_in = 1;
        } else {
            a0 = src0.addr + i;
        }
        if (two_src) {
            if (src1.space == TPU_DRAM) {
                tpu_move(buf1, src1.addr + i, len, TPU_DMA_FILL);
                a1 = buf1;
                staged_in = 1;
            } else {
                a1 = src1.addr + i;
            }
        }
        ad = (dst.space == TPU_DRAM) ? bufd : (dst.addr + dst_off);

        if (staged_in)
            tpu_wait(TPU_U_DMA);

        if (pack) {
            tpu_vpu(TPU_V_QUANT4, ad, a0, 0u, len, rq);
        } else {
            tpu_vpu(widen_op, tmp32, a0, a1, len, 0u);
            tpu_vpu(narrow_op, ad, tmp32, 0u, len, rq);
        }

        if (dst.space == TPU_DRAM) {
            tpu_wait(TPU_U_VPU);
            tpu_move(bufd, dst.addr + dst_off, pack ? (len >> 1) : len,
                     TPU_DMA_SPILL);
            tpu_wait(TPU_U_DMA);
        } else if (staged_in) {
            tpu_wait(TPU_U_VPU);        /* the staging buffers are reused */
        }
    }
    tpu_wait(TPU_U_VPU);
    ws->next = mark;
}

/* dst = narrow(a + b). `narrow_op` is REQUANT or DYT — the two differ only in
 * the clip, and choosing DYT here is how a kernel says "this residual add is
 * also norm1". */
static inline void tpu_add_narrow(unsigned narrow_op, tpu_buf dst, tpu_buf a,
                                  tpu_buf b, uint32_t n, uint32_t rq,
                                  tpu_arena *ws)
{
    tpu_ew(TPU_V_ADD, narrow_op, dst, a, b, n, rq, ws);
}

/* dst = requant(relu(src)). */
static inline void tpu_relu_narrow(tpu_buf dst, tpu_buf src, uint32_t n,
                                   uint32_t rq, tpu_arena *ws)
{
    tpu_ew(TPU_V_RELU, TPU_V_REQUANT, dst, src, TPU_SP(0), n, rq, ws);
}

/* dst = quant4(src): int8 in, 4 bits out, so the destination advances half as
 * fast as the source. This is what turns an activation into a legal weight
 * operand without a repacking pass or a host round trip. */
static inline void tpu_pack4(tpu_buf dst, tpu_buf src, uint32_t n, uint32_t rq,
                             tpu_arena *ws)
{
    tpu_ew(0u, TPU_V_QUANT4, dst, src, TPU_SP(0), n, rq, ws);
}

/* int32 -> int4-in-int8 over a scratchpad block, chunked. Used by the matmul
 * below when a split contraction leaves int32 partials the store could not
 * narrow; a kernel that has its own int32 block can call it directly. */
static inline void tpu_requant_rows(uint32_t dst, uint32_t dst_row,
                                    uint32_t src32, uint32_t src_row,
                                    uint32_t rows, uint32_t cols, uint32_t rq)
{
    /* Dense on both sides is one flat pass over the block; otherwise one pass
     * per row, since a VPU op walks its operands contiguously. */
    const unsigned dense = (dst_row == cols) && (src_row == cols * 4u);
    const uint32_t passes = dense ? 1u : rows;
    const uint32_t per    = dense ? rows * cols : cols;

    for (uint32_t r = 0; r < passes; r++) {
        const uint32_t d = dst + r * dst_row, s = src32 + r * src_row;
        for (uint32_t i = 0; i < per; i += TPU_CHUNK) {
            const uint32_t len = (per - i < TPU_CHUNK) ? (per - i) : TPU_CHUNK;
            tpu_vpu(TPU_V_REQUANT, d + i, s + i * 4u, 0u, len, rq);
        }
    }
    tpu_wait(TPU_U_VPU);
}

/* ---- matmul -------------------------------------------------------------- */

/* C[m][n] = A[m][k] @ W[k][n], at any size, with each operand in either memory.
 *
 *   A   int8, `a.row` bytes per row (>= k)
 *   W   int4 row-major, two per byte, `w.row` bytes per row (>= n/2)
 *   C   int8 when `rq` is a {m0,n} word, int32 when `rq` is 0 (m0 = 0 is not a
 *       legal multiplier, so zero is free as the sentinel). `c.row` is bytes
 *       either way, so an int8 C has c.row >= n and an int32 C has c.row >= 4n.
 *
 * k must be a multiple of TPU_ROWS and n a multiple of TPU_COLS: the array
 * consumes whole tiles and has no lane masking, so a partial tile would read
 * neighbouring data rather than zeros. Pad the tensor (adder.c's output head
 * pads 13 columns to 16) rather than passing a short one. */
typedef struct {
    uint32_t m, k, n;
    tpu_buf  a;
    tpu_buf  w;
    tpu_buf  c;
    uint32_t rq;
} tpu_gemm;

/* Arena bytes one block of this shape would need. The C term is the interesting
 * one: an *unsplit* contraction never needs an int32 buffer at all, because the
 * array holds its partials in result_buf across the whole k loop and narrows on
 * the single store. Splitting k is what forces 4 bytes per output element into
 * the arena, which is why the block chooser gives it up last. */
__attribute__((always_inline))
static inline uint32_t tpu_gemm_need(const tpu_gemm *g, uint32_t mb,
                                     uint32_t kb, uint32_t nb)
{
    const unsigned rq = (g->rq != 0u), split = (kb < g->k);
    uint32_t need = 0u;

    if (g->a.space == TPU_DRAM) need += mb * kb;
    if (g->w.space == TPU_DRAM) need += (kb * nb) >> 1;
    if (rq && g->c.space == TPU_DRAM) need += mb * nb;
    if ((!rq && g->c.space == TPU_DRAM) || (rq && split)) need += mb * nb * 4u;
    return need;
}

/* Largest block the arena holds, starting from the largest the hardware allows.
 * Halve whichever extent is biggest until it fits; the contraction goes last
 * because it is the only one whose split *adds* a buffer. Everything divides by
 * TPU_ROWS / TPU_COLS, which are compile-time constants — the core is built
 * with ENABLE_DIV(0) and nothing here may emit a runtime divide.
 *
 * The early-out is not just a speed-up of this function. It is what lets a call
 * site with a compile-time shape fold: with `kb == k` the C term of
 * `tpu_gemm_need` stops depending on `g->rq`, so the whole test is constant and
 * gcc deletes the search loop, the block loops and every staging branch behind
 * it. That matters because the producer is a 5-to-9-clock-per-instruction
 * PicoRV32 with no cache, and everything this function does happens *between* a
 * barrier and the next push, where it is fully exposed
 * (docs/picorv32_migration.md §9.10). */
__attribute__((always_inline))
static inline void tpu_gemm_blocks(const tpu_gemm *g, uint32_t budget,
                                   uint32_t *pmb, uint32_t *pkb, uint32_t *pnb)
{
    uint32_t mb  = (g->m < TPU_TLEN_MAX) ? g->m : TPU_TLEN_MAX;
    uint32_t ktb = g->k / TPU_ROWS;
    uint32_t ntb = g->n / TPU_COLS;

    if (g->m <= TPU_TLEN_MAX && ktb <= TPU_TILES_MAX && ntb <= TPU_TILES_MAX &&
        tpu_gemm_need(g, g->m, g->k, g->n) <= budget) {
        *pmb = g->m;
        *pkb = g->k;
        *pnb = g->n;
        return;
    }

    if (ktb > TPU_TILES_MAX) ktb = TPU_TILES_MAX;
    if (ntb > TPU_TILES_MAX) ntb = TPU_TILES_MAX;
    /* A dimension shorter than one tile is a caller bug the assertions below
     * name; clamping keeps the block loops from stepping by zero meanwhile. */
    if (ktb == 0u) ktb = 1u;
    if (ntb == 0u) ntb = 1u;

    while (tpu_gemm_need(g, mb, ktb * TPU_ROWS, ntb * TPU_COLS) > budget) {
        if (ntb > 1u && ntb * TPU_COLS >= mb) ntb = (ntb + 1u) >> 1;
        else if (mb > 1u)                     mb  = (mb + 1u) >> 1;
        else if (ktb > 1u)                    ktb = (ktb + 1u) >> 1;
        else                                  break;
    }
    *pmb = mb;
    *pkb = ktb * TPU_ROWS;
    *pnb = ntb * TPU_COLS;
    TPU_ASSERT(tpu_gemm_need(g, *pmb, *pkb, *pnb) <= budget,
               "matmul: the arena cannot hold even a single-tile block");
}

__attribute__((always_inline))
static inline void tpu_matmul(const tpu_gemm *g, tpu_arena *ws)
{
    if (g->m == 0u || g->k == 0u || g->n == 0u)
        return;
    TPU_ASSERT(g->k % TPU_ROWS == 0u, "matmul: k must be a multiple of TPU_ROWS");
    TPU_ASSERT(g->n % TPU_COLS == 0u, "matmul: n must be a multiple of TPU_COLS");

    uint32_t mb, kb, nb;
    tpu_gemm_blocks(g, ws->end - ws->next, &mb, &kb, &nb);

    const unsigned rq        = (g->rq != 0u);
    const unsigned split     = (kb < g->k);
    const unsigned stage_a   = (g->a.space == TPU_DRAM);
    const unsigned stage_w   = (g->w.space == TPU_DRAM);
    const unsigned stage_c8  = rq && (g->c.space == TPU_DRAM);
    const unsigned stage_c32 = (!rq && g->c.space == TPU_DRAM) || (rq && split);
    /* The store narrows only when the whole contraction ran in one dispatch.
     * A split leaves int32 partials, and those come down through the VPU. */
    const unsigned mm_rq     = rq && !stage_c32;
    /* With one contraction block the weight depends only on n0, so it can be
     * staged once and reused down the whole m loop. */
    const unsigned hoist_w   = stage_w && !split;

    const uint32_t mark   = ws->next;
    const uint32_t sp_a   = stage_a   ? tpu_alloc(ws, mb * kb)          : 0u;
    const uint32_t sp_w   = stage_w   ? tpu_alloc(ws, (kb * nb) >> 1)   : 0u;
    const uint32_t sp_c8  = stage_c8  ? tpu_alloc(ws, mb * nb)          : 0u;
    const uint32_t sp_c32 = stage_c32 ? tpu_alloc(ws, mb * nb * 4u)     : 0u;

    /* MXU_GEOM is sticky in this unit's queue, so it is re-pushed only when a
     * block's shape actually differs from the last one — which is exactly the
     * remainder blocks at the far edge of each axis. */
    uint32_t g_arow = 0u, g_crow = 0u, g_wrow = 0u;
    uint32_t g_kt = 0u, g_nt = 0u, g_tlen = 0u;
    unsigned have_geom = 0;

    for (uint32_t n0 = 0; n0 < g->n; n0 += nb) {
        const uint32_t nbi = (g->n - n0 < nb) ? (g->n - n0) : nb;

        if (hoist_w) {
            tpu_move2d(sp_w, nbi >> 1, g->w.addr + (n0 >> 1), g->w.row,
                       g->k, nbi >> 1, TPU_DMA_FILL);
            tpu_wait(TPU_U_DMA);
        }

        for (uint32_t m0 = 0; m0 < g->m; m0 += mb) {
            const uint32_t mbi = (g->m - m0 < mb) ? (g->m - m0) : mb;

            for (uint32_t k0 = 0; k0 < g->k; k0 += kb) {
                const uint32_t kbi = (g->k - k0 < kb) ? (g->k - k0) : kb;
                uint32_t act, wgt, out, arow, wrow, crow;
                unsigned filled = 0;

                if (stage_w && !hoist_w) {
                    tpu_move2d(sp_w, nbi >> 1,
                               g->w.addr + k0 * g->w.row + (n0 >> 1), g->w.row,
                               kbi, nbi >> 1, TPU_DMA_FILL);
                    filled = 1;
                }
                if (stage_a) {
                    tpu_move2d(sp_a, kbi, g->a.addr + m0 * g->a.row + k0,
                               g->a.row, mbi, kbi, TPU_DMA_FILL);
                    filled = 1;
                }
                if (filled)
                    tpu_wait(TPU_U_DMA);

                if (stage_a) { act = sp_a; arow = kbi; }
                else { act = g->a.addr + m0 * g->a.row + k0; arow = g->a.row; }

                if (stage_w) { wgt = sp_w; wrow = nbi >> 1; }
                else { wgt = g->w.addr + k0 * g->w.row + (n0 >> 1); wrow = g->w.row; }

                /* `crow` is always the int32 row stride: the array derives the
                 * int8 store stride as crow/4 (mxu.sv res_row_stride_i8). */
                if (stage_c32)     { out = sp_c32; crow = nbi * 4u; }
                else if (stage_c8) { out = sp_c8;  crow = nbi * 4u; }
                else if (rq)       { out = g->c.addr + m0 * g->c.row + n0;
                                     crow = g->c.row * 4u; }
                else               { out = g->c.addr + m0 * g->c.row + n0 * 4u;
                                     crow = g->c.row; }

                TPU_ASSERT(arow <= 0xFFFFu && crow <= 0xFFFFu && wrow <= 0xFFFFu,
                           "matmul: a GEOM stride does not fit 16 bits");

                if (!have_geom || arow != g_arow || crow != g_crow ||
                    wrow != g_wrow || kbi / TPU_ROWS != g_kt ||
                    nbi / TPU_COLS != g_nt || mbi != g_tlen) {
                    g_arow = arow; g_crow = crow; g_wrow = wrow;
                    g_kt = kbi / TPU_ROWS; g_nt = nbi / TPU_COLS; g_tlen = mbi;
                    have_geom = 1;
                    tpu_mxu_geom(g_arow, g_crow, g_wrow, g_kt, g_nt, g_tlen);
                }

                tpu_mxu_mm(out, act, wgt,
                           TPU_MM_TILED | (k0 ? TPU_MM_ACC : 0u) |
                               (mm_rq ? TPU_MM_RQ : 0u),
                           g->rq);

                /* Only a reused staging buffer forces a barrier inside the k
                 * loop; back-to-back dispatches out of resident operands stay
                 * queued. */
                if (stage_a || (stage_w && !hoist_w))
                    tpu_wait(TPU_U_MXU);
            }
            tpu_wait(TPU_U_MXU);

            if (stage_c32 && rq)
                tpu_requant_rows(stage_c8 ? sp_c8
                                          : g->c.addr + m0 * g->c.row + n0,
                                 stage_c8 ? nbi : g->c.row,
                                 sp_c32, nbi * 4u, mbi, nbi, g->rq);

            if (g->c.space == TPU_DRAM) {
                const uint32_t width = stage_c8 ? nbi : nbi * 4u;
                tpu_move2d(stage_c8 ? sp_c8 : sp_c32, width,
                           g->c.addr + m0 * g->c.row + (stage_c8 ? n0 : n0 * 4u),
                           g->c.row, mbi, width, TPU_DMA_SPILL);
                tpu_wait(TPU_U_DMA);
            }
        }
    }
    ws->next = mark;
}

#endif /* TPULIB_H */
