/* tpulib.h — primitives over tpu.h's macro-ops. See docs/fw.md. */
#ifndef TPULIB_H
#define TPULIB_H

#include "tpu.h"

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

/* ---- the machine (tpu_top.sv's parameters) ------------------------------- */

#ifndef TPU_ADDR_W
#define TPU_ADDR_W 16
#endif
#define TPU_SPAD_BYTES (1u << TPU_ADDR_W)

#ifndef TPU_BANK_WORDS
#define TPU_BANK_WORDS 1024u
#endif
#define TPU_BANK_BYTES (TPU_BANK_WORDS * TPU_WORD_BYTES)

#define TPU_ALIGN_DOWN(v, a) ((v) & ~((uint32_t)(a) - 1u))
#define TPU_ALIGN_UP(v, a)   TPU_ALIGN_DOWN((v) + (uint32_t)(a) - 1u, (a))

/* ---- buffers ------------------------------------------------------------- */

/* A DRAM tensor: where it starts and how far apart its rows are. A zero
 * row_bytes means densely packed, the same convention the DMA and MXU strides
 * use. */
typedef struct {
    uint32_t addr;
    uint32_t row_bytes;
} tpu_buf;

#define TPU_ROWS(dram_addr, bytes_per_row) \
    ((tpu_buf){ (uint32_t)(dram_addr), (uint32_t)(bytes_per_row) })
#define TPU_AT(dram_addr) TPU_ROWS((dram_addr), 0u)

static inline tpu_buf tpu_off(tpu_buf buf, uint32_t bytes)
{
    buf.addr += bytes;
    return buf;
}

/* ---- the arena ----------------------------------------------------------- */

/* How much scratchpad a primitive may spend. Each one lays itself out inside
 * this and nothing outlives the call, so a kernel never names a staging
 * address. The base must be bank-aligned: tpu_matmul's A, B and C slots are
 * whole banks because the MXU reads A and B on the same clock and a bank serves
 * one read per clock. */
typedef struct {
    uint32_t base;
    uint32_t bytes;
} tpu_arena;

static inline void tpu_arena_init(tpu_arena *arena, uint32_t base, uint32_t bytes)
{
    arena->base  = base;
    arena->bytes = bytes;
}

/* ---- raw DMA ------------------------------------------------------------- */

/* One flat run between DRAM and a scratchpad address the kernel picked. Does
 * NOT fence — this is the path for the CPU's mailbox, where the caller batches
 * several and waits once. */
static inline void tpu_move_bytes(uint32_t spad_addr, uint32_t dram_addr,
                                  uint32_t bytes, unsigned direction)
{
    tpu_dma(spad_addr, dram_addr, bytes * 2u, 1u, 0u, 0u, direction);
}

/* ---- matmul -------------------------------------------------------------- */

/* C = requant(A @ B), or @ B' when transposed. B is [depth][cols] plain and
 * [cols][depth] transposed. With `accumulate`, C's existing contents are added
 * — as int4, since the MXU has no int32 store. */
typedef struct {
    uint32_t rows;
    uint32_t depth;
    uint32_t cols;
    tpu_buf  a;
    tpu_buf  b;
    tpu_buf  c;
    unsigned transpose;
    unsigned accumulate;
    uint32_t rq_word;
} tpu_gemm;

/* One dispatch is an N x N output block; this walks rows and columns, lets the
 * array take the whole contraction. always_inline is load-bearing — see
 * docs/fw.md and CLAUDE.md. */
__attribute__((always_inline))
static inline void tpu_matmul(const tpu_gemm *gemm, tpu_arena *arena)
{
    const uint32_t depth_bytes = gemm->depth / 2u;
    const uint32_t block_bytes = TPU_N * depth_bytes;   /* one N-row operand block */

    const uint32_t a_row = gemm->a.row_bytes ? gemm->a.row_bytes : depth_bytes;
    const uint32_t c_row = gemm->c.row_bytes ? gemm->c.row_bytes : gemm->cols / 2u;
    const uint32_t b_row = gemm->b.row_bytes ? gemm->b.row_bytes
                         : (gemm->transpose ? depth_bytes : gemm->cols / 2u);

    /* A staged C row is the column count rounded out to whole output blocks:
     * the last block writes N columns whether or not they are all live. */
    const uint32_t c_slot_row = TPU_ALIGN_UP(gemm->cols, TPU_N) / 2u;

    TPU_ASSERT(gemm->depth % TPU_N == 0, "depth is not a whole array word");
    TPU_ASSERT(gemm->cols % 2u == 0, "an odd column count spills half a byte");
    TPU_ASSERT(gemm->depth <= 0xFFFFu, "contraction longer than the len field");
    TPU_ASSERT(arena->base % TPU_BANK_BYTES == 0, "arena base is not bank aligned");

    /* Three bank-disjoint slots, because the MXU reads A and B on the same
     * clock and a bank serves one read per clock. B holds one column block —
     * [depth][N] plain, [N][depth] transposed, the same bytes either way. */
    const uint32_t usable       = TPU_ALIGN_DOWN(arena->bytes, TPU_BANK_BYTES);
    const uint32_t b_slot_bytes = TPU_ALIGN_UP(block_bytes, TPU_BANK_BYTES);
    const uint32_t a_min        = TPU_ALIGN_UP(block_bytes, TPU_BANK_BYTES);
    const uint32_t c_min        = TPU_ALIGN_UP(TPU_N * c_slot_row, TPU_BANK_BYTES);

    TPU_ASSERT(usable >= b_slot_bytes + a_min + c_min,
               "arena too small: A, B and C each need a whole N-row block");

    /* Whatever is left over buys panel rows, which cost a row in each of A and
     * C. Split it in that ratio so neither slot is what caps the panel. */
    const uint32_t spare   = usable - b_slot_bytes - a_min - c_min;
    const uint32_t c_extra = TPU_ALIGN_DOWN((spare / (depth_bytes + c_slot_row))
                                            * c_slot_row, TPU_BANK_BYTES);

    const uint32_t c_slot_bytes = c_min + c_extra;
    const uint32_t a_slot_bytes = usable - b_slot_bytes - c_slot_bytes;
    const uint32_t a_slot       = arena->base;
    const uint32_t b_slot       = a_slot + a_slot_bytes;
    const uint32_t c_slot       = b_slot + b_slot_bytes;

    const uint32_t a_rows = TPU_ALIGN_DOWN(a_slot_bytes / depth_bytes, TPU_N);
    const uint32_t c_rows = TPU_ALIGN_DOWN(c_slot_bytes / c_slot_row, TPU_N);
    uint32_t panel_rows = (a_rows < c_rows) ? a_rows : c_rows;

    if (panel_rows > gemm->rows)
        panel_rows = gemm->rows;

    /* Sticky inside the MXU's queue and invariant across every block here. */
    tpu_mxu_geom(depth_bytes,
                 gemm->transpose ? depth_bytes : TPU_WORD_BYTES,
                 c_slot_row, gemm->depth);

    for (uint32_t r0 = 0; r0 < gemm->rows; r0 += panel_rows) {
        const uint32_t c_dram = gemm->c.addr + r0 * c_row;
        uint32_t nrows = gemm->rows - r0;

        if (nrows > panel_rows)
            nrows = panel_rows;

        tpu_dma(a_slot, gemm->a.addr + r0 * a_row,
                gemm->depth, nrows, a_row, depth_bytes, TPU_DMA_FILL);
        if (gemm->accumulate)
            tpu_dma(c_slot, c_dram, gemm->cols, nrows, c_row,
                    c_slot_row, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);

        for (uint32_t c0 = 0; c0 < gemm->cols; c0 += TPU_N) {
            uint32_t ncols = gemm->cols - c0;

            if (ncols > TPU_N)
                ncols = TPU_N;

            /* A short last block leaves the unstaged lanes stale; they land in
             * C columns past `ncols`, which the spill below never reads. */
            if (gemm->transpose)
                tpu_dma(b_slot, gemm->b.addr + c0 * b_row,
                        gemm->depth, ncols, b_row, depth_bytes, TPU_DMA_FILL);
            else
                tpu_dma(b_slot, gemm->b.addr + c0 / 2u,
                        ncols, gemm->depth, b_row, TPU_WORD_BYTES, TPU_DMA_FILL);
            tpu_wait(TPU_U_DMA);

            for (uint32_t sub = 0; sub < nrows; sub += TPU_N)
                tpu_mxu_mm(c_slot + sub * c_slot_row + c0 / 2u,
                           a_slot + sub * depth_bytes,
                           b_slot,
                           (gemm->transpose ? TPU_MM_T : 0u) |
                           (gemm->accumulate ? TPU_MM_ACC : 0u),
                           gemm->rq_word);
            tpu_wait(TPU_U_MXU);
        }

        tpu_dma(c_slot, c_dram, gemm->cols, nrows, c_row,
                c_slot_row, TPU_DMA_SPILL);
        tpu_wait(TPU_U_DMA);
    }
}

/* The same GEMM with one column block of C staged instead of a whole C row.
 * That is the only structural difference from tpu_matmul, and it moves the
 * arena's spare bytes from C's width into the row panel: a panel row costs
 * `depth/2 + TPU_WORD_BYTES` here against `depth/2 + cols/2` there, so a wide
 * problem gets a deeper panel and reads B fewer times. It pays for that with
 * one spill per column block instead of one per panel, and it has no
 * transposed-B or accumulate path. See docs/fw.md for when that trade loses. */
// __attribute__((always_inline))
// static inline void tpu_matmul_wide(const tpu_gemm *gemm, tpu_arena *arena)
// {
//     const uint32_t depth_bytes = gemm->depth / 2u;
//     const uint32_t block_bytes = TPU_N * depth_bytes;   /* one N-row operand block */

//     const uint32_t a_row = gemm->a.row_bytes ? gemm->a.row_bytes : depth_bytes;
//     const uint32_t b_row = gemm->b.row_bytes ? gemm->b.row_bytes : gemm->cols / 2u;
//     const uint32_t c_row = gemm->c.row_bytes ? gemm->c.row_bytes : gemm->cols / 2u;

//     TPU_ASSERT(gemm->depth % TPU_N == 0, "depth is not a whole array word");
//     TPU_ASSERT(gemm->cols % 2u == 0, "an odd column count spills half a byte");
//     TPU_ASSERT(gemm->depth <= 0xFFFFu, "contraction longer than the len field");
//     TPU_ASSERT(arena->base % TPU_BANK_BYTES == 0, "arena base is not bank aligned");
//     TPU_ASSERT(!gemm->transpose, "tpu_matmul_wide has no transposed-B path");
//     TPU_ASSERT(!gemm->accumulate, "tpu_matmul_wide has no accumulate path");

//     /* Three bank-disjoint slots, as in tpu_matmul, but a staged C row is one
//      * output block wide whatever `cols` is. */
//     const uint32_t usable       = TPU_ALIGN_DOWN(arena->bytes, TPU_BANK_BYTES);
//     const uint32_t b_slot_bytes = TPU_ALIGN_UP(block_bytes, TPU_BANK_BYTES);
//     const uint32_t a_min        = TPU_ALIGN_UP(block_bytes, TPU_BANK_BYTES);
//     const uint32_t c_min        = TPU_ALIGN_UP(TPU_N * TPU_WORD_BYTES, TPU_BANK_BYTES);

//     TPU_ASSERT(usable >= b_slot_bytes + a_min + c_min,
//                "arena too small: A, B and C each need a whole N-row block");

//     /* A panel row costs a row in each of A and C; split the spare in that
//      * ratio so neither slot is what caps the panel. */
//     const uint32_t spare   = usable - b_slot_bytes - a_min - c_min;
//     const uint32_t c_extra = TPU_ALIGN_DOWN((spare / (depth_bytes + TPU_WORD_BYTES))
//                                             * TPU_WORD_BYTES, TPU_BANK_BYTES);

//     const uint32_t c_slot_bytes = c_min + c_extra;
//     const uint32_t a_slot_bytes = usable - b_slot_bytes - c_slot_bytes;
//     const uint32_t a_slot       = arena->base;
//     const uint32_t b_slot       = a_slot + a_slot_bytes;
//     const uint32_t c_slot       = b_slot + b_slot_bytes;

//     const uint32_t a_rows = TPU_ALIGN_DOWN(a_slot_bytes / depth_bytes, TPU_N);
//     const uint32_t c_rows = TPU_ALIGN_DOWN(c_slot_bytes / TPU_WORD_BYTES, TPU_N);
//     uint32_t panel_rows = (a_rows < c_rows) ? a_rows : c_rows;

//     if (panel_rows > gemm->rows)
//         panel_rows = gemm->rows;

//     tpu_mxu_geom(depth_bytes, TPU_WORD_BYTES, TPU_WORD_BYTES, gemm->depth);

//     for (uint32_t r0 = 0; r0 < gemm->rows; r0 += panel_rows) {
//         uint32_t nrows = gemm->rows - r0;

//         if (nrows > panel_rows)
//             nrows = panel_rows;

//         tpu_dma(a_slot, gemm->a.addr + r0 * a_row,
//                 gemm->depth, nrows, a_row, depth_bytes, TPU_DMA_FILL);

//         for (uint32_t c0 = 0; c0 < gemm->cols; c0 += TPU_N) {
//             uint32_t ncols = gemm->cols - c0;

//             if (ncols > TPU_N)
//                 ncols = TPU_N;

//             tpu_dma(b_slot, gemm->b.addr + c0 / 2u,
//                     ncols, gemm->depth, b_row, TPU_WORD_BYTES, TPU_DMA_FILL);
//             tpu_wait(TPU_U_DMA);        /* also retires the previous spill */

//             for (uint32_t sub = 0; sub < nrows; sub += TPU_N)
//                 tpu_mxu_mm(c_slot + sub * TPU_WORD_BYTES,
//                            a_slot + sub * depth_bytes, b_slot, 0u,
//                            gemm->rq_word);
//             tpu_wait(TPU_U_MXU);

//             /* A short last block left the unstaged lanes stale; they sit in C
//              * columns past `ncols`, which this spill never reads. */
//             tpu_dma(c_slot, gemm->c.addr + r0 * c_row + c0 / 2u,
//                     ncols, nrows, c_row, TPU_WORD_BYTES, TPU_DMA_SPILL);
//         }
//     }

//     tpu_wait(TPU_U_DMA);
// }

/* ---- elementwise --------------------------------------------------------- */

/* `count` int4 elements, streamed through the arena a chunk at a time. Rows do
 * not matter here: every caller's operands are densely packed, so a tensor is a
 * flat vector and `tpu_buf.row_bytes` is ignored.
 *
 * The narrow is fused into the op, so nothing int32 is ever staged. */
static inline void tpu_elementwise(unsigned op, tpu_buf dst, tpu_buf src0,
                                   tpu_buf src1, uint32_t count,
                                   uint32_t rq_word, tpu_arena *arena)
{
    const unsigned binary = (op == TPU_V_ADD) || (op == TPU_V_DYT);
    const uint32_t usable = TPU_ALIGN_DOWN(arena->bytes, TPU_WORD_BYTES);

    uint32_t chunk = TPU_ALIGN_DOWN(usable / 3u, TPU_WORD_BYTES) * 2u;

    TPU_ASSERT(op != TPU_V_DOT, "DOT writes an int32 scalar, not a vector");
    TPU_ASSERT(count % 2u == 0, "an odd count writes a nibble it did not read");

    if (chunk > TPU_VCHUNK_MAX)
        chunk = TPU_VCHUNK_MAX;
    TPU_ASSERT(chunk >= TPU_N, "arena too small to stage three vector chunks");

    /* The slot bases are fixed and word-aligned, so every VPU address is,
     * whatever the chunk length. */
    const uint32_t slot_bytes = chunk / 2u;
    const uint32_t s0_slot    = arena->base;
    const uint32_t s1_slot    = s0_slot + slot_bytes;
    const uint32_t dst_slot   = s1_slot + slot_bytes;

    for (uint32_t i = 0; i < count; i += chunk) {
        uint32_t n = count - i;

        if (n > chunk)
            n = chunk;

        tpu_dma(s0_slot, src0.addr + i / 2u, n, 1u, 0u, 0u, TPU_DMA_FILL);
        if (binary)
            tpu_dma(s1_slot, src1.addr + i / 2u, n, 1u, 0u, 0u, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);

        tpu_vpu(op, dst_slot, s0_slot, binary ? s1_slot : 0u, n, rq_word);
        tpu_wait(TPU_U_VPU);

        tpu_dma(dst_slot, dst.addr + i / 2u, n, 1u, 0u, 0u, TPU_DMA_SPILL);
        tpu_wait(TPU_U_DMA);
    }
}

static inline void tpu_add(tpu_buf dst, tpu_buf a, tpu_buf b, uint32_t count,
                           uint32_t rq_word, tpu_arena *arena)
{
    tpu_elementwise(TPU_V_ADD, dst, a, b, count, rq_word, arena);
}

/* The residual add with hardtanh's symmetric clip. See docs/vpu.md. */
static inline void tpu_dyt(tpu_buf dst, tpu_buf a, tpu_buf b, uint32_t count,
                           uint32_t rq_word, tpu_arena *arena)
{
    tpu_elementwise(TPU_V_DYT, dst, a, b, count, rq_word, arena);
}

static inline void tpu_relu(tpu_buf dst, tpu_buf src, uint32_t count,
                            uint32_t rq_word, tpu_arena *arena)
{
    tpu_elementwise(TPU_V_RELU, dst, src, src, count, rq_word, arena);
}

static inline void tpu_requant(tpu_buf dst, tpu_buf src, uint32_t count,
                               uint32_t rq_word, tpu_arena *arena)
{
    tpu_elementwise(TPU_V_REQUANT, dst, src, src, count, rq_word, arena);
}

/* ---- copy ---------------------------------------------------------------- */

/* `rows` x `cols` int4, DRAM to DRAM through the arena. The DMA has DRAM on one
 * side by construction, so this is a fill and a spill, and a row stride on each
 * side means a sub-block moves in one pair of commands. */
static inline void tpu_copy(tpu_buf dst, tpu_buf src, uint32_t rows,
                            uint32_t cols, tpu_arena *arena)
{
    const uint32_t row_bytes = cols / 2u;
    const uint32_t src_row   = src.row_bytes ? src.row_bytes : row_bytes;
    const uint32_t dst_row   = dst.row_bytes ? dst.row_bytes : row_bytes;
    uint32_t block = arena->bytes / (row_bytes ? row_bytes : 1u);

    TPU_ASSERT(cols % 2u == 0, "an odd column count moves half a byte");
    TPU_ASSERT(block >= 1u, "arena too small to stage one row");

    if (block > rows)
        block = rows;

    for (uint32_t r = 0; r < rows; r += block) {
        uint32_t n = rows - r;

        if (n > block)
            n = block;

        tpu_dma(arena->base, src.addr + r * src_row, cols, n, src_row,
                row_bytes, TPU_DMA_FILL);
        tpu_wait(TPU_U_DMA);
        tpu_dma(arena->base, dst.addr + r * dst_row, cols, n, dst_row,
                row_bytes, TPU_DMA_SPILL);
        tpu_wait(TPU_U_DMA);
    }
}

#endif /* TPULIB_H */
