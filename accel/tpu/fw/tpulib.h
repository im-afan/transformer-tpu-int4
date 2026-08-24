/* tpulib.h — size-independent primitives, one level up from tpu.h.
 *
 * tpu.h packs single macro-ops, and every one of them has a hardware limit a
 * kernel would otherwise have to be shaped around:
 *
 *   token rows <= 32       mxu.sv's result buffer is MAX_TOKENS deep
 *   tile counts <= 255     depth_tiles / col_tiles are 8-bit fields
 *   DMA bytes <= 65535     one DMA command moves at most 64 KB - 1
 *   VPU count <= 1023      10 bits, and QUANT4 additionally needs it even
 *
 * A primitive here takes a problem of any size, splits it into pieces the
 * hardware accepts, stages operands in and results out, and fences between the
 * units on the way. What it does NOT hide is where a tensor lives — that stays
 * the caller's decision, because it is the one that costs clocks.
 *
 * RESIDENCY. A `tpu_buf` is an address plus which memory it is in. A
 * scratchpad-resident operand is used in place: the MXU addresses a sub-block
 * of a larger matrix natively through the three geometry strides, so a kernel
 * whose activations fit pays nothing for going through this layer. A
 * DRAM-resident operand is staged a block at a time through the caller's arena.
 *
 * BARRIERS. Every primitive is self-fencing — it returns only once its commands
 * have retired — so composing two is always safe. Ordering within one unit's
 * queue is free (the queue is in-order); only cross-unit dependencies cost a
 * tpu_wait, and those are taken only when something was actually staged.
 *
 * INLINING. `tpu_matmul`, `tpu_gemm_blocks` and `tpu_gemm_arena_bytes` are
 * `always_inline` for speed, not style. Everything a primitive computes before
 * its first push runs with no unit busy (the caller just fenced), so it is
 * exposed clock for clock on a PicoRV32 that takes 5-9 clocks per instruction
 * with no cache. With the shape a compile-time constant at the call site, gcc
 * folds the block chooser, the block loops and every staging branch away, and
 * the call becomes the two stores it would have been by hand. Measured:
 * adder.c is 1992 bytes and 453 778 clocks with the inlining, 6420 bytes and
 * 597 936 clocks without (docs/picorv32_migration.md 9.10).
 *
 * NOT HERE: a scratchpad-to-scratchpad move. The DMA has DRAM on one side by
 * construction, so `tpu_transpose_int8` goes out to DRAM and back, and a
 * split-contraction matmul narrows its int32 partials with a VPU pass rather
 * than a copy.
 */
#ifndef TPULIB_H
#define TPULIB_H

#include "tpu.h"

/* Checked in the -DTPU_TRACE build only, where a bad tiling is cheapest to find
 * (`make trace PROG=<kernel>` needs no board and no simulator). On the device
 * it compiles to nothing: there is no output path to report on. */
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
 * so a kernel built for a different array need not fork this file. */
#ifndef TPU_ROWS
#define TPU_ROWS 8              /* contraction rows per array pass */
#endif
#ifndef TPU_COLS
#define TPU_COLS 8              /* output columns per array pass   */
#endif

#define TPU_TOKENS_MAX    32u      /* token rows per dispatch (mxu.sv MAX_TOKENS) */
#define TPU_TILES_MAX     255u     /* depth_tiles / col_tiles are 8 bits          */
#define TPU_DMA_BYTES_MAX 0xFF00u  /* the 16-bit byte count, kept round           */

/* Elements per VPU pass. Bounded by the 10-bit count, kept even for QUANT4 (two
 * nibbles share a byte) and a multiple of the 16 int32 lanes so a chunk never
 * straddles a partial access. Also sets the int32 staging temp, at 4x this. */
#ifndef TPU_CHUNK
#define TPU_CHUNK 512u
#endif

/* ---- where a tensor lives ------------------------------------------------ */
#define TPU_SPAD 0u
#define TPU_DRAM 1u

/* `row_bytes` is the stride between rows of the tensor this buffer names, read
 * only by the 2-D primitives (the elementwise ops treat operands as flat). It
 * is a BYTE stride whatever the element width is, so an int32 [m][n] has
 * row_bytes = 4*n and an int4 [m][n] has row_bytes = n/2. */
typedef struct {
    uint32_t addr;
    uint32_t row_bytes;
    uint32_t memory;            /* TPU_SPAD or TPU_DRAM */
} tpu_buf;

#define TPU_SPAD_AT(a)          ((tpu_buf){ (uint32_t)(a), 0u, TPU_SPAD })
#define TPU_DRAM_AT(a)          ((tpu_buf){ (uint32_t)(a), 0u, TPU_DRAM })
#define TPU_SPAD_ROWS(a, bytes) ((tpu_buf){ (uint32_t)(a), (uint32_t)(bytes), TPU_SPAD })
#define TPU_DRAM_ROWS(a, bytes) ((tpu_buf){ (uint32_t)(a), (uint32_t)(bytes), TPU_DRAM })

/* ---- the staging arena --------------------------------------------------- */
/* A bump allocator over one scratchpad region. Every primitive allocates on the
 * way in and rewinds on the way out, so the high-water mark is the largest
 * single primitive rather than the sum of them. */
typedef struct {
    uint32_t next_free;
    uint32_t limit;
} tpu_arena;

static inline void tpu_arena_init(tpu_arena *arena, uint32_t base,
                                  uint32_t bytes)
{
    arena->next_free = base;
    arena->limit     = base + bytes;
}

static inline uint32_t tpu_arena_alloc(tpu_arena *arena, uint32_t bytes)
{
    uint32_t addr = arena->next_free;

    bytes = (bytes + 3u) & ~3u;         /* int32 staging wants 4-byte starts */
    TPU_ASSERT(addr + bytes <= arena->limit,
               "arena overflow — give the kernel a bigger one");
    arena->next_free = addr + bytes;
    return addr;
}

/* ---- moves --------------------------------------------------------------- */

/* `bytes` between scratchpad and DRAM, any length. One command per 64 KB. */
static inline void tpu_move(uint32_t spad_addr, uint32_t dram_addr,
                            uint32_t bytes, unsigned direction)
{
    while (bytes) {
        uint32_t chunk = (bytes > TPU_DMA_BYTES_MAX) ? TPU_DMA_BYTES_MAX : bytes;

        tpu_dma(spad_addr, dram_addr, chunk, direction);
        spad_addr += chunk;
        dram_addr += chunk;
        bytes     -= chunk;
    }
}

/* A `rows` x `cols` byte block, honouring both row strides. Dense on both sides
 * is one transfer; anything else is one transfer per row, because a DMA range
 * is contiguous and a strided block is not. Keeping a staged block dense is
 * therefore worth real commands, which is why every staging buffer below is
 * allocated at exactly the block's width. */
static inline void tpu_move2d(uint32_t spad_addr, uint32_t spad_row_bytes,
                              uint32_t dram_addr, uint32_t dram_row_bytes,
                              uint32_t rows, uint32_t cols, unsigned direction)
{
    if (spad_row_bytes == cols && dram_row_bytes == cols) {
        tpu_move(spad_addr, dram_addr, rows * cols, direction);
        return;
    }
    for (uint32_t r = 0; r < rows; r++)
        tpu_move(spad_addr + r * spad_row_bytes,
                 dram_addr + r * dram_row_bytes, cols, direction);
}

/* dst[col][row] = src[row][col], int8, via `dram_scratch` (rows*cols bytes).
 *
 * The DMA is the only unit that transposes and one of its sides is always DRAM,
 * so this is a transposing spill followed by a plain fill. No barrier between
 * them: both ride the DMA's own in-order queue.
 *
 * The spill is split by rows when the block exceeds one DMA command. That works
 * because dma.sv writes dst[col*dst_row_bytes + row]: a slice covering rows
 * [row0, row0+n) lands at `dram_scratch + row0` as long as dst_row_bytes stays
 * the FULL row count. */
static inline void tpu_transpose_int8(uint32_t dst_addr, uint32_t dst_row_bytes,
                                      uint32_t src_addr, uint32_t src_row_bytes,
                                      uint32_t rows, uint32_t cols,
                                      uint32_t dram_scratch)
{
    TPU_ASSERT(cols <= TPU_DMA_BYTES_MAX,
               "transpose: a source row exceeds one DMA");
    if (rows == 0u || cols == 0u)
        return;

    for (uint32_t row0 = 0; row0 < rows; ) {
        uint32_t rows_in_cmd = 0, bytes = 0;

        while (row0 + rows_in_cmd < rows && bytes + cols <= TPU_DMA_BYTES_MAX) {
            rows_in_cmd++;
            bytes += cols;
        }
        tpu_dma_transpose(src_addr + row0 * src_row_bytes,
                          dram_scratch + row0, bytes, TPU_DMA_SPILL,
                          cols, src_row_bytes, rows);
        row0 += rows_in_cmd;
    }
    /* The DRAM image is now [cols][rows], dense. */
    tpu_move2d(dst_addr, dst_row_bytes, dram_scratch, rows, cols, rows,
               TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);
}

/* ---- elementwise --------------------------------------------------------- */

/* dst = narrow(widen(src0, src1)) over `count` elements, chunked at TPU_CHUNK.
 *
 * The pair is fused rather than run as two full passes because a widening op
 * writes int32 and only an explicit narrow comes back down (docs/vpu.md): two
 * separate passes would need an int32 temp the size of the whole tensor.
 * QUANT4 is the one op with no widening half — int8 in, 4 bits out — and is
 * selected by passing it as `narrow_op`.
 *
 * Any operand may be in DRAM; it is then streamed through the arena a chunk at
 * a time, which is what makes this work on a tensor larger than the scratchpad.
 * All-resident operands stage nothing and take no barrier until the drain on
 * the way out. */
static inline void tpu_elementwise(unsigned widen_op, unsigned narrow_op,
                                   tpu_buf dst, tpu_buf src0, tpu_buf src1,
                                   uint32_t count, uint32_t rq_word,
                                   tpu_arena *arena)
{
    const unsigned packing    = (narrow_op == TPU_V_QUANT4);
    const unsigned needs_src1 = !packing && (widen_op == TPU_V_ADD);

    TPU_ASSERT(!packing || (count % 2u) == 0u,
               "quant4: count must be even (a nibble pair shares a byte)");
    if (count == 0u)
        return;

    const uint32_t arena_mark = arena->next_free;

    /* Everything resident: no staging, no barrier until the drain, and the loop
     * body is the two pushes and nothing else. Kept as its own path rather than
     * a degenerate case of the streaming one below, because a per-chunk
     * residency test costs the CPU more clocks than the VPU spends on the
     * chunk — and because it folds to the hand-written loop when the shapes are
     * compile-time constants. */
    if (dst.memory == TPU_SPAD && src0.memory == TPU_SPAD &&
        (!needs_src1 || src1.memory == TPU_SPAD)) {
        const uint32_t wide_tmp =
            packing ? 0u : tpu_arena_alloc(arena, TPU_CHUNK * 4u);

        for (uint32_t i = 0; i < count; i += TPU_CHUNK) {
            const uint32_t chunk =
                (count - i < TPU_CHUNK) ? (count - i) : TPU_CHUNK;

            if (packing) {
                tpu_vpu(TPU_V_QUANT4, dst.addr + (i >> 1), src0.addr + i, 0u,
                        chunk, rq_word);
            } else {
                /* A one-source widening op ignores src1 (vpu.sv needs_src1),
                 * but it still rides in the command — send zero rather than a
                 * live address, so a trace of a `relu` reads as one. */
                tpu_vpu(widen_op, wide_tmp, src0.addr + i,
                        needs_src1 ? src1.addr + i : 0u, chunk, 0u);
                tpu_vpu(narrow_op, dst.addr + i, wide_tmp, 0u, chunk, rq_word);
            }
        }
        tpu_wait(TPU_U_VPU);
        arena->next_free = arena_mark;
        return;
    }

    const uint32_t src0_stage =
        (src0.memory == TPU_DRAM) ? tpu_arena_alloc(arena, TPU_CHUNK) : 0u;
    const uint32_t src1_stage = (needs_src1 && src1.memory == TPU_DRAM)
                                    ? tpu_arena_alloc(arena, TPU_CHUNK) : 0u;
    const uint32_t dst_stage =
        (dst.memory == TPU_DRAM) ? tpu_arena_alloc(arena, TPU_CHUNK) : 0u;
    const uint32_t wide_tmp =
        packing ? 0u : tpu_arena_alloc(arena, TPU_CHUNK * 4u);

    for (uint32_t i = 0; i < count; i += TPU_CHUNK) {
        const uint32_t chunk = (count - i < TPU_CHUNK) ? (count - i) : TPU_CHUNK;
        const uint32_t dst_off = packing ? (i >> 1) : i;
        uint32_t src0_spad, src1_spad = 0u, dst_spad;
        unsigned staged_inputs = 0;

        if (src0.memory == TPU_DRAM) {
            tpu_move(src0_stage, src0.addr + i, chunk, TPU_DMA_FILL);
            src0_spad = src0_stage;
            staged_inputs = 1;
        } else {
            src0_spad = src0.addr + i;
        }
        if (needs_src1) {
            if (src1.memory == TPU_DRAM) {
                tpu_move(src1_stage, src1.addr + i, chunk, TPU_DMA_FILL);
                src1_spad = src1_stage;
                staged_inputs = 1;
            } else {
                src1_spad = src1.addr + i;
            }
        }
        dst_spad = (dst.memory == TPU_DRAM) ? dst_stage : (dst.addr + dst_off);

        if (staged_inputs)
            tpu_wait(TPU_U_DMA);

        if (packing) {
            tpu_vpu(TPU_V_QUANT4, dst_spad, src0_spad, 0u, chunk, rq_word);
        } else {
            tpu_vpu(widen_op, wide_tmp, src0_spad, src1_spad, chunk, 0u);
            tpu_vpu(narrow_op, dst_spad, wide_tmp, 0u, chunk, rq_word);
        }

        if (dst.memory == TPU_DRAM) {
            tpu_wait(TPU_U_VPU);
            tpu_move(dst_stage, dst.addr + dst_off,
                     packing ? (chunk >> 1) : chunk, TPU_DMA_SPILL);
            tpu_wait(TPU_U_DMA);
        } else if (staged_inputs) {
            tpu_wait(TPU_U_VPU);        /* the staging buffers are reused */
        }
    }
    tpu_wait(TPU_U_VPU);
    arena->next_free = arena_mark;
}

/* dst = narrow(a + b). `narrow_op` is REQUANT or DYT — the two differ only in
 * the clip, and choosing DYT here is how a kernel says "this residual add is
 * also norm1". */
static inline void tpu_add_narrow(unsigned narrow_op, tpu_buf dst, tpu_buf a,
                                  tpu_buf b, uint32_t count, uint32_t rq_word,
                                  tpu_arena *arena)
{
    tpu_elementwise(TPU_V_ADD, narrow_op, dst, a, b, count, rq_word, arena);
}

/* dst = requant(relu(src)). */
static inline void tpu_relu_narrow(tpu_buf dst, tpu_buf src, uint32_t count,
                                   uint32_t rq_word, tpu_arena *arena)
{
    tpu_elementwise(TPU_V_RELU, TPU_V_REQUANT, dst, src, TPU_SPAD_AT(0), count,
                    rq_word, arena);
}

/* dst = quant4(src): int8 in, 4 bits out, so the destination advances half as
 * fast as the source. This is what turns an activation into a legal weight
 * operand without a repacking pass or a host round trip. */
static inline void tpu_pack4(tpu_buf dst, tpu_buf src, uint32_t count,
                             uint32_t rq_word, tpu_arena *arena)
{
    tpu_elementwise(0u, TPU_V_QUANT4, dst, src, TPU_SPAD_AT(0), count, rq_word,
                    arena);
}

/* int32 -> int4-in-int8 over a scratchpad block, chunked. Used by the matmul
 * below when a split contraction leaves int32 partials the store could not
 * narrow; a kernel holding its own int32 block can call it directly. */
static inline void tpu_requant_rows(uint32_t dst_addr, uint32_t dst_row_bytes,
                                    uint32_t src32_addr,
                                    uint32_t src32_row_bytes, uint32_t rows,
                                    uint32_t cols, uint32_t rq_word)
{
    /* Dense on both sides is one flat pass over the block; otherwise one pass
     * per row, since a VPU op walks its operands contiguously. */
    const unsigned dense = (dst_row_bytes == cols) &&
                           (src32_row_bytes == cols * 4u);
    const uint32_t pass_count     = dense ? 1u : rows;
    const uint32_t elems_per_pass = dense ? rows * cols : cols;

    for (uint32_t r = 0; r < pass_count; r++) {
        const uint32_t dst_row_addr = dst_addr + r * dst_row_bytes;
        const uint32_t src_row_addr = src32_addr + r * src32_row_bytes;

        for (uint32_t i = 0; i < elems_per_pass; i += TPU_CHUNK) {
            const uint32_t chunk = (elems_per_pass - i < TPU_CHUNK)
                                       ? (elems_per_pass - i) : TPU_CHUNK;

            tpu_vpu(TPU_V_REQUANT, dst_row_addr + i, src_row_addr + i * 4u, 0u,
                    chunk, rq_word);
        }
    }
    tpu_wait(TPU_U_VPU);
}

/* ---- matmul -------------------------------------------------------------- */

/* out[rows][cols] = act[rows][depth] @ wgt[depth][cols], at any size, with each
 * operand in either memory.
 *
 *   act   int8, act.row_bytes >= depth
 *   wgt   int4 row-major, two per byte, wgt.row_bytes >= cols/2
 *   out   int8 when `rq_word` is a {m0,n} literal, int32 when it is 0 (m0 = 0
 *         is not a legal multiplier, so zero is free as the sentinel).
 *         out.row_bytes is bytes either way: >= cols for int8, >= 4*cols for
 *         int32.
 *
 * `depth` must be a multiple of TPU_ROWS and `cols` of TPU_COLS: the array
 * consumes whole tiles and has no lane masking, so a partial tile would read
 * neighbouring data rather than zeros. Pad the tensor (adder.c pads the output
 * head's 13 columns to 16) rather than passing a short one. */
typedef struct {
    uint32_t rows;              /* M */
    uint32_t depth;             /* K, the contraction */
    uint32_t cols;              /* N */
    tpu_buf  act;
    tpu_buf  wgt;
    tpu_buf  out;
    uint32_t rq_word;
} tpu_gemm;

/* Arena bytes one block of this shape would need. The output term is the
 * interesting one: an UNSPLIT contraction needs no int32 buffer at all, because
 * the array holds its partials in result_buf across the whole depth loop and
 * narrows on the single store. Splitting the depth is what forces 4 bytes per
 * output element into the arena, which is why the block chooser gives it up
 * last. */
__attribute__((always_inline))
static inline uint32_t tpu_gemm_arena_bytes(const tpu_gemm *gemm, uint32_t rows,
                                            uint32_t depth, uint32_t cols)
{
    const unsigned requantize = (gemm->rq_word != 0u);
    const unsigned split      = (depth < gemm->depth);
    uint32_t bytes = 0u;

    if (gemm->act.memory == TPU_DRAM) bytes += rows * depth;
    if (gemm->wgt.memory == TPU_DRAM) bytes += (depth * cols) >> 1;
    if (requantize && gemm->out.memory == TPU_DRAM) bytes += rows * cols;
    if ((!requantize && gemm->out.memory == TPU_DRAM) || (requantize && split))
        bytes += rows * cols * 4u;
    return bytes;
}

/* The largest block the arena holds, starting from the largest the hardware
 * allows. Halve whichever extent is biggest until it fits; the contraction goes
 * last because it is the only one whose split ADDS a buffer. Every division is
 * by TPU_ROWS / TPU_COLS, which are compile-time constants — the core is built
 * with ENABLE_DIV(0) and nothing here may emit a runtime divide.
 *
 * The early-out is not just a speed-up of this function: with the whole shape
 * fitting, the output term of tpu_gemm_arena_bytes stops depending on
 * `rq_word`, so the whole test is constant and gcc deletes the search loop, the
 * block loops and every staging branch behind it. */
__attribute__((always_inline))
static inline void tpu_gemm_blocks(const tpu_gemm *gemm, uint32_t arena_bytes,
                                   uint32_t *rows_out, uint32_t *depth_out,
                                   uint32_t *cols_out)
{
    uint32_t rows = (gemm->rows < TPU_TOKENS_MAX) ? gemm->rows : TPU_TOKENS_MAX;
    uint32_t depth_tiles = gemm->depth / TPU_ROWS;
    uint32_t col_tiles   = gemm->cols / TPU_COLS;

    if (gemm->rows <= TPU_TOKENS_MAX && depth_tiles <= TPU_TILES_MAX &&
        col_tiles <= TPU_TILES_MAX &&
        tpu_gemm_arena_bytes(gemm, gemm->rows, gemm->depth, gemm->cols)
            <= arena_bytes) {
        *rows_out  = gemm->rows;
        *depth_out = gemm->depth;
        *cols_out  = gemm->cols;
        return;
    }

    if (depth_tiles > TPU_TILES_MAX) depth_tiles = TPU_TILES_MAX;
    if (col_tiles   > TPU_TILES_MAX) col_tiles   = TPU_TILES_MAX;
    /* A dimension shorter than one tile is a caller bug the assertions in
     * tpu_matmul name; clamping keeps the block loops from stepping by zero
     * meanwhile. */
    if (depth_tiles == 0u) depth_tiles = 1u;
    if (col_tiles   == 0u) col_tiles   = 1u;

    while (tpu_gemm_arena_bytes(gemm, rows, depth_tiles * TPU_ROWS,
                                col_tiles * TPU_COLS) > arena_bytes) {
        if (col_tiles > 1u && col_tiles * TPU_COLS >= rows)
            col_tiles = (col_tiles + 1u) >> 1;
        else if (rows > 1u)
            rows = (rows + 1u) >> 1;
        else if (depth_tiles > 1u)
            depth_tiles = (depth_tiles + 1u) >> 1;
        else
            break;
    }
    *rows_out  = rows;
    *depth_out = depth_tiles * TPU_ROWS;
    *cols_out  = col_tiles * TPU_COLS;
    TPU_ASSERT(tpu_gemm_arena_bytes(gemm, *rows_out, *depth_out, *cols_out)
                   <= arena_bytes,
               "matmul: the arena cannot hold even a single-tile block");
}

__attribute__((always_inline))
static inline void tpu_matmul(const tpu_gemm *gemm, tpu_arena *arena)
{
    if (gemm->rows == 0u || gemm->depth == 0u || gemm->cols == 0u)
        return;
    TPU_ASSERT(gemm->depth % TPU_ROWS == 0u,
               "matmul: depth must be a multiple of TPU_ROWS");
    TPU_ASSERT(gemm->cols % TPU_COLS == 0u,
               "matmul: cols must be a multiple of TPU_COLS");

    uint32_t rows_per_block, depth_per_block, cols_per_block;
    tpu_gemm_blocks(gemm, arena->limit - arena->next_free, &rows_per_block,
                    &depth_per_block, &cols_per_block);

    const unsigned requantize = (gemm->rq_word != 0u);
    const unsigned split_depth = (depth_per_block < gemm->depth);
    const unsigned stage_act = (gemm->act.memory == TPU_DRAM);
    const unsigned stage_wgt = (gemm->wgt.memory == TPU_DRAM);
    const unsigned stage_out_int8  = requantize &&
                                     (gemm->out.memory == TPU_DRAM);
    const unsigned stage_out_int32 =
        (!requantize && gemm->out.memory == TPU_DRAM) ||
        (requantize && split_depth);
    /* The store narrows only when the whole contraction ran in one dispatch. A
     * split leaves int32 partials, and those come down through the VPU. */
    const unsigned narrow_on_store = requantize && !stage_out_int32;
    /* With one contraction block the weight depends only on the column block,
     * so it can be staged once and reused down the whole row loop. */
    const unsigned stage_wgt_once = stage_wgt && !split_depth;

    const uint32_t arena_mark = arena->next_free;
    const uint32_t act_buf = stage_act
        ? tpu_arena_alloc(arena, rows_per_block * depth_per_block) : 0u;
    const uint32_t wgt_buf = stage_wgt
        ? tpu_arena_alloc(arena, (depth_per_block * cols_per_block) >> 1) : 0u;
    const uint32_t out8_buf = stage_out_int8
        ? tpu_arena_alloc(arena, rows_per_block * cols_per_block) : 0u;
    const uint32_t out32_buf = stage_out_int32
        ? tpu_arena_alloc(arena, rows_per_block * cols_per_block * 4u) : 0u;

    /* MXU_GEOM is sticky in this unit's queue, so it is re-pushed only when a
     * block's shape actually differs from the last one — which is exactly the
     * remainder blocks at the far edge of each axis. */
    uint32_t last_act_row = 0u, last_out_row = 0u, last_wgt_row = 0u;
    uint32_t last_depth_tiles = 0u, last_col_tiles = 0u, last_token_rows = 0u;
    unsigned have_geom = 0;

    for (uint32_t col_base = 0; col_base < gemm->cols;
         col_base += cols_per_block) {
        const uint32_t cols = (gemm->cols - col_base < cols_per_block)
                                  ? (gemm->cols - col_base) : cols_per_block;

        if (stage_wgt_once) {
            tpu_move2d(wgt_buf, cols >> 1,
                       gemm->wgt.addr + (col_base >> 1), gemm->wgt.row_bytes,
                       gemm->depth, cols >> 1, TPU_DMA_FILL);
            tpu_wait(TPU_U_DMA);
        }

        for (uint32_t row_base = 0; row_base < gemm->rows;
             row_base += rows_per_block) {
            const uint32_t rows = (gemm->rows - row_base < rows_per_block)
                                      ? (gemm->rows - row_base)
                                      : rows_per_block;

            for (uint32_t depth_base = 0; depth_base < gemm->depth;
                 depth_base += depth_per_block) {
                const uint32_t depth =
                    (gemm->depth - depth_base < depth_per_block)
                        ? (gemm->depth - depth_base) : depth_per_block;
                uint32_t act_addr, wgt_addr, out_addr;
                uint32_t act_row, wgt_row, out_row;
                unsigned filled = 0;

                if (stage_wgt && !stage_wgt_once) {
                    tpu_move2d(wgt_buf, cols >> 1,
                               gemm->wgt.addr + depth_base * gemm->wgt.row_bytes
                                   + (col_base >> 1),
                               gemm->wgt.row_bytes, depth, cols >> 1,
                               TPU_DMA_FILL);
                    filled = 1;
                }
                if (stage_act) {
                    tpu_move2d(act_buf, depth,
                               gemm->act.addr + row_base * gemm->act.row_bytes
                                   + depth_base,
                               gemm->act.row_bytes, rows, depth, TPU_DMA_FILL);
                    filled = 1;
                }
                if (filled)
                    tpu_wait(TPU_U_DMA);

                if (stage_act) {
                    act_addr = act_buf;
                    act_row  = depth;
                } else {
                    act_addr = gemm->act.addr
                             + row_base * gemm->act.row_bytes + depth_base;
                    act_row  = gemm->act.row_bytes;
                }

                if (stage_wgt) {
                    wgt_addr = wgt_buf;
                    wgt_row  = cols >> 1;
                } else {
                    wgt_addr = gemm->wgt.addr
                             + depth_base * gemm->wgt.row_bytes
                             + (col_base >> 1);
                    wgt_row  = gemm->wgt.row_bytes;
                }

                /* out_row is always the int32 row stride: the array derives the
                 * int8 store stride as out_row/4 (mxu.sv res_row_stride_i8). */
                if (stage_out_int32) {
                    out_addr = out32_buf;
                    out_row  = cols * 4u;
                } else if (stage_out_int8) {
                    out_addr = out8_buf;
                    out_row  = cols * 4u;
                } else if (requantize) {
                    out_addr = gemm->out.addr
                             + row_base * gemm->out.row_bytes + col_base;
                    out_row  = gemm->out.row_bytes * 4u;
                } else {
                    out_addr = gemm->out.addr
                             + row_base * gemm->out.row_bytes + col_base * 4u;
                    out_row  = gemm->out.row_bytes;
                }

                TPU_ASSERT(act_row <= 0xFFFFu && out_row <= 0xFFFFu &&
                               wgt_row <= 0xFFFFu,
                           "matmul: a geometry stride does not fit 16 bits");

                if (!have_geom || act_row != last_act_row ||
                    out_row != last_out_row || wgt_row != last_wgt_row ||
                    depth / TPU_ROWS != last_depth_tiles ||
                    cols / TPU_COLS != last_col_tiles ||
                    rows != last_token_rows) {
                    last_act_row     = act_row;
                    last_out_row     = out_row;
                    last_wgt_row     = wgt_row;
                    last_depth_tiles = depth / TPU_ROWS;
                    last_col_tiles   = cols / TPU_COLS;
                    last_token_rows  = rows;
                    have_geom = 1;
                    tpu_mxu_geom(last_act_row, last_out_row, last_wgt_row,
                                 last_depth_tiles, last_col_tiles,
                                 last_token_rows);
                }

                tpu_mxu_mm(out_addr, act_addr, wgt_addr,
                           TPU_MM_TILED | (depth_base ? TPU_MM_ACC : 0u) |
                               (narrow_on_store ? TPU_MM_RQ : 0u),
                           gemm->rq_word);

                /* Only a reused staging buffer forces a barrier inside the
                 * depth loop; back-to-back dispatches out of resident operands
                 * stay queued. */
                if (stage_act || (stage_wgt && !stage_wgt_once))
                    tpu_wait(TPU_U_MXU);
            }
            tpu_wait(TPU_U_MXU);

            if (stage_out_int32 && requantize)
                tpu_requant_rows(stage_out_int8
                                     ? out8_buf
                                     : gemm->out.addr
                                           + row_base * gemm->out.row_bytes
                                           + col_base,
                                 stage_out_int8 ? cols : gemm->out.row_bytes,
                                 out32_buf, cols * 4u, rows, cols,
                                 gemm->rq_word);

            if (gemm->out.memory == TPU_DRAM) {
                const uint32_t width = stage_out_int8 ? cols : cols * 4u;

                tpu_move2d(stage_out_int8 ? out8_buf : out32_buf, width,
                           gemm->out.addr + row_base * gemm->out.row_bytes
                               + (stage_out_int8 ? col_base : col_base * 4u),
                           gemm->out.row_bytes, rows, width, TPU_DMA_SPILL);
                tpu_wait(TPU_U_DMA);
            }
        }
    }
    arena->next_free = arena_mark;
}

#endif /* TPULIB_H */
