/* tpu.h — the MMIO command plane, as seen from firmware.
 *
 * Mirrors rtl/cpu_subsys.sv (the address map) and rtl/cmd_{mxu,vpu,dma}.sv (the
 * command fields). Each builder packs one 128-bit macro-op exactly as the
 * decoder reads it; there is no driver abstraction here.
 *
 * Address map:
 *   0x8000_0000 + 0x10*unit   4-word command port, COMMITS ON WORD 3
 *   0x8000_0040               status: issued/retired/level per unit
 *   0x8000_0070               write = done; read = unit_idle
 *   0x9000_0000               the scratchpad, as CPU-addressable words
 *
 * A full queue withholds the write response on word 3, so the CPU stalls inside
 * the store and flow control needs no software. Cross-unit ordering IS
 * software's job: the three queues are independent, so anything the MXU reads
 * must be tpu_wait(TPU_U_DMA)'d first.
 */
#ifndef TPU_H
#define TPU_H

#include <stdint.h>

#define TPU_MMIO     0x80000000u
#define TPU_SPAD_WIN 0x90000000u   /* the scratchpad, as CPU memory */

#define TPU_U_MXU 0u
#define TPU_U_VPU 1u
#define TPU_U_DMA 2u

#define TPU_CMD(unit) ((volatile uint32_t *)(TPU_MMIO + 0x10u * (unit)))
#define TPU_STAT(i)   (*(volatile uint32_t *)(TPU_MMIO + 0x40u + 4u * (i)))

/* Per unit: issued, retired, queue level. Four words apart, in unit order. */
#define TPU_ISSUED(unit)  TPU_STAT(4u * (unit) + 0u)
#define TPU_RETIRED(unit) TPU_STAT(4u * (unit) + 1u)
#define TPU_LEVEL(unit)   TPU_STAT(4u * (unit) + 2u)

#define TPU_DONE (*(volatile uint32_t *)(TPU_MMIO + 0x70u))

/* ---- the four target-specific primitives ---------------------------------
 *
 * Everything below these four is plain uint32_t arithmetic, which is what lets
 * -DTPU_TRACE compile the *same* kernel source with the host compiler and have
 * it print its command trace instead of executing it. Only these four are
 * swapped, so no builder is duplicated and the trace cannot drift from the
 * firmware. New builders written in terms of tpu_push are traced for free.
 *
 * THE SCRATCHPAD WINDOW (tpu_spad_ld / tpu_spad_st) is not about commands:
 * cpu_subsys.sv decodes 0x9xxx_xxxx onto the scratchpad's S port, so the CPU can
 * read what the array computed and branch on it (fw/infer.c's argmax). Two
 * rules, both from cpu_subsys.sv:
 *
 *   32-BIT ACCESSES ONLY, 4-byte aligned. The S port has no byte strobes, so an
 *   `sb` or `sh` also overwrites its neighbours in the word.
 *
 *   FENCE FIRST. A load is not a command: it does not wait for the queues, so
 *   tpu_wait() the unit that produces the data or the read is stale. The S port
 *   also loses arbitration to the MXU and VPU, so a read taken mid-matmul
 *   stalls the core inside the load.
 */
#ifdef TPU_TRACE

/* Implemented by mock/tpu_trace.c. Declared, not defined, so a builder that
 * bypasses tpu_push fails to link rather than silently escaping the trace. */
void tpu_trace_push(unsigned unit, uint32_t word0, uint32_t word1,
                    uint32_t word2, uint32_t word3);
void tpu_trace_wait(unsigned unit);
uint32_t tpu_trace_spad_ld(uint32_t byte_addr);
void tpu_trace_spad_st(uint32_t byte_addr, uint32_t value);

static inline void tpu_push(unsigned unit, uint32_t word0, uint32_t word1,
                            uint32_t word2, uint32_t word3)
{
    tpu_trace_push(unit, word0, word1, word2, word3);
}

static inline void tpu_wait(unsigned unit)
{
    tpu_trace_wait(unit);
}

static inline uint32_t tpu_spad_ld(uint32_t byte_addr)
{
    return tpu_trace_spad_ld(byte_addr);
}

static inline void tpu_spad_st(uint32_t byte_addr, uint32_t value)
{
    tpu_trace_spad_st(byte_addr, value);
}

#else

/* Push one command. The store to word 3 assembles the 128 bits and enqueues, so
 * a unit can never see a torn command. */
static inline void tpu_push(unsigned unit, uint32_t word0, uint32_t word1,
                            uint32_t word2, uint32_t word3)
{
    volatile uint32_t *port = TPU_CMD(unit);
    port[0] = word0;
    port[1] = word1;
    port[2] = word2;
    port[3] = word3;
}

/* Block until everything pushed to `unit` has retired. The counters free-run
 * across runs (only rst_n clears them), so compare as a signed difference
 * rather than against a fixed sequence number. */
static inline void tpu_wait(unsigned unit)
{
    uint32_t issued = TPU_ISSUED(unit);

    while ((int32_t)(issued - TPU_RETIRED(unit)) > 0)
        ;
}

/* One scratchpad word, by byte address. `volatile` so polling a tensor the DMA
 * is still filling cannot be hoisted out of the poll loop. */
static inline uint32_t tpu_spad_ld(uint32_t byte_addr)
{
    return *(volatile uint32_t *)(TPU_SPAD_WIN + byte_addr);
}

static inline void tpu_spad_st(uint32_t byte_addr, uint32_t value)
{
    *(volatile uint32_t *)(TPU_SPAD_WIN + byte_addr) = value;
}

#endif /* TPU_TRACE */

/* ---- DMA (cmd_dma.sv) --------------------------------------------------- */

#define TPU_DMA_MOVE  0x01u
#define TPU_DMA_FILL  0u        /* DRAM -> scratchpad */
#define TPU_DMA_SPILL 1u        /* scratchpad -> DRAM */

/* `bytes` between scratchpad `spad_addr` and 19-bit `dram_addr`. Linear: the
 * transpose geometry words are left zero. */
static inline void tpu_dma(uint32_t spad_addr, uint32_t dram_addr,
                           uint32_t bytes, unsigned direction)
{
    tpu_push(TPU_U_DMA,
             TPU_DMA_MOVE | ((uint32_t)direction << 8) | (spad_addr << 16),
             dram_addr,
             bytes,
             0u);
}

/* The transposing mode (docs/dma.md 5). Same byte count, but the source is read
 * row-major over `src_cols` columns at `src_row_bytes` stride while the
 * destination is written down columns at `dst_row_bytes`:
 *
 *     dst[col * dst_row_bytes + row] = src[row * src_row_bytes + col]
 *
 * This is what turns an activation the array produced row-major into a weight
 * operand. `src_cols == 0` degenerates to one row (a strided gather), which is
 * dma.sv's zero-means-unset fallback rather than a fault. */
static inline void tpu_dma_transpose(uint32_t spad_addr, uint32_t dram_addr,
                                     uint32_t bytes, unsigned direction,
                                     uint32_t src_cols, uint32_t src_row_bytes,
                                     uint32_t dst_row_bytes)
{
    tpu_push(TPU_U_DMA,
             TPU_DMA_MOVE | ((uint32_t)direction << 8) | (1u << 9) |
                 (spad_addr << 16),
             dram_addr,
             bytes | (src_cols << 16),
             src_row_bytes | (dst_row_bytes << 16));
}

/* ---- MXU (cmd_mxu.sv) --------------------------------------------------- */

#define TPU_MXU_GEOM 0x01u
#define TPU_MXU_MM   0x02u

#define TPU_MM_ACC   (1u << 8)   /* add into the existing int32 output */
#define TPU_MM_RQ    (1u << 9)   /* narrow the store via the {m0,n} literal */
#define TPU_MM_TILED (1u << 10)  /* use the strides and tile counts below */

/* Operand geometry for every matmul that follows it in this unit's queue.
 * Strides are bytes:
 *
 *   act_row_bytes = K            int8 activation
 *   out_row_bytes = N * 4        ALWAYS the int32 stride — a requantized store
 *                                derives its int8 stride as out_row_bytes/4
 *   wgt_row_bytes = N / 2        int4 weights, two nibbles per byte
 *
 * `depth_tiles` and `col_tiles` are the array's passes over K and N (8 bits
 * each); `token_rows` is M, 6 bits. */
static inline void tpu_mxu_geom(uint32_t act_row_bytes, uint32_t out_row_bytes,
                                uint32_t wgt_row_bytes, uint32_t depth_tiles,
                                uint32_t col_tiles, uint32_t token_rows)
{
    tpu_push(TPU_U_MXU,
             TPU_MXU_GEOM | (act_row_bytes << 16),
             out_row_bytes | (wgt_row_bytes << 16),
             depth_tiles | (col_tiles << 8) | (token_rows << 16),
             0u);
}

/* One matmul: out = act @ wgt over the geometry above. `rq_word` is the {m0,n}
 * literal and is read only when TPU_MM_RQ is set. */
static inline void tpu_mxu_mm(uint32_t out_addr, uint32_t act_addr,
                              uint32_t wgt_addr, uint32_t flags,
                              uint32_t rq_word)
{
    tpu_push(TPU_U_MXU,
             TPU_MXU_MM | flags | (out_addr << 16),
             act_addr | (wgt_addr << 16),
             rq_word,
             0u);
}

/* ---- VPU (cmd_vpu.sv) --------------------------------------------------- */

/* The VPU has one command. 0x02 was VPU_GEOM, carrying the vecmatmul macro
 * op's row/column geometry; both were removed (rtl/vpu.sv header) and the
 * opcode is a retired hole. */
#define TPU_VPU_OP   0x01u

/* vpu.sv's VOP_* encodings — the opcode the unit decodes. 13 was VECMM. */
#define TPU_V_DOT     0u
#define TPU_V_ADD     1u
#define TPU_V_RELU    3u
#define TPU_V_REQUANT 10u
#define TPU_V_DYT     16u
#define TPU_V_QUANT4  17u

/* One vector op over `count` elements. `rq_word` is the {m0,n} literal and is
 * read only by REQUANT / DYT / QUANT4.
 *
 * Element widths are per-op and are the caller's business (docs/vpu.md):
 * ADD/RELU read int8 and write int32; REQUANT/DYT read int32 and write int8;
 * QUANT4 reads int8 and writes 4 bits, so its destination advances HALF as fast
 * as its source and `count` must be even. */
static inline void tpu_vpu(unsigned op, uint32_t dst_addr, uint32_t src0_addr,
                           uint32_t src1_addr, uint32_t count, uint32_t rq_word)
{
    tpu_push(TPU_U_VPU,
             TPU_VPU_OP | (op << 8) | (dst_addr << 16),
             src0_addr | (src1_addr << 16),
             count | (rq_word << 16),
             0u);
}

#endif /* TPU_H */
