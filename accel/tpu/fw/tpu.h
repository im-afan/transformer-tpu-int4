/* tpu.h — the MMIO command plane, as seen from firmware. See docs/fw.md. */
#ifndef TPU_H
#define TPU_H

#include <stdint.h>

/* Fixed by the bitstream; overridable so a kernel for a different array
 * geometry need not fork this file. One scratchpad word is N/2 bytes. */
#ifndef TPU_N
#define TPU_N 8
#endif
#define TPU_WORD_BYTES (TPU_N / 2)

/* vpu_vlen is a 10-bit field in the VPU macro-op and must be even, so a longer
 * elementwise pass is several commands. A chunk of a whole number of array
 * words keeps its byte count a whole number of scratchpad words too. This is a
 * property of the encoding, not of tpulib.h — a kernel writing raw commands has
 * to chunk as well, and a `vlen` past this truncates silently. */
#define TPU_VLEN_MAX   1023u
#define TPU_VCHUNK_MAX ((TPU_VLEN_MAX / TPU_N) * TPU_N)

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

/* ---- the four target-specific primitives; see docs/fw.md ---------------- */
#ifdef TPU_TRACE

/* Implemented by mock/tpu_trace.c; declared not defined so a builder that
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

/* `rows` rows of `len` int4 elements, with an independent row stride on each
 * side. A row is (len+1)/2 bytes; a zero stride means densely packed. DRAM
 * addressing is 19 bits, the whole part. */
static inline void tpu_dma(uint32_t spad_addr, uint32_t dram_addr,
                           uint32_t len, uint32_t rows, uint32_t dram_stride,
                           uint32_t spad_stride, unsigned direction)
{
    tpu_push(TPU_U_DMA,
             TPU_DMA_MOVE | ((uint32_t)direction << 8) | (spad_addr << 16),
             dram_addr,
             len | (rows << 16),
             dram_stride | (spad_stride << 16));
}

/* ---- MXU (cmd_mxu.sv) --------------------------------------------------- */

#define TPU_MXU_GEOM 0x01u
#define TPU_MXU_MM   0x02u

#define TPU_MM_ACC (1u << 8)   /* C = clip4(requant(A @ B) + C_old) */
#define TPU_MM_T   (1u << 9)   /* B is transposed in the matmul */

/* Strides and the contraction length for every matmul that follows in this
 * unit's queue. All three operands are row-major packed int4, so a stride is
 * elements/2 bytes and must be a multiple of TPU_WORD_BYTES. A zero stride
 * means the densely packed default:
 *
 *   a_stride   len/2                A is [N][len]
 *   b_stride   N/2, or len/2 when transposed
 *   c_stride   N/2                  C is [N][N]
 *
 * `len` is 16 bits, so a contraction never has to be split. */
static inline void tpu_mxu_geom(uint32_t a_stride, uint32_t b_stride,
                                uint32_t c_stride, uint32_t len)
{
    tpu_push(TPU_U_MXU,
             TPU_MXU_GEOM | (a_stride << 16),
             b_stride | (c_stride << 16),
             len,
             0u);
}

/* One N x N output block: C = requant(A @ B) over the geometry above. A, B and
 * C must lie in different scratchpad banks — they are read on the same clock
 * and a bank serves one requester per clock. */
static inline void tpu_mxu_mm(uint32_t c_addr, uint32_t a_addr,
                              uint32_t b_addr, uint32_t flags,
                              uint32_t rq_word)
{
    tpu_push(TPU_U_MXU,
             TPU_MXU_MM | flags | (c_addr << 16),
             a_addr | (b_addr << 16),
             rq_word,
             0u);
}

/* ---- VPU (cmd_vpu.sv) --------------------------------------------------- */

/* The VPU has one command. 0x02 was VPU_GEOM, carrying the vecmatmul macro
 * op's row/column geometry; both were removed and the opcode is a retired
 * hole. */
#define TPU_VPU_OP   0x01u

/* vpu.sv's VOP_* encodings. 13 was VECMM, 17 was QUANT4. */
#define TPU_V_DOT     0u
#define TPU_V_ADD     1u
#define TPU_V_RELU    3u
#define TPU_V_REQUANT 10u
#define TPU_V_DYT     16u
#define TPU_V_ARGMAX  18u

/* One vector op over `count` int4 elements. Every operand and every
 * elementwise result is packed int4, and the narrow is fused into the op, so
 * `rq_word` is read by all of them except DOT ({1,0} is the identity).
 *
 * `count` must be even and every address a multiple of TPU_WORD_BYTES: two
 * elements share a byte and the write strobe is per byte, so a half-filled
 * tail byte takes nibble 0 rather than keeping what was there. DOT and ARGMAX
 * are the ops whose destination is not int4 — each writes one int32 scalar,
 * ARGMAX's being an index into `count` rather than a value. */
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
