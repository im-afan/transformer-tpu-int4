/* dma_roundtrip.c — a matrix out of DRAM, through the scratchpad and back, one
 * tile at a time. Nothing computes: the run is the DMA and the cost of feeding
 * it, to be read against sram.sv's 1 clock/byte on a fill and 2 on a spill. A
 * tile narrower than the matrix makes every row of it a separate range, which
 * is what a strided move costs. */
#include "tpu.h"

/* Shape and the address map come from generate.py as -D; the defaults below are
 * only for a bare `make`. */
#ifndef ROWS
#define ROWS  128               /* the matrix is [ROWS][COLS] int4 */
#endif
#ifndef COLS
#define COLS  128
#endif
#ifndef TROWS
#define TROWS 32                /* one DMA command moves [TROWS][TCOLS] */
#endif
#ifndef TCOLS
#define TCOLS 32
#endif

#define ROW_BYTES  (COLS / 2)   /* the matrix's row pitch in DRAM */
#define TILE_BYTES (TCOLS / 2)  /* the staging tile is densely packed */

#ifndef DR_SRC
#define DR_SRC 0x00000u
#endif
#ifndef DR_DST
#define DR_DST 0x02000u
#endif
#ifndef SP_BUF
#define SP_BUF 0x00000u
#endif

int main(void)
{
    /* No fence inside the loop. One queue in program order already orders a
     * tile's spill behind its own fill, and a full queue stalls the store — so
     * this measures the DMA rather than the barrier the CPU would pay per
     * tile. Every tile reuses the one staging buffer. */
    for (unsigned i = 0; i < ROWS; i += TROWS) {
        unsigned trows = (ROWS - i) < TROWS ? (ROWS - i) : TROWS;

        for (unsigned j = 0; j < COLS; j += TCOLS) {
            unsigned tcols = (COLS - j) < TCOLS ? (COLS - j) : TCOLS;
            unsigned offset = i * ROW_BYTES + j / 2;

            tpu_dma(SP_BUF, DR_SRC + offset, tcols, trows, ROW_BYTES, 0u,
                    TPU_DMA_FILL);
            tpu_wait(TPU_U_DMA);
            tpu_dma(SP_BUF, DR_DST + offset, tcols, trows, ROW_BYTES, 0u,
                    TPU_DMA_SPILL);
            tpu_wait(TPU_U_DMA);
        }
    }
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
