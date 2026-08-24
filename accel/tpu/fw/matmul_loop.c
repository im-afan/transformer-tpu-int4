/* matmul_loop.c — the same C = A @ W as matmul.c, with the tile grid walked in
 * firmware instead of by the MXU.
 *
 * matmul.c issues ONE tiled matmul and the array walks the grid itself, so the
 * int32 partials stay in its result buffer for a whole contraction and reach
 * the scratchpad once per output tile. Here the grid is a C `for` pair and each
 * tile is its own dispatch, so the partials round-trip through the scratchpad
 * between contraction tiles: the first depth tile initialises the int32 tile
 * and the rest add into it with TPU_MM_ACC. That is the C traffic the hardware
 * tile loop was built to delete (docs/macro_ops.md 4.2).
 *
 * A requantized result would set TPU_MM_RQ on the LAST depth tile only, since
 * the narrow has to see the finished sum. Not done here: matmul.c leaves C
 * int32 and this must match it byte for byte, so host/run_fw_matmul.py checks
 * either one unchanged:
 *
 *     make -C accel/tpu/fw PROG=matmul_loop
 *     make -C accel/tpu/fw run PROG=matmul_loop PORT=COM5
 *
 * TPU_MM_TILED is still set on every dispatch, with both tile counts at 1. The
 * flag selects the configured strides over the single-tile constants (mxu.sv,
 * `act_row_stride_sel`); at a count of 1 the hardware's loop runs exactly one
 * pass, so nothing here is hardware-managed. Dropping the flag would force
 * dense, tile-shaped operands instead — which is the other way to write this,
 * at one DMA per tile. Keeping A, W and C resident and flat is what makes this
 * comparable to matmul.c.
 *
 *   A : [M][K] int8, row-major      act_row = K
 *   W : [K][N] int4, row-major      wgt_row = N/2
 *   C : [M][N] int32                out_row = N*4
 */
#include "tpu.h"

#define ARRAY_ROWS 8            /* MXU geometry — fixed by the bitstream */
#define ARRAY_COLS 8

/* Shape, overridable exactly as in matmul.c — the two must be built with the
 * same three numbers to be comparable. M <= 32 (mxu.sv MAX_TOKENS). */
#ifndef M
#define M      8                /* token rows */
#endif
#ifndef KTILES
#define KTILES 4                /* array passes over the contraction */
#endif
#ifndef NTILES
#define NTILES 2                /* array passes over the output columns */
#endif

#define K (KTILES * ARRAY_ROWS)
#define N (NTILES * ARRAY_COLS)

#define ACT_ROW K               /* bytes: int8 */
#define WGT_ROW (N / 2)         /* bytes: int4, two nibbles per byte */
#define OUT_ROW (N * 4)         /* bytes: int32 */

/* Per-tile address steps. The hardware derives these from the tile indices;
 * with the counts pinned at 1 they are this loop's own address arithmetic.
 * Row-major weights swap the two weight steps relative to the column-major
 * version — an output-column tile steps ALONG a weight row, a depth tile steps
 * DOWN whole weight rows. Same swap as mxu.sv's wgt_ntile_step/wgt_ktile_step. */
#define ACT_DEPTH_STEP ARRAY_ROWS           /* 8  — one depth tile along a row */
#define WGT_DEPTH_STEP (ARRAY_ROWS * WGT_ROW)
#define WGT_COL_STEP   (ARRAY_COLS / 2)     /* 4  — one column tile along a row */
#define OUT_COL_STEP   (ARRAY_COLS * 4)     /* 32 — one column tile of int32 C  */

#define A_BYTES (M * K)
#define W_BYTES (K * WGT_ROW)
#define C_BYTES (M * N * 4)

/* Same address in DRAM and in the scratchpad. */
#define A_ADDR 0x0000u
#define W_ADDR 0x2000u
#define C_ADDR 0x4000u

int main(void)
{
    unsigned col_tile, depth_tile;

    /* operands in */
    tpu_dma(A_ADDR, A_ADDR, A_BYTES, TPU_DMA_FILL);
    tpu_dma(W_ADDR, W_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);        /* the MXU queue is not ordered against the DMA's */

    /* Strides describe the whole matrix; the counts describe one array pass. */
    tpu_mxu_geom(ACT_ROW, OUT_ROW, WGT_ROW, 1u, 1u, M);

    /* Columns outer, depth inner — the order the hardware loop uses, so the two
     * paths differ only in who walks the grid. No fence inside: one unit's
     * queue is in-order, so a later accumulate cannot pass the store that
     * initialised the tile. Past the queue's 8 entries the ninth push stalls
     * the CPU inside the store, so there is no software flow control here
     * either. */
    for (col_tile = 0; col_tile < NTILES; col_tile++)
        for (depth_tile = 0; depth_tile < KTILES; depth_tile++)
            tpu_mxu_mm(C_ADDR + col_tile * OUT_COL_STEP,
                       A_ADDR + depth_tile * ACT_DEPTH_STEP,
                       W_ADDR + col_tile * WGT_COL_STEP
                              + depth_tile * WGT_DEPTH_STEP,
                       TPU_MM_TILED | (depth_tile ? TPU_MM_ACC : 0u),
                       0u);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(C_ADDR, C_ADDR, C_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
