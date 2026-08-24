/* matmul.c — C = A @ W, the whole contraction in one dispatch.
 *
 * The C counterpart of the deleted tiled_matmul_hw.tpu: same problem, same
 * layout, same addresses, but the commands come from PicoRV32 firmware instead
 * of the scalar unit. Operands are staged in and the result out by the DMA, so
 * the host only ever touches DRAM.
 *
 *   A : [M][K] int8, row-major      act_row = K
 *   W : [K][N] int4, row-major      wgt_row = N/2
 *   C : [M][N] int32                out_row = N*4
 *
 * The MXU walks the tile grid itself (columns outer, depth inner), so the int32
 * partials stay in its result buffer and reach the scratchpad once per output
 * tile. The geometry must match host/run_fw_matmul.py, which builds the
 * operands and checks the result.
 */
#include "tpu.h"

#define ARRAY_ROWS 8            /* MXU geometry — fixed by the bitstream */
#define ARRAY_COLS 8

/* Shape, overridable from the Makefile (`make M=8 KTILES=4 NTILES=2`) so the
 * sweep in ../tb/run_fw_sweep.sh can walk it; the testbench and the host script
 * take the same three numbers. M <= 32 (mxu.sv MAX_TOKENS, and the token-row
 * field is 6 bits). */
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

#define A_BYTES (M * K)
#define W_BYTES (K * WGT_ROW)
#define C_BYTES (M * N * 4)

/* Same address in DRAM and in the scratchpad. The bases are spaced for shapes
 * well past the default: A <= 8 KB, W <= 8 KB, C <= 48 KB of the 64 KB
 * scratchpad. */
#define A_ADDR 0x0000u
#define W_ADDR 0x2000u
#define C_ADDR 0x4000u

int main(void)
{
    /* operands in */
    tpu_dma(A_ADDR, A_ADDR, A_BYTES, TPU_DMA_FILL);
    tpu_dma(W_ADDR, W_ADDR, W_BYTES, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);        /* the MXU queue is not ordered against the DMA's */

    /* the contraction */
    tpu_mxu_geom(ACT_ROW, OUT_ROW, WGT_ROW, KTILES, NTILES, M);
    tpu_mxu_mm(C_ADDR, A_ADDR, W_ADDR, TPU_MM_TILED, 0u);
    tpu_wait(TPU_U_MXU);

    /* result out */
    tpu_dma(C_ADDR, C_ADDR, C_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
