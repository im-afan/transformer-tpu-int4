/* spadwin.c — the scratchpad window (cpu_subsys.sv's 0x9xxx_xxxx), on its own.
 * `python accel/test/tests/spadwin/generate.py -b rtl`, ~2 s.
 * See accel/tpu/docs/fw.md. */
#include "tpulib.h"

#define VEC_WORDS 16            /* int32 words the CPU scans */
#define ROW_BYTES 16            /* bytes in one table row    */

/* ---- DRAM ---------------------------------------------------------------- */
#define DR_VEC   0x0000u        /* [VEC_WORDS] int32     — host   */
#define DR_TABLE 0x0100u        /* [13][ROW_BYTES] int8  — host   */
#define DR_OUT   0x0200u        /* the results below     — device */

/* ---- scratchpad ---------------------------------------------------------- */
#define SP_VEC 0x0000u          /* [VEC_WORDS] int32                    */
#define SP_OUT 0x0100u          /* {index, value, readback} int32       */
#define SP_ROW 0x0200u          /* [ROW_BYTES] int8, the gathered row   */

int main(void)
{
    unsigned best_index = 0, i;
    int32_t best_value;

    tpu_move_bytes(SP_VEC, DR_VEC, VEC_WORDS * 4, TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);        /* fence: the window reads memory, not a queue */

    best_value = (int32_t)tpu_spad_ld(SP_VEC);
    for (i = 1; i < VEC_WORDS; i++) {
        int32_t word = (int32_t)tpu_spad_ld(SP_VEC + i * 4);

        if (word > best_value) {  /* strictly: a tie takes the lowest index */
            best_value = word;
            best_index = i;
        }
    }

    tpu_spad_st(SP_OUT + 0u, best_index);
    tpu_spad_st(SP_OUT + 4u, (uint32_t)best_value);
    /* Read one back: a write the port accepted and dropped would otherwise only
     * be caught by the spill, which reads through the DMA's port rather than
     * this one. */
    tpu_spad_st(SP_OUT + 8u, tpu_spad_ld(SP_OUT + 0u));

    tpu_move_bytes(SP_ROW, DR_TABLE + best_index * ROW_BYTES, ROW_BYTES,
                   TPU_DMA_FILL);
    tpu_wait(TPU_U_DMA);

    tpu_move_bytes(SP_OUT, DR_OUT, 12u, TPU_DMA_SPILL);
    tpu_move_bytes(SP_ROW, DR_OUT + 16u, ROW_BYTES, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;
}
