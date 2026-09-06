/* argmax.c — tpu_argmax over four lengths and two arena sizes.
 * `python accel/test/tests/argmax/generate.py -b rtl`, ~2 s.
 * See accel/tpu/docs/fw.md. */
#include "tpulib.h"

/* The four lengths, the DRAM map and both arenas come from generate.py as -D.
 * The defaults below are only for a bare `make`; they are what the chunk fold
 * was written against, so keep the properties when changing them. */
#ifndef LEN_MULTI
#define LEN_MULTI 2500          /* several VPU chunks, the last one ragged */
#endif
#ifndef LEN_EXACT
#define LEN_EXACT 1016          /* TPU_VCHUNK_MAX at N=8: one whole chunk  */
#endif
#ifndef LEN_WORD
#define LEN_WORD  8             /* one scratchpad word, no tail lanes      */
#endif
#ifndef LEN_PAIR
#define LEN_PAIR  2             /* one lane pair, the shortest legal vector */
#endif
/* A reduction writes an int32 scalar and no nibbles, so unlike every
 * elementwise op it takes an odd length. infer.c's head_argmax needs it. */
#ifndef LEN_ODD
#define LEN_ODD   1013          /* odd, and several chunks under a small arena */
#endif
#ifndef LEN_VOCAB
#define LEN_VOCAB 13            /* the vocabulary, exactly what head_argmax reduces */
#endif

#define I4(cols) ((cols) / 2)

/* ---- DRAM ---------------------------------------------------------------- */
#ifndef DR_MULTI
#define DR_MULTI 0x00000u       /* [LEN_MULTI] int4 — host */
#endif
#ifndef DR_EXACT
#define DR_EXACT 0x01000u
#endif
#ifndef DR_WORD
#define DR_WORD  0x01800u
#endif
#ifndef DR_PAIR
#define DR_PAIR  0x01880u
#endif
#ifndef DR_ODD
#define DR_ODD   0x01900u
#endif
#ifndef DR_VOCAB
#define DR_VOCAB 0x01B00u
#endif
#ifndef DR_OUT
#define DR_OUT   0x01B80u       /* [NPROBLEM] int32 — device */
#endif

/* ---- scratchpad ---------------------------------------------------------- */
#ifndef SP_ARENA
#define SP_ARENA 0x0000u
#endif
#ifndef SP_OUT
#define SP_OUT   0x8000u        /* clear of both arenas */
#endif

/* One bank leaves the chunk capped by the 10-bit vlen field; the small one caps
 * it well below that, so the same problems run both fold paths. */
#ifndef ARENA_BYTES
#define ARENA_BYTES TPU_BANK_BYTES
#endif
#ifndef ARENA_SMALL_BYTES
#define ARENA_SMALL_BYTES 128u
#endif

#define NLEN     6
#define NPROBLEM (2 * NLEN)

int main(void)
{
    static const uint32_t addr[NLEN] = { DR_MULTI, DR_EXACT, DR_WORD, DR_PAIR,
                                         DR_ODD, DR_VOCAB };
    static const uint32_t len[NLEN]  = { LEN_MULTI, LEN_EXACT, LEN_WORD, LEN_PAIR,
                                         LEN_ODD, LEN_VOCAB };
    static const uint32_t bytes[2]   = { ARENA_BYTES, ARENA_SMALL_BYTES };

    tpu_arena arena;
    unsigned a, i;

    for (a = 0; a < 2u; a++) {
        tpu_arena_init(&arena, SP_ARENA, bytes[a]);
        for (i = 0; i < NLEN; i++)
            tpu_spad_st(SP_OUT + (a * NLEN + i) * 4u,
                        tpu_argmax(TPU_AT(addr[i]), len[i], &arena));
    }

    tpu_move_bytes(SP_OUT, DR_OUT, NPROBLEM * 4u, TPU_DMA_SPILL);
    tpu_wait(TPU_U_DMA);

    return 0;                   /* start.S raises `done` from here */
}
