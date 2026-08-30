/* tpu_trace.c — host-side implementation of tpu.h's four primitives, for
 * -DTPU_TRACE builds. See docs/fw.md. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void tpu_trace_push(unsigned unit, uint32_t word0, uint32_t word1,
                    uint32_t word2, uint32_t word3)
{
    printf("CMD %u %08x %08x %08x %08x\n", unit, word0, word1, word2, word3);
}

void tpu_trace_wait(unsigned unit)
{
    printf("WAIT %u\n", unit);
}

uint32_t tpu_trace_spad_ld(uint32_t byte_addr)
{
    static int warned = 0;
    char reply[64];
    unsigned long value;

    /* stdout is a pipe here (block buffered), so flush before blocking on the reply. */
    printf("SRD %08x\n", byte_addr);
    fflush(stdout);

    if (!fgets(reply, sizeof reply, stdin)) {
        if (!warned) {
            fprintf(stderr, "tpu_trace: scratchpad read with no driver attached "
                            "— returning 0 (see mock/tpu_trace.c)\n");
            warned = 1;
        }
        return 0u;
    }
    value = strtoul(reply, NULL, 16);
    return (uint32_t)value;
}

void tpu_trace_spad_st(uint32_t byte_addr, uint32_t value)
{
    /* No reply: the driver applies it to the ISS scratchpad, and a kernel that
     * reads back what it just wrote gets it from there. */
    printf("SWR %08x %08x\n", byte_addr, value);
    fflush(stdout);
}
