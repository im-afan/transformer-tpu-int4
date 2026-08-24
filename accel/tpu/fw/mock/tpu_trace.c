/* tpu_trace.c — the host-side implementation of tpu.h's four primitives.
 *
 * Link this against any firmware kernel with -DTPU_TRACE and the host compiler,
 * and running the result prints the kernel's command trace instead of executing
 * it. The kernel source is unmodified:
 *
 *   cc -DTPU_TRACE -I.. mock/tpu_trace.c matmul.c -o matmul.trace
 *   ./matmul.trace > trace.txt
 *
 * Format, one record per line, all hex, consumed by
 * accel/tpulang/fw_vectors.py:
 *
 *   CMD  <unit> <w0> <w1> <w2> <w3>     one 128-bit macro-op
 *   WAIT <unit>                         a producer barrier on that unit
 *   SRD  <addr>                         the CPU read a scratchpad word
 *   SWR  <addr> <val>                   the CPU wrote one
 *
 * WAIT has no hardware effect — it is a spin on the retired counter — but it is
 * recorded because cross-unit ordering is software's job, so a missing barrier
 * is a real firmware bug and the trace is where it shows. The ISS executes
 * commands in trace order, which is the strongest ordering any correct barrier
 * placement can produce, so a trace missing a barrier still yields correct
 * golden images here and diverges on the RTL. That asymmetry is deliberate: the
 * images stay a statement of intent, and the RTL run is what tests ordering.
 *
 * CO-EXECUTION. SRD is the one record that needs an ANSWER: a kernel that
 * argmaxes a tensor (fw/infer.c) issues commands whose addresses depend on what
 * the array computed, so its trace cannot be produced by a program with no
 * model of the machine. So this one asks instead of modelling. After printing
 * SRD it blocks on stdin for a line of hex, and the driver on the other end of
 * the pipe (fw_vectors.py -x) is the ISS: it executes each CMD as it arrives
 * and answers the read out of its own scratchpad.
 *
 * With no driver — a plain `make trace`, stdin at EOF — a read returns 0 and
 * warns once. The command sequence is then still the right shape but the
 * numbers in it are meaningless, which is all `make trace` can give for a
 * kernel that branches on its own results.
 */
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

    /* The kernel is about to block on the reply, so everything it has printed
     * has to be on the wire first — stdout is a pipe here, hence block
     * buffered, and the driver cannot answer a read it has not seen. */
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
