/* memops.c — the functions a freestanding gcc may call without being asked:
 * memcpy/memset, which the ABI lowers to implicitly, and the 32-bit unsigned
 * division helpers, which it emits for a division it chose not to open-code.
 * This toolchain ships no rv32 libgcc, so the kernel carries them.
 * See docs/fw.md. */
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
uint32_t __udivsi3(uint32_t numerator, uint32_t denominator);
uint32_t __umodsi3(uint32_t numerator, uint32_t denominator);

void *memcpy(void *dst, const void *src, size_t n)
{
    unsigned char *dst_byte = (unsigned char *)dst;
    const unsigned char *src_byte = (const unsigned char *)src;

    if ((((uintptr_t)dst_byte | (uintptr_t)src_byte) & 3u) == 0u) {
        uint32_t *dst_word = (uint32_t *)dst;
        const uint32_t *src_word = (const uint32_t *)src;

        for (; n >= 4u; n -= 4u)
            *dst_word++ = *src_word++;
        dst_byte = (unsigned char *)dst_word;
        src_byte = (const unsigned char *)src_word;
    }
    while (n--)
        *dst_byte++ = *src_byte++;
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    unsigned char *dst_byte = (unsigned char *)dst;
    const unsigned char fill_byte = (unsigned char)c;

    if (((uintptr_t)dst_byte & 3u) == 0u) {
        const uint32_t fill_word = (uint32_t)fill_byte * 0x01010101u;
        uint32_t *dst_word = (uint32_t *)dst;

        for (; n >= 4u; n -= 4u)
            *dst_word++ = fill_word;
        dst_byte = (unsigned char *)dst_word;
    }
    while (n--)
        *dst_byte++ = fill_byte;
    return dst;
}

/* Shift-subtract, because the core is built ENABLE_DIV(0) and `div` traps. gcc
 * open-codes a constant divisor as a magic multiply and only calls these when
 * it predicted the block cold and preferred the smaller libcall. */
static uint32_t udivmod(uint32_t numerator, uint32_t denominator,
                        uint32_t *remainder)
{
    uint32_t quotient = 0u;
    uint32_t quotient_bit = 1u;

    if (denominator == 0u) {
        if (remainder)
            *remainder = numerator;
        return 0xFFFFFFFFu;
    }

    while (denominator <= numerator && (denominator & 0x80000000u) == 0u) {
        denominator <<= 1;
        quotient_bit <<= 1;
    }
    while (quotient_bit) {
        if (numerator >= denominator) {
            numerator -= denominator;
            quotient |= quotient_bit;
        }
        denominator >>= 1;
        quotient_bit >>= 1;
    }

    if (remainder)
        *remainder = numerator;
    return quotient;
}

uint32_t __udivsi3(uint32_t numerator, uint32_t denominator)
{
    return udivmod(numerator, denominator, NULL);
}

uint32_t __umodsi3(uint32_t numerator, uint32_t denominator)
{
    uint32_t remainder;

    udivmod(numerator, denominator, &remainder);
    return remainder;
}
