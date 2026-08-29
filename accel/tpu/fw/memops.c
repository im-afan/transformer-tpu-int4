/* memops.c — the two functions a freestanding gcc may call without being
 * asked (memcpy/memset, lowered to implicitly by the ABI). See docs/fw.md for
 * why no shipped kernel currently links this. */
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);

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
