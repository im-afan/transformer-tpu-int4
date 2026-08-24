/* memops.c — the two functions a freestanding gcc may call without being asked.
 *
 * `-ffreestanding -fno-builtin` stops gcc from recognising memcpy/memset in
 * source, but not from emitting calls to them: the ABI lowers a struct
 * assignment or a large aggregate initializer to `memcpy` whatever the flags
 * say. Nothing here links libc or libgcc, so without this file a kernel that
 * uses a struct fails at link with "undefined reference to memcpy".
 *
 * Both are word-wise where the pointers and length allow, because these sit on
 * the CPU's issue path where every clock is exposed.
 *
 * NO SHIPPED KERNEL LINKS THIS TODAY. tpulib.h's `always_inline` entry points
 * fold every descriptor into constants, so none is ever materialized or copied,
 * and --gc-sections drops this file. It stays because that is a property of
 * these kernels, not of the library: a kernel with genuinely runtime shapes
 * gets the general path, gets a descriptor in memory, and needs this.
 */
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
