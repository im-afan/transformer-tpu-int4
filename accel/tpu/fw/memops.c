/* memops.c — the two functions a freestanding gcc may call without being asked.
 *
 * `-ffreestanding -fno-builtin` stops gcc from *recognising* memcpy/memset in
 * source, but it does not stop it emitting calls to them: a struct assignment
 * or an aggregate initializer larger than a few words is lowered to `memcpy`
 * by the ABI, whatever the flags say. Nothing here linked against libc or
 * libgcc, so before this file existed a kernel that used a struct failed at
 * link with "undefined reference to memcpy" — which is what tpulib.h's
 * descriptors did the moment adder.c started filling one per matmul.
 *
 * Both are word-wise when the pointers and the length allow it, because these
 * are on the CPU's issue path, where every clock is exposed.
 *
 * **No shipped kernel links this today.** The `always_inline` on tpulib.h's
 * three shape-facing entry points folds every descriptor here into constants,
 * so none of them is ever materialized and none is ever copied; --gc-sections
 * then drops this file. It stays because that is a property of *these* kernels,
 * not of the library: a kernel whose shapes are genuinely runtime values gets
 * the general path, gets a descriptor in memory, and gets a link error without
 * this. It was the live path for a full `make fw FWPROG=adder` run (526 959
 * checks, 0 errors) before the inlining landed, which is where it was tested.
 */
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);

void *memcpy(void *dst, const void *src, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;

    if ((((uintptr_t)d | (uintptr_t)s) & 3u) == 0u) {
        uint32_t *dw = (uint32_t *)dst;
        const uint32_t *sw = (const uint32_t *)src;
        for (; n >= 4u; n -= 4u)
            *dw++ = *sw++;
        d = (unsigned char *)dw;
        s = (const unsigned char *)sw;
    }
    while (n--)
        *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char b = (unsigned char)c;

    if (((uintptr_t)d & 3u) == 0u) {
        const uint32_t w = (uint32_t)b * 0x01010101u;
        uint32_t *dw = (uint32_t *)dst;
        for (; n >= 4u; n -= 4u)
            *dw++ = w;
        d = (unsigned char *)dw;
    }
    while (n--)
        *d++ = b;
    return dst;
}
