#!/usr/bin/env python3
"""vector_generator.py — a kernel's operands and its expected result.

A generator owns three things and nothing else: the image DRAM is loaded with
once, the per-run patches, and what the answer should be. It never runs the
kernel, so its golden is independent of the ISS, the RTL and the board alike.
See accel/test/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Q4_MIN, Q4_MAX = -8, 7


# =============================================================================
# Images.
# =============================================================================
def i4_row(cols: int) -> int:
    """Bytes in a row-major int4 row — the C side's I4()."""
    return cols // 2


def put_rowmajor_i4(img: dict, base: int, rows: int, cols: int, fn) -> None:
    """An int4 block, row-major, two nibbles per byte, low nibble first."""
    row_bytes = i4_row(cols)
    for r in range(rows):
        for c in range(cols):
            addr = base + r * row_bytes + c // 2
            nib = fn(r, c) & 0xF
            cur = img.get(addr, 0)
            img[addr] = ((cur & 0xF0) | nib) if c % 2 == 0 else ((cur & 0x0F) | (nib << 4))


def put_rowmajor_i8(img: dict, base: int, rows: int, cols: int, stride: int,
                    fn) -> None:
    for r in range(rows):
        for c in range(cols):
            img[base + r * stride + c] = fn(r, c) & 0xFF


def put_i32(img: dict, base: int, values) -> None:
    """int32, little-endian, densely packed from `base`."""
    for i, v in enumerate(values):
        for b in range(4):
            img[base + i * 4 + b] = (int(v) >> (8 * b)) & 0xFF


def zero_range(img: dict, base: int, length: int) -> None:
    """Make a region explicitly zero.

    The board's SRAM keeps whatever the last run left in it and the ISS starts
    from zeros, so anything a kernel reads has to be in the image for the two to
    agree. That is what this is for.
    """
    for a in range(base, base + length):
        img.setdefault(a, 0)


def get_i4(image, base: int, i: int) -> int:
    code = (image[base + (i >> 1)] >> (4 * (i & 1))) & 0xF
    return code - 16 if code >= 8 else code


def get_i32(image, addr: int) -> int:
    v = sum(image[addr + b] << (8 * b) for b in range(4))
    return v - (1 << 32) if v >= (1 << 31) else v


# =============================================================================
# Synthetic operand values. Deterministic, and shaped so a wrong index is
# visible: an activation spans a narrow range, a weight spans the whole int4
# grid including -8, and `w_hash` has no arithmetic structure in either index.
# =============================================================================
def a_val(m: int, k: int) -> int:
    return ((m * 3 + k * 5) % 9) - 4


def w_val(k: int, n: int) -> int:
    return ((k * 5 + n * 3) % 16) - 8


def w_hash(r: int, c: int, salt: int) -> int:
    """A stand-in for a trained weight: 32-bit integer mixing, no structure."""
    v = (r * 2654435761 + c * 2246822519 + salt * 3266489917) & 0xFFFFFFFF
    v ^= v >> 15
    return ((v * 2654435761 >> 13) & 0xF) - 8


# =============================================================================
# The shared fixed point: clip((acc*m0 + 2**(n-1)) >> n).
# =============================================================================
def rq_word(m0: int, n: int) -> int:
    """The {m0,n} literal a command carries: m0 in the low 12 bits, n above."""
    return (n << 12) | m0


def narrow(acc: int, word: int, lo: int = Q4_MIN) -> int:
    m0, n = word & 0xFFF, (word >> 12) & 0xF
    v = (acc * m0 + ((1 << (n - 1)) if n else 0)) >> n
    return max(lo, min(Q4_MAX, v))


def dyt(acc: int, word: int) -> int:
    """`narrow` clipped symmetrically to +-7, which is what DyT's hardtanh is."""
    return narrow(acc, word, lo=-Q4_MAX)


# =============================================================================
# $readmemh byte images.
# =============================================================================
def write_hex(path: str, image: dict, header: str = "") -> None:
    """Sparse image: an `@addr` directive at every discontinuity."""
    with open(path, "w") as f:
        if header:
            f.write(f"// {header}\n")
        prev = None
        for addr in sorted(image):
            if prev is None or addr != prev + 1:
                f.write(f"@{addr:05x}\n")
            f.write(f"{image[addr]:02x}\n")
            prev = addr


def read_hex(path: str) -> dict:
    """The inverse. `xx` (a byte the writer never touched) is skipped."""
    image: dict = {}
    addr = 0
    with open(path) as f:
        for raw in f:
            line = raw.split("//")[0].strip()
            for tok in line.split():
                if tok.startswith("@"):
                    addr = int(tok[1:], 16)
                elif "x" in tok.lower():
                    addr += 1
                else:
                    image[addr] = int(tok, 16)
                    addr += 1
    return image


# =============================================================================
# The generator contract.
# =============================================================================
@dataclass
class Case:
    """One run of the kernel.

    `patch` is written into DRAM on top of whatever is already there — the
    static image on the first run, the previous case's leftovers after that,
    which is exactly the board's behaviour. `golden` is what the bytes in
    `check_ranges` must read back as.
    """

    name: str
    patch: dict = field(default_factory=dict)
    golden: dict = field(default_factory=dict)
    check_ranges: list = field(default_factory=list)   # [(base, length), ...]


class VectorGenerator:
    """Subclass this and implement `static` and `cases`."""

    #: -D flags the kernel is compiled with, and the C config header's contents.
    defines: dict = {}

    def static(self) -> dict:
        """The image DRAM is loaded with once, as {addr: byte}.

        Must be dense over every byte the kernel reads. `zero_range` is how a
        region the kernel reads before writing gets into it.
        """
        raise NotImplementedError

    def cases(self):
        """Yield `Case`s, in order. One is a plain unit test; many is a sweep."""
        raise NotImplementedError

    def writable_ranges(self) -> list:
        """DRAM the kernel may write without it being checked — scratch and
        intermediates. Anything changed outside these and `check_ranges` is a
        stray write and fails. Empty means the kernel writes only its result.
        """
        return []
