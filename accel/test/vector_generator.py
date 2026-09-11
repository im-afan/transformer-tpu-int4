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

# The machine, mirroring tpu.h / tpulib.h. A kernel that maps a tensor to the
# same address in DRAM and in the scratchpad is bounded by the scratchpad, and
# wants its operands in different banks — scratchpad.sv serves one reader per
# bank per clock, so A, B and C sharing one is a stall per beat.
TPU_N = 8
TPU_WORD_BYTES = TPU_N // 2
TPU_BANK_BYTES = 1024 * TPU_WORD_BYTES          # 4 KB
TPU_SPAD_BYTES = 1 << 16                        # ADDR_W = 16

M0_W, N_W = 12, 4                               # the requant word's two fields


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
    return (n << M0_W) | m0


RQ_ONE = rq_word(1, 0)              # an identity pass: the input is already
                                    # on the output's grid and scale


def fixed_point(mult: float, what: str = "") -> tuple:
    """The {m0, n} pair closest to real multiplier `mult`.

    The op computes a multiplier of `m0/2**n` with `m0 < 4096` and `n <= 15`, so
    this takes the largest `n` that keeps `m0` in range — every extra shift is
    another bit of precision on a multiplier usually much smaller than 1.
    """
    import sys

    if not mult > 0:
        raise SystemExit(f"{what or 'requant'}: multiplier {mult} is not "
                         f"positive — m0 is unsigned")
    best = None
    for n in range(1 << N_W):
        m0 = int(mult * (1 << n) + 0.5)          # round-half-up, like the RTL
        if 0 < m0 < (1 << M0_W):
            best = (m0, n)
    if best is None:
        if mult >= 1:
            print(f"  WARNING {what}: m = {mult:.4g} exceeds the representable "
                  f"4095, clamping", file=sys.stderr)
            return ((1 << M0_W) - 1, 0)
        print(f"  WARNING {what}: m = {mult:.4g} underflows m0 = 0 at n = 15; "
              f"this tensor will be all zeros", file=sys.stderr)
        return (0, 0)
    return best


def fit_rq(accumulators, what: str = "") -> int:
    """The {m0,n} that lands the largest of `accumulators` on the top of the
    int4 grid — the multiplier that uses the range without clipping.

    This is what makes a kernel's shape a knob. A requant word tuned by hand for
    one contraction length saturates or collapses at another, and an all-zero
    golden passes against any datapath at all; deriving it from the accumulators
    the golden just produced cannot drift from the shape. Same rule
    `export.logit_rq_word` uses for the output head.
    """
    peak = max((abs(int(a)) for a in accumulators), default=0)
    if peak == 0:
        return RQ_ONE
    return rq_word(*fixed_point(Q4_MAX / peak, what or "fit_rq"))


def narrow(acc: int, word: int, lo: int = Q4_MIN) -> int:
    m0, n = word & 0xFFF, (word >> 12) & 0xF
    v = (acc * m0 + ((1 << (n - 1)) if n else 0)) >> n
    return max(lo, min(Q4_MAX, v))


def dyt(acc: int, word: int) -> int:
    """`narrow` clipped symmetrically to +-7, which is what DyT's hardtanh is."""
    return narrow(acc, word, lo=-Q4_MAX)


# =============================================================================
# tpu_flashattention's block size. A golden that models the kernel's key
# blocking has to pick the same B the firmware will, so this is
# tpu_flash_fit/tpu_flash_bytes written twice — if the two ever drift, every
# flash case fails, which is the point.
# =============================================================================
def flash_bytes(block: int, head_dim: int) -> int:
    """Four bank-aligned [B][head_dim] slots and one bank-aligned score region
    holding P and the mask block."""
    def up(v):
        return (v + TPU_BANK_BYTES - 1) // TPU_BANK_BYTES * TPU_BANK_BYTES
    return 4 * up(block * (head_dim // 2)) + up(block * block)


def flash_block(arena_bytes: int, keys: int, head_dim: int) -> int:
    """The largest whole-array-word key block the arena holds, walked down from
    the whole key axis."""
    usable = arena_bytes // TPU_BANK_BYTES * TPU_BANK_BYTES
    block = keys // TPU_N * TPU_N
    while block > TPU_N and flash_bytes(block, head_dim) > usable:
        block -= TPU_N
    if block < TPU_N or flash_bytes(block, head_dim) > usable:
        raise SystemExit(
            f"flash: a {arena_bytes}-byte arena holds no block of a "
            f"{keys}x{head_dim} head — give it more banks")
    return block


# =============================================================================
# The address map. A kernel's operands are laid out here, in Python, and handed
# to the compiler as -D — so the C carries defaults for a bare build and the
# generator's numbers are the ones that ran.
# =============================================================================
class AddressMap:
    """A sequential allocator at bank granularity.

    These kernels give a tensor the **same** address in DRAM and in the
    scratchpad, so one map covers both and the scratchpad is the tighter bound.
    Allocating a whole bank per tensor is why the small kernels' operands sit
    0x1000 apart: a bank serves one reader per clock, and a matmul reads A, B
    and C at once.
    """

    def __init__(self, align: int = TPU_BANK_BYTES, base: int = 0,
                 limit: int = TPU_SPAD_BYTES, what: str = "kernel"):
        self.align, self.limit, self.what = align, limit, what
        self.next = base
        self.slots: dict = {}

    def alloc(self, name: str, nbytes: int) -> int:
        """Place `name` and return its address. Zero-length allocations still
        take a slot, so a shape that degenerates does not alias two tensors."""
        addr = self.next
        step = max(nbytes, 1)
        self.next = (addr + step + self.align - 1) // self.align * self.align
        if self.next > self.limit:
            raise SystemExit(
                f"{self.what}: the operand map needs {self.next} bytes of a "
                f"{self.limit}-byte scratchpad — this shape does not fit. "
                f"Lower it, or give the kernel a staging arena instead of "
                f"mapping every tensor.")
        self.slots[name] = addr
        return addr

    def __getitem__(self, name: str) -> int:
        return self.slots[name]

    def defines(self) -> dict:
        """`{NAME: "0x....u"}`, ready to be handed to the compiler."""
        return {name: f"0x{addr:05x}u" for name, addr in self.slots.items()}

    def summary(self) -> str:
        return ", ".join(f"{n}=0x{a:05x}" for n, a in self.slots.items())


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
