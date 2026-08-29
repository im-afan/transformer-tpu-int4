#!/usr/bin/env python3
"""iss.py — instruction-set simulator for the TPU's 128-bit macro-ops.

Executes the commands a firmware image pushes, against a model of the
scratchpad, DRAM and the three units, matching ``accel/tpu/rtl/*.sv``. Every
dispatch is *atomic* — read its operands, compute, write the result back — so
no cycle-level modelling is needed to reproduce the memory state a real run
leaves behind. ``exec_command`` and ``run_trace`` are the way in.

It exists to produce the golden vectors the testbenches check the DUT against:
load the same program and input tensors, run this, and the final DRAM image is
what the hardware must reproduce byte for byte.

Numerics are bit-exact with the RTL. **Everything is int4, packed two per byte,
low nibble first** — the MXU's three operands and the VPU's, alike:

  * MXU: ``C[N][N] = requant(A[N][len] @ B)``, int32 accumulate, requant to
    int4 forced on store (mxu.sv ``requant4``). ``accumulate`` adds the stored
    int4 back and re-clips, so it is a fused int4 add, not an int32 partial.
  * VPU: the narrow is fused into every op (vpu.sv ``narrow4``); ``DYT`` is the
    same fixed point with a symmetric +-7 clip, which is its hardtanh rather
    than an approximation of it. ``DOT`` is the one op writing int32.

DMA is a 2-D byte copy between DRAM and the scratchpad, ``rows`` rows of
``len`` int4 with an independent row stride on each side (dma.sv). The two
memories are **different sizes**: the scratchpad is ``2**addr_w`` and DRAM is
``2**mem_addr_w`` (the Cmod A7's 512K x 8 part), so each side of a transfer is
masked in its own space (``_a`` vs ``_d``). A spill is what makes a byte
host-visible, so the golden outputs are exactly the DRAM bytes it wrote
(tracked in ``dram_written``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# vpu.sv VOP_*. The gaps are retired holes: 2, 4-9, 11-15 went with the
# softmax/LayerNorm/GELU datapath, and 17 (QUANT4) went when the MXU began
# storing int4 itself. They are not reused, so a stale binary decodes to an
# unknown op rather than to a different one.
VOP_DOT, VOP_ADD, VOP_RELU, VOP_REQUANT, VOP_DYT = 0, 1, 3, 10, 16

# The int4 grid: mxu.sv/vpu.sv Q4_MIN/Q4_MAX and model/transformer.py
# INT4_QMIN/INT4_QMAX are this pair.
Q4_MIN, Q4_MAX = -8, 7


def s8(b: int) -> int:
    b &= 0xFF
    return b - 0x100 if b >= 0x80 else b


def s32(v: int) -> int:
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


class ISSError(Exception):
    pass


@dataclass
class TPU:
    """A byte-addressed scratchpad + DRAM + the three units.

    Geometry mirrors the DUT parameters in ``tpu_top.sv``.
    """

    rows: int = 8          # array size N (rows == cols; the block is N x N)
    cols: int = 8
    addr_w: int = 16       # scratchpad byte-address width
    mem_addr_w: int = 19   # external DRAM byte-address width
    m0_w: int = 12         # requant fixed-point multiplier width
    n_w: int = 4           # requant shift width

    mem: bytearray = field(init=False)      # on-chip scratchpad
    dram: bytearray = field(init=False)     # external DRAM
    written: set = field(init=False)        # scratchpad bytes a unit wrote
    dram_written: set = field(init=False)   # DRAM bytes a spill wrote

    def __post_init__(self):
        if self.rows != self.cols:
            raise ISSError("the array is square: rows must equal cols")
        self.depth = 1 << self.addr_w
        self.dram_depth = 1 << self.mem_addr_w
        self.mem = bytearray(self.depth)
        self.dram = bytearray(self.dram_depth)
        self.written = set()
        self.dram_written = set()
        self.word_bytes = self.rows // 2        # one scratchpad bank word
        # MXU_GEOM, sticky inside that unit's queue. Nothing else writes it.
        self.g_astride = self.g_bstride = self.g_cstride = self.g_len = 0

    # ---- memory access (addresses wrap in their own space, like the RTL) -----
    def _a(self, addr: int) -> int:
        return addr & (self.depth - 1)

    def _d(self, addr: int) -> int:
        """Mask a **DRAM** byte address. Distinct from :meth:`_a` because the
        two memories are different sizes; a DRAM address masked to 16 bits
        aliases back into the low window instead of reaching the rest of the
        chip, which is the one way this model can silently disagree with the
        RTL."""
        return addr & (self.dram_depth - 1)

    def rd_u32(self, addr: int) -> int:
        return sum(self.mem[self._a(addr + k)] << (8 * k) for k in range(4))

    def rd_i32(self, addr: int) -> int:
        return s32(self.rd_u32(addr))

    def rd_i8(self, addr: int) -> int:
        return s8(self.mem[self._a(addr)])

    def wr_i8(self, addr: int, val: int, *, track: bool = True) -> None:
        a = self._a(addr)
        self.mem[a] = val & 0xFF
        if track:
            self.written.add(a)

    def wr_i32(self, addr: int, val: int, *, track: bool = True) -> None:
        val &= 0xFFFFFFFF
        for k in range(4):
            a = self._a(addr + k)
            self.mem[a] = (val >> (8 * k)) & 0xFF
            if track:
                self.written.add(a)

    # ---- packed int4, two per byte, low nibble first -------------------------
    def rd_i4(self, base: int, i: int) -> int:
        byte = self.mem[self._a(base + (i >> 1))]
        code = (byte >> (4 * (i & 1))) & 0xF
        return code - 16 if code >= 8 else code

    def wr_i4_byte(self, base: int, b: int, lo: int, hi: int) -> None:
        """One destination byte: two nibbles at once.

        The RTL's write strobe is per byte, so this is the finest thing any
        unit writes. A caller with only a low nibble to place passes ``hi = 0``
        — that is what makes an odd-length tail zero-fill rather than preserve.
        """
        self.wr_i8(base + b, ((hi & 0xF) << 4) | (lo & 0xF))

    @staticmethod
    def _nib(rowint: int, j: int) -> int:
        """Element ``j`` of a packed int4 row already read as an integer."""
        code = (rowint >> (4 * j)) & 0xF
        return code - 16 if code >= 8 else code

    # ---- the shared fixed point ---------------------------------------------
    def _narrow(self, acc: int, m0: int, n: int, lo: int) -> int:
        """``clip((acc*m0 + round) >> n)``. Python's ``>>`` floors, like
        Verilog's ``>>>``."""
        rnd = 0 if n == 0 else (1 << (n - 1))
        shifted = (acc * m0 + rnd) >> n
        if shifted > Q4_MAX:
            return Q4_MAX
        if shifted < lo:
            return lo
        return shifted

    def requant4(self, acc: int, m0: int, n: int) -> int:
        return self._narrow(acc, m0, n, Q4_MIN)

    def dyt4(self, acc: int, m0: int, n: int) -> int:
        """The same rescale as a requant, clipped symmetrically. ``hardtanh`` is odd, so its
        floor has to be the negative of its ceiling; int4's -8 would put the
        saturated end at -8/7 = -1.143. The multiplier carries
        ``alpha * s_in * 7``, which is what makes the clip coincide with the
        hardtanh rather than merely resemble it."""
        return self._narrow(acc, m0, n, -Q4_MAX)

    @staticmethod
    def _clip4(v: int) -> int:
        return max(Q4_MIN, min(Q4_MAX, v))

    def _rq_split(self, word: int) -> tuple:
        return (word & ((1 << self.m0_w) - 1),
                (word >> self.m0_w) & ((1 << self.n_w) - 1))

    # =========================================================================
    # The units.
    # =========================================================================
    def _matmul(self, c_base: int, a_base: int, b_base: int, *,
                transpose: bool, accumulate: bool, rq_word: int) -> None:
        """``C[N][N] = requant(A[N][len] @ B)``, or ``@ B'`` when transposed.

        The array always writes N rows of N, whatever the caller's live extent
        is: there is no row count in the dispatch. Zero strides mean the
        densely packed default, exactly as mxu.sv resolves them.
        """
        n_arr = self.rows
        length = self.g_len & 0xFFFF
        if length == 0:
            return
        m0, sh = self._rq_split(rq_word)

        len_bytes = length >> 1
        a_stride = self.g_astride or len_bytes
        b_stride = self.g_bstride or (len_bytes if transpose else self.word_bytes)
        c_stride = self.g_cstride or self.word_bytes

        for i in range(n_arr):
            a_row = a_base + i * a_stride
            out = []
            for j in range(n_arr):
                acc = 0
                if transpose:
                    b_row = b_base + j * b_stride
                    for k in range(length):
                        acc += self.rd_i4(a_row, k) * self.rd_i4(b_row, k)
                else:
                    for k in range(length):
                        acc += (self.rd_i4(a_row, k)
                                * self.rd_i4(b_base + k * b_stride, j))
                v = self.requant4(acc, m0, sh)
                if accumulate:
                    v = self._clip4(v + self.rd_i4(c_base + i * c_stride, j))
                out.append(v)
            # One whole row per store: the C port writes a full word.
            for b in range(n_arr // 2):
                self.wr_i4_byte(c_base + i * c_stride, b, out[2 * b], out[2 * b + 1])

    def _dma(self, *, spad: int, dram: int, spill: bool, length: int,
             rows: int, dram_stride: int, spad_stride: int) -> None:
        """``rows`` rows of ``length`` int4 between DRAM and the scratchpad.

        A row is ``(length + 1) // 2`` bytes; a zero stride means densely
        packed rows, the same zero-is-not-set convention the MXU strides use.
        """
        row_bytes = (length + 1) // 2
        if rows == 0 or length == 0:
            return
        dstride = dram_stride or row_bytes
        sstride = spad_stride or row_bytes

        for r in range(rows):
            s_row = spad + r * sstride
            d_row = dram + r * dstride
            for b in range(row_bytes):
                if spill:
                    a = self._d(d_row + b)
                    self.dram[a] = self.mem[self._a(s_row + b)]
                    self.dram_written.add(a)
                else:
                    self.mem[self._a(s_row + b)] = self.dram[self._d(d_row + b)]

    def _vpu(self, vop: int, dst: int, src0: int, src1: int, vlen: int,
             rq_word: int) -> None:
        """One vector op over ``vlen`` packed int4 elements.

        The narrow is fused into every op, so there is no int32 intermediate.
        ``DOT`` is the exception: it reduces to one int32 scalar.
        """
        if vlen == 0:
            return
        m0, sh = self._rq_split(rq_word)

        if vop == VOP_DOT:
            acc = 0
            for i in range(vlen):
                acc += self.rd_i4(src0, i) * self.rd_i4(src1, i)
            self.wr_i32(dst, s32(acc))
            return

        def value(i: int) -> int:
            a = self.rd_i4(src0, i)
            if vop in (VOP_ADD, VOP_DYT):
                return a + self.rd_i4(src1, i)
            if vop == VOP_RELU:
                return a if a > 0 else 0
            return a                     # REQUANT: a plain rescale

        narrow = self.dyt4 if vop == VOP_DYT else self.requant4

        # Byte at a time, because that is what the write strobe covers: an odd
        # vlen leaves the unfilled slot of the last byte at 0 rather than
        # preserving it.
        for b in range((vlen + 1) // 2):
            lo = narrow(value(2 * b), m0, sh)
            hi = narrow(value(2 * b + 1), m0, sh) if 2 * b + 1 < vlen else 0
            self.wr_i4_byte(dst, b, lo, hi)

    # =========================================================================
    # Command front end. Mirrors cmd_{mxu,vpu,dma}.sv field for field.
    # =========================================================================
    U_MXU, U_VPU, U_DMA = 0, 1, 2

    MXU_GEOM, MXU_MM = 0x01, 0x02
    VPU_CMD_OP = 0x01          # 0x02 (GEOM) retired with the vecmatmul macro op
    DMA_MOVE = 0x01

    _VOPS = (VOP_DOT, VOP_ADD, VOP_RELU, VOP_REQUANT, VOP_DYT)

    def exec_command(self, unit: int, w0: int, w1: int, w2: int, w3: int) -> None:
        """Execute one 128-bit macro-op.

        An unknown opcode is *discarded*, not an error — cmd_{mxu,vpu,dma}.sv
        pop it with a ``$display`` and carry on, and a model that raised
        instead would disagree with the hardware about a malformed stream.
        """
        op = w0 & 0xFF
        rq_mask = (1 << (self.m0_w + self.n_w)) - 1

        if unit == self.U_MXU:
            if op == self.MXU_GEOM:
                self.g_astride = (w0 >> 16) & 0xFFFF
                self.g_bstride = w1 & 0xFFFF
                self.g_cstride = (w1 >> 16) & 0xFFFF
                self.g_len = w2 & 0xFFFF
            elif op == self.MXU_MM:
                self._matmul((w0 >> 16) & 0xFFFF, w1 & 0xFFFF,
                             (w1 >> 16) & 0xFFFF,
                             accumulate=bool(w0 & (1 << 8)),
                             transpose=bool(w0 & (1 << 9)),
                             rq_word=w2 & rq_mask)

        elif unit == self.U_VPU:
            if op == self.VPU_CMD_OP:
                vop = (w0 >> 8) & 0x1F
                if vop not in self._VOPS:
                    return
                self._vpu(vop, (w0 >> 16) & 0xFFFF, w1 & 0xFFFF,
                          (w1 >> 16) & 0xFFFF, w2 & 0x3FF,
                          (w2 >> 16) & rq_mask)

        elif unit == self.U_DMA:
            if op == self.DMA_MOVE:
                self._dma(spad=(w0 >> 16) & 0xFFFF,
                          dram=w1 & (self.dram_depth - 1),
                          spill=bool(w0 & (1 << 8)),
                          length=w2 & 0xFFFF,
                          rows=(w2 >> 16) & 0xFFFF,
                          dram_stride=w3 & 0xFFFF,
                          spad_stride=(w3 >> 16) & 0xFFFF)

    def run_trace(self, records) -> list:
        """Execute a producer's command trace; return the commands, in order.

        ``records`` is the parsed output of a ``-DTPU_TRACE`` firmware build
        (see ``fw/mock/tpu_trace.c``): ``("CMD", unit, w0, w1, w2, w3)`` and
        ``("WAIT", unit)``.

        WAIT is a no-op here, and deliberately so. This model has no
        concurrency: it retires every command completely before looking at the
        next, which is the strongest ordering any barrier placement can
        produce. So a firmware that *omits* a needed cross-unit barrier still
        gets correct golden images out of this — and then diverges on the RTL,
        where the three queues really are independent. Keeping the images a
        statement about intent, and letting the RTL run be the thing that tests
        ordering, is what makes a mismatch mean something specific.
        """
        cmds = []
        for rec in records:
            if rec[0] != "CMD":
                continue
            _, unit, w0, w1, w2, w3 = rec
            self.exec_command(unit, w0, w1, w2, w3)
            cmds.append((unit, w0, w1, w2, w3))
        return cmds


def parse_trace(text: str) -> list:
    """Parse ``fw/mock/tpu_trace.c`` output into records for :meth:`TPU.run_trace`."""
    out = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        if f[0] == "CMD" and len(f) == 6:
            out.append(("CMD", int(f[1], 16) if f[1].startswith("0x") else int(f[1]),
                        *(int(x, 16) for x in f[2:6])))
        elif f[0] == "WAIT" and len(f) == 2:
            out.append(("WAIT", int(f[1])))
        else:
            raise ISSError(f"trace line {lineno}: cannot parse {line!r}")
    return out
