#!/usr/bin/env python3
"""fw_vectors.py — golden vectors for a *firmware* kernel, from its command trace.

The `gen_vectors.py` of the CPU producer, and the last piece of phase 3
(`accel/tpu/docs/picorv32_migration.md` §8). Where `gen_vectors.py` assembles a
`.tpu` program and runs it through `iss.py`'s instruction decoder, this reads the
command trace a natively-compiled firmware kernel emitted and runs it through
`iss.py`'s *command* decoder — the same op bodies underneath, so the two paths
cannot disagree about numerics without disagreeing everywhere.

    make -C ../tpu/fw PROG=matmul matmul.trace
    python fw_vectors.py -x ../tpu/fw/matmul.trace -o ../tpu/tb/vectors_fw

Three files come out, all `$readmemh`-able:

    fw_dram_in.hex    the operand image the testbench seeds DRAM with
    fw_dram_exp.hex   every DRAM byte the run wrote, and what it should be
    fw_cmds.hex       the expected command trace: 5 words per command
                      (unit, w0, w1, w2, w3), terminated by a 0xFFFFFFFF unit

The third is what makes this more than a rewrite of the old inline checks. The
testbench monitors the arbitrated command write inside `tpu_top` and diffs the
real PicoRV32's command stream against this one, so a failure says *which
command* diverged rather than only that the answer was wrong — which is the
capability the abandoned RV32IM interpreter was really being bought for
(§8.1).

**The operand formulas live here and nowhere else.** They used to be duplicated
between `fw_matmul_tb.sv` and `host/run_fw_matmul.py`, which is exactly the
drift this phase exists to remove.

**The trace is produced here too** (`-x`), rather than redirected into a file by
the caller. That is not tidiness: `fw/infer.c` reads its own results back over
the scratchpad window and branches on them, so its command stream is not a
function of the program alone and a producer with no model of the machine cannot
emit it. Running the kernel binary as a co-process — the ISS executing each
command as it arrives and answering the kernel's scratchpad reads out of its own
memory — is what makes a data-dependent kernel traceable at all. See
:func:`coexecute`. `-t` still reads a trace someone else captured.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from iss import TPU, parse_trace  # noqa: E402

# Geometry. Must match the array the testbench instantiates and the numbers the
# firmware was built with (fw/Makefile takes the same three).
ROWS, COLS = 8, 8

A_ADDR, W_ADDR, C_ADDR = 0x0000, 0x2000, 0x4000


def a_val(m: int, k: int) -> int:
    """An int8 activation. Bounded to [-4, 4] so a chain of requants has somewhere
    to land instead of pinning at the clip."""
    return ((m * 3 + k * 5) % 9) - 4


def w_val(k: int, n: int) -> int:
    """An int4 weight in [-8, 7].

    Spans the **whole** grid rather than a ternary `%3 - 1`: -8 is the one value
    with no positive counterpart, so a run that never produces it cannot catch a
    nibble sign-extended as unsigned.
    """
    return ((k * 5 + n * 3) % 16) - 8


def w_hash(r: int, c: int, salt: int) -> int:
    """An int4 weight in [-8, 7] with no arithmetic structure in either index.

    A stand-in for a trained weight, for the one kernel whose operands are big
    enough that a `(a*r + b*c) % 16` pattern aliases with the tensor's own
    strides. Deterministic and platform-independent — plain 32-bit integer
    mixing, no RNG and no seeding.
    """
    v = (r * 2654435761 + c * 2246822519 + salt * 3266489917) & 0xFFFFFFFF
    v ^= v >> 15
    return ((v * 2654435761 >> 13) & 0xF) - 8


def put_rowmajor_i8(img: dict, base: int, rows: int, cols: int, stride: int,
                    fn) -> None:
    for r in range(rows):
        for c in range(cols):
            img[base + r * stride + c] = fn(r, c) & 0xFF


def put_rowmajor_i4(img: dict, base: int, rows: int, cols: int, fn) -> None:
    """An int4 weight block, row-major, two nibbles per byte, low nibble first.

    The single definition of the layout mxu.sv reads and iss.TPU._nib decodes.
    """
    wrow = (cols * 4) // 8
    for r in range(rows):
        for c in range(cols):
            addr = base + r * wrow + c // 2
            cur = img.get(addr, 0)
            nib = fn(r, c) & 0xF
            img[addr] = ((cur & 0xF0) | nib) if c % 2 == 0 else ((cur & 0x0F) | (nib << 4))


# =============================================================================
# Running the kernel.
# =============================================================================
def coexecute(tpu: TPU, exe: str, quiet: bool = False) -> tuple:
    """Run a `-DTPU_TRACE` kernel binary against `tpu`; return (cmds, lines).

    The kernel prints one record per line and this executes them as they arrive,
    which is the same thing :meth:`TPU.run_trace` does to a captured file —
    except for the one record that needs an answer.

    ``SRD <addr>`` is the CPU reading a scratchpad word through cpu_subsys.sv's
    0x9xxx_xxxx window. A kernel that argmaxes its own logits and then gathers an
    embedding row at ``base + tok*D`` puts that token into the *address* of a
    later command, so its trace cannot be produced by a program that does not
    know what the array computed. So the kernel asks: it blocks on stdin and this
    replies out of the model's scratchpad. ``SWR`` is the same window in the
    write direction and needs no reply, only application.

    The upshot is that the returned command stream is a real forward pass's, and
    the RTL has to reproduce it exactly — including every address the firmware
    derived from a token it chose. If the hardware picks a different token
    anywhere, the trace diverges at that command rather than merely producing a
    different answer, which is a much more specific failure.

    `lines` is the transcript, with the two scratchpad records commented out so
    the file stays `parse_trace`-able (and `cmd_timeline.py`-able).
    """
    proc = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, bufsize=1)
    cmds, lines, n_rd, n_wr = [], [], 0, 0
    try:
        for line in proc.stdout:
            f = line.split()
            if not f:
                continue
            if f[0] == "CMD" and len(f) == 6:
                unit, w = int(f[1]), [int(x, 16) for x in f[2:6]]
                tpu.exec_command(unit, *w)
                cmds.append((unit, *w))
                lines.append(line)
            elif f[0] == "WAIT" and len(f) == 2:
                lines.append(line)
            elif f[0] == "SRD" and len(f) == 2:
                val = tpu.rd_u32(int(f[1], 16))
                proc.stdin.write(f"{val:08x}\n")
                proc.stdin.flush()
                lines.append(f"# SRD {f[1]} -> {val:08x}\n")
                n_rd += 1
            elif f[0] == "SWR" and len(f) == 3:
                # track=False: this is the CPU writing, not a unit, and
                # `written` is the compute-side record.
                tpu.wr_i32(int(f[1], 16), int(f[2], 16), track=False)
                lines.append(f"# SWR {f[1]} {f[2]}\n")
                n_wr += 1
            else:
                raise SystemExit(f"coexecute: cannot parse {line!r}")
    finally:
        if proc.stdin:
            proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"coexecute: {exe} exited {rc}")
    if (n_rd or n_wr) and not quiet:
        print(f"co-execution: {n_rd} scratchpad reads answered, {n_wr} writes "
              f"applied")
    return cmds, lines


# =============================================================================
# Per-kernel operand images.
#
# One builder per firmware kernel, keyed by its name. This is where a kernel's
# DRAM layout is defined *once* — `host/run_fw_matmul.py` imports the matmul one
# rather than carrying its own copy, and the testbench gets it as a file. A new
# kernel adds a function here and nothing else.
# =============================================================================
def operands_matmul(args) -> dict:
    """matmul.c / matmul_loop.c: A[M][K] int8, W[K][N] int4, both at their own bases."""
    k, n = args.ktiles * ROWS, args.ntiles * COLS
    img: dict = {}
    put_rowmajor_i8(img, A_ADDR, args.M, k, k, a_val)
    put_rowmajor_i4(img, W_ADDR, k, n, w_val)
    return img


def operands_ffn(args) -> dict:
    """ffn.c: X[T][D] int8, W1[D][F] int4, W2[F][D] int4."""
    T, D, F = 8, 8, 16
    img: dict = {}
    put_rowmajor_i8(img, 0x0000, T, D, D, a_val)
    put_rowmajor_i4(img, 0x0400, D, F, w_val)
    # A different salt so a builder mix-up shows up as a wrong answer, not a
    # coincidentally-equal one.
    put_rowmajor_i4(img, 0x0800, F, D, lambda r, c: ((r * 3 + c * 7) % 16) - 8)
    return img


def operands_mha(args) -> dict:
    """mha.c: X[T][D] int8 and three [D][DH] int4 projections, distinct salts."""
    T, D, DH = 8, 8, 8
    img: dict = {}
    put_rowmajor_i8(img, 0x0000, T, D, D, a_val)
    put_rowmajor_i4(img, 0x0400, D, DH, w_val)
    put_rowmajor_i4(img, 0x0500, D, DH, lambda r, c: ((r * 3 + c * 7) % 16) - 8)
    put_rowmajor_i4(img, 0x0600, D, DH, lambda r, c: ((r * 11 + c * 5) % 16) - 8)
    return img


def operands_adder(args) -> dict:
    """adder.c: the whole model's DRAM image, with **synthetic** weights.

    Deliberately not a checkpoint. `make fw FWPROG=adder` is a datapath
    regression — does the CPU issue the right commands and does the array
    compute what the ISS says — and tying it to an untracked `.pt` would make it
    unrunnable the moment the model is retrained. `adder_export.py` stages the
    real thing into this same map.

    The map is adder.c's, and this is the only other place it is written down:

        0x00000 X0     [T][D]   int8    0x00800 mask  [T][T]  int8
        0x01400 W_fc   [D][16]  int4    0x01800 logits[T][16] int32 (out)
        0x02000 + L*0x6000: Wq [D][D], +0x0800 Wk, +0x1000 Wv, +0x1800 Wo,
                            +0x2000 W1 [D][F], +0x4000 W2 [F][D]

    Every tensor gets its own salt so a mis-addressed weight shows up as a wrong
    answer rather than as a coincidentally equal one, and the padding columns of
    W_fc carry **live** weights: staged as zeros they would agree with a kernel
    that strided its second output tile wrongly, because both would be zero.

    The weights come from :func:`w_hash` rather than the `(a*r + b*c) % 16`
    pattern the smaller kernels use: `c*b mod 16` has a period dividing 16, so a
    linear pattern makes blocks that differ only by a multiple-of-16 column
    offset bit-identical — and the four [D][D] projections here are exactly that
    kind of neighbour, which would make an addressing bug between them
    invisible.
    """
    T, D, DFF, VPAD, LAYERS = 32, 64, 256, 16, 4
    img: dict = {}

    put_rowmajor_i8(img, 0x00000, T, D, D, a_val)
    for t in range(T):
        for s in range(T):
            # 0 where s <= t, -8 above. S is int4, so S-8 <= -1 for every S in
            # range and ReLU takes a masked entry to exactly zero.
            img[0x00800 + t * T + s] = (0 if s <= t else -8) & 0xFF
    put_rowmajor_i4(img, 0x01400, D, VPAD, lambda r, c: w_hash(r, c, 0))

    for l in range(LAYERS):
        base = 0x02000 + l * 0x06000
        for i, off in enumerate((0x0000, 0x0800, 0x1000, 0x1800)):   # Wq Wk Wv Wo
            put_rowmajor_i4(img, base + off, D, D,
                            lambda r, c, s=6 * l + i + 1: w_hash(r, c, s))
        put_rowmajor_i4(img, base + 0x2000, D, DFF,
                        lambda r, c, s=6 * l + 5: w_hash(r, c, s))
        put_rowmajor_i4(img, base + 0x4000, DFF, D,
                        lambda r, c, s=6 * l + 6: w_hash(r, c, s))
    return img


# ---- spadwin.c --------------------------------------------------------------
SW_VEC, SW_TAB, SW_OUT = 0x0000, 0x0100, 0x0200
SW_N, SW_ROWB, SW_ROWS = 16, 16, 13


def _sw_vec(i: int) -> int:
    """A permutation of 0..15 scaled out of int8 range, so the max is unique.

    5 is coprime to 16, so `(5i+3) % 16` hits every value once: exactly one word
    is the largest, and it is at index 12 rather than at either end.
    """
    return ((i * 5 + 3) % 16) * 137 - 900


def _sw_tab(r: int, c: int) -> int:
    return ((r * 7 + c * 3) % 251) - 128


def operands_spadwin(args) -> dict:
    """spadwin.c: a vector for the CPU to scan and a table for it to gather from."""
    del args
    img: dict = {}
    for i in range(SW_N):
        for b in range(4):
            img[SW_VEC + i * 4 + b] = (_sw_vec(i) >> (8 * b)) & 0xFF
    put_rowmajor_i8(img, SW_TAB, SW_ROWS, SW_ROWB, SW_ROWB, _sw_tab)
    return img


def reference_spadwin(tpu: TPU) -> None:
    """What the CPU should have found, computed here instead.

    The ISS cannot check this one against itself at all: the argmax happens in
    the *firmware*, and every command that follows it is already conditioned on
    the answer. So the only statement worth making is the independent one.
    """
    vals = [_sw_vec(i) for i in range(SW_N)]
    best = max(range(SW_N), key=lambda i: (vals[i], -i))

    def dram_i32(addr):
        v = sum(tpu.dram[addr + b] << (8 * b) for b in range(4))
        return v - (1 << 32) if v >= (1 << 31) else v

    want = {SW_OUT + 0: best, SW_OUT + 4: vals[best], SW_OUT + 8: best}
    bad = [f"[0x{a:04x}] = {dram_i32(a)}, expected {w}"
           for a, w in sorted(want.items()) if dram_i32(a) != w]
    for c in range(SW_ROWB):
        got = tpu.dram[SW_OUT + 16 + c]
        exp = _sw_tab(best, c) & 0xFF
        if got != exp:
            bad.append(f"gathered row byte {c} = 0x{got:02x}, expected 0x{exp:02x} "
                       f"(row {best})")
    if bad:
        for b in bad[:8]:
            print(f"  REF FAIL {b}", file=sys.stderr)
        raise SystemExit("spadwin: the scratchpad window did not behave — the "
                         "CPU's read, its write, or the DMA address it derived")
    print(f"reference: the CPU found max {vals[best]} at index {best} and "
          f"gathered row {best}")


# ---- infer.c ----------------------------------------------------------------
# Its DRAM map. Everything it shares with adder.c is at the same address; what
# was the embedded X0 is now the embedding table and the token sequence, because
# this kernel embeds and argmaxes on the device.
IN_EMB, IN_TOK, IN_MASK, IN_WFC, IN_LOG = (0x00000, 0x00400, 0x00800,
                                           0x01400, 0x01800)
IN_T, IN_D, IN_DFF, IN_NH, IN_LAYERS = 32, 64, 256, 4, 4
IN_VOCAB, IN_VPAD, IN_PROMPT = 13, 16, 15
IN_DH = IN_D // IN_NH

# The synthetic prompt: "321+54" then pads to 14 and '=', which is the shape
# numbers_data emits (digits least-significant first, answer at EQUALS_POS=15).
# The weights below are not a checkpoint, so this decodes to nothing — it is
# here because a prompt that looks like a prompt makes a wrong gather obvious.
IN_PROMPT_IDS = [3, 2, 1, 10, 5, 4] + [12] * 8 + [11]

# fw/adder_rq.h's table, which fw/infer.c compiles in. DUPLICATED FROM THAT
# HEADER on purpose: the reference below has to know the fixed point to predict
# a single byte, and there is no path from a C macro to here. If the two drift
# the reference fails loudly on the first requant, which is the failure mode to
# want.
IN_RQ = {"Q": (1, 5), "K": (1, 5), "V": (1, 5), "KP": (1, 0), "VP": (1, 0),
         "S": (1, 3), "ID": (1, 0), "P": (1, 0), "A": (1, 5), "O": (1, 5),
         "XO": (1, 0), "X1": (1, 1), "H": (1, 5), "HR": (1, 0), "F": (1, 6),
         "X2": (1, 1)}


def emb_val(v: int, d: int) -> int:
    """An int4 embedding row. Distinct per token id, or a wrong gather is invisible."""
    return ((v * 7 + d * 3) % 9) - 4


def operands_infer(args) -> dict:
    """infer.c: the embedding table, the prompt ids, the mask, the head, the weights.

    Same synthetic weights as :func:`operands_adder`, at the same addresses and
    the same salts, so the two kernels are running one model — that is what lets
    a divergence between them mean something.
    """
    del args
    img: dict = {}

    put_rowmajor_i8(img, IN_EMB, IN_VOCAB, IN_D, IN_D, emb_val)
    for i, tok in enumerate(IN_PROMPT_IDS):        # int32, little-endian
        for b in range(4):
            img[IN_TOK + i * 4 + b] = (tok >> (8 * b)) & 0xFF
    for t in range(IN_T):
        for s in range(IN_T):
            img[IN_MASK + t * IN_T + s] = (0 if s <= t else -8) & 0xFF
    put_rowmajor_i4(img, IN_WFC, IN_D, IN_VPAD, lambda r, c: w_hash(r, c, 0))

    for l in range(IN_LAYERS):
        base = 0x02000 + l * 0x06000
        for i, off in enumerate((0x0000, 0x0800, 0x1000, 0x1800)):   # Wq Wk Wv Wo
            put_rowmajor_i4(img, base + off, IN_D, IN_D,
                            lambda r, c, s=6 * l + i + 1: w_hash(r, c, s))
        put_rowmajor_i4(img, base + 0x2000, IN_D, IN_DFF,
                        lambda r, c, s=6 * l + 5: w_hash(r, c, s))
        put_rowmajor_i4(img, base + 0x4000, IN_DFF, IN_D,
                        lambda r, c, s=6 * l + 6: w_hash(r, c, s))
    return img


# ---- tiled.c ----------------------------------------------------------------
# Its DRAM map, written once and read by both the operand builder and the
# reference below.
TL_A1, TL_W1, TL_C1, TL_C2 = 0x00000, 0x00600, 0x00800, 0x00E00
TL_A3, TL_W3, TL_C3 = 0x01400, 0x01600, 0x01700
TL_M1, TL_K1, TL_N1 = 40, 32, 32
TL_M3, TL_K3, TL_N3 = 8, 64, 8


def _tl_a3(r: int, c: int) -> int:
    return ((r * 7 + c * 3) % 9) - 4


def _tl_w3(r: int, c: int) -> int:
    return ((r * 3 + c * 7) % 16) - 8


def operands_tiled(args) -> dict:
    """tiled.c: two matmuls and one elementwise pass, all DRAM to DRAM."""
    del args
    img: dict = {}
    put_rowmajor_i8(img, TL_A1, TL_M1, TL_K1, TL_K1, a_val)
    put_rowmajor_i4(img, TL_W1, TL_K1, TL_N1, w_val)
    put_rowmajor_i8(img, TL_A3, TL_M3, TL_K3, TL_K3, _tl_a3)
    put_rowmajor_i4(img, TL_W3, TL_K3, TL_N3, _tl_w3)
    return img


OPERANDS = {
    "matmul": operands_matmul,
    "matmul_loop": operands_matmul,
    "ffn": operands_ffn,
    "mha": operands_mha,
    "adder": operands_adder,
    "infer": operands_infer,
    "spadwin": operands_spadwin,
    "tiled": operands_tiled,
}


# =============================================================================
# Independent references.
#
# The golden DRAM image is whatever the ISS computed, which checks the RTL
# against the ISS and nothing else. That is the right check for a kernel whose
# job is to drive the datapath: the two implementations of `matmul` are
# independent, so agreeing means something.
#
# It is NOT enough for a kernel whose job is to drive a *loop*. If tpulib.h
# tiles a matmul wrongly — a stale stride, a block base off by a tile — the ISS
# executes the wrong commands exactly as faithfully as the hardware does, and
# both agree on the wrong answer. So a kernel may register a reference here, and
# it is checked against the ISS's DRAM before any vector file is written.
# =============================================================================
def _ref_matmul(a, w, m: int, k: int, n: int, rq_m0: int, rq_n: int) -> list:
    """C = requant(A @ W), plain Python. `a`/`w` are index functions."""
    out = []
    for i in range(m):
        row = []
        for j in range(n):
            acc = sum(a(i, t) * w(t, j) for t in range(k))
            v = (acc * rq_m0 + (1 << (rq_n - 1) if rq_n else 0)) >> rq_n
            row.append(max(-8, min(7, v)))
        out.append(row)
    return out


def reference_tiled(tpu: TPU) -> None:
    """tiled.c's three results, computed without the ISS or the kernel."""
    c1 = _ref_matmul(a_val, w_val, TL_M1, TL_K1, TL_N1, 1, 4)
    c3 = _ref_matmul(_tl_a3, _tl_w3, TL_M3, TL_K3, TL_N3, 1, 4)
    want = {}
    for i in range(TL_M1):
        for j in range(TL_N1):
            want[TL_C1 + i * TL_N1 + j] = c1[i][j] & 0xFF
            want[TL_C2 + i * TL_N1 + j] = max(c1[i][j], 0) & 0xFF
    for i in range(TL_M3):
        for j in range(TL_N3):
            want[TL_C3 + i * TL_N3 + j] = c3[i][j] & 0xFF

    bad = [(a, tpu.dram[a], v) for a, v in sorted(want.items()) if tpu.dram[a] != v]
    if bad:
        for a, got, exp in bad[:8]:
            print(f"  REF FAIL dram[0x{a:05x}] = 0x{got:02x}, expected 0x{exp:02x}",
                  file=sys.stderr)
        raise SystemExit(f"tiled: {len(bad)} of {len(want)} result bytes disagree "
                         f"with the independent reference — the kernel's tiling "
                         f"is wrong, not just the hardware's copy of it")
    print(f"reference: {len(want)} result bytes match an independent matmul")


def _rq(acc, mn, lo=-8, hi=7):
    """`clip((acc*m0 + 2**(n-1)) >> n)` — iss.requant8, over a numpy array.

    numpy's `>>` on a signed integer is arithmetic, i.e. it floors, which is
    what Verilog's `>>>` and Python's own `>>` do. dyt is this with lo=-7.
    """
    import numpy as np

    m0, n = mn
    v = (acc.astype(np.int64) * m0 + ((1 << (n - 1)) if n else 0)) >> n
    return np.clip(v, lo, hi)


def reference_infer(tpu: TPU) -> None:
    """infer.c's generated tokens and logits, computed WITHOUT a KV cache.

    `tiled.c` has an independent reference for one reason and this needs one for
    the same reason, more so: every bug specific to this kernel — a cache column
    written at the wrong offset, the mask row of the wrong position, an argmax
    over the wrong words, an embedding gathered from the wrong row — is a bug
    the ISS reproduces as faithfully as the hardware would, because the ISS is
    executing the commands the kernel *asked* for.

    So this recomputes the whole model from scratch at every step, over the
    prefix the device has generated so far, in plain integer numpy: no cache, no
    tiling, and no requant table but `IN_RQ` above. If the device's cache is
    right, a cached step and a full recompute are the same arithmetic — that is
    the claim the kernel is making, and this is the check of it.
    """
    try:
        import numpy as np
    except ImportError:     # not a declared dep of the repo, but torch's own
        raise SystemExit("infer's reference needs numpy (which torch installs)")

    def wblk(rows, cols, salt):
        return np.array([[w_hash(r, c, salt) for c in range(cols)]
                         for r in range(rows)], dtype=np.int64)

    emb = np.array([[emb_val(v, d) for d in range(IN_D)]
                    for v in range(IN_VOCAB)], dtype=np.int64)
    wfc = wblk(IN_D, IN_VPAD, 0)
    lay = [{"q": wblk(IN_D, IN_D, 6 * l + 1), "k": wblk(IN_D, IN_D, 6 * l + 2),
            "v": wblk(IN_D, IN_D, 6 * l + 3), "o": wblk(IN_D, IN_D, 6 * l + 4),
            "w1": wblk(IN_D, IN_DFF, 6 * l + 5), "w2": wblk(IN_DFF, IN_D, 6 * l + 6)}
           for l in range(IN_LAYERS)]

    # How many tokens the run actually generated, from what it spilled — so a
    # kernel built with -DINFER_GEN=n needs no second copy of n over here.
    n_gen = len([a for a in tpu.dram_written
                 if IN_TOK + IN_PROMPT * 4 <= a < IN_TOK + IN_T * 4]) // 4
    if n_gen == 0:
        raise SystemExit("infer: the run generated no tokens at all")

    toks = list(IN_PROMPT_IDS)
    want_tok, want_log = {}, {}

    for step in range(n_gen):
        pos = IN_PROMPT - 1 + step          # the row whose logits pick the next
        X = emb[toks[:pos + 1]]             # [pos+1][D], int4 codes

        for w in lay:
            n = X.shape[0]
            Q = _rq(X @ w["q"], IN_RQ["Q"])
            K = _rq(X @ w["k"], IN_RQ["K"])
            V = _rq(X @ w["v"], IN_RQ["V"])
            A = np.zeros((n, IN_D), dtype=np.int64)
            for h in range(IN_NH):
                sl = slice(h * IN_DH, (h + 1) * IN_DH)
                S = _rq(Q[:, sl] @ K[:, sl].T, IN_RQ["S"])          # [n][n]
                # The causal mask, as the kernel applies it: -8 against an int4
                # score is at most -1, and ReLU takes it to exactly zero. The
                # cache's uninitialized tail dies the same way, which is why
                # this reference can simply not have one.
                S = np.tril(_rq(S, IN_RQ["ID"]))
                P = _rq(np.maximum(S, 0), IN_RQ["P"])
                A[:, sl] = _rq(P @ V[:, sl], IN_RQ["A"])
            O = _rq(A @ w["o"], IN_RQ["O"])
            XO = _rq(X + O, IN_RQ["XO"])
            X1 = _rq(XO + X, IN_RQ["X1"], lo=-7)                    # dyt
            H = _rq(X1 @ w["w1"], IN_RQ["H"])
            HR = _rq(np.maximum(H, 0), IN_RQ["HR"])
            F = _rq(HR @ w["w2"], IN_RQ["F"])
            X = _rq(X1 + F, IN_RQ["X2"], lo=-7)                     # dyt

        logits = X[pos] @ wfc                                       # int32, raw
        want_log[pos] = logits
        nxt = int(np.argmax(logits[:IN_VOCAB]))                     # ties -> lowest
        want_tok[pos + 1] = nxt
        toks.append(nxt)

    # ---- compare ------------------------------------------------------------
    def dram_i32(addr):
        v = sum(tpu.dram[addr + b] << (8 * b) for b in range(4))
        return v - (1 << 32) if v >= (1 << 31) else v

    bad = []
    for pos, want in sorted(want_tok.items()):
        got = dram_i32(IN_TOK + pos * 4)
        if got != want:
            bad.append(f"token[{pos}] = {got}, expected {want}")
    n_log = 0
    for pos, want in sorted(want_log.items()):
        for j in range(IN_VPAD):
            got = dram_i32(IN_LOG + (pos * IN_VPAD + j) * 4)
            n_log += 1
            if got != int(want[j]):
                bad.append(f"logit[{pos}][{j}] = {got}, expected {int(want[j])}")

    if bad:
        for b in bad[:8]:
            print(f"  REF FAIL {b}", file=sys.stderr)
        raise SystemExit(f"infer: {len(bad)} values disagree with the "
                         f"cache-free reference — the KV cache, the mask window "
                         f"or the argmax is wrong, not just the datapath's copy "
                         f"of it")

    # A run that emitted one token over and over, or all-zero logits, would
    # match a reference that did the same and test nothing. Say what it was.
    seq = [want_tok[p] for p in sorted(want_tok)]
    if len(set(seq)) == 1:
        print(f"  WARNING: every generated token is {seq[0]} — the synthetic "
              f"weights have collapsed and this check is weak", file=sys.stderr)
    print(f"reference: {len(seq)} generated tokens and {n_log} logits match a "
          f"cache-free recompute; sequence {seq}")


REFERENCES = {
    "tiled": reference_tiled,
    "infer": reference_infer,
    "spadwin": reference_spadwin,
}


def build_operands(tpu: TPU, args) -> dict:
    """Write the kernel's operand image into the model's DRAM; return it."""
    if args.kernel not in OPERANDS:
        raise SystemExit(f"no operand builder for kernel {args.kernel!r} — "
                         f"add one to OPERANDS in {__file__}")
    image = OPERANDS[args.kernel](args)
    for addr, byte in image.items():
        tpu.dram[addr] = byte
    return image


def operand_image(m_rows: int, k: int, n: int) -> dict:
    """The matmul kernels' image, for `host/run_fw_matmul.py`.

    Kept as a positional-argument shim because the host script knows its shape
    as three numbers, not as parsed args.
    """
    img: dict = {}
    put_rowmajor_i8(img, A_ADDR, m_rows, k, k, a_val)
    put_rowmajor_i4(img, W_ADDR, k, n, w_val)
    return img


def write_hex(path: str, byte_map: dict, header: str) -> None:
    """Sparse `$readmemh` byte image: an `@addr` directive per discontinuity."""
    with open(path, "w") as f:
        f.write(f"// {header}\n")
        prev = None
        for addr in sorted(byte_map):
            if prev is None or addr != prev + 1:
                f.write(f"@{addr:05x}\n")
            f.write(f"{byte_map[addr]:02x}\n")
            prev = addr


def write_cmds(path: str, cmds: list, header: str) -> None:
    with open(path, "w") as f:
        f.write(f"// {header}\n")
        for unit, w0, w1, w2, w3 in cmds:
            f.write(f"{unit:08x} {w0:08x} {w1:08x} {w2:08x} {w3:08x}\n")
        f.write("ffffffff 00000000 00000000 00000000 00000000\n")   # terminator


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-x", "--exec", dest="exe",
                    help="a -DTPU_TRACE firmware binary to run against the ISS, "
                         "answering its scratchpad reads (the only way to trace "
                         "a kernel that branches on its own results)")
    ap.add_argument("-t", "--trace",
                    help="...or a command trace already captured from one")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("-k", "--kernel", default="matmul",
                    help="which kernel's operand image to build "
                         f"({', '.join(sorted(OPERANDS))})")
    ap.add_argument("-M", type=int, default=8, help="token rows")
    ap.add_argument("--ktiles", type=int, default=4)
    ap.add_argument("--ntiles", type=int, default=2)
    args = ap.parse_args()

    if bool(args.exe) == bool(args.trace):
        raise SystemExit("pass exactly one of -x (run the kernel binary) and "
                         "-t (read a captured trace)")

    tpu = TPU(rows=ROWS, cols=COLS)
    dram_in = build_operands(tpu, args)

    os.makedirs(args.out, exist_ok=True)
    if args.exe:
        # The operands are already in DRAM, which they have to be *before* the
        # first command runs: a kernel that reads its own results is executing
        # against this model as it goes, not replaying into it afterwards.
        cmds, lines = coexecute(tpu, args.exe)
        src = os.path.join(args.out, f"{args.kernel}.trace.txt")
        with open(src, "w") as f:
            f.writelines(lines)
    else:
        src = args.trace
        with open(src) as f:
            records = parse_trace(f.read())
        cmds = tpu.run_trace(records)

    if args.kernel in REFERENCES:
        REFERENCES[args.kernel](tpu)

    # Everything the run spilled back to DRAM, straight out of the model's own
    # write tracking — no second reference implementation to keep in step.
    dram_exp = {a: tpu.dram[a] for a in sorted(tpu.dram_written)}

    shape = f"{args.kernel}, int4 row-major weights"
    write_hex(os.path.join(args.out, "fw_dram_in.hex"), dram_in,
              f"firmware operands: {shape}")
    write_hex(os.path.join(args.out, "fw_dram_exp.hex"), dram_exp,
              f"firmware golden DRAM output (ISS-computed): {shape}")
    write_cmds(os.path.join(args.out, "fw_cmds.hex"), cmds,
               f"expected command trace, {len(cmds)} commands, from {src}")

    print(f"{len(cmds)} commands, {len(dram_in)} operand bytes in, "
          f"{len(dram_exp)} bytes out -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
