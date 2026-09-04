# accel/tpu/fw/ — firmware layer notes

Background and design rationale for the files in `accel/tpu/fw/`. Code comments
there stay to one line; anything longer lives here.

## tpu.h — the command plane

Mirrors `rtl/cpu_subsys.sv` (address map) and `rtl/cmd_{mxu,vpu,dma}.sv`
(command fields). Each builder packs one 128-bit macro-op exactly as the
decoder reads it — no driver abstraction.

Address map://
- `0x8000_0000 + 0x10*unit` — 4-word command port, commits on word 3
- `0x8000_0040` — status: issued/retired/level per unit
- `0x8000_0070` — write = done; read = unit_idle
- `0x9000_0000` — the scratchpad, as CPU-addressable words

A full queue withholds the write response on word 3, so the CPU stalls inside
the store — flow control needs no software. Cross-unit ordering IS software's
job: the three queues are independent, so anything the MXU reads must be
`tpu_wait(TPU_U_DMA)`'d first.

Only four primitives are target-specific (`tpu_push`, `tpu_wait`,
`tpu_spad_ld`, `tpu_spad_st`). Everything else is plain `uint32_t` arithmetic,
which is what lets `-DTPU_TRACE` compile the same kernel source with the host
compiler and print its command trace instead of executing it. New builders
written in terms of `tpu_push` are traced for free.

The scratchpad window (`tpu_spad_ld`/`tpu_spad_st`) is not a command:
`cpu_subsys.sv` decodes `0x9xxx_xxxx` onto the scratchpad's S port directly.
Two rules follow from that:
- 32-bit accesses only, 4-byte aligned — the S port has no byte strobes, so an
  `sb`/`sh` also overwrites its neighbours in the word.
- Fence first — a load is not a command, so it doesn't wait for the queues;
  `tpu_wait()` the producing unit or the read is stale. The S port also loses
  arbitration to the MXU and VPU, so a read mid-matmul stalls the core inside
  the load.

## tpulib.h — primitives over tpu.h's macro-ops

Every operand lives in DRAM. A primitive blocks the problem into pieces the
hardware accepts, stages them through the caller's arena, and fences before it
returns, so composing two is always safe. `tpu_move_bytes` is the exception —
a raw DMA the caller fences itself.

`tpu_matmul`'s row loop runs over panels sized to the arena rather than single
N-row blocks, and a panel's whole C stays resident until the column loop is
done — so a weight stream and a result spill are each paid once per panel, not
once per output block. `tpu_matmul_wide` is the same loop with a one-block-wide
C; the section below is when to reach for which.

`tpu_matmul`, `tpu_gemm_blocks`-equivalent chooser logic, and friends are
`always_inline`, which is load-bearing: with the shape constant at the call
site gcc folds the block arithmetic away, and the PicoRV32 runs it with no
unit busy. See CLAUDE.md's note on `always_inline` firmware primitives.

### `tpu_matmul_wide` — one column block of C on chip instead of a whole row

Same GEMM, same `tpu_gemm` struct, same arena. The one structural change is that
a staged C row is `TPU_N` wide rather than `cols` wide, so the arena's spare
bytes buy row-panel depth instead of C width:

| | a panel row costs | staged C |
| --- | --- | --- |
| `tpu_matmul` | `depth/2 + align_up(cols, N)/2` | `[panel_rows][cols]` |
| `tpu_matmul_wide` | `depth/2 + TPU_WORD_BYTES` | `[panel_rows][N]` |

B is re-read `ceil(rows / panel_rows)` times, and that re-read is the whole
weight stream. So a deeper panel is fewer passes over B — which is the win, and
the only win. It is paid for with `cols/N` spill commands per panel instead of
one, so the same C bytes leave in `cols/N` times as many DMA ranges.

`tests/tiled/`'s fifth pass is the regression, and `MM5_WIDE=0`
(`--mm5-general`) runs the identical problem through `tpu_matmul` against the
same golden — that is the A/B.

#### B is double-buffered

The B slot is two bank-aligned halves. The fill for column block `c0+N` is
issued *after* the barrier that made block `c0` resident, so it streams under
block `c0`'s matmuls, and the half it writes was last read by block `c0-N`,
which retired an iteration earlier. The halves are a bank apart because the
matmul outranks the DMA at the scratchpad — sharing one would cost the prefetch
a beat for every beat the array reads.

That second half costs a bank, which comes out of the row panel: at the live
`infer` arena a `depth=128` wide panel is 768 rows against 832 single-buffered.
`TPU_WGT_PREFETCH=0` compiles the prefetch out and gives the bank back, which is
the A/B. An arena with no room for the second half single-buffers on its own
rather than failing.

#### Where it is not legal

These are `TPU_SHAPE_ASSERT`s: a **compile error** when the shape is constant at
the call site, which is the normal case for an `always_inline` primitive, and a
`TPU_ASSERT` otherwise. **`TPU_ASSERT` compiles to nothing outside the
`TPU_TRACE` build**, so a shape that only the run knows is caught on the ISS and
corrupts silently on the RTL and the board.

1. **`depth` not a multiple of `TPU_N`, or past `0xFFFF`.** The contraction is
   taken in one dispatch, exactly as in `tpu_matmul`.
2. **An odd `cols`.** A row is two elements per byte.
3. **An arena under three banks**, or under whatever one N-row block of each of
   A, B and C costs at this `depth`. A, B and C are read on the same clock and a
   bank serves one requester per clock, so each needs a whole bank of its own.

`transpose` and `accumulate` both work now — the transposed fill is `ncols` rows
of `depth` with `b_stride = depth/2`, and accumulate fills the C column block
from DRAM alongside its block of B, one fill per column block rather than per
panel.

A `cols` that is *not* a multiple of `TPU_N` is fine — the last column block
computes `TPU_N` wide and spills `ncols`, the same way `tpu_matmul` does. This is
the one restriction the hand-written `tiled_simple.c::tiled_matmul_optimized`
has that the library function does not.

#### Where it is legal and still a loss

6. **`depth/2` already dominates `cols/2`.** Then A's slot, not C's, is what caps
   the panel, both functions pick the same `panel_rows`, and the extra spill
   commands are pure cost. `tiled`'s default fifth pass — 20x512 @ 512x52 in a
   3-bank arena — is exactly this: both settle on a 16-row panel, and the run is
   **92 commands wide against 80 general**.
7. **The problem already fits one panel of `tpu_matmul`** (`rows <= panel_rows`).
   Both read B once; only the spill count differs, so wide can only lose.
8. **`cols` near `TPU_N`.** There is nothing to reclaim from C's width.
9. **A wide C over a shallow panel.** The spill command count is
   `ceil(rows/panel_rows) * cols/N`, and each of those is a fence for the
   producer. Past some width the issue overhead outruns the B reuse — the
   crossover is a property of the shape, so measure it rather than assume.

Range count itself is close to free: `tiled_simple` at 128x128 @ 128x512 moves
16 512 DMA ranges and lands at 106 688 clocks against a 106 496-clock floor, so
`sram.sv` charges ~0 per range beyond its bytes. The cost of point 9 is the
**commands**, not the ranges.

#### What it is worth when it applies

The A/B in `tests/tiled/`, at `--rows5 128 --depth5 64 --cols5 512
--arena-banks 6`. `tpu_matmul` settles on a 48-row panel there (a panel row
costs it `32 + 256` bytes) and reads B three times; `tpu_matmul_wide` gets 128
rows (`32 + 4`) and reads it once. Whole-kernel totals through the RTL, so
passes 1-4 are the same work in both columns:

| | commands | run | DMA busy | idle | MXU busy |
| --- | --- | --- | --- | --- | --- |
| `--mm5-general` | 1259 | 281 811 | 142 099 | 42 181 | 95 866 |
| `tpu_matmul_wide` | 1190 | **221 779** | **109 514** | **14 734** | 95 866 |

Identical MXU busy — it is the same arithmetic — and 60 032 clocks off the run,
32 585 of it the two extra passes over B and 27 447 of it the barriers those
passes cost the CPU.

`tests/tiled_simple/` is the same algorithm hand-written out of raw `tpu.h`
commands, without the library's arena arithmetic, and it is the cleaner
measurement. At 128x128 @ 128x512 through the RTL:

| | commands | run | DMA busy | idle |
| --- | --- | --- | --- | --- |
| one row block at a time | 3089 | 997 146 | 601 088 | 239 386 |
| the whole of A staged | 1154 | **273 112** | **106 688** | 9 752 |

3.65x, and the DMA lands on the single-pass floor: A once, B once, C once. What
was removed is 15 re-reads of the weight stream.

`tiled_matmul_optimized` is now the only kernel in that test: the one-row-block
variant it was measured against is gone, and the superblock height is derived in
the C from `SPAD_SIZE` and the shape rather than passed in as `SUPER_ROWS`. It
takes `transpose` and `acc`, which pick up `TPU_MM_T` (W stored `[N][K]`, filled
as `TPU_N` rows of `K` and given a `b_stride` of `K/2`) and `TPU_MM_ACC` (each C
block filled from DRAM alongside its column block of W, one fill per superblock
column rather than per row block). `generate.py`'s `--transpose` / `--acc` are
the A/B.

It also double-buffers W. The region is two bank-aligned halves; the fill for
column block *j+1* is issued after the barrier that made block *j* resident, so
it streams under block *j*'s matmuls, and the half it writes was last read by
block *j-1*, which retired an iteration earlier. The halves are a bank apart
because a bank serves one requester per clock and the matmul outranks the DMA —
sharing one would cost the prefetch a beat for every beat the array reads.
`DOUBLE_BUFFER_W=0` (`--single-buffer`) compiles it back to one half and the
fill between the barriers, which is the A/B; it also gives the row superblock a
bank back, so the two arms do not stage the same number of rows.

### `tests/wide/` — the regression for `tpu_matmul_wide`

One `tpu_matmul_wide`, DRAM to DRAM, at the shape the double buffer was written
against: `20x512 @ 512x52` in a 4-bank arena, so the row panel is 16 and repeats,
there are seven column blocks (the parity ends odd and has to reset for the
second panel), and the last block is ragged. The golden is a plain Python
matmul.

Everything else is a flag over the same problem: `--transpose` stores B as
`[N][K]` and runs it with `TPU_MM_T`, `--acc` seeds `DR_C` and runs with
`TPU_MM_ACC` — the two paths the function did not used to have. `--single-buffer`
compiles the prefetch out, and `--arena-banks 3` leaves no room for the second
half so it single-buffers on its own. Through the RTL:

| | commands | run | overlap |
| --- | --- | --- | --- |
| 3 banks, single-buffered | 52 | 47 190 | 0 |
| 4 banks, double-buffered | 52 | **37 084** | 9 576 |
| 4 banks, `TPU_WGT_PREFETCH=0` | 37 | 32 220 | 0 |
| 4 banks, `--transpose --acc` | 66 | 37 646 | 9 704 |

The first two rows are the prefetch on its own — identical work and identical
commands, and the overlap is the whole 10 106-clock difference. The third row is
the second half's other side: with the bank back the panel goes 16 rows to 32,
the pass over B halves, and the DMA clocks it saves beat what the overlap buys.
**Which way it lands is a property of the shape and the arena, so measure it.**

## infer.c — the model generating

The int4 adder model as inference: prefill, then decode against a KV cache.
Token ids in, token ids out, one run. `infer_block(rows, first_pos)` runs
`rows` new positions per sequence, so the row count is the only difference
between the training shape and the generative one.

**Phases, for benchmarking.** `INFER_PREFILL`/`INFER_DECODE` select which half
of a generation this image runs; both default to 1 (the whole thing, what the
accuracy path builds). They come from the config header, so
`tests/infer/generate.py --phase prefill|decode|both|split` is what sets them.
The counters can't be read mid-run — they reset at `G` and freeze at the
halt — so an image that runs one half IS the measurement of that half.

- `--phase split` runs the prefill-only and the decode-only image in turn and
  prints each phase's clocks divided by the tokens it covered. Two images, so
  the sum counts the prompt load and the id spill twice: at `d=32/f=64/L=2,
  PROMPT=16, --gen 3` it is 267 998 clocks against 263 316 for one whole run,
  1.8% high.
- The decode-only image starts from the prompt's last token instead of the one
  the prefill would have produced, so its ids are noise and nothing is checked.
  A step's cost isn't data-dependent, so the clocks are the same clocks.
- The prefill-only image still produces the reference's first token, so that
  one token and its logits are checked against the golden like any other case.
- `head_argmax` is `noinline`. With both phases gcc outlines it anyway; a
  phase-only image has one call site, and inlining it into `main` grew `main`
  to where gcc expands `tpu_gemm_fit`'s `spare / (depth_bytes + c_slot_row)`
  as a `__udivsi3` call, which does not link — there is no libgcc here. It also
  keeps every phase measuring the same head. The whole-run image is
  byte-identical with the attribute (14 708 bytes); prefill-only is 8 372 and
  decode-only 7 908.

**Batching.** `BATCH` independent sequences share every weight stream. X is
`[BATCH][rows][D]`, sequence-major, so the three projections, `Wo` and both FFN
matmuls run once over `BATCH*rows` rows. Attention stays per sequence — each
has its own KV cache and its own append into it.

**Which matmul each site gets.** All of them go through `INFER_MM`, which is
`tpu_matmul_wide` — including `Q@K^T`, now that the wide layout has a
transposed-B path. `tpu_matmul` has no call site left in this kernel, so gcc
emits only one of the two. `-DINFER_MM_WIDE=0` (`generate.py --general`) puts
every site back on `tpu_matmul` against the same golden, which is the A/B.

**What it buys, and it is not the row panel.** At the live shape — `d=128`,
`f=512`, a 61 440-byte usable arena, `BATCH=1`, `BLOCK=32` — every site already
fits one row panel on either layout, so the deeper panel removes no pass over B:

| site | rows | shape | general panel | wide panel | passes over B |
| --- | --- | --- | --- | --- | --- |
| `Wq`/`Wk`/`Wv`/`Wo` | 32 | 128 x 128 | 448 | 768 | 1 either way |
| `Q@K^T` | 32 | 32 x 64 (T) | 768 | 768 | 1 either way |
| `P@V` | 32 | 128 x 32 | 704 | 768 | 1 either way |
| FF1 | 32 | 128 x 512 | 160 | 768 | 1 either way |
| FF2 | 32 | 512 x 128 | 176 | 192 | 1 either way |
| head | 1 | 128 x 16 | 768 | 768 | 1 either way |

The win is the B prefetch, which `tpu_matmul` does not have at all. Through the
RTL, `--gen 3` at that shape, one problem, byte-identical DRAM on both arms:

| | clocks | mxu | dma | idlec | ovlap | commands |
| --- | --- | --- | --- | --- | --- | --- |
| `--general` | 3 489 116 | 755 094 | 1 985 938 | 654 196 | 0 | 8 682 |
| wide | **2 890 294** | 755 094 | 1 992 355 | 632 652 | **583 695** | 10 821 |

Identical array work and within 0.3% on DMA clocks: the whole 598 822-clock
difference is overlap the general layout cannot express, worth **17.2%** of the
run. It is paid for in commands (+24.6%, the `cols/N` spills per panel) and in
image — 14 708 bytes of the 16 KB firmware RAM against 12 204, so the stack has
~1.6 KB rather than ~3.9 KB.

FF1 is where the row panel would start to matter too: it wants `BATCH*BLOCK >
160`, so `--batch 8` at `BLOCK=32` takes its passes over `W1` from 2 to 1 on the
wide layout and leaves them at 2 on the general one.

**Where tensors live.** Every tensor is in DRAM, every element int4, two per
byte. `tpulib.h` owns the scratchpad: it stages each primitive's operands in
and results out, so `infer.c` names no scratchpad address except the mailbox
the CPU itself reads.

**The KV cache.** Both halves are `[T][D]` int4 — exactly how K and V leave
their projections — because the MXU transposes its second operand itself. So
an append is a copy of `rows` rows, `Q@K^T` is a transposed matmul against the
K cache and `P@V` a plain one against the V cache, with no repack and no
transposing DMA between them. Nothing is zeroed: cache rows past the current
position reach S as garbage, but S is int4 and the mask is -8, so a masked
score is at most -1 and ReLU takes it to exactly zero. The causal mask is what
makes an uninitialized cache safe.

**The host.** Stages `DR_TOKENS` (prompt ids), `DR_EMBED`, `DR_MASK`,
`DR_HEAD_WGT` and the six weight blocks per layer; presses `'G'`; reads
`DR_TOKENS[seq][PROMPT..]` back. The argmax and embedding gather are on the
device — `cpu_subsys.sv` maps the scratchpad at `0x9xxx_xxxx`, and a DMA takes
an address the CPU computed. The host tokenizes and nothing else.

**DRAM layout.** A computed chain (`DR_ALIGN` cascade), so a shape change
re-lays it out and `DR_END`'s assert says whether the result still fits. Two
rotating temporaries are the whole activation working set — an intermediate
dies as soon as its consumer has read it:
- `DR_TMP_A`: `K_new -> A -> X+O -> F`
- `DR_TMP_B`: `V_new -> O -> X1`

`DR_SCRATCH` holds whichever phase of the layer is running (`Q`/`S` during
attention, `H` once the FFN starts) — attention's working set and the FFN's
hidden layer never coexist, so they're one region and the layer costs the
larger of the two rather than the sum. This is the only aliasing in the map;
`X`, `TMP_A`, `TMP_B` are live across the whole layer.

If `DR_END <= DR_LAYER0` fails: lower `BLOCK` first (it scales `X`, `TMP_A`,
`TMP_B` and the scratch union, and costs one weight stream per extra pass),
then lower `BATCH`.

**Scratchpad.** The CPU has no path to DRAM, so what it reads (prompt ids,
logits) and writes (the chosen token) lives in a fixed mailbox the scratchpad
window reaches. That's the only fixed allocation — the rest is arena.

**`--bench` — timing without weights.** `tests/infer/generate.py --bench` runs
the same kernel with nothing staged into DRAM: no weights, no embedding table,
no mask, no zeroed cache, and no golden compare. What a step costs is not
data-dependent, so the counters are the same ones a checked run reports, and
the load — ~400 KB of weights, which dominates `-b rtl-uart` and `-b board` —
is gone. The prompt ids are still patched in (an id outside the table would
make the embedding gather read DRAM the map does not own), and they are built
in the generator rather than taken from the dataset, so `--bench` needs neither
torch nor a checkpoint. It defaults to one case, and every address in the map
counts as writable, since with an empty image the stray-write check has no
baseline to compare against.

    python accel/test/tests/infer/generate.py -b rtl --synthetic --gen 3 --bench

## ffn.c / mha.c / matmul.c — datapath smoke tests

Small, fixed-shape kernels that exercise one piece of the pipeline each, with
operands supplied by each kernel's `generate.py` (and the ISS, which runs the kernel's own
command trace through `iss.py` for the golden result).

- `matmul.c` — `C = requant(A @ W)`, the block grid walked in firmware; shape
  overridable from the Makefile so `tb/run_fw_sweep.sh` can sweep it.
- `ffn.c` — `H = requant(X@W1)`, `relu(H)`, `Y = requant(H@W2)`; the first
  kernel to depend on MXU-then-VPU ordering (`tpu_wait(TPU_U_MXU)` before the
  VPU push, since the VPU's queue isn't ordered against the MXU's).
- `mha.c` — one head of ReLU attention: `Q/K/V` projections, `S = Q@K^T`
  (transposed), `P = relu(S)`, `A = P@V`. Exercises both units, both DMA
  directions, and the MXU's transpose flag. No causal mask — a datapath/ISA
  test, not the model.
  - Why K needs the transpose flag and V doesn't: for `out = A @ B` the array
    reads B's row `k` contiguous over the output columns. `Q@K^T` contracts
    over the head dim, so its B is `K^T[h][s]` — K read down its columns,
    while K leaves its projection row-major; that's what `transpose` does, in
    the array, with no pass in between. `P@V` contracts over keys, so its B is
    `V[s][h]` — exactly how V left its projection. Free either way.

All three kernels' `{m0,n}` requant shifts are tuned against measured
accumulator ranges, not guessed — a golden answer of all zeros (from an
over-large shift) or a fully-saturated one both pass against any datapath at
all, so the comment at each site records the range that ruled that out.

## spadwin.c — the scratchpad window on its own

Every other kernel drives the compute units; this one drives the memory they
share, from the CPU — because `infer.c` depends on a path nothing else
exercises: the CPU reading what the array computed and writing what no unit
produced. `cpu_smoke_tb` proves the command aperture; this is the same kind of
proof for the data window (`python accel/test/run_suite.py -b rtl -k spadwin`).

Three things, none of which a DMA can do alone:
- **Read** — a block fills from DRAM and the CPU reads all 16 words back,
  finding the largest (an argmax over a tensor — what `infer.c` needs and the
  VPU has no op for, since `REDUCEMAX` went with the softmax datapath).
- **Write** — the index and value go back through the window, then are read
  straight back to prove the write landed rather than merely being accepted
  (the S port has no byte strobes, so both are 32-bit accesses at 4-aligned
  addresses).
- **Gather** — the CPU fills a table row at `DR_TABLE + index*16`, a DMA whose
  address it computed from data it read.

The host-staged pattern makes the maximum unique and puts it at index 12, so a
read returning zero/a constant/the wrong word can't accidentally agree, and the
gathered row sits near the end of the table rather than at its base.

## tiled.c — tpulib.h past the point where the scratchpad helps

`infer.c` exercises the library on shapes that mostly fit; this kernel is the
other half. The arena is exactly three banks — the smallest a matmul can run
in, since A, B and C each need one of their own — so every loop in
`tpu_matmul` and `tpu_elementwise` runs more than once to produce a right
answer.

Five problems, chosen for what they force rather than for what they compute:
1. `C1 = A1 @ W1`, 16x1024 @ 1024x16 — a whole 1024-element contraction in one
   dispatch (`len` is 16 bits, so the array never splits one) against operands
   that don't fit a bank; both the row loop and column loop run twice.
2. `C2 = relu(C1)`, then `C3 = add(V1, V2)` over 2500 elements — the
   elementwise chunk loop, with a tail that isn't a whole chunk; both operands
   are in DRAM and stream through the arena.
3. `C4 = A4 @ B4'`, 12x64 @ 20x64 transposed — the MXU's transpose flag, and
   extents that are NOT multiples of the array: the block computes padded and
   only the live rows/columns spill, the path that keeps decode's rows=1
   honest.
4. `C5 = A5 @ W5`, 20x512 @ 512x52 through `tpu_matmul_wide` — more rows than
   one panel of a 3-bank arena, so its row loop runs twice, and a last column
   block that is 4 of 8 wide. `--mm5-general` runs the same problem through
   `tpu_matmul` against the same golden.

This is the one kernel that alone wouldn't test anything, because a tiling bug
is something the ISS would reproduce as faithfully as the RTL. So
`tests/tiled/generate.py` also carries an independent Python matmul and checks the
ISS against a plain Python matmul before any vectors are written.

## memops.c — memcpy/memset and 32-bit unsigned division

`-ffreestanding -fno-builtin` stops gcc from recognizing memcpy/memset in
source, but not from emitting calls to them: the ABI lowers a struct
assignment or large aggregate initializer to `memcpy` regardless of the flags.
Nothing here links libc or libgcc, so without this file a kernel using a
struct fails at link with "undefined reference to memcpy". Both are word-wise
where pointers and length allow, since these sit on the CPU's issue path where
every clock is exposed.

No shipped kernel links this today — `tpulib.h`'s `always_inline` entry points
fold every descriptor into constants, so none is ever materialized or copied,
and `--gc-sections` drops the file. It stays because that's a property of
these kernels, not of the library: a kernel with genuinely runtime shapes gets
the general path, gets a descriptor in memory, and needs this.

### `__udivsi3` / `__umodsi3`

The core is built `ENABLE_DIV(0)`, so `div`/`rem` trap, and `-march=rv32ic_zmmul`
is what tells gcc that. That leaves gcc two ways to do a division: open-code a
constant divisor as a magic multiply (`mulhu`, a shift), or call a libgcc
helper. It picks the helper for a block it predicted **cold**, where it
optimizes for size, and then the link fails — nothing here links libgcc, and
this toolchain has no rv32 multilib of it to link even if it did.

That is not hypothetical. `tpu_gemm_fit` divides by `depth/2 + c_slot_row`, and
in the prefill-only `infer` image (`generate.py --phase prefill|split`, so
`INFER_DECODE=0`) 6 of the 7 `/68` sites open-code and the 7th — the
`head_argmax` after the last prefill pass, which gcc reads as cold because
nothing follows it — becomes `undefined reference to __udivsi3`. `-Os` makes it
10 of 10. So the fix belongs here, not at the call site: gcc's block-frequency
guess is not something a kernel can be written against.

Shift-subtract, no `div`, one bit per iteration. `__udivsi3` is 74 bytes and
`--gc-sections` drops both from every image that folds all its divisions —
`--phase both` and `--phase decode` link with neither.

## infer_config.h — the whole configuration, generated

`accel/test/export.py` writes it and `infer.c` includes it. Three things in one
file: the shape, every DRAM address, and 14 `{m0,n}` requant words per layer in
the block order `infer.c`'s enum declares (`INFER_RQ_SITES` is what catches the
two lists drifting; the old `KP`/`VP` holes are gone with the `quant4` passes
the MXU's int4 store replaced).

A requant word is `m0` in the low 12 bits and `n` above; the op computes
`clip((acc*m0 + 2**(n-1)) >> n)` — `[-8,7]` for every op but DYT, which clips
to `[-7,7]`.

**There is no checked-in default any more.** `tests/infer/generate.py` produces
the header either way:

```
python accel/test/tests/infer/generate.py --synthetic --gen 3 -n 1   # no checkpoint
python accel/test/tests/infer/generate.py --model-path model/saved/int4_d128_f512_l4.pt
```

`--synthetic` uses a hand-picked table over mixed-hash weights, which makes the
kernel a self-contained datapath regression with no model file involved. Too
small a shift and every tensor pins at the clip; too large and the model
collapses to zeros — and an all-zero golden passes against any datapath at all,
which is why the test warns when every generated token is the same.

Each shift is one bit per doubling of the contraction that feeds it. At
`d=64 / f=256 / T=64`: `Q`/`K`/`V`/`O`/`H` contract over `D`, `S` over
`head_dim`, `A` over `T`, `F` over `DFF`; `X1`/`X2` add two int4 tensors and are
bounded by 16.

`INFER_RQ_LOGIT` is the output head's shift, not per-layer. The MXU requantizes
on store, so the logits are int4 and this decides whether an argmax over them
can separate anything: too small a shift pins every logit at the clip and the
answer is token 0 every time. On the synthetic operands the raw head
accumulator spans about ±40, so a shift of 3 lands it across the grid.

A checkpoint's head accumulator is nowhere near ±40, so a real run derives the
word instead. `export.logit_rq_word` takes it from the head weights alone, no
calibration data: the residual stream reaching the head is a DyT output, so its
codes are bounded by 7, and the largest accumulator column `j` can reach is
`7 * sum_d |w[d][j]|`. Mapping that bound onto the top of the int4 grid is a
multiplier that cannot clip. On `int4_d64_f256_l4.pt` the bound is 1071 against
a measured span of ±734, so the word is `{m0=214, n=15}` — about 1/153, where
the synthetic shift of 3 clipped 78% of the logits and preserved only 68% of
the reference argmaxes.

## mock/tpu_trace.c — the host-side implementation of tpu.h's four primitives

Link this against any firmware kernel with `-DTPU_TRACE` and the host
compiler, and running the result prints the kernel's command trace instead of
executing it. The kernel source is unmodified:

```
cc -DTPU_TRACE -I.. mock/tpu_trace.c matmul.c -o matmul.trace
./matmul.trace > trace.txt
```

Format, one record per line, all hex, consumed by `accel/test/backends.py`:

```
CMD  <unit> <w0> <w1> <w2> <w3>     one 128-bit macro-op
WAIT <unit>                         a producer barrier on that unit
SRD  <addr>                         the CPU read a scratchpad word
SWR  <addr> <val>                   the CPU wrote one
```

`WAIT` has no hardware effect — it is a spin on the retired counter — but it
is recorded because cross-unit ordering is software's job, so a missing
barrier is a real firmware bug and the trace is where it shows. The ISS
executes commands in trace order, which is the strongest ordering any correct
barrier placement can produce, so a trace missing a barrier still yields
correct golden images here and diverges on the RTL. That asymmetry is
deliberate: the images stay a statement of intent, and the RTL run is what
tests ordering.

**Co-execution.** `SRD` is the one record that needs an answer: a kernel that
argmaxes a tensor (`fw/infer.c`) issues commands whose addresses depend on
what the array computed, so its trace cannot be produced by a program with no
model of the machine. So this one asks instead of modelling. After printing
`SRD` it blocks on stdin for a line of hex, and the driver on the other end of
the pipe (`ISSBackend._coexecute`) is the ISS: it executes each `CMD` as it
arrives and answers the read out of its own scratchpad.

With no driver — a plain `make trace`, stdin at EOF — a read returns 0 and
warns once. The command sequence is then still the right shape but the
numbers in it are meaningless, which is all `make trace` can give for a
kernel that branches on its own results.
</content>
