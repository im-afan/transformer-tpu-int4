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
once per output block.

`tpu_matmul`, `tpu_gemm_blocks`-equivalent chooser logic, and friends are
`always_inline`, which is load-bearing: with the shape constant at the call
site gcc folds the block arithmetic away, and the PicoRV32 runs it with no
unit busy. See CLAUDE.md's note on `always_inline` firmware primitives.

## infer.c — the model generating

The int4 adder model as inference: prefill, then decode against a KV cache.
Token ids in, token ids out, one run. `infer_block(rows, first_pos)` runs
`rows` new positions per sequence, so the row count is the only difference
between the training shape and the generative one.

**Phases, for benchmarking.** `INFER_PREFILL`/`INFER_DECODE` select which half
of a generation this image runs; both default to 1 (the whole thing, what the
accuracy path builds). `make PROG=infer PHASE=prefill|decode|both` sets them.
The counters can't be read mid-run — they reset at `G` and freeze at the
halt — so an image that runs one half IS the measurement of that half. The
decode-only image starts from the prompt's last token instead of the one the
prefill would have produced: a step's cost isn't data-dependent, so the clocks
match and the emitted tokens aren't scored.

**Batching.** `BATCH` independent sequences share every weight stream. X is
`[BATCH][rows][D]`, sequence-major, so the three projections, `Wo` and both FFN
matmuls run once over `BATCH*rows` rows. Attention stays per sequence — each
has its own KV cache and its own append into it.

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

## ffn.c / mha.c / matmul.c — datapath smoke tests

Small, fixed-shape kernels that exercise one piece of the pipeline each, with
operands supplied by `accel/tpulang/fw_vectors.py` (which runs the kernel's own
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
proof for the data window (`cd accel/tpu/tb && make fw FWPROG=spadwin`, ~1s).

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

Four problems, chosen for what they force rather than for what they compute:
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

This is the one kernel that alone wouldn't test anything, because a tiling bug
is something the ISS would reproduce as faithfully as the RTL. So
`fw_vectors.py` also carries an independent `reference_tiled` and checks the
ISS against a plain Python matmul before any vectors are written.

## memops.c — memcpy/memset for a freestanding build

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

## infer_rq.h — infer.c's default requant table

16 `{m0,n}` words per layer, in the block order `infer.c`'s enum declares. The
word is `m0` in the low 12 bits and `n` above; the op computes
`clip((acc*m0 + 2**(n-1)) >> n)` — `[-8,7]` for every op but DYT, which clips
to `[-7,7]`. `KP`/`VP` are retired holes from the `quant4` passes the MXU's
int4 store replaced.

**These are not a checkpoint's scales.** They're tuned for the synthetic
operands `accel/tpulang/fw_vectors.py` stages, so `make fw FWPROG=infer` is a
self-contained datapath regression with no model file involved. Too small and
every tensor pins at the clip; too large and the model collapses to zeros — and
an all-zero golden answer passes against any datapath at all, which is why
`fw_vectors.py`'s reference warns when the generated sequence is constant.

**Why this isn't `adder_rq.h`.** The two kernels shared one table while they
were the same model. Both are `adder_int4_wide` now (`d=128, f=512`), but
`adder.c` runs the whole `T=128` sequence and `infer.c` runs `T=64`, so `RQ_A`
(set by the contraction over keys) differs; the rest of the row is the same
arithmetic. A separate header stops a later shape change to one kernel from
silently moving the other's table. Each shift here is one bit per doubling of
the contraction that feeds it — how these carried over from the old d=64/T=32
table: Q/K/V/O/H gained one (D 64->128), S gained one (head_dim 16->32), A
gained one (T 32->64), F gained one (DFF 256->512).

A real run overrides this file wholesale:

```
python accel/tpulang/infer_export.py --model-path model/saved/int4_d128_f512_l4.pt
```

which writes the same `ADDER_RQ_INIT` macro from the checkpoint's learned
`ActQuant` scales and `Int4Linear` weight scales, and builds against it with
`-DADDER_RQ_H`.
</content>
