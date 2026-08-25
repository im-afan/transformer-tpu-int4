# Firmware (PicoRV32 command producer)

C kernels for the CPU in [`../rtl/cpu_subsys.sv`](../rtl/cpu_subsys.sv) — the **only**
producer of the 128-bit macro-ops the MXU/VPU/DMA queues consume.

| File | Contents |
| --- | --- |
| `tpu.h` | The MMIO aperture and one builder per command. No abstraction — fields are packed exactly as `cmd_mxu.sv` / `cmd_vpu.sv` / `cmd_dma.sv` decode them |
| `tpulib.h` | **The primitives**: matmul, elementwise, transpose and block moves at any size, over operands in either memory |
| `matmul.c` | `C[8x16] = A[8x32] @ W[32x16]`, DMA in, one `matmul_t`, DMA out |
| `matmul_loop.c` | The same product with the 4x2 tile grid walked in C — 8 single-tile dispatches, `.acc` across the contraction |
| `ffn.c` | The feed-forward block, `X@W1 -> relu -> requant -> @W2`. First kernel to issue a VPU command |
| `mha.c` | One head of ReLU attention: all three DMA modes including the transposing spill, plus the `quant4` pack |
| `tiled.c` | `tpulib.h` past the scratchpad: three DRAM-to-DRAM problems sized so the row, column and contraction loops all have to run |
| `spadwin.c` | The **scratchpad window** on its own: the CPU reads a tensor back, writes one, and issues a DMA at an address it computed |
| `infer.c` | **The whole model, generating**: prefill then decode against a KV cache, argmax and embedding on the device. `d=128`, `f=512`, `T=64` |
| `adder.c` | The whole model in the **training** shape — every position at once, teacher-forced. Still at the old `d=64` / `f=256` / `T=32` (see below) |
| `adder_rq.h`, `infer_rq.h` | 16 requant `{m0,n}` words per layer, one header per kernel. The checked-in copies are tuned for synthetic operands; a real checkpoint overrides them |
| `memops.c` | `memcpy`/`memset`, which gcc emits calls to whatever the flags say. Nothing links it today |
| `mock/tpu_trace.c` | The host-side `tpu_push`/`tpu_wait`, so `-DTPU_TRACE` turns any kernel into its own trace producer |
| `start.S`, `link.ld` | Reset entry (`gp`/`sp`, zero `.bss`, `main`, raise `done`) and the 16 KB firmware RAM at address 0 |
| `bin2hex.py` | `.bin` -> one 32-bit word per line, for `'I'` and for `$readmemh` |

**Two layers, and which one a kernel uses is a real choice.** `matmul.c`,
`matmul_loop.c`, `ffn.c`, `mha.c` and `spadwin.c` are written straight against `tpu.h` —
they are ISA tests, and the point is that every field is visible. `adder.c`, `infer.c` and
`tiled.c` are written against `tpulib.h`, because they are *programs*.

## ⚠️ `adder.c` has not been migrated

`infer.c`, `../../tpulang/fw_vectors.py`, `../../tpulang/adder_export.py` and
`../host/run_adder.py` all moved to `adder_int4_wide` (`d=128`, `f=512`). **`adder.c` did
not** — it is still `T=32`, `D=64`, `DFF=256`, with the old DRAM map
(`DR_LAYER0 = 0x02000`, stride `0x06000`) against the exporters' `0x20000` / `0x18000`.

Consequences until it is migrated:

- `make fw FWPROG=adder` and `adder_export.py` do not line up — the golden operands are
  staged at the wide map.
- `infer.c` is the migrated teacher-forcing-free path, and is what to use.
- `infer.c` is currently compiled with `LAYERS 2`, not the config's 4. `fw_vectors.py`'s
  `IN_LAYERS` is 4. Set it back before trusting an accuracy number.

## Build

Needs a bare-metal RISC-V gcc — `brew install riscv64-elf-gcc`, or the xPack
`riscv-none-elf-gcc`. The Makefile autodetects the prefix; override with `CROSS=`.

```bash
make -C accel/tpu/fw                       # -> matmul.hex (PROG defaults to matmul)
make -C accel/tpu/fw PROG=infer            # infer.hex
make -C accel/tpu/fw PROG=infer dis        # disassembly
make -C accel/tpu/fw PROG=infer size
make -C accel/tpu/fw PROG=infer trace      # native command trace, host cc only
make -C accel/tpu/fw PROG=infer run PORT=COM5
```

`-march=rv32ic_zmmul -mabi=ilp32` matches how `cpu_subsys.sv` parameterizes the core:
compressed on, fast multiplier on, **divider off** — a `div` or `rem` traps as an illegal
instruction, so plain `rv32imc` is wrong. Nothing is linked (`-nostdlib`, no libgcc). With
a gcc older than 12 (no `zmmul`), use `-march=rv32ic` and keep multiplication out too.

### Build-time knobs

| | |
| --- | --- |
| `M=`, `KTILES=`, `NTILES=` | `matmul.c` / `matmul_loop.c` problem shape |
| `GEN=n` | `infer.c`: tokens to generate (default `T - PROMPT` = 32) |
| `BATCH=n` | `infer.c`: independent sequences sharing one weight stream. Costs 48 KB of DRAM per sequence for the cache; the `DR_END` assert is the ceiling |
| `BLOCK=n` | `infer.c`: prefill rows per sequence per pass (default 32, the array's dispatch limit). Lower it to shrink every activation buffer, at one weight stream per extra pass — this is what makes `BATCH>1` fit |
| `PHASE=prefill\|decode\|both` | `infer.c`: which half of a generation the image runs. Both is the default and is what the accuracy path builds; the other two are benchmarks, because the counters reset at `'G'` and cannot be read mid-run |
| `RQ=<path>` | Requant table header, overriding the checked-in one |
| `TPU_WGT_PREFETCH=0` | Compile the weight prefetch out (the A/B) |

### Sizes

Against 16 KB of firmware RAM, all text, no `.data`:

| kernel | bytes |
| --- | --- |
| `matmul` | 240 |
| `matmul_loop` | 344 |
| `ffn` | 408 |
| `mha` | 648 |
| `tiled` | 1736 |
| `adder` (narrow) | 1992 |
| `infer` (wide, prefetch on) | **15 772** |

**`infer` is close to the ceiling**: 15 772 of 16 384 bytes, and the stack grows down
from the top of the same RAM, so there are ~600 bytes of headroom. Roughly 1.2 KB of that
is the weight prefetch and its pipeline (`-DTPU_WGT_PREFETCH=0` builds at 11 782 on the
same toolchain); the rest is `infer_block` inlined twice, once for the prefill's M=32 and
once for a decode step's M=1. If it stops fitting, the knob is `infer_block`'s
`always_inline`, and giving it up costs runtime rather than correctness.

Clang builds this code about 4.6x larger than gcc (`zig cc -target
riscv32-freestanding-none`: `adder` at 9120 bytes against gcc's 1992), so a clang build of
`infer` will not fit.

## Simulate it

Two testbenches run any kernel through the whole core against the same ISS golden vectors.
They differ only in how the bytes get in.

`../tb/fw_matmul_tb.sv` loads the image through `FW_INIT` (`$readmemh`, no UART), seeds
DRAM by backdoor, releases the CPU and checks the result. This is the fast one:

```bash
cd accel/tpu/tb
make fw                     # matmul.hex
make fw FWPROG=matmul_loop  # the software tile loop, same expectations
make fw FWPROG=tiled        # the block loops, checked against Python  (~20 s)
make fw FWPROG=spadwin      # the CPU's scratchpad window              (~1 s)
make fw FWPROG=infer GEN=3  # prefill + 2 decode steps
make fw FWPROG=infer        # the whole generation, ~16 M clocks
make fwtime FWPROG=<kernel> # the same run + a per-command timeline
```

The golden vectors come from the kernel's own native build, and `fw_vectors.py` **runs**
that build (`-x`) rather than reading a captured trace — see "Tracing a kernel that
branches" below.

`../tb/fw_uart_tb.sv` touches nothing but the two serial pins: `'I'` loads the firmware,
`'W'` writes the operands including the weights, `'G'` starts the core, `'T'` reads the
counters, `'R'` reads results back. That is exactly what the board host does, so a pass
says the board path is wired end to end.

```bash
make fwuart FWPROG=ffn          # ~12 s
make fwuart FWPROG=ffn RERUN=1  # load and run twice, no reset between
```

`RERUN=1` is a real regression, not a formality: it caught `tpu_top.sv` clearing
`cpu_run` against the *previous* run's stale `cpu_done`, so the second `'G'` released the
core for one cycle and re-reset it.

`fwuart` costs ~6x what `fw` does, because at the default `FWUART_CPB=16` a byte is 160
core clocks. `FWUART_CPB=8` halves it; below 8 the receiver's mid-bit sample stops being
mid-bit.

**A long Icarus run prints nothing until it halts**, which makes "slow" and "deadlocked"
look identical from outside. Redirect the log to a file rather than piping through
`tail`/`head`.

---

## `tpulib.h` — the primitives

`tpu.h` packs one macro-op, and every limit in it is a hardware field width:

| limit | where it comes from |
| --- | --- |
| `t_len <= 32` | `mxu.sv`'s result buffer is `MAX_TOKENS` deep — one dispatch covers 32 rows of A |
| `k_tiles`, `n_tiles <= 255` | 8-bit fields in `MXU_GEOM` |
| `dma_len <= 65535` | 16 bits, so one transfer is under 64 KB |
| `vpu_vlen <= 1023` | 10 bits, and `quant4` additionally needs it even |
| 64 KB of scratchpad | shared by every resident tensor |

`tpulib.h` is the loop that hides them:

| | |
| --- | --- |
| `tpu_matmul(&gemm, &arena)` | `C[m][n] = A[m][k] @ W[k][n]`, any size, each operand in either memory. Blocks in rows, columns and the contraction; stages what is in DRAM; requants on store when it can and through the VPU when a split contraction stopped it |
| `tpu_add_narrow`, `tpu_relu_narrow`, `tpu_pack4` | the widening/narrowing VPU pairs, chunked at `vlen`, streaming through the arena when an operand is in DRAM |
| `tpu_transpose_int8` | `dst[c][r] = src[r][c]` with the source resident |
| `tpu_transpose_dram_int8` | the same with **both** sides in DRAM, staged `rows_per_pass` rows at a time |
| `tpu_move`, `tpu_move2d` | a linear or strided block between the two memories |
| `tpu_buf`, `tpu_buf_off` | an address plus which memory it is in; `_off` slices without losing either |
| `tpu_arena` | a bump allocator over one scratchpad region. Each primitive takes what it needs and rewinds, so the high-water mark is the largest primitive, not their sum |

Every primitive is **self-fencing** — it returns only once its commands have retired — so
composing two is always safe.

What it does **not** hide is where a tensor lives, because that is what costs clocks. A
scratchpad-resident operand is used in place (the MXU addresses a sub-block natively
through the three `GEOM` strides); a DRAM-resident one is staged a block at a time.

### Specialization is load-bearing

Everything a primitive computes before its first push is exposed clock for clock — the
caller has just fenced — and the PicoRV32 runs 5–9 clocks per instruction with no cache.
So ~200 instructions of block arithmetic costs more than the array spends on the dispatch
they produce.

Measured on `adder.c`: routing its matmuls through a helper with runtime `m`/`k`/`n` cost
**597 936 clocks** against **453 778** for the same commands from constant shapes — and
the image was *larger*, 6420 bytes against 1992. `always_inline` on `tpu_matmul`,
`tpu_gemm_blocks` and `tpu_gemm_arena_bytes` is what lets gcc fold the chooser, the block
loops and every staging branch at a call site whose dimensions are `#define`s.

**If you add a firmware abstraction, check the disassembly, not the command count.**

### The weight prefetch — the one place two units run at once

Every other primitive fences. `tpu_gemm.prefetch` double-buffers the staged weight so
block *n+1*'s fill streams **under** block *n*'s dispatch, turning the exposed cost per
block from `DMA + MXU` into `max(DMA, MXU)`.

```c
tpu_matmul(&(const tpu_gemm){ ..., .rq_word = rq[RQ_H], .prefetch = 1 }, &arena);
```

It is a request, not a mode. `tpu_matmul` engages it only when all of these hold:

| | |
| --- | --- |
| the weight is DRAM-resident | there is nothing to stage otherwise |
| the activation is **not** | the act staging buffer is single, so refilling it every block forces exactly the MXU fence the prefetch exists to hide behind |
| two buffers fit the arena | `tpu_gemm_prefetch_depth` finds the deepest block leaving room for two, plus the int32 partials |
| there is more than one block | a weight that fits the arena whole has no successor to fetch |

**It splits the contraction, never the columns.** `tpu_move2d` stages a block in *one*
transfer only while the block spans a whole row of the tensor. Split the columns and it
degenerates to one DMA command per contraction row — 512 commands of 32 bytes for
`infer.c`'s `[512][128]` `W2`.

The contraction split costs the int32 partials: the array can only narrow on a store that
saw the whole depth, so one VPU pass has to bring them down. That is `rows*cols*4` bytes
of arena and `rows*cols` VPU elements, and it is why this is opt-in per call site.

Where it fires in `infer.c` today:

| | asked | gets | why |
| --- | --- | --- | --- |
| `W2` `[512][128]` | yes | 16 x `[32][128]`, 2 KB each, double-buffered | partials are `BLOCK*D*4` = 16 KB, which the arena has |
| `W1` `[128][512]` | yes | no — falls back to a 2-way **column** split | partials would be `BLOCK*DFF*4` = **64 KB**, more than the whole scratchpad |
| `Wq/Wk/Wv/Wo` `[128][128]` | no | one 8 KB dispatch, fits the arena whole | nothing to prefetch |

W1's fallback is fine: its column block is still 128 bytes per contraction row, so the
fill streams at full rate. The pathological 32-byte rows belong to `W2`, which is the one
that gets the contraction split instead.

Caveat: at `M = 1` — every decode step, which is most of a run — the array work per block
is a few hundred clocks against a 2 KB fill, so there is little to hide and the int32 pass
may cost more than the overlap saves. `make fwtime FWPROG=infer GEN=3` and read `idlec`.

### `tiled.c` — the paths the model kernels do not take

Three DRAM-to-DRAM problems with a deliberately undersized arena, so the row loop, the
column loop, the contraction split and the DRAM-streaming elementwise path all have to run.

```bash
cd accel/tpu/tb && make fw FWPROG=tiled     # 80 404 clocks, 525 474 checks, 0 errors
```

It is also one of two kernels with an **independent** reference. The golden DRAM image is
whatever `iss.py` computed, which checks the RTL against the ISS and nothing else — the
right check for a kernel driving the datapath, and not enough for one driving a *loop*,
because a mis-tiled matmul is something the ISS reproduces as faithfully as the hardware.
So `fw_vectors.py` carries `reference_tiled`, a plain Python matmul, and checks the ISS
against it before any vector file is written.

---

## `infer.c` — the model generating

`adder_int4_wide` — `d=128`, `f=512`, four heads of 32, int4 weights and activations, no
bias — over a `T=64` sequence: a 32-token prompt and the answer after it.

```bash
cd accel/tpu/tb && make fw FWPROG=infer GEN=3    # 3 tokens, for iterating
python accel/tpulang/infer_export.py -n 256      # accuracy, generating, on the ISS
python accel/tpu/host/run_adder.py -p COM5 -n 64 # ...on the board
```

```
prefill   the 32 prompt tokens, in BLOCK = 32 row passes — one pass at this prompt
          length. Their K and V land in the cache; the last row's logits are the
          first answer digit.
decode    31 steps of M=1. The new token's K and V are appended and attention
          contracts against the whole cache.
```

**Both halves are the same code.** `infer_block(rows, first_pos)` runs `rows` new rows
starting at `first_pos`; the prefill calls it as `(32, 0)` and a decode step as `(1, t)`,
and `always_inline` specializes the two. The cache, the mask and `tpulib.h`'s block loops
do not care how many rows arrive at once, so **M is the only difference between the
training shape and the generation shape.**

That is also what lets the prefill be chunked when the prompt is longer than a pass: pass
*b* runs all four layers before *b+1* starts, so *b+1*'s layer-L attention finds *b*'s
layer-L K and V already in the cache.

### Batching

`BATCH` independent sequences share every weight stream. X is `[BATCH][rows][D]` —
sequence-major — so the three projections, `Wo` and both FFN matmuls run once over
`BATCH*rows` rows and each weight is staged once for all of them. Attention stays per
sequence, because each has its own cache.

That is the whole point at decode: a step is one row of arithmetic against ~390 KB of
weights, so `BATCH` rows cost the same DMA as one. What it costs is DRAM — 48 KB of cache
per sequence at four layers — and `DR_END`'s assert says whether a given `BATCH` still
fits under the weights.

### Where tensors live

**Every tensor's home is DRAM**, activations included, and the kernel is correct reading
and writing them there and nowhere else. The DRAM map is a *computed chain* off the shape
(`DR_ALIGN(previous + its size)`), so changing `T`, `D`, `DFF`, `HEADS`, `LAYERS` or
`BATCH` re-lays it out and one `_Static_assert` says whether it still fits.

The 64 KB scratchpad holds two things:

| | |
| --- | --- |
| the **mailbox**, 320 B | the only fixed allocation. The head's logits and the token sequence — what the CPU has to read and write, since it has no path to DRAM |
| the **arena** | staging that the primitives allocate and rewind, sized by one block of one matmul and never by the model |

On top of that, a compile-time **promotion cascade** gives a scratchpad copy to the tensors
that save the most DMA per byte of arena, stopping as soon as a promotion would leave less
than one dense `[D][D]` weight block plus the elementwise chunk buffers. At `d=128` /
`f=512` / `BLOCK=32` all eight are promoted and 22 208 bytes of arena are left.

**Nothing here is a correctness condition.** Every tensor is a `tpu_buf`, every primitive
takes one, and dropping every promotion computes the same bytes more slowly — which is why
nothing asserts that an activation fits on chip.

The activation working set is three rotating temporaries plus the residual stream, because
an intermediate dies the moment its consumer has read it:

| | |
| --- | --- |
| `DR_X` | the residual stream, live across all layers |
| `DR_Q` | Q, live from the projection to the last head's scores |
| `DR_TMP_A` | K_new -> A -> X+O -> F |
| `DR_TMP_B` | V_new -> O -> X1 |

`S` and `H` are the other two, and both elementwise chains run **in place** — safe by
construction, because a VPU pair reads its chunk into the arena's int32 temp before it
writes the same chunk's worth of destination, and the two commands are in order on one unit.

### The KV cache

12 KB per layer per sequence, in DRAM with everything else:

| | |
| --- | --- |
| `DR_K_CACHE[l]` | `[D][T]` int8 — K **transposed**, 8192 B |
| `DR_V_CACHE[l]` | `[T][D]` int4 — V packed as a weight operand, 4096 B |
| `DR_KT` | the layer's K cache packed to int4, 4096 B — rebuilt each layer-step |

**Each half is in the orientation its matmul wants, because what a cache costs is the
append.**

- **V is free.** `P @ V` contracts over keys, so its weight is `V[s][h]` — row `s`
  contiguous over `h`, exactly how V leaves its projection. The append *is* the pack: one
  `quant4`.
- **K is a column.** `Q @ K^T`'s weight is `K^T[h][s]` — row `h` contiguous over `s` — so
  the cache is column-major and appending writes one byte into each of D rows. That
  scatter is the transposing DMA, a single `spill.t` (`tdrow = T`) straight out of the
  projection, one command whatever M is.

K stays int8 and is re-`quant4`ed whole once per layer-step. That looks wasteful — 8192
elements packed to use at most `t+1` columns — but it is 8 KB of DMA against the ~96 KB of
weights the same layer-step streams, and the alternative does not exist: the nibble for
`(d, t)` sits in the middle of a byte and no op writes half a byte.

**Nothing is zeroed and nothing needs to be.** Cache columns past the current position
hold whatever the last problem left there and reach S as garbage — but S is int4 and the
mask is `-8`, so a masked score is at most `-1` and ReLU takes it to exactly zero. The
mask that makes attention causal is what makes an uninitialized cache safe. That is why
`infer_export.py` and `run_adder.py` run every problem through one instance rather than a
fresh one: the test, not a shortcut.

### The argmax and the gather are on the device

`cpu_subsys.sv` decodes `0x9xxx_xxxx` onto the scratchpad's S port.

| | |
| --- | --- |
| argmax | the head writes 13 int32 logits to the scratchpad **mailbox**; the CPU reads them with `tpu_spad_ld` and compares |
| gather | the embedding table is a DRAM tensor and a DMA takes a *computed* address, so `DR_EMB + tok*D` is the gather. The index never leaves the CPU |

So the host stages the weights, the mask, the head, the embedding table and the prompt's
token ids, presses `'G'` once, and reads a finished sequence out of `DR_TOKENS`. Nothing
round-trips per token. What the host still does is tokenize.

[`spadwin.c`](spadwin.c) is that window on its own, in 304 bytes and 4 commands — run it
first if anything here misbehaves.

### Tracing a kernel that branches on its own results

The command stream depends on what the array computed: the token the argmax picked lands
in the *address* of the next gather. So a producer with no model of the machine cannot
emit this kernel's trace, and `make trace PROG=infer` gives a structurally-right,
numerically-meaningless one (a read with no driver attached returns 0).

The real trace comes from co-execution: `fw_vectors.py -x` runs the `-DTPU_TRACE` binary
as a **co-process**, executing each command on `iss.py` as it arrives and answering the
kernel's scratchpad reads out of the model's own memory. That is what `make fw
FWPROG=infer` does, and it makes the golden command stream a real forward pass's — so if
the RTL picks a different token anywhere, the run fails at *that command* rather than
merely producing a different answer.

`fw_vectors.py` also carries `reference_infer`, which recomputes the whole model from
scratch at every step, in integer numpy, with **no cache at all**, and checks every
generated token and every logit. A cache column written at the wrong offset, the mask row
of the wrong position, an argmax over the wrong words — the ISS reproduces all of those
exactly as faithfully as the hardware would.

### Measured

From `accel/tpu/fw/perf_notes.md`, on the board at 12 MHz, `L=4, d=128, f=512`,
32 generated tokens:

```
per token   69.4 ms (832 837 core clocks)
occupancy   mxu 45.5%  mload 16.0%  vpu 2.8%  dma 54.1%  idlec 17.5%  ovlap 19.9%

                          clocks       ms     mxu   mload    vpu     dma   idlec   ovlap
prefill (32 rows)        1768763  147.397   48.6%    7.5%  10.2%   30.9%   16.5%    6.2%
one decode step (M=1)     802645   66.887   45.2%   16.6%   2.3%   55.8%   17.5%   20.9%
whole generation        26650766 2220.897   45.5%   16.0%   2.8%   54.1%   17.5%   19.9%
```

- A prompt token costs 55 274 clocks in the prefill against 802 645 in a decode step
  (14.5x) — decode is DMA-bound on the weight stream, prefill is not.
- Roofline for the prefill: 394 KB of weights at 1 byte/clock plus 32 KB of KV is ~426 000
  clocks of memory, against ~390 000 clocks of array work at 64 int4 ops/clock. So even
  with no overlap the prefill should be ~800 k clocks; it is 1.77 M.
- The accuracy in that file is against the **untrained** dummy checkpoint and is noise.

---

## `adder.c` — the training shape

One forward pass over the whole sequence: every position at once, against a causal mask,
teacher-forced. Same layers as `infer.c`, no cache.

**Still at `d=64`, `f=256`, `T=32`, four layers** — see the warning at the top. What it
measured at that shape:

```
534 commands, 1992 bytes of firmware, 453 778 clocks, 526 959 checks, 0 errors
counters: run=453778 mxu=206361 mload=33440 vpu=84352 dma=131200
          idlec=31864 qfull=0 ovlap=0
```

`idlec` — clocks with no unit busy at all — was **7.0%**, which is what the CPU costs as a
command producer on a real workload. `ovlap` was 0: it fences after every cross-unit
dependency.

Where those clocks went (`make fwtime FWPROG=adder`):

| phase | MXU | VPU | DMA | CPU | share |
| --- | ---: | ---: | ---: | ---: | ---: |
| weight fills | — | — | 98 400 | 6 878 | **23.2%** |
| FFN `W1` | 63 492 | — | — | 1 084 | 14.2% |
| FFN `W2` | 60 420 | — | — | 1 080 | 13.6% |
| projections | 47 628 | — | — | 3 056 | 11.2% |
| FFN `relu -> requant` | — | 24 704 | — | 732 | 5.6% |
| K transpose (spill + fill) | — | — | 25 104 | 424 | 5.6% |
| attention mask | — | 16 448 | — | 2 384 | 4.2% |
| `Wo` | 15 876 | — | — | 1 024 | 3.7% |
| attention `relu` | — | 12 352 | — | 2 768 | 3.3% |
| attention `S` | 9 488 | — | — | 4 096 | 3.0% |
| attention `A` | 8 464 | — | — | 3 712 | 2.7% |
| DyT + residuals | — | 24 672 | — | 1 984 | 6.0% |
| the rest | 993 | 6 176 | 7 696 | 2 486 | 3.9% |

Two things the totals hide:

- **Moving weights costs more than either FFN matmul.** 98 400 DMA clocks, re-fetched
  every forward because the arena cannot hold a layer. Nothing but the arena depends on a
  weight fill, which is what the prefetch above went after.
- **The CPU's 31 864 clocks are the fences, not the commands.** A command pushed on top of
  a busy unit costs **13.3** exposed clocks; one that follows a `tpu_wait` costs **183.6**,
  because `tpu_wait` polls a counter over AXI4-Lite and the CPU is mid-poll when the unit
  goes idle. 143 barriers, 82% of the CPU total, and 48 of them are the three-way fence in
  the per-head attention loop.

**What the library cost.** The hand-written version — every address a constant, no block
loops — was 518 commands, 1544 bytes and 439 917 clocks. Going through `tpulib.h` cost
+3.2% of the run and +448 bytes, and the units do byte-identical work: the whole delta is
16 more commands (three separate `Wq`/`Wk`/`Wv` fills instead of one fused block) and 36
more barriers.

### Layout facts, both the opposite of the retired ternary kernel's

Weights went row-major, which inverted the attention transpose:

- **`Q @ K^T` needs the transpose.** Its weight is `K^T[h][s]`, so row `h` must be
  contiguous over `s` — K column-major, and K leaves its projection row-major. It goes out
  through the DMA's `.t` spill and back in as **bytes**, because a packed int4 nibble is
  half of one; then `quant4` packs it.
- **`P @ V` does not.** It contracts over keys, so its weight is `V[s][h]` — exactly how V
  left its projection. Free.

Other layout notes:

| | |
| --- | --- |
| the weights | six dense row-major blocks per layer: `Wq`, `Wk`, `Wv`, `Wo`, then `W1` and `W2`. `Wq\|Wk\|Wv` are **not** one fused `[D][3D]` block — that was free when the kernel staged its own weights, but a column slice of a fused block is strided, so `tpu_matmul` would fetch it a row at a time |
| the mask | int8 `0`/`-8`. S is already int4, so `S-8 <= -1` whatever the scale is, and ReLU takes a masked entry to **exactly** zero |
| the head | 13 logits padded to a whole 16-wide tile, so the second output tile cannot land on the next token's row. Its destination is DRAM, so `tpu_matmul` stages the int32 block and spills it. Read 13 of every 16 words back |

## The requant table is a compile-time input

The `{m0,n}` word is a literal in the macro-op, so there is no path by which the device
could fetch it from memory — the 16 words per layer have to be in the image.

- `adder_rq.h` / `infer_rq.h` are the checked-in defaults, tuned for the **synthetic**
  operands `../../tpulang/fw_vectors.py` stages. That makes `make fw FWPROG=<kernel>` a
  self-contained datapath regression with no `.pt` involved.
- Tuning them matters: too small and every tensor pins at the clip; too large and the model
  collapses to zeros — and a golden answer of all zeros passes against any datapath at all,
  which is why `fw_vectors.py` warns when the generated sequence is constant.
- Each shift is one bit per doubling of the contraction that feeds it.
- The two kernels keep **separate** headers because they are not the same sequence length,
  and `RQ_A`'s shift is set by the contraction over keys.
- A real checkpoint overrides the file wholesale:

```bash
python accel/tpulang/infer_export.py --model-path model/saved/int4_d128_f512_l4.pt
python accel/tpulang/adder_export.py --dump-rq -n 0     # just the 16 words per layer
```

Both derive the table from the model's learned `ActQuant` scales and `Int4Linear` weight
scales, write the same `ADDER_RQ_INIT` macro, and build against it with `-DADDER_RQ_H`.

## Three things that are software's problem now

- **Cross-unit ordering.** Each unit has its own queue, so a `matmul` will start on top of
  a DMA that has not finished. `tpu_wait(unit)` — retired caught up with issued — is the
  fence.
- **Queue-ordered geometry.** `MXU_GEOM` sticks until the next one, but only within the
  MXU's own command stream, so it cannot be corrupted by another unit or an earlier
  program the way the old `cfg` registers could.
- **The scratchpad window is unsynchronized.** `tpu_spad_ld` / `tpu_spad_st` bypass the
  queues entirely — they are loads and stores — so a read of a tensor a queued command has
  not written yet is simply stale, and needs the same `tpu_wait` a dependent command would.
  Two more rules from `cpu_subsys.sv`: 32-bit accesses only (the S port has no byte
  strobes, so an `sb` writes its three neighbours) and 4-byte aligned.

Flow control is *not* software's problem: a full queue withholds the write response on the
fourth word and the CPU stalls inside the store.
