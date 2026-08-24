# Firmware (PicoRV32 command producer)

C kernels for the CPU in [`../rtl/cpu_subsys.sv`](../rtl/cpu_subsys.sv) — the
**only** producer of the 128-bit macro-ops the MXU/VPU/DMA queues consume
(`../docs/picorv32_migration.md` §4) — the scalar unit and its `.tpu` programs
were deleted with phase 5.

| File | Contents |
| --- | --- |
| `tpu.h` | the MMIO aperture and one builder per command. No abstraction — the fields are packed exactly as `cmd_mxu.sv` / `cmd_vpu.sv` / `cmd_dma.sv` decode them |
| `tpulib.h` | **the primitives**: matmul, elementwise, transpose and block moves at any size, over operands in either memory. The block loops that hide `tpu.h`'s field widths |
| `matmul.c` | `C[8x16] = A[8x32] @ W[32x16]`, DMA in, one `matmul_t`, DMA out |
| `matmul_loop.c` | the same product with the 4x2 tile grid walked in C — 8 single-tile dispatches, `.acc` across the contraction — instead of by the array |
| `ffn.c` | the feed-forward block, `X@W1 → relu → requant → @W2`. The first kernel to issue a **VPU** command |
| `mha.c` | one head of ReLU attention: all three DMA modes including the transposing spill, plus the `quant4` pack that turns an activation into a weight operand |
| `tiled.c` | `tpulib.h` past the scratchpad: three DRAM-to-DRAM problems sized so the row, column and contraction loops all have to run |
| `spadwin.c` | the **scratchpad window** on its own: the CPU reads a tensor back, writes one, and issues a DMA at an address it computed from what it read |
| `adder.c` | **the whole shipped model** — four transformer layers and the output head, 534 commands, one run |
| `infer.c` | the same model as **inference**: prefill, then decode against a statically allocated KV cache, argmax and embedding on the device. Token ids in, token ids out, 4606 commands, one run |
| `adder_rq.h` | the 16 requant `{m0,n}` words per layer, shared by `adder.c` and `infer.c`. The checked-in copy is tuned for the synthetic operands; a real checkpoint overrides it (below) |
| `memops.c` | `memcpy`/`memset`, which gcc emits calls to whatever the flags say. Nothing links it today — see its header |
| `mock/tpu_trace.c` | the host-side `tpu_push`/`tpu_wait`, so `-DTPU_TRACE` turns any kernel here into its own command-trace producer |
| `start.S` | reset entry: `gp`/`sp`, zero `.bss`, `main`, then raise `done` |
| `link.ld` | the 16 KB firmware RAM at address 0 |
| `bin2hex.py` | `.bin` → one 32-bit word per line, for `'I'` and for `$readmemh` |

Two layers, and which one a kernel is written against is a real choice.
`matmul.c`, `matmul_loop.c`, `ffn.c`, `mha.c` and `spadwin.c` are written
straight against `tpu.h`: they are ISA tests, and the point of them is that
every field is visible. `adder.c`, `infer.c` and `tiled.c` are written against
`tpulib.h`, because they are *programs* — what they need visible is the model,
not the field widths.

## Build

Needs a bare-metal RISC-V gcc — `brew install riscv64-elf-gcc` (what these were
built with: 16.2.0, binutils 2.47), or the xPack `riscv-none-elf-gcc`. The
Makefile autodetects the prefix; override with `CROSS=`. Nothing else: the
newlib the formula installs is never linked.

Current sizes, all text, no `.data`/`.bss`, against 16 KB of firmware RAM:
`matmul` 240 bytes, `matmul_loop` 344, `ffn` 408, `mha` 648, `tiled` 1736,
`adder` 1992. The whole four-layer model is 2 KB of RISC-V because the per-layer
body is a loop over a base register, not unrolled — depth costs no instructions
at all.

`spadwin` and `infer` have only been built with **clang** (`zig cc
-target riscv32-freestanding-none`, on a machine with no riscv gcc): 304 bytes
and 15088. Clang is about 4.6x gcc on this code — it builds `adder` at 9120
against gcc's 1992 — so expect roughly 3.5 KB from the gcc above for `infer`,
which carries two specialized copies of the layer body (prefill's M=15 and
decode's M=1). It fits either way; if a future toolchain disagrees, the knob is
`infer_block`'s `always_inline`, and giving it up costs runtime rather than
correctness (see `tpulib.h`'s note on why).

```bash
make -C accel/tpu/fw            # -> matmul.hex
make -C accel/tpu/fw dis        # disassembly
make -C accel/tpu/fw PROG=foo   # foo.c instead
```

`-march=rv32ic_zmmul -mabi=ilp32` matches how `cpu_subsys.sv` parameterizes the
core: compressed on, fast multiplier on, **divider off** — a `div` or `rem`
would decode as an illegal instruction and trap, so plain `rv32imc` is wrong
here. Nothing is linked (`-nostdlib`, no libgcc), so a kernel that needs `/` or
`%` needs the RTL to enable the divider first. With a gcc older than 12 (no
`zmmul`), use `-march=rv32ic` and keep multiplication out of the kernel too.

## Simulate it

Two testbenches run any kernel here through the whole core, against the same ISS
golden vectors. They differ only in how the image and the operands get in.

`../tb/fw_matmul_tb.sv` loads the image through `cpu_subsys.sv`'s `FW_INIT`
(`$readmemh`, no UART), seeds DRAM by backdoor, releases the CPU and checks the
int32 C. This is the fast one and the one to iterate on:

```bash
cd accel/tpu/tb
make fw                     # ../fw/matmul.hex
make fw FWPROG=matmul_loop  # the software tile loop, same expectations
make fw FWPROG=tiled        # tpulib.h's block loops, checked against Python
make fw FWPROG=spadwin      # the CPU's scratchpad window, ~1 s
make fw FWPROG=infer        # prefill + decode with a KV cache, 4606 commands
```

The golden vectors come out of the kernel's own native build either way, but
`fw_vectors.py` now **runs** that build (`-x`) instead of being handed a trace
captured by a shell redirect — `infer.c` reads its own logits back and puts the
token it chose into the address of a later command, so its trace only exists if
something answers those reads. Kernels that read nothing see no difference.

Both pass: `matmul` halts after 2 209 clocks (MXU 289, DMA 1 548, 372 with no
unit busy), `matmul_loop` after 2 586 (MXU 392, same DMA, 646 idle). The 103
extra MXU clocks are the int32 partials round-tripping through the scratchpad
between contraction tiles, which is exactly what the hardware tile loop exists
to delete. Breakdown in
[`../docs/picorv32_migration.md`](../docs/picorv32_migration.md) §9.6.

`make fwtime FWPROG=<kernel>` runs the same simulation and adds a per-command
timeline: where the run's clocks went, per unit and per phase, and what the CPU
cost to issue each class of command. It works on any kernel here.

`../tb/fw_uart_tb.sv` touches nothing but the two serial pins: `'I'` loads the
firmware, `'W'` writes the operands — weights included — `'G'` starts the core,
`'T'` reads the counters and `'R'` reads the results back. That is exactly the
sequence `host/run_fw_matmul.py` runs on the board, so a pass says the board path
is wired end to end and not only that the datapath computes:

```bash
make fwuart FWPROG=ffn          # ~12 s
make fwuart FWPROG=adder        # the whole model, 102 KB of weights over the wire
make fwuart FWPROG=ffn RERUN=1  # load and run a second time, no reset in between
```

`adder` passes: 1 980 firmware bytes in one `'I'`, 101 888 operand bytes in 26
`'W'` frames, 4 096 result bytes back in 2 `'R'` frames, 531 102 checks, 0 errors,
534/534 commands — and a counter block **bit-identical** to `make fw`'s
(`run=453 777 mxu=206 361 vpu=84 352 dma=131 200 idlec=31 864`). The serial path
adds nothing to the run; it only changes how the bytes arrive.

The link is the whole cost of that target: at `FWUART_CPB=16` clocks per bit a
byte is 160 core clocks, so `adder` is 17.8 M simulated clocks — **~21 minutes of
Icarus** against `make fw`'s ~3 — of which 16.3 M is serial traffic and 454 k is
the compute. `FWUART_CPB=8` halves it; below 8 the receiver's mid-bit sample
stops being mid-bit (it sits at `CPB/2` clocks past a two-flop synchroniser).
Iterate with `make fw`; run this one to prove the board path.

`RERUN=1` is the regression for restarting the core without a reset — see
[`../docs/picorv32_migration.md`](../docs/picorv32_migration.md) §9.11 for the
bug it found.

`matmul.c` and `matmul_loop.c` take their shape from the build, and either
testbench follows: `make fw M=... KTILES=... NTILES=...` (or `make fwuart ...`)
rebuilds the firmware, its native trace and the golden operands at that shape —
until recently it rebuilt only the first two and staged the *default* shape's
operands, which did not fail: the ISS read the same zeros the hardware did, so
the run passed having tested almost nothing. `make fwsweep` walks a range of
shapes and tabulates what each costs the CPU:

```bash
make -C accel/tpu/fw M=8 KTILES=16 NTILES=16   # one shape (make clean first)
cd accel/tpu/tb && make fwsweep                # the whole sweep, ~40 s
```

`matmul.c` issues 5 commands at every shape and its CPU cost is flat at ~380
clocks across a 205x range of array work; `matmul_loop.c` pays ~85 clocks per
dispatch, of which only `max(0, 85 - MXU-clocks-per-tile)` is exposed — nothing
at all once M is past ~24. Full table in §9.7.

## `tpulib.h` — the primitives

`tpu.h` packs one macro-op. Every limit that shows up in it is a hardware field
width, and a kernel written straight against it has to be shaped around all of
them at once:

| limit | where it comes from |
| --- | --- |
| `t_len <= 32` | `mxu.sv`'s result buffer is `MAX_TOKENS = 1 << TOK_W` deep — one dispatch covers 32 rows of A |
| `k_tiles`, `n_tiles` `<= 255` | 8-bit fields in `MXU_GEOM` |
| `dma_len <= 65535` | 16 bits, so one transfer is under 64 KB |
| `vpu_vlen <= 1023` | 10 bits, and `quant4` additionally needs it even |
| 64 KB of scratchpad | shared by every resident tensor |

`tpulib.h` is the loop that hides them. A primitive takes a problem of any size,
splits it into pieces the hardware will take, moves operands in and results out,
and fences between the units on the way:

| | |
| --- | --- |
| `tpu_matmul(&gemm, &arena)` | `C[m][n] = A[m][k] @ W[k][n]`, any size, each operand in either memory. Blocks in rows, columns and the contraction; stages what is in DRAM; requants on store when it can and through the VPU when a split contraction stopped it |
| `tpu_add_narrow`, `tpu_relu_narrow`, `tpu_pack4` | the widening/narrowing VPU pairs, chunked at `vlen`, streaming through the arena when an operand is in DRAM |
| `tpu_transpose_int8` | `dst[c][r] = src[r][c]`, out through DRAM and back, split by rows past 64 KB |
| `tpu_move`, `tpu_move2d` | a linear or strided block between the two memories |
| `tpu_arena` | a bump allocator over one scratchpad region. Each primitive takes what it needs and rewinds, so the high-water mark is the largest primitive, not their sum |

What it does **not** hide is where a tensor lives. A `tpu_buf` is an address
plus its memory, and the choice stays the kernel's, because it is the one that
costs clocks: a scratchpad-resident operand is used in place (the MXU addresses
a sub-block of a larger matrix natively, through the three `GEOM` strides) while
a DRAM-resident one is staged a block at a time. That is why `adder.c` can go
through this layer and keep every activation resident.

**It is written to be specialized, and that is load-bearing.** Everything a
primitive computes before its first push is exposed clock for clock — the caller
has just fenced — and the PicoRV32 runs 5-9 clocks per instruction with no
cache, so ~200 instructions of block arithmetic costs more than the array spends
on the dispatch they produce. Measured: routing `adder.c`'s matmuls through a
helper with runtime `m`/`k`/`n` cost **597 936 clocks** against 453 778 for the
same commands issued from constant shapes. `always_inline` on the three entry
points that see the shape is what lets gcc fold the chooser, the block loops and
every staging branch at a call site whose dimensions are `#define`s — and it
makes the image *smaller*, 1992 bytes against 6420. `../docs/picorv32_migration.md`
§9.10 has the breakdown.

### `tiled.c` — the paths `adder.c` does not take

`adder.c` exercises the library with everything resident and only the weights
streaming, which is the easy half: no operand is ever bigger than the arena, so
every block loop runs once. [`tiled.c`](tiled.c) is the other half — three
DRAM-to-DRAM problems with a deliberately undersized arena, so the row loop, the
column loop, the contraction split and the DRAM-streaming elementwise path all
have to run to get a right answer.

```bash
cd accel/tpu/tb && make fw FWPROG=tiled     # 80 404 clocks, 525 474 checks, 0 errors
```

It is also the one kernel with an **independent** reference. The golden DRAM
image is whatever `iss.py` computed, which checks the RTL against the ISS and
nothing else — the right check for a kernel whose job is to drive the datapath,
and not enough for one whose job is to drive a *loop*, because a mis-tiled
matmul is something the ISS reproduces as faithfully as the hardware. So
`fw_vectors.py` carries `reference_tiled`, a plain Python matmul, and checks the
ISS against it before any vector file is written.

## `adder.c` — the whole model

`model/transformer.py::adder_int4_vanilla` (`d=64`, `f=256`, `layers=4`,
`q_heads=kv_heads=4`, `head_dim=16`, `vocab=13`, int4 weights *and* int4
activations, no bias) as one program, composed out of `tpulib.h`: **534
commands, 1992 bytes of firmware, 453 778 clocks.**

```bash
cd accel/tpu/tb && make fw FWPROG=adder     # ~3 min, 526 959 checks, 0 errors
```

```
counters: run=453778 mxu=206361 mload=33440 vpu=84352 dma=131200
          idlec=31864 qfull=0 ovlap=0
command trace: 534 commands, 534 expected
```

`idlec` — clocks with no unit busy at all — is **7.0%** of the run, which is what
the CPU costs as a command producer on a real workload. `ovlap` is 0 because the
kernel fences after every cross-unit dependency; nothing is overlapped yet.

**What the library cost.** The hand-written version of this kernel — every
address and stride a constant, no block loops, no residency to resolve — was
518 commands, 1544 bytes and 439 917 clocks. Going through `tpulib.h` costs
**+3.2% of the run and +448 bytes**, and the array, the VPU and the DMA do
byte-identical work: the whole 13 861-clock delta is 16 more commands (three
separate weight fills for `Wq`/`Wk`/`Wv` instead of one fused block) and 36 more
barriers. In exchange nothing in the kernel depends on the model fitting in
64 KB any more. It is only 3% because the library folds — called through a
helper with runtime `m`/`k`/`n` the same kernel costs **597 936 clocks**; see
the `always_inline` note above and `../docs/picorv32_migration.md` §9.10.

### Where the clocks go

```bash
cd accel/tpu/tb && make fwtime FWPROG=adder     # the same run + the timeline
```

`fw_matmul_tb.sv` will dump a per-command timeline (`+CMDLOG=<path>`) and
[`../tb/cmd_timeline.py`](../tb/cmd_timeline.py) turns it into these. The two
columns it uses — clocks each unit was busy on each command, and the
no-unit-busy clocks before each push — reconstruct the perf counters exactly,
and the tool checks that on every run.

| phase | MXU | VPU | DMA | CPU | share |
| --- | ---: | ---: | ---: | ---: | ---: |
| weight fills | — | — | 98 400 | 6 878 | **23.2%** |
| FFN `W1` | 63 492 | — | — | 1 084 | 14.2% |
| FFN `W2` | 60 420 | — | — | 1 080 | 13.6% |
| projections | 47 628 | — | — | 3 056 | 11.2% |
| FFN `relu → requant` | — | 24 704 | — | 732 | 5.6% |
| K transpose (spill + fill) | — | — | 25 104 | 424 | 5.6% |
| attention mask (`+mask`, `RQ_ID`) | — | 16 448 | — | 2 384 | 4.2% |
| `Wo` | 15 876 | — | — | 1 024 | 3.7% |
| attention `relu` | — | 12 352 | — | 2 768 | 3.3% |
| attention `S` | 9 488 | — | — | 4 096 | 3.0% |
| attention `A` | 8 464 | — | — | 3 712 | 2.7% |
| DyT norm1 / norm2 / residual | — | 24 672 | — | 1 984 | 6.0% |
| the rest (packs, head, logits) | 993 | 6 176 | 7 696 | 2 486 | 3.9% |

Two things this says that the totals do not:

- **Moving weights costs more than either FFN matmul.** 98 400 DMA clocks is
  24 KB per layer at ~1 clock/byte, re-fetched every forward because one 8 KB
  arena cannot hold a layer. Nothing but the arena it lands in depends on a
  weight fill, so this is the one part of the run that double-buffering could
  hide almost entirely.
- **The CPU's 31 864 clocks are the fences, not the commands.** A command pushed
  on top of a busy unit costs **13.3** exposed clocks; a command that follows a
  `tpu_wait` costs **183.6**, because `tpu_wait` polls a counter over AXI4-Lite
  and the CPU is mid-poll when the unit goes idle. 143 barriers, 82% of the CPU
  total, and **48 of them are the three-way fence in the per-head attention
  loop**. Full breakdown in
  [`../docs/picorv32_migration.md`](../docs/picorv32_migration.md) §9.9-§9.10.

Two layout facts are the whole design, and both are the *opposite* of the
retired ternary kernel's, because weights went row-major:

- **`Q @ K^T` needs the transpose.** Its weight is `K^T[h][s]`, so row `h` must
  be contiguous over `s` — that is K column-major, and K leaves its projection
  row-major. It goes out through the DMA's `.t` spill and back in, as **bytes**,
  because a packed int4 nibble is half of one; then `quant4` packs it.
- **`P @ V` does not.** It contracts over keys, so its weight is `V[s][h]`, row
  `s` contiguous over `h` — exactly how V left its projection. Free.

Everything else worth knowing:

| | |
| --- | --- |
| scratchpad | one 8 KB arena at `0x0000` — the weight block, the elementwise int32 temp and the head's logits all stage through it — with every activation resident above it; top byte used is `0xC7FF` of 64 KB |
| DRAM | 106 KB — `X0`, the mask, `W_fc`, the logits, and `0x2000 + L*0x6000` per layer |
| the weights | six dense row-major blocks per layer: `Wq`, `Wk`, `Wv`, `Wo` at `+0x0000`/`0x0800`/`0x1000`/`0x1800`, then `W1` and `W2`. `Wq|Wk|Wv` used to be one fused `[D][3D]` block, which was free when the kernel staged its own weights and is not now — a column slice of a fused block is strided, so `tpu_matmul` would fetch it a row at a time |
| the mask | int8 `0`/`-8`. S is already int4, so `S-8 <= -1` whatever `s_s` is and ReLU takes a masked entry to **exactly** zero |
| the head | 13 logits padded to a whole 16-wide tile, so the second output tile cannot land on the next token's row. Its destination is DRAM, so `tpu_matmul` stages the int32 block and spills it — the kernel issues no explicit spill at all. Read 13 of every 16 int32 words back |
| the host | the token embedding (no gather in the ISA) and the argmax (nothing returns an index). Nothing else |

**The requant table is a compile-time input.** The `{m0,n}` word is a literal in
the macro-op, so unlike the retired scalar ISA there is no path by which the
device could fetch it from memory — the 16 words per layer have to be in the
image. `adder_rq.h` is the checked-in default and is tuned for the *synthetic*
operands `../../tpulang/fw_vectors.py` stages, so `make fw FWPROG=adder` is a
self-contained datapath regression with no `.pt` involved. A real checkpoint
goes through [`../../tpulang/adder_export.py`](../../tpulang/adder_export.py),
which derives the table from the model's learned `ActQuant` scales, compiles the
same `adder.c` against it, and runs the trace on `iss.py`:

```bash
python accel/tpulang/adder_export.py -n 256          # accuracy on the addition task
python accel/tpulang/adder_export.py --dump-rq -n 0  # just the 16 words per layer
```

Measured on `model/saved/int4_d64_f256_l4.pt`, 256 problems: **100.00%
exact-sequence, 100.00% token** — identical to the PyTorch QAT model it came
from, with 0 of 4352 scored argmax positions differing.

## `infer.c` — the same model, generating

`adder.c` is one T=32 forward pass: every position at once, against a causal
mask, over a sequence somebody already knows. That is the *training* shape.
Generating an answer that way costs a whole 32-token forward per token emitted
and recomputes K and V for the prefix every time.

[`infer.c`](infer.c) is the generative shape — **prefill, then decode against a
statically allocated KV cache** — and closes the loop on the device:

```bash
cd accel/tpu/tb && make fw FWPROG=infer          # 4606 commands
cd accel/tpu/tb && make fw FWPROG=infer GEN=3    # 3 tokens instead of 17, for speed
python accel/tpulang/infer_export.py -n 256      # accuracy, generating
```

```
prefill   the 15 prompt tokens (numbers_data.EQUALS_POS) in ONE block. Their K
          and V land in the cache; the last row's logits are the first answer
          digit.
decode    16 steps of M=1. The new token's K and V are appended and attention
          contracts against the whole cache.
```

Both are the same code. `infer_block(rows, first_pos)` runs `rows` new rows starting at
position `t0`, prefill calls it as `(15, 0)` and a decode step as `(1, t)`, and
`always_inline` specializes the two. The cache, the mask and `tpulib.h`'s block
loops do not care how many rows arrive at once, so **M is the only difference
between the training shape and the generation shape on this machine.**

### The cache, and why each half is stored the way it is

3 KB per layer, scratchpad-resident, never moved:

| | |
| --- | --- |
| `SP_K_CACHE[l]` | `[D][T]` int8 — K **transposed**, 2048 B |
| `SP_V_CACHE[l]` | `[T][D]` int4 — V packed as a weight operand, 1024 B |

Each is in the orientation its matmul needs, because what a cache costs is the
*append*:

- **V is free.** `P @ V` contracts over keys, so its weight is `V[s][h]` — row
  `s` contiguous over `h`, exactly how V leaves its projection. Appending token
  `t` is one `quant4` writing one 32-byte row.
- **K is a column.** `Q @ K^T`'s weight is `K^T[h][s]` — row `h` contiguous over
  *s* — so the cache is column-major and appending writes one byte into each of
  D rows. That scatter is the transposing DMA (`fill.t` with `tdrow = T`), one
  command whatever M is.

K is kept int8 and re-`quant4`ed whole into a scratch `[D][T]` once per layer per
step. That looks wasteful — 2048 elements packed to use at most `t+1` columns —
and it is 4 VPU commands against the ~24 000 DMA clocks the same layer spends
fetching weights. The alternative does not exist: the nibble for `(d, t)` sits in
the middle of a byte and no op writes half a byte.

**Nothing is zeroed, and nothing needs to be.** Cache columns past the current
position hold whatever the last problem left there and reach S as garbage — but
S is int4 and the mask is `-8`, so a masked score is at most `-1` and ReLU takes
it to exactly zero. The mask that makes attention causal is the same mask that
makes an uninitialized cache safe, which is why `infer_export.py` runs every
problem through one TPU instance rather than a fresh one: not a shortcut, the
test.

### What moved from the host to the device

`adder.c`'s header calls the token embedding and the argmax the host's job
"structurally": the ISA has no gather and nothing returns an index. Both are
true of the *command* ISA and neither is true of the machine — `cpu_subsys.sv`
decodes `0x9xxx_xxxx` onto the scratchpad's S port, and always did.

| | |
| --- | --- |
| argmax | the head writes 13 int32 logits to the **scratchpad**, and the CPU reads them back with `tpu_spad_ld` and compares |
| gather | the embedding table is a DRAM tensor and a DMA takes a *computed* address, so `DR_EMB + tok*D` is the gather. The index never leaves the CPU |

So the host stages the weights, the mask, the head, the embedding table and the
prompt's token **ids**, presses `'G'` once, and reads a finished sequence out of
`DR_TOKENS`. Nothing round trips per token. What it still does is tokenize.
[`spadwin.c`](spadwin.c) is that window on its own, in 304 bytes and 4 commands
— run it first if anything here misbehaves.

### Tracing a kernel that branches on its own results

The command stream now depends on what the array computed: the token the argmax
picked is in the *address* of the next gather. So a producer with no model of the
machine cannot emit this kernel's trace, and `make trace PROG=infer` gives a
structurally-right, numerically-meaningless one (a read with no driver attached
returns 0).

The real trace comes from co-execution: `fw_vectors.py -x` runs the `-DTPU_TRACE`
binary as a **co-process**, executing each command on `iss.py` as it arrives and
answering the kernel's scratchpad reads out of the model's own memory. That is
what `make fw FWPROG=infer` does, and it makes the golden command stream a real
forward pass's — so if the RTL picks a different token anywhere, the run fails at
*that command* rather than merely producing a different answer.

`fw_vectors.py` also carries `reference_infer`, which recomputes the whole model
from scratch at every step, in integer numpy, with **no cache at all**, and
checks every generated token and every logit. That is the check that matters
here for the same reason `tiled.c` has one: a cache column written at the wrong
offset, the mask row of the wrong position, an argmax over the wrong words — the
ISS reproduces all of them exactly as faithfully as the hardware would.

### Measured

`python accel/tpulang/infer_export.py -n 256` on
`model/saved/int4_d64_f256_l4.pt`, generating the whole 17-token answer field
from the prompt ids and nothing else:

| | exact-sequence | token |
| --- | --- | --- |
| QAT model, greedy | 100.00% | 100.00% |
| this kernel, generating | **100.00%** | **100.00%** |

and the device chose the *same sequence* as the PyTorch model on 256 of 256
problems. Unlike `adder_export.py`'s number this is not teacher forcing: the
device is fed the prompt and its own output, so one wrong digit would derail
every digit after it.

## Run it on the board

```bash
make -C accel/tpu/fw run PORT=COM5
make -C accel/tpu/fw run PROG=matmul_loop PORT=COM5   # the software tile loop
# or: python accel/tpu/host/run_fw_matmul.py -p COM5
python accel/tpu/host/run_fw_matmul.py --dry-run   # operands + reference, no board
```

[`../host/run_fw_matmul.py`](../host/run_fw_matmul.py) loads the image over
`'I'`, writes A and W into DRAM, releases the core with `'G'`, then reads the
int32 result back and checks it against a Python matmul. Both commands take the
firmware RAM by setting bit 12 of their word address (`tpu_uart.FW_BASE`); the
CPU always resets to firmware address 0.

`matmul.c` and `matmul_loop.c` share the problem, the layout and the result
address, so the same script checks either — pass `--fw matmul_loop.hex`, or let
`make run PROG=` do it. The pair is the firmware version of
[`tiled_matmul_hw.tpu`](../../tpulang/examples/tiled_matmul_hw.tpu) vs
[`tiled_matmul.tpu`](../../tpulang/examples/tiled_matmul.tpu): one `matmul_t`
against 8 single-tile dispatches with `.acc` across the contraction.

The board must be running `board=cmod_a7` and the bitstream must be new enough
to contain `cpu_subsys.sv`.

## Three things that are software's problem now

- **Cross-unit ordering.** Each unit has its own queue, so a `matmul` will start
  on top of a DMA that has not finished. `tpu_wait(unit)` — retired caught up
  with issued — is the fence.
- **Queue-ordered geometry.** `MXU_GEOM` sticks until the next one, but only
  within the MXU's own command stream, so it cannot be corrupted by another unit
  or an earlier program the way the old `cfg` registers could.
- **The scratchpad window is unsynchronized.** `tpu_spad_ld` / `tpu_spad_st`
  bypass the queues entirely — they are loads and stores — so a read of a tensor
  a queued command has not written yet is simply stale, and needs the same
  `tpu_wait` a dependent command would. Two more rules, both from
  `cpu_subsys.sv`: 32-bit accesses only (the S port has no byte strobes, so an
  `sb` writes its three neighbours) and 4-byte-aligned.

Flow control is *not* software's problem: a full queue withholds the write
response on the fourth word and the CPU stalls inside the store.
