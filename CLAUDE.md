# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Practices (VERY IMPORTANT)
- Do NOT run non-trivial commands unless told to do so. This includes: running code, installing packages, pushing to git 
- Respond to me concisely, and don't tell me things I didn't ask for. After making edits, create a summary of every change you made, but keep it short: concise bullet points for everything, not full paragraphs.
- Don't use unneccessary jargon that just creates further confusion. For example, you once told me that a test has "teeth" when referring to a modified testbench... what does that even mean? Just keep the responses straightforward and simple.
- Keep code self-documenting. Instead of writing long comments explaining everything, make variable names clear. Feel free to make names as long as needed!
-  Keep comments to a minimum. You do not need to narrate every single line of code with a 4-line paragraph. The comments only confuse me more. Don't justify design choices in code, either. Just describe what it does (if it's confusing) and move on; leave the justification to the docs. Don't make a huge line of text at the beginning of a file, either. Leave all that to docs.
  - make a doc file for code changes instead of making huge comments
- Make docs in markdown, but do not use markdown syntax; only bullet points and hashtags; I rarely read docs in actual rendered markdown, I just read the source.
- Write all docs and comments in a concise style, in my voice. Look at sw_rewrite.md for a reference on style
- Double-check before claiming something as fact. Don't state things confidently without a source - if you can't verify it, say so or go verify it first. Trust verified evidence over a single conflicting source.

## What this is

A small decoder-only transformer trained to do multi-digit **addition** (character-level),
plus a custom SystemVerilog TPU that runs it end to end on an FPGA.

- Everything is **int4** — weights *and* activations, trained with QAT.
- `model/` is the golden reference. Every accelerator in `accel/` is validated against it.
- The TPU runs on a **Digilent Cmod A7-35T** and generates autoregressively: prefill, then
  decode against a KV cache, with the argmax and embedding lookup on the device.

### Live shape

`model/transformer.py::adder_int4_wide` — `d=128, f=512, layers=4, q_heads=kv_heads=4`
(so `head_dim=32`), int4 weights and activations, no bias. `numbers_data` supplies
`EQUALS_POS = 64` and `MAX_TOKENS = 128`.

`accel/test/tests/infer/infer.c` is that model as inference, and is the kernel everything
else points at. Its shape and its whole DRAM map come from a generated `infer_config.h`.

### Things that are currently out of step — check before trusting them

- **There is no trained checkpoint at the wide shape.** `python -m model.make_dummy_checkpoint`
  writes an untrained one at `model/saved/int4_d128_f512_l4.pt` so the export, staging and
  RTL paths can run. Any accuracy it scores is chance — `fw/perf_notes.md`'s 0.00% is that.
- **`numpy` and `torch` are needed for the `infer` test and for `export.py`.** The other
  seven kernels need only a host C compiler.

## Commands

Python is a package rooted at the repo, so **run from the repo root** with `-m`:

```bash
python -m model.train --arch int4_wide          # vanilla | gqa | int4_vanilla | int4_wide
python -m model.tests.test_inference --arch int4_wide
python -m model.make_dummy_checkpoint           # untrained .pt, for plumbing only
```

TPU stack — **one command producer, PicoRV32 firmware built out of `accel/tpu/fw/`**. There
is no assembler and no `.tpu` language. Everything runs through `accel/test`:

```bash
python accel/test/run_suite.py                  # 8 kernels on the ISS, ~1 s
python accel/test/run_suite.py -b rtl           # ...through the whole core, ~20 s
python accel/test/run_suite.py -b rtl -k tiled -v
python accel/test/run_suite.py -b rtl-uart      # ...loaded over the simulated UART, ~6x
python accel/test/run_suite.py -b board -p COM5

python accel/test/tests/matmul/generate.py -b rtl -M 32 --ktiles 8 --ntiles 4
python accel/test/tests/ffn/generate.py -b rtl -T 32 -d 64 -f 256
python accel/test/tests/infer/generate.py -b iss --synthetic --gen 3 -n 1
python accel/test/tests/infer/generate.py -b iss -n 256        # accuracy, generating
python accel/test/tests/infer/generate.py -b rtl --synthetic --gen 3 --phase split
python -m accel.test.export --dump-rq --model-path model/saved/int4_d128_f512_l4.pt

cd accel/tpu/tb && make list                    # RTL block testbenches
cd accel/tpu/tb && make TEST=mxu                # one of them
```

`infer`'s knobs are `--gen`, `--batch`, `--block`, `-T/--prompt`, `--phase`, and
(synthetic only) `-d/-f/-L/--heads`. They land in the generated header, not in `-D` flags.

**`--phase prefill|decode|both|split` is how prefill and decode are measured apart.** The
counters reset at the launch and freeze at the halt, so an image that runs one half *is*
the measurement of that half; `split` runs both phase-only images in turn and prints each
phase's clocks per token. Decode-only checks nothing (it starts from the prompt's last
token, so its ids are noise; the clocks are not data-dependent and are the same clocks);
prefill-only still produces the reference's first token and is checked on it.

`ISSBackend` **runs** the `-DTPU_TRACE` binary as a co-process rather than reading a
captured trace, because `infer.c` argmaxes its own logits and the token it picks lands in
the *address* of the next DMA — its command stream is not a function of the program alone.
The ISS answers each `SRD` out of its own scratchpad.

Deps are `torch` (+ jupyter for `model/notebook.ipynb`) and `pyserial` for the host driver.
They live in the conda base env — `conda activate` before running anything that imports them. There is no requirements.txt and no `.venv/`.

## Architecture

### `model/transformer.py`

- **5-D attention layout.** Q/K/V are `[batch, tokens, kv_heads, heads_per_q, head_dim]`
  (K/V drop `heads_per_q`). GQA is broadcasting `kv_heads` over `heads_per_q`. The math is
  the einsum pair `"btkgh,bskh->btskg"` and `"btskg,bskh->btkgh"`. This is the contract every
  accelerator implements.
- **ReLU attention, not softmax.** `P = relu(S + causal_mask)`, mask built as
  `triu(ones([T,T]) * -1e9, diagonal=1)`. No normalization over the source axis at all.
  (The code still carries the comment "revert to softmax if training bad".)
- **No positional encoding.** The only position signal is the causal mask; with no
  source-axis normalization the *magnitude* of `P @ V` carries the count of visible keys.
  The same removal under softmax would leave the model position-blind.
- **No LayerNorm.** `norm1`/`norm2` are `DyT` — `hardtanh(x * alpha)`, one learned scalar,
  no gamma/beta.
- **Double residual.** `MultiHeadAttention.forward` ends in `return O + X` and
  `Transformer.forward` adds `X` again, so the attention residual is `2X + O`. The
  checkpoints were fitted to it.
- **Attention is inlined** in `MultiHeadAttention.forward`. `use_custom_attention` still
  threads down but is accepted and ignored — there is no CUDA path from the model.
- **The padding mask is threaded through and never applied.** Only the causal mask is used.
- **int4 weights (`Int4Linear`).** `[-8, 7]` scaled by `absmax/7`, `RoundClip` as the STE.
  The scale is **detached**. The weight is `[in_dim, out_dim]` with `x @ w` — the transpose
  of `nn.Linear`, so int4 and float checkpoints are not interchangeable. `make_linear(...,
  use_int4)` selects it, **including `Model.fc`**; the head's output is never requantized,
  since an argmax does not care about scale. `TernaryLinear` is retained but no config
  builds one.
- **int4 activations are QAT, not post-hoc.** Every requant site is an `ActQuant` with one
  per-tensor LSQ-learned scale. Sites the hardware *pins* share one **instance** —
  `q_o`/`q_xo` are the residual stream's `x_quant`, `q_p is q_s`, `q_hr is q_h` — so the
  sharing in `__init__` is load-bearing. DyT outputs are pinned to `fixed_scale=1/7` with
  `qmin=-7` (hardtanh bounds them analytically, symmetric because hardtanh is odd).
  `set_quant_enabled(model, False)` turns them all off — **necessary for a real float
  baseline**, since they are live by default.
- **MoE is gone.**
- Named configs at the bottom are the source of truth and are wired to `--arch`.

### `model/numbers_data.py`

- **Digits are reversed** (`REVERSE_DIGITS = True`): `123+45=168` is emitted as
  `321+54=861`. Carries propagate from the ones digit up, which is the direction an
  autoregressive model can see, and it fixes place value to position.
- **`EQUALS_POS = 64`** is the first *answer* position; `=` is at 63. Operands are padded so
  this holds regardless of length. Implies `max_digits <= (EQUALS_POS - 2) // 2` = 31, and
  the generator raises rather than silently shifting.
- `equals_pos` is a per-call argument. `accel/test/export.py` pins it to the kernel's `PROMPT`.
- Padding token is `'N'` (`PAD_ID = 12`); `tokenize` builds an additive `-1e9` mask from the
  pad positions, which the model ignores.
- `_sample_number` samples uniformly over **digit count**, not value.
- `MAX_INTEGER = 99999` is vestigial.

### `model/train.py`

- `batch_size` vs `mini_batch_size` is **gradient accumulation**;
  `optim.step()` fires every `batch_size // mini_batch_size` micro-steps.
- **Gradient clipping** (`--grad_clip`, default 1.0) applies to the *accumulated* gradient
  immediately before the step; reported `grad_norm` stats are pre-clip.
- The loss spans `EQUALS_POS-1 : -1` against `EQUALS_POS:`, so the trailing pads are part of
  the objective on purpose.
- Checkpoints go to `model/saved/test_model_*.pt`, last 3 kept, gitignored.

## The TPU (`accel/tpu/`)

Everything is int4 on both sides: the RTL and `iss.py` take **int4 weights in a row-major
4-bit packed layout**, and every narrow clips to `[-8, 7]`. Row-major inverted the attention
transpose relative to the retired ternary kernel — `P@V`'s weight operand is now the free
one and `Q@K^T`'s is the one needing the transposing DMA.

### Dispatch

- **A dispatch is a 128-bit macro-op pushed into a per-unit queue**, not a wire bundle plus a
  global config file. `cmd_queue.sv` + `cmd_{mxu,vpu,dma}.sv` sit in front of each unit; a
  command carries its own operands and geometry, so nothing an earlier dispatch — or an
  earlier *program* — left in a register can reach it.
- **The requant `{m0,n}` is a literal in the command**, not a scratchpad address. Same 16
  bits, and it deleted a two-state fetch in both `mxu.sv` and `vpu.sv`. The consequence:
  **the requant table has to be compiled into the firmware image.**
- **`scratchpad.sv`'s exclusivity invariant is gone.** It held only because issue-and-wait
  serialized the units. Arbitration is real, with grants ordered so the requester that cannot
  stall wins: reads `A>W>C>V>s>DMA`, writes `C>V>s>DMA`. The VPU freezes for a clock when
  denied; the DMA parks fill bytes in a skid buffer and, past that, stops the SRAM read
  stream. **If you make two units run at once, check every path into the scratchpad takes
  its grant back** — a denied requester that ignores it loses the access silently.
- **Perf counters are 10 wide**: run, mxu, mload, vpu, dma, swait, vmm, idlec, qfull, ovlap.
  `swait` and `vmm` are **retired slots tied low**, kept so the UART `'T'` reply does not
  renumber. `idlec` (no unit busy) is what instruction overhead costs. `qfull` is
  structurally 0 — both producers gate `cmd_we` with `!cmd_full`.
- **Retired opcodes are holes, not free space.** `0x02`, `0x05`, `0x0A`–`0x0F`, `0x1B`,
  `0x1E`, `0x20`; VPU op selector 13; VPU command `0x02`; `cfg` indices 9 and 10–14. A stale
  binary decodes to an unknown op rather than a different one. `0x22` is `quant4`; new ops
  go at `0x23`+.

### The VPU has six ops and nothing else

`VOP_DOT`, `ADD`, `RELU`, `REQUANT`, `DYT`, `QUANT4`.

- **`QUANT4`** (`0x22`) is `requant`'s fixed point clipped to `[-8, 7]` and written **4 bits
  wide**, two elements per byte, in the MXU's weight encoding. It is what lets an activation
  be a weight operand, and therefore what put `Q@K^T` and `P@V` on the array. `vlen` must be
  a multiple of 2, and the destination advances half as fast as the source.
- **`VECMATMUL` was removed** once that left it with no caller (it was 36.4% of the whole run
  when attention ran on it). Gone with it: the `mm_*` sequencer, `cfg` 10–14, the `VPU_GEOM`
  command, and `vpu_mm_busy`.
- **`VOP_DOT` stayed** with no caller: it *is* the reduction path and the only int8 x int8
  reduction the ISA has.
- `GELU`, `EXP`, `SQUARE`, `ELEMENT_MUL`, `SCALAR_*`, `REDUCEMAX`, `REDUCESUM` and the
  `SOFTMAX` macro op were removed with both activation ROMs, `rtl/luts/`, `accel/tpulang/luts.py` (all deleted),
  the restoring divider, the softmax sequencer and the reduction path's max fold.
  Measured by OOC synthesis of `vpu` alone: **10012 -> 5162 LUTs (−48%), 897 -> 667 FFs,
  90 -> 32 DSPs (−64%)**.

### Memory

- **`sram.sv` moves ranges, not bytes.** One request is a start address, a byte count and a
  stride. **1 clock/byte on fills, 2 on spills**, against ~8 before; the full model went
  1 226 722 -> 541 590 clocks (2.27x) with byte-identical output, and DMA fell from 66% of
  the run to 24%.
- Two things to know before touching it: **writes are two clocks because WE# is generated on
  the falling edge**, so its rising edge lands half a clock clear of the address/data change
  (the failure mode is a byte written to its *neighbour*, and `sram_tb` and `dma_tb` check
  for it); and the `stride` exists for the transposing spill, whose destination is
  column-major and would otherwise be one range per byte. `CLOCKS_PER_ACCESS` is now *extra*
  clocks per beat and is 0 on every board.
- **19-bit DRAM addressing.** It used to be 16, which silently confined a program to the low
  64 KB of a 512 KB part. **When something addresses DRAM, check it is not using `ADDR_W`** —
  the testbenches' DRAM byte maps were sized off it too, so high expectations were dropped by
  `$readmemh` without a word.

### Synthesis

Last `make bit`: **17 606 LUTs of 20 800 (85%)**, 10 903 FFs, 68 BRAM primitives, 46 DSPs,
WNS **+34.98 ns**, all constraints met — but that bitstream predates the last edits to
`tpu_top.sv` / `vpu.sv` / `cmd_vpu.sv`, so re-synthesize before quoting it. Board geometry is **8x8**, `VPU_BYTES=32`,
`ADDR_W=16` (64 KB scratchpad), `MEM_ADDR_W=19`, 12 MHz.

Per block: `mxu` 7588 LUTs, `vpu` 4021, `scratchpad` 2413, `cpu_subsys` 1573, the three
command queues 815, `dma` 493.

### `fw/tpulib.h` — the primitive layer, written to be specialized

- `tpu_matmul` blocks a GEMM in rows (`t_len <= 32`), columns and the contraction, stages
  whichever operands are in DRAM through a caller-supplied `tpu_arena`, and requants on store
  when the contraction was unsplit or through the VPU when it was not.
- **`tpu_matmul_wide` is `tpu_matmul` with a one-column-block C.** It spends the
  arena's spare bytes on row-panel depth instead of C width, so a wide GEMM reads
  its weight stream fewer times — but it issues `cols/N` spills per panel instead
  of one, and it double-buffers B, which costs a bank. Transposed-B and
  accumulate both work. `docs/fw.md` has the cases where it is a loss;
  `tests/tiled/`'s `--mm5-general` and `tests/wide/` are the A/Bs.
- **`infer.c` uses `tpu_matmul_wide` at every site**, `Q@K^T` included.
  `-DINFER_MM_WIDE=0` (`tests/infer/generate.py --general`) is the A/B. On the RTL
  at `d=128/f=512, --gen 3`: **2 890 294 clocks against 3 489 116 (−17.2%)**, with
  identical `mxu` and DMA clocks — the whole difference is 583 695 clocks of
  prefetch overlap that `tpu_matmul` cannot express. It costs +24.6% commands and
  ~2.5 KB of the 16 KB firmware image.
- `tpu_add_narrow` / `tpu_relu_narrow` / `tpu_pack4` chunk the VPU pairs at `vlen`;
  `tpu_transpose_int8`, `tpu_transpose_dram_int8`, `tpu_move2d` cover the rest.
- **Every primitive is self-fencing** — it returns only once its commands have retired — so
  composing two is always safe.
- **`tpu_matmul`, `tpu_gemm_blocks` and `tpu_gemm_arena_bytes` are `always_inline`, and that
  is load-bearing.** Everything a primitive computes before its first push is exposed clock
  for clock (the caller has just fenced) and the PicoRV32 is ~5–9 clocks per instruction with
  no cache, so ~200 instructions of block arithmetic costs more than the array spends on the
  dispatch. With the shape constant at the call site gcc folds the chooser, the loops and
  every staging branch away; without the fold the same kernel runs **597 936 clocks instead
  of 453 778** and the image is *larger*. **If you add a firmware abstraction, check the
  disassembly, not the command count.**
- **The one place two units run at once is `tpu_gemm.prefetch`**: double-buffer the staged
  weight so block *n+1*'s fill streams under block *n*'s matmul. Opt-in per call site, and it
  splits the **contraction, never the columns** — a column split makes `tpu_move2d` issue one
  DMA per contraction row. It costs the int32 partials a split contraction forces.
  `TPU_WGT_PREFETCH=0` compiles it out, which is the A/B.
- `tests/tiled/` is the regression for the block loops themselves: DRAM to DRAM, a tiny arena,
  and `tests/tiled/generate.py` checking every backend against a plain Python matmul —
  because a mis-tiled matmul is something the ISS would reproduce as faithfully as the RTL.

### `tests/infer/infer.c` — the model generating

A **32-token prefill**, then 31 decode steps of M=1 against a KV cache in DRAM. One
`infer_block(rows, first_pos)` serves both, so **M is the only difference between the
training shape and the generation shape**.

- **Every tensor's home is DRAM.** The scratchpad holds a staging arena and a 320-byte
  mailbox; scratchpad copies are a compile-time promotion cascade with a DRAM fallback.
  Nothing asserts that an activation fits on chip.
- **The cache is 12 KB per layer per sequence, and each half is stored in the orientation its
  matmul wants**, because the append is what a cache costs. V's is free (one `quant4` row,
  exactly how V leaves its projection); K's is a *column* of a column-major `[D][T]`, which
  is the transposing DMA (`spill.t`, `tdrow = T`) — one command for any M. K stays int8 and
  is re-packed whole each layer-step, because the nibble for `(d, t)` is half a byte and no
  op writes half a byte.
- **Nothing is zeroed and nothing needs to be.** The mask that makes attention causal (`-8`
  against an int4 score, then ReLU) is also what makes the uninitialized tail of the cache
  exactly zero. So the scratchpad is not cleared between problems, on the board or in
  the suite — that is the test, not a shortcut. (The KV cache *is* zeroed once, in the
  static image, so the first problem sees the same memory on every backend; nothing
  touches it after that.)
- **The argmax and the embedding gather are on the device.** `cpu_subsys.sv` maps the
  scratchpad at `0x9xxx_xxxx`, so the head writes its logits there, the CPU reads them back
  (`tpu_spad_ld`) and argmaxes, and the gather is a DMA at `DR_EMB + tok*D`. The host
  tokenizes and nothing else. `tests/spadwin/` is that window's own regression; the window
  is **unsynchronized** — a load is not a command, so it needs the same `tpu_wait` a
  dependent command would, 32 bits wide and 4-byte aligned (the S port has no byte strobes).
- **`BATCH` sequences share every weight stream.** X is `[BATCH][rows][D]`, so the
  projections, `Wo` and both FFN matmuls run once over `BATCH*rows` rows. Attention stays per
  sequence. That is the point at decode, where a step is one row of arithmetic against
  ~390 KB of weights.
- **The image is 14 708 bytes of a 16 KB firmware RAM** at `d=128 / f=512`, with the
  stack growing down from the same RAM — ~1.6 KB of headroom. It is 14 804 at
  `d=64 / f=256`. The wide-everywhere `INFER_MM` is what costs it: `-DINFER_MM_WIDE=0`
  is 12 204 and `-DTPU_WGT_PREFETCH=0` is 11 808. **Check `size` on the .elf before
  adding a call site.**
- **Its shape and its whole DRAM map come from the generated `infer_config.h`.** There is
  no `DR_ALIGN` chain in the C any more, and `DR_LAYER0` is wherever the activations end
  rather than a hardcoded `0x20000`.

### `accel/test/` — the verification suite

- `iss.py` — bit-exact with the RTL; the instruction decoder is gone, `exec_command` /
  `run_trace` are the way in.
- `backends.py` — `ISSBackend` (native build + co-execution), `RTLBackend` (RISC-V image
  through the whole core in Icarus, `uart=True` for the serial load path), `TPUBackend`
  (the board). All three: `build`, `load` once, `run(patch)` per case.
- `vector_generator.py` — the `VectorGenerator` / `Case` contract, `AddressMap`, `fit_rq`,
  and the packing / requant / `$readmemh` helpers.
- `program.py` — `TPUProgram(source, backend, generator).run_program()`: build, load,
  per-case compare, plus the stray-write check. Each `Result` carries that case's perf
  counters; `benchmark()` / `format_benchmark()` aggregate them, and `report()` and
  `run_suite.py` print them. Keys and order are the board's `'T'` reply on every backend
  that has them, so `rtl`, `rtl-uart` and `board` are directly comparable — they overlap
  and do not partition the run.
- `export.py` — checkpoint -> `infer_config.h` (shape, the whole DRAM map, the requant
  table) + the static DRAM image. **Python owns the addresses; `infer.c` computes none.**
- `tests/<name>/` — one kernel's `.c` and its `generate.py`, in one folder.
  **A kernel's shape, address map and requant words come from its generator as `-D`**; the
  `.c` carries `#ifndef` defaults for a bare `make`. `fit_rq` derives each requant word
  from the accumulators the golden produced, so a shape change cannot silently saturate or
  collapse a tensor. Every shape has a flag: `-M/--ktiles/--ntiles`, `-T/-d/-f`,
  `--head-dim`, `--depth1/--vec/--arena-banks`, `-w/--row-bytes`.
  **The build directory is keyed on a hash of the flags** — `make` compares timestamps and
  cannot see a changed `-D`, so without that a sweep runs the previous shape's image
  against this shape's golden.
- `run_suite.py` — all of them, on one backend.

Three rules the backends depend on: the static image must be **dense over everything the
kernel reads** (the ISS starts zeroed, the board's SRAM does not); DRAM **carries over**
between cases and is not reset; and a write outside `check_ranges ∪ writable_ranges` fails
the case on `iss` and `rtl`.

### `accel/cuda/`

Legacy. It implements *softmax* attention and the model does ReLU attention; nothing in
`model/` loads it. It still passes its own self-contained test.

## When touching attention numerics

Keep these in sync: `model/transformer.py` (reference), `accel/test/tests/infer/infer.c`,
`accel/test/tests/infer/generate.py`'s integer reference, `accel/test/export.py`'s requant
derivation, `iss.py`, and the RTL.

## Long simulations

- **The ISS is where you iterate.** `python accel/test/run_suite.py` is under a second and
  covers eight kernels; `tests/infer/generate.py -b iss --gen 4 -n 4` is the model. The RTL
  run is what proves the *hardware* agrees.
- `run_suite.py -b rtl` is the smoke test that dispatch still
  works at all. `spadwin` is the only thing exercising the CPU's scratchpad window.
- `infer` on the RTL is DMA-bound at ~830 k clocks per generated token — **use `--gen 3`
  while iterating.** `run_suite.WATCHDOG_NS` sizes each kernel's watchdog.
- `-b rtl-uart` costs ~6x `-b rtl` on the same kernel, because at `UART_CPB=16` a byte is
  160 core clocks. `8` halves it; below 8 the receiver's mid-bit sample stops being mid-bit.
  `RTLBackend(rerun=True)` loads and runs twice with no reset in between — a real
  regression: it is what caught `tpu_top.sv` clearing `cpu_run` against the previous run's
  stale `cpu_done`.
- **A long Icarus run prints nothing until it halts**, which makes "slow" and "deadlocked"
  look identical from outside. Redirect the log to a file rather than piping through
  `tail`/`head`, which buffer the whole stream.
- `vvp <kernel>.vvp +CMDLOG=<path>` adds a per-command timeline; `tb/cmd_timeline.py` reconstructs
  the perf counters from it exactly and checks that on every run.

## Measured performance

From `accel/tpu/fw/perf_notes.md`, on the board at 12 MHz, `L=4, d=128, f=512`, 32 tokens:

```
per token   69.4 ms (832 837 core clocks)
occupancy   mxu 45.5%  mload 16.0%  vpu 2.8%  dma 54.1%  idlec 17.5%  ovlap 19.9%

                          clocks       ms     mxu   mload    vpu     dma   idlec   ovlap
prefill (32 rows)        1768763  147.397   48.6%    7.5%  10.2%   30.9%   16.5%    6.2%
one decode step (M=1)     802645   66.887   45.2%   16.6%   2.3%   55.8%   17.5%   20.9%
```

- A prompt token costs 55 274 clocks in the prefill against 802 645 in a decode step
  (14.5x) — decode is DMA-bound on the weight stream.
- Roofline for the prefill: 394 KB of weights at 1 byte/clock plus 32 KB of KV is ~426 000
  clocks of memory against ~390 000 clocks of array work at 64 int4 ops/clock, so even with
  no overlap it should be ~800 k. It is 1.77 M.
- The accuracy in that file is against the untrained dummy checkpoint and is noise.

Older, on the narrow `adder.c` (534 commands, 453 778 clocks): `mxu = 206 361`,
`dma = 131 200`, `vpu = 84 352`, `idlec = 31 864` (7.0%, what the CPU costs as a producer).
Two things the totals hid — **moving weights (23.2%) cost more than either FFN matmul**, and
**82% of the CPU's time was the `tpu_wait` barriers, not building commands** (13.3 exposed
clocks for a command pushed onto a busy unit, 183.6 for one after a barrier).

## Why the model is QAT

**It is not int8-quantizable post-hoc, and QAT is what fixed it.**

- Quantization error is *absolute*, and a per-tensor scale is pinned by the maximum, so what
  matters is `median/max`. With no normalization that collapses with depth: 1.3% of layer 0's
  residual stream rounded to exactly zero, 91.6% of layer 3's `A` did.
- The usual outlier check misses it — the *top* of the distribution is well behaved.
- The model is not fragile to perturbation in general: it tolerates ±10% *multiplicative*
  jitter everywhere at 100% exact-sequence, ~100x more error than int8 injects.
- DyT fixed the diagnosed mechanism and still scored 0.00%, because what it exposed was the
  residual *addend* — `vecadd` puts `X` and `O = Wo(A)` on one scale while they differed by
  up to 7259x.
- Per-channel scales would fix it and **this ISA cannot express them**: `requant` takes one
  `{m0,n}` per dispatch.
- QAT needs no hardware change: instead of finding scales the trained weights tolerate, it
  trains weights that tolerate the scales.

Three consequences worth internalizing:

- **A "float" row is no longer a ceiling.** Removing the quantizers from a QAT checkpoint
  gives a network the weights were never trained for.
- **Saturation rates stopped being a health check.** A layer can learn to use the requant as
  a `sign()` and clip 99.88% of a tensor deliberately.
- **Scales come from the checkpoint, not from calibration.** Re-deriving by absmax moves
  every rounding grid the weights were fitted against; on a QAT checkpoint that scores 0.00%.

Related, from tuning the synthetic requant words: moving `RQ_A` by two shifts took `A` from
45% saturated to 99.3% *exactly zero*. Each layer's requant multiplies the residual stream by
a constant, so an error compounds geometrically with nothing renormalizing it.

## Solved: intermittent UART corruption

**The fault was on the host, not the FPGA.** Reading the serial port while the USB-serial
bridge is still transmitting corrupts the byte in flight — the host's IN requests disturb the
FT2232H's transmit bit timing by roughly half a bit, so the device decodes `sent[k]` **or**
`sent[k-1]` for every bit `k`, usually `value << 1`.

Fixed in `accel/test/tpu_uart.py`: `TPUUart._send` waits for a frame to clear the wire before
anything reads. The old code read immediately after `ser.write(frame)`, so the read landed on
the *header of every command*, where a corrupted length does the most damage.

The evidence, on the echo image, 64-byte bursts, 6400 bytes per arm:

| host behaviour | corrupted | positions |
| --- | --- | --- |
| read immediately after write | 72/6400 | 1–6 |
| sleep past the whole burst, then read | **0/6400** | — |
| sleep over only the *first half*, then read | 77/6400 | **33–46** |

Delaying the read moves the damage to where the read begins.

**Ruled out — do not re-litigate.** Anything in the RTL; `uart_interface`'s `SEND_STATUS`
blind window; external SRAM and its pins (`cmod_a7_bram` reproduces identically); byte value
and predecessor; position in the burst per se; baud mismatch and sampling phase (swept ±3%,
flat); long low runs on the line; the activity LED's load step; `reset_input_buffer()`;
metastability, crosstalk, timing closure.

**The simulation blind spot.** `tb/uart_memory_cosim_tb.sv`'s host clocks every bit for
exactly `CPB` clocks, so it has *zero* baud error — a real 115200 host against a 12 MHz
CPB=104 device runs at 104.1667. No simulation could ever see this — `fw_uart_tb.sv` has
the same property. Worth remembering before trusting a green simulation about anything
analogue or timing-related. (`uart_memory_cosim_tb.sv` and its host driver are deleted.)

**Still worth doing.** `UART_RX_TIMEOUT = 0` is what turns one corrupted byte into a
permanently wedged link. `20 * UART_CPB` in `boards/*/board.tcl` makes it cost one legible
timeout instead. That is hardening, not the fix.

`docs/uart_selftest.md` has the echo self-test. **Reflash `board=cmod_a7` before running
anything on `-b board`** — it times out against the echo bitstream, and
bitstreams under `synth/build/` are not rebuilt by `mode=program`.
