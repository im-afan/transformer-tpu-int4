# Synthesis & deployment (Vivado)

How to get `rtl/` onto a **Digilent Cmod A7-35T** (`xc7a35t-cpg236-1`).

The build is **non-project batch mode**: no project is version controlled, no `.xpr` is ever
opened. `synth/vivado/build.tcl` is the whole recipe, and everything lands in
`synth/build/<board>/`, which is gitignored.

> **Vivado 2024.2** (win64, build 5239630) — the version the checked-in results were
> produced with. Tcl commands and default strategies drift between releases.

## 1. Quickstart

```bash
cd accel/tpu/synth/vivado

vivado -mode batch -source build.tcl -tclargs mode=rtl      # does it elaborate?  (seconds)
vivado -mode batch -source build.tcl -tclargs mode=synth    # how big is it?      (minutes)
vivado -mode batch -source build.tcl                        # full build to a bitstream
vivado -mode batch -source build.tcl -tclargs mode=program  # flash the board
vivado -mode batch -source build.tcl -tclargs mode=help
```

**On Windows**, `vivado` is `vivado.bat`, and cmd.exe treats `=` as a delimiter — so
`-tclargs mode=rtl` reaches Tcl as two words. `build.tcl` accepts that split form, so all
three of these work:

```powershell
-tclargs mode=rtl       # split by cmd, handled
-tclargs "mode=rtl"     # quoted, arrives intact
-tclargs mode rtl       # explicit space form
```

Vivado must be on `PATH` — run from the Vivado HLx Command Prompt, or call `vivado.bat` by
full path.

## 2. What each file does

| File | Role |
| --- | --- |
| `synth/vivado/build.tcl` | The flow. Board-agnostic; parses args, dispatches on `mode=` |
| `synth/vivado/sources.tcl` | The RTL file list, declared once, kept in the same order as `tb/Makefile`'s list so sim and synth stay visibly in sync |
| `synth/vivado/boards/cmod_a7/board.tcl` | Everything target-specific: part, top module, clock, geometry generics, flash part |
| `synth/vivado/boards/cmod_a7/cmod_a7_top.sv` | Board wrapper — brings out only clock/reset/LEDs/UART/SRAM |
| `constraints/cmod_a7.xdc` | Pins, the 12 MHz clock, timing exceptions. The **only** xdc the build reads |
| `constraints/constraings-cmod.xdc` | Digilent's master, fully commented out. Pin reference only |

To add a board, copy `boards/cmod_a7/` and run `board=<name>`. `build.tcl` needs no change.

## 3. Modes

| `mode=` | Does | Use when |
| --- | --- | --- |
| `rtl` | elaboration only | Always first. Catches port/width/parameter mistakes in seconds |
| `ooc module=<name>` | out-of-context synth of one module | "How big is `mxu` on its own?" |
| `synth` | full synthesis + reports | Area check before a 20-minute place-and-route |
| `impl` | synth + opt/place/phys_opt/route | Timing closure |
| `bit` *(default)* | everything, ending in `<top>.bit` | Producing a bitstream |
| `mcs` | `bit`, then a QSPI image | Making it persist across power cycles |
| `program` | downloads an existing `.bit`. **Builds nothing** | Re-flashing |

`program` builds nothing on purpose, so re-flashing cannot silently rebuild with different
arguments than the `.bit` was made with.

Reports go to `synth/build/<board>/reports/`. The script prints a utilization summary and an
explicit **WNS/WHS verdict**, because a design that misses timing still writes a
valid-looking bitstream.

## 4. Configuration

All overrides are `key=value` after `-tclargs`, order independent. Unknown keys are a hard
error, so typos fail loudly.

```bash
vivado -mode batch -source build.tcl -tclargs mode=synth rows=4 cols=4 addr_w=13
vivado -mode batch -source build.tcl -tclargs mode=ooc module=vpu
vivado -mode batch -source build.tcl -tclargs part=xc7a15t-cpg236-1
```

Defaults live in `boards/cmod_a7/board.tcl`:

| Knob | Default | Note |
| --- | --- | --- |
| `rows` / `cols` | 8 / 8 | **Not** `tpu_top`'s 128x128 defaults |
| `addr_w` | 16 | Scratchpad byte-address width -> 64 KB |
| `vpu_bytes` | 32 | 256-bit VPU port. **Not** `tpu_top`'s own default of 64 |
| `mem_addr_w` | 19 | Cmod A7 cellular SRAM is 512K x 8 |
| `clk_mhz` | 12 | The board's only oscillator |
| `cpb` | derived | `clk_mhz * 1e6 / baud` = **104** |
| `uart_rx_timeout` | 0 | Disabled — see the hardening note in `uart_host.md` |

**Why 8x8.** Every kernel and golden vector is written against it. It is also the only size
in the right order of magnitude for this part: a 128x128 array is 16 384 PEs each holding a
32-bit partial sum, ~524 000 flip-flops against the A7-35T's 41 600.

**Why 12 MHz.** The board has one oscillator. Feeding it straight to the core avoids an
MMCM, the Clocking Wizard and any `.xci` IP in the repo, and closure at an 83 ns period is
essentially free. The host is unaffected — only the on-chip divisor changes, and `board.tcl`
derives it from `clk_mhz` so the two cannot drift.

If you add an MMCM, revisit the SRAM false paths in the xdc. At 12 MHz they are safe; at
100 MHz they would be hiding a real violation.

## 5. Sizing

### It fits. `make bit` completes, timing met.

Last measured build (`board=cmod_a7 mode=bit`, 8x8, `addr_w=16`, `xc7a35t-cpg236-1`).
Note the bitstream predates the last edits to `tpu_top.sv` / `vpu.sv` / `cmd_vpu.sv`, so
re-run `mode=synth` before quoting these as current:

| Resource | Used | Available | % |
| --- | --- | --- | --- |
| Slice LUTs | **17 606** | 20 800 | **85%** |
| Slice registers (FF) | 10 903 | 41 600 | 26% |
| Block RAM | 64 x RAMB18 + 4 x RAMB36 | 50 tiles | 72% |
| DSP48E1 | 46 | 90 | 51% |

WNS **+34.98 ns** against the 83.3 ns period, WHS +0.026 ns, 0 failing endpoints, all
constraints met.

Per block, post-route:

| Block | LUTs | FFs | BRAM | DSP |
| --- | ---: | ---: | --- | ---: |
| `mxu` | 7 588 | 5 842 | — | 18 |
| `vpu` | 4 021 | 503 | — | 24 |
| `scratchpad` | 2 413 | 9 | 64 x RAMB18 | — |
| `cpu_subsys` (PicoRV32 + AXI) | 1 573 | 793 | 4 x RAMB36 | 4 |
| `cmd_dma` / `cmd_vpu` / `cmd_mxu` | 415 / 212 / 188 | 950 / 899 / 793 | — | — |
| `dma` | 493 | 132 | — | — |
| `uart_interface` + rx + tx | 415 | 602 | — | — |
| `perf_counters` | 212 | 289 | — | — |
| `sram_controller` | 91 | 78 | — | — |

**`mxu` dominates** at 43% of the design's LUTs — not the 8x8 array itself (~2k FFs of
partial sums) but `result_buf`, `MAX_TOKENS` x `COLS` of int32. `TOK_W` is the first knob
to reach for.

### History

The design has been over budget twice and is not now. For the record:

| `mode=bit` DRC, same board and geometry | LUT as Logic required |
| --- | --- |
| before the VPU trim | 26 032 (5 232 over) |
| after the VPU trim | 21 657 (857 over) |
| after the MXU shrank | 13 342 |
| today, with the command plane and PicoRV32 | **17 606** |

The 857-over state stopped reproducing when the MXU came down from 14 476 LUTs post-synth
to 5 930 post-route; the command queues and the CPU have since added back ~4 200.

**If it goes over again**, cheapest first: build at `rows=4 cols=4` (the vectors assume
8x8); shrink `TOK_W`; constrain operand addresses to 64-byte alignment to delete the
scratchpad's two barrel rotates; then `addr_w`; then a larger part.

### What the range interface cost

`sram.sv` taking whole ranges and `dma.sv` streaming them is the one area change with a
clean before/after (same board, same geometry, `mode=synth`):

| post-synth | before | after | delta |
| --- | ---: | ---: | ---: |
| whole design, LUTs | 13 892 | 13 958 | **+66** |
| whole design, FFs | 7 507 | 7 541 | +34 |
| `dma` | 146 | 762 | +616 |
| `sram_controller` | 9 | 79 | +70 |

The two blocks grew by 686 LUTs and the design grew by 66 — cross-hierarchy LUT combining
doing its job. 0.3% of the LUT budget for 2.27x on the full model.

### Why the scratchpad is banked

It used to be a flat 64 KB byte array with six independent read ports, each taking an
unaligned byte window at a runtime address — which does not fit any part in this family.
Block RAM has two ports, so `ram_style="block"` could not be honoured; Vivado fell back to
registers (**524 288 FFs** vs 41 600 available) or to LUTRAM replicated per read port
(~3 Mbit vs 400 Kbit), on top of 64 parallel 64Ki-to-1 byte multiplexers.

Banking gives each byte lane its own genuine BRAM. Storage efficiency is deliberately poor —
512 Kbit of data in ~1150 Kbit of primitives, because each bank is 8 Kbit deep and a RAMB18
is 18 Kbit. That is the price of byte granularity at arbitrary alignment, and the two
64-byte barrel rotates are the LUT cost. See [scratchpad.md](scratchpad.md) §5.

### Order of attack when changing geometry

1. `mode=rtl` — confirm the wrapper and generics elaborate.
2. `mode=ooc module=mxu`, then `scratchpad`, `vpu`, `dma`, `cpu_subsys` — real per-block
   numbers.
3. `mode=synth` — check the totals against the sum.
4. `mode=impl`, read the WNS/WHS verdict, then `mode=bit`.

## 6. Bring-up after flashing

The wrapper exposes no parallel host port, so everything happens over the serial link.

- `led[0]` = core busy, `led[1]` = program halted. `btn[0]` is an active-high manual reset.
- Use the port the FT2232 bridge enumerates as. Baud is 115200.

```bash
# link and SRAM first
python accel/test/run_suite.py -b board -p COM5 -k matmul

# then a kernel, then the model
python accel/test/tests/infer/generate.py -b board -p COM5 -n 8
```

Known functional gap, matching simulation: `nb_*` (the inter-TPU link) is stubbed in
`tpu_top`, so a LINK op is a completing no-op.

**The design reads no `$readmemh` file by default.** The VPU's activation LUTs went with the
`gelu`/`exp` instructions, and `rtl/luts/` was deleted. `SPAD_INIT` (an optional scratchpad
preload) is the only such path left and is empty. `read_design` still cds into `accel/tpu`
before reading the RTL, which matters if you ever set it: a `$readmemh` that misses only
*warns* and leaves the memory zero-filled, which looks exactly like logic that computes zero.

**Bitstreams under `synth/build/` are not rebuilt by `mode=program`.** Check their mtime
against `rtl/` before concluding anything from a board run.

## 7. Why no `.xpr` in git

`.xpr`, `.runs/`, `.cache/`, `.sim/`, `.hw/`, `.ip_user_files/` are regenerable,
version-sensitive and merge badly — an `.xpr` is XML full of absolute paths and a Vivado
version stamp. The reviewable surface is instead four small text files: `build.tcl`,
`sources.tcl`, `board.tcl` and the `.xdc`.

To keep a working bitstream around, attach it to a release rather than committing it —
`*.bit` and `*.mcs` are ignored on purpose.
