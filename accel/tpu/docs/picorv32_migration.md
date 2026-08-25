# PicoRV32 + macro-op dispatch — design record and measurements

The design record for how the TPU got its current control plane. The **ISA reference** is
[macro_ops.md](macro_ops.md); this file is the reasoning and the numbers.

## Status

**Done.** The scalar unit, `assembler.py`, `gen_vectors.py`, `torch_ref.py`, `pytpu.py` and
every `examples/*.tpu` are deleted. PicoRV32 firmware is the only command producer, and
[`../fw/`](../fw/README.md) carries the whole model.

| Phase | State |
| --- | --- |
| 0 | **done** — counters reworked; `swait` supplemented by `idlec`/`qfull`/`ovlap` |
| 1 | **done** — `cmd_queue.sv` + `cmd_{mxu,vpu,dma}.sv`; the scalar unit packed its config registers into commands and pushed them |
| 2 | **done** — scratchpad grants, VPU stall, DMA skid buffer + `sram.sv` backpressure, queue depth 8 |
| 3 | **done** — `-DTPU_TRACE` mock `tpu.h`, `iss.py` re-fronted onto command traces, RTL command-trace monitor, `fw_vectors.py` |
| 4 | **done** — `cpu_subsys.sv`, then `tpu.h`, `tpulib.h` and the model kernels |
| 5 | **done** — scalar unit and the whole tpulang toolchain deleted |
| 6 | **done** — a second architecture (`infer.c`, KV-cached decode) as a new `.c` file, no RTL change |

### What passes today

| Test | Result |
| --- | --- |
| `make TEST=mxu` / `vpu` / `scratchpad` / `sram` | 208 / 357 / 50 / 4872 checks, 0 errors |
| `make TEST=dma` | 16 737 checks, 0 errors |
| `make TEST=cmd_queue` | 123 checks, 0 errors |
| `make TEST=cpu_smoke` | 68 checks, 0 errors |
| `make cosim` (the real host driver against the RTL) | 11 of 11 |
| `make fw` / `FWPROG=matmul_loop` | 539 / 574 checks, 0 errors; 2 210 / 2 587 clocks |
| `make fw FWPROG=ffn` / `mha` | 0 errors — first VPU and transposing-DMA commands from firmware |
| `make fw FWPROG=tiled` | 525 474 checks, 0 errors, 80 404 clocks, 237/237 commands, ISS checked against an independent Python matmul |
| `make fw FWPROG=adder` | 526 959 checks, 0 errors, 453 778 clocks, 534/534 commands |
| `make fwuart FWPROG=<kernel>` | 0 errors on every kernel; `adder` 531 102 checks, counter block identical to `make fw`'s |
| `make fwsweep` | 22 of 22, 0 failures |
| `make all` | 15 of 15 |

`make fw FWPROG=spadwin` / `FWPROG=infer` were written on a machine with no RISC-V gcc and
an Icarus too old for this tree, so the ISS and their independent references are what stands
behind them.

## Why

Three limits, none of them hardware:

- **No compiler.** The retired `adder_model.tpu` was 901 lines for one model, and `pytpu.py`
  was 528 lines of staged emitter to make that writable.
- **Global config registers.** Sizes came from `setcfg` state, so stale config was the most
  common silent bug in a program, and the units read it at `start`.
- **Issue-and-wait.** A dispatch parked the producer until the unit reported done, so
  nothing ever overlapped.

The fix for all three is the same: a self-contained 128-bit command in a per-unit queue,
pushed by a CPU running C.

## Measurements

### The command plane cost +0.35%

Like-for-like on the four-layer ternary model, same program, same vectors, same 8x8 geometry:

| | baseline | with the command plane | delta |
| --- | ---: | ---: | ---: |
| run | 690 705 | **693 107** | **+2 401 (+0.35%)** |
| MXU | 405 433 | 404 409 | −1 024 |
| VPU | 52 160 | 51 648 | −512 |
| DMA | 226 198 | 227 820 | +1 622 |
| issue overhead | 6 914 (1.00%) | **9 229 (1.33%)** | +2 315 |

Three effects pulling in different directions:

- **Issue overhead is +2 315 clocks.** A dispatch spends a clock pushing per command (two
  for a matmul, which pushes geometry as well), plus a clock in the unit's front end.
- **The units got faster** by 1 536 clocks, because the requant literal deleted a scratchpad
  round trip from every requantizing dispatch.
- **The DMA got slower** by 1 622 clocks, because a fill drains its skid buffer before the
  range is declared finished. That is the price of being able to overlap at all, paid per
  range — which is why the transposing transfers (one range per row) carry most of it.

Byte-identical output. Hold that against the ~1.45x the overlap is worth, none of which was
claimed at that point: `ovlap` read 0.

### The whole model on the C producer

`make fw FWPROG=adder`, four layers plus the head, 534 commands:

| | clocks | share |
| --- | ---: | ---: |
| whole run | 453 778 | 100% |
| MXU | 206 361 | 45.5% |
| DMA | 131 200 | 28.9% |
| VPU | 84 352 | 18.6% |
| **`idlec` = the CPU issuing commands** | **31 864** | **7.0%** |

Two things follow, and neither is the queue:

- **`qfull` and `ovlap` are both 0.** Every primitive fences after every cross-unit
  dependency, and every one of those is a real dependency — removing them needs
  double-buffering, not a looser barrier. That is what `tpulib.h`'s weight prefetch is.
- **The VPU is 18.6% of the run**, and its cost tracks element count almost exactly:
  196 608 elements over 84 352 clocks is 0.43 each. The FFN's `relu -> requant` pair is **a
  third** of that traffic on its own — 32 commands and 16 384 elements per layer to compute a
  ReLU — and it is that size only because `relu` writes int32 and nothing narrows on the
  writeback path. **A fused `relu.rq` would delete half those commands and all of the int32
  traffic, worth ~6% of the whole run** — the largest item here that is a design choice
  rather than arithmetic.

### 82% of the CPU's time is barriers, not building commands

`make fwtime FWPROG=<kernel>` dumps a per-command timeline (`+CMDLOG=`) and
`tb/cmd_timeline.py` tables it. The two columns it uses are accumulated in the testbench
rather than derived, because they have to reconstruct the perf counters exactly — they do,
bit-identically, on every kernel, and the tool checks it.

| `adder` | clocks | share of the CPU's 31 864 |
| --- | ---: | ---: |
| 143 commands that follow a `tpu_wait`, 183.6 each | **26 256** | **82.4%** |
| ...of which the barrier's marginal cost | *24 352* | *76.4%* |
| 390 commands pushed back-to-back, 13.3 each | 5 192 | 16.3% |
| boot + tail | 416 | 1.3% |

- A command pushed on top of a busy unit is nearly free — 13.3 exposed clocks against the
  ~85 it costs to build, because the rest hides under the unit still running. `vpu requant`
  and `vpu dyt` cost **1.1** each: the queue absorbs them entirely.
- `tpu_wait` costs 183.6 because it polls a counter over AXI4-Lite, and the CPU is mid-poll
  when the unit goes idle.
- **48 of the 143 barriers are the three-way fence in the per-head attention loop.**

Where the clocks go by phase is in [`../fw/README.md`](../fw/README.md). The headline:
**moving weights costs more than either FFN matmul.**

### Issue overhead is O(1) in the problem, with the hardware tile walk

`make fwsweep` rebuilds both matmul kernels at each shape:

| shape M x K x N | kernel | run | MXU | DMA | `idlec` | cmds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8x8x8 | `matmul` | 1 039 | 43 | 604 | **392** | 5 |
| 8x32x32 | `matmul` | 3 520 | 577 | 2 572 | **371** | 5 |
| | `matmul_loop` | 4 413 | 784 | 2 572 | **1 057** | 20 |
| 8x128x128 | `matmul` | 22 537 | 8 833 | 13 324 | **380** | 5 |
| | `matmul_loop` | 35 222 | 12 928 | 13 324 | **8 970** | 260 |
| 32x32x32 | `matmul` | 10 911 | 1 057 | 9 484 | **370** | 5 |
| | `matmul_loop` | 11 782 | 1 840 | 9 484 | **458** | 20 |

- **`matmul.c` issues five commands whatever the shape**, and `idlec` is 370–409 clocks
  across a 205x range of array work and a 4x range of M. The spread is not a trend — it is
  gcc materializing different address constants.
- **The C loop costs ~85 clocks per dispatch.** At 8x128x128 the queue level never exceeded
  1 and not one push landed while the MXU was busy, so the CPU's cost is recoverable
  directly: 91 clocks at 8x32x32, 87 at 8x64x64, 84 at 8x128x128. Against an estimate of
  ~26, the factor is the fetch path, not the command format.
- **What it costs the run is `max(0, cpu - array)` per dispatch**, so the M axis matters more
  than the tile axis:

| M | MXU/dispatch | CPU/dispatch | exposed |
| ---: | ---: | ---: | ---: |
| 8 | 50 | ~85 | 34 |
| 16 | 71 | ~85 | 22 |
| 32 | 115 | ~85 | **6** |

The crossover is around **M ~ 24** on the 8x8 array. Above it, software tiling is free in
CPU terms — though not overall, since the int32 writeback penalty is untouched by M.

**A deeper queue would not help.** Depth is 8 and the level never reached 2.

### What `tpulib.h` costs, and why `always_inline` is load-bearing

Same simulation, same operands, same array:

| | commands | image | clocks | `idlec` |
| --- | ---: | ---: | ---: | ---: |
| hand-written against `tpu.h` | 518 | 1 544 B | 439 917 | 18 035 (4.1%) |
| through `tpulib.h`, runtime `m`/`k`/`n` | 534 | 2 784 B | **597 936** | 176 022 (29.4%) |
| through `tpulib.h`, shapes constant at the call site | 534 | 1 992 B | **453 778** | 31 864 (7.0%) |

The MXU, VPU and DMA columns are **byte-identical across all three**. The whole spread is
the producer.

**The middle row is the lesson.** A general matmul has to choose block sizes, resolve
residency, compute three base addresses and decide whether the store can narrow — about 200
instructions. On a PicoRV32 with no cache that is ~1 400 clocks, and *all of it lands between
a barrier and the next push*, where it is exposed clock for clock. Per-barrier cost went
140.3 -> **1 143.8**. The library did not issue more commands; it issued the same commands
1 000 clocks later each.

The third row is the same library with the decisions made at compile time. A transformer's
dimensions are `#define`s, so a call site that spells its shape out gives the compiler
everything: the block chooser folds to a constant, the loops fold to one iteration, every
space test resolves, and the descriptor never reaches memory. gcc will not do it unprompted
at `-Os`, so the three entry points that see the shape are `always_inline`. That makes the
image **smaller** — 1 992 B against 6 420 B without the fold — because the general paths
become dead code at every site.

What remains against the hand-written kernel is **+3.2%**: 16 more commands (`Wq`/`Wk`/`Wv`
are three dense blocks rather than one fused `[D][3D]` one, because a column slice of a
fused block is strided) and 36 more barriers, half of them the drain at the end of every
elementwise primitive. That is the price of the self-fencing contract.

**The general rule:** on this machine the producer's cost is paid in *instructions between a
barrier and a push*, not in commands issued. Any firmware abstraction is free if it folds and
expensive if it does not — so check the disassembly, not the command count.

### The serial path adds nothing and hides nothing

`make fw` proves the datapath. It does not prove the *board* path: the image arrives through
`FW_INIT`, operands are poked into the SRAM model, results read back the same way.
`tb/fw_uart_tb.sv` (`make fwuart`) ties off everything but `uart_rx`.

| | `make fw` | `make fwuart` |
| --- | --- | --- |
| firmware image | `FW_INIT` `$readmemh` | `'I'` frames |
| operands | poked into the chip model | `'W'` frames |
| start | `host_run` pin | `'G'` |
| results | read out of the chip model | `'R'` frames, compared as they arrive |
| counters | hierarchical reference | `'T'` reply, decoded |

Both use the same ISS-generated vector files, so nothing about the golden data is duplicated.
On `adder`: **531 102 checks, 0 errors, 534/534 commands, and a counter block bit-identical
to `make fw`'s** (`run=453 777 mxu=206 361 vpu=84 352 dma=131 200 idlec=31 864`).

The one check that cannot go over the wire is the full-DRAM sweep — 512 KB at 10x`UART_CPB`
clocks a byte is more simulation than every kernel put together. It stays a backdoor read of
the same memory `'W'` just wrote, extending the `'R'` check rather than replacing it.

**Cost:** 17.8 M simulated clocks, ~21 minutes of Icarus against `make fw`'s ~3, of which
16.3 M is serial traffic. `FWUART_CPB=8` halves it; below 8 the receiver's mid-bit sample
stops being mid-bit.

### The restart bug `RERUN=1` found

`make fwuart RERUN=1` loads and runs the same kernel a second time with no reset — what two
back-to-back host invocations do on the board. It failed on the second `'G'`:

```
FAIL core never started after 'G' (done=0)
FAIL command count: RTL issued 0, expected 19
```

`cpu_done` is level-held after a run, and `cpu_subsys.sv` only re-arms it on the clock
*after* it sees `cpu_run` rise. `tpu_top.sv`'s run latch cleared `cpu_run` on
`cpu_run && cpu_done` — true on that very clock, against the **previous** run's stale
`done_r`. So the core was released for exactly one cycle and put straight back into reset,
and the host saw ACK for `'I'`, `'W'` and `'G'` and then an idle board holding the *first*
run's counters and results.

Fixed by gating the clear on the run having actually been taken up, one clock later.

**This is invisible to `make fw`, `make fwsweep` and `cpu_smoke_tb`** — every one of them
runs exactly one program per reset.

## Verification: why there is no RV32IM interpreter

`iss.py` is bit-exact with the RTL, and a RISC-V core cannot be modelled by an assembler plus
`iss.py` as they stood. The answer was to **move the verification contract from the
instruction stream to the command stream**. The macro-op trace is the ISA boundary, and three
producers must agree on it:

1. **The firmware kernel, compiled natively against a mock `tpu.h`.** `tpu.h` funnels every
   MMIO access through two inline functions, so `-DTPU_TRACE` swaps those for a trace emitter
   and the *actual* `.c` source becomes the producer, with its real control flow, compiled by
   the host compiler.
2. **`iss.py`, re-fronted.** Its `_matmul` / `_vpu` / `_dma` bodies are the golden numerics
   and are kept verbatim; only the decode front end changed, from 32-bit words to 128-bit
   commands.
3. **The RTL**, via a sim-only monitor on the arbitrated command write (`p_cmd_*`, where both
   producers converged).

`fw_vectors.py` turns 1 into 2 into expected images, and `fw_matmul_tb.sv` checks *two*
things: the memory image, and 3 against 1. A mismatch localizes immediately to either the
firmware or the datapath.

### Co-execution, for kernels that branch on their own results

`infer.c` argmaxes its own logits over the scratchpad window and puts the token it chose in
the *address* of the next DMA, so its command stream is not a function of the program alone.
`fw_vectors.py -x` therefore **runs** the trace binary as a co-process, executing each command
on the ISS as it arrives and answering the kernel's scratchpad reads out of the model's
memory. A kernel that reads nothing sees no difference.

That also makes the golden command stream a real forward pass's: if the RTL picks a different
token anywhere, the run fails at *that command* rather than merely producing a different
answer.

### Independent references

The golden DRAM image is whatever `iss.py` computed, which checks the RTL against the ISS and
nothing else. That is the right check for a kernel whose job is to drive the datapath, and
**not enough** for one whose job is to drive a loop or a cache — a mis-tiled matmul, a cache
column at the wrong offset, or an argmax over the wrong words is something the ISS reproduces
as faithfully as the hardware.

So `fw_vectors.py` carries `reference_tiled` (a plain Python matmul) and `reference_infer`
(the whole model recomputed in integer numpy at every step, with no cache at all), and checks
the ISS against them before any vector file is written.

## Area

Post-route on the Cmod A7, current tree: **17 606 LUTs of 20 800 (85%)**, 10 903 FFs, 68
BRAM primitives, 46 DSPs, WNS +34.98 ns. The plane's own share:

| Block | LUTs | FFs |
| --- | ---: | ---: |
| `cpu_subsys` (PicoRV32 + AXI4-Lite) | 1 573 | 793 |
| `cmd_dma` + `cmd_vpu` + `cmd_mxu` | 815 | 2 642 |
| `perf_counters` | 212 | 289 |

Timing was never a risk: PicoRV32 closes far above 100 MHz on this part, and the board's only
oscillator is 12 MHz.

## What is left

- **Overlap.** `ovlap` is nonzero now only because of `tpulib.h`'s weight prefetch. Doing it
  generally, in hardware, is [scheduler_plan.md](scheduler_plan.md).
- **A fused `relu.rq`** — the largest single design-choice item on the measurement table.
- **`fw/adder.c` has not been migrated to the wide model.** See
  [`../fw/README.md`](../fw/README.md).
