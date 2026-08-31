# accel/test — the verification suite

## Purpose: one way to run a firmware kernel, on three things, against a golden none of them produced

Before this, a kernel's expected answer came out of `iss.py` and the RTL was checked
against it; the board had its own script; the export path had two more. A wrong ISS and a
wrong kernel agreed. Now the golden is written by hand next to the kernel, and the ISS, the
RTL and the FPGA are three interchangeable ways of producing an answer to compare it to.

## Flow

```
tests/<name>/<name>.c        the kernel
tests/<name>/generate.py     its VectorGenerator + its CLI
                                    |
        VectorGenerator.static()  ---+---  the image DRAM is loaded with, once
        VectorGenerator.cases()   ---+---  per-run patch, golden, ranges to check
                                    |
                              TPUProgram
                                    |
             +----------------------+----------------------+
             |                      |                      |
        ISSBackend             RTLBackend             TPUBackend
      native cc + iss.py    riscv gcc + Icarus     riscv gcc + UART
```

### The three backends

| | builds | runs on | sees all of DRAM | command trace | perf counters |
| --- | --- | --- | --- | --- | --- |
| `iss` | host `cc`, `-DTPU_TRACE` | `iss.py` | yes | yes | no cycle model |
| `rtl` | RISC-V gcc | Icarus, whole core | yes | yes | yes, + dispatch counts |
| `rtl-uart` | RISC-V gcc | Icarus, loaded over the simulated link | yes | yes | yes, read back over `'T'` |
| `board` | RISC-V gcc | the Cmod A7 | no | no | yes, over `'T'` |

- **`iss` is where you iterate.** Seconds. It is the native build of the same `.c`, with
  `tpu.h`'s two MMIO primitives swapped for a trace emitter, so nothing but those two
  primitives differs between it and the image the board runs.
- **`rtl` is what proves the hardware agrees.** Its testbench checks two things on top of
  what `TPUProgram` compares: every byte of DRAM against the golden, and **the command
  stream the RISC-V build issued against the one the native build issued**. Both of those
  goldens come from an `ISSBackend` running inside `RTLBackend`, which is what makes it an
  RTL-vs-ISS check rather than a second opinion from the same model.
- **`rtl-uart` adds the load path.** The image and the operands go over the two serial pins
  instead of being backdoored. About 6x the run time, and the only thing besides the board
  that exercises `uart_interface.sv` under a real kernel.
- **`board` is the answer.** No stray-write check — the link can only read ranges you name.

### The contract

```python
class VectorGenerator:
    defines: dict                  # -D the kernel is compiled with: its shape,
                                   # its address map, its requant words
    def static(self) -> dict                  # {addr: byte}, loaded once
    def cases(self) -> Iterable[Case]         # Case(name, patch, golden, check_ranges)
    def writable_ranges(self) -> list         # DRAM the kernel may scribble on

class Backend:
    def build(self, source, defines, include_dirs)
    def load(self, image)                     # persists across runs
    def run(self, patch, read_ranges) -> RunResult
```

`RunResult` carries `dram` (the requested ranges), and `cmds`, `counters`, `written` where
the backend has them.

`TPUProgram(source, backend, generator).run_program()` builds, loads the static image once,
then per case writes the patch, runs, and compares. It returns one `Result` per case;
`passed()` is the verdict.

## Benchmarking

`run_program()` carries the device's own performance counters back on every case, for the
backends that have them:

```python
results  = prog.run_program()
results[0].counters        # {'run': 42225, 'mxu': 4730, 'dma': 31501, ...}
prog.read_timers()         # one dict per case
prog.timer_totals()        # them summed
prog.benchmark(clk_mhz=12) # clocks, ms, per-counter share, min/mean/max per case
print(prog.format_benchmark())
```

```
  run                                       42225 clocks  3.519 ms @ 12 MHz
  MXU busy                                   4730   11.2%
    of which weight load                        0    0.0%
  VPU busy                                   1665    3.9%
  DMA busy                                  31501   74.6%
  no unit busy (issue overhead)              4329   10.3%
  producer stalled on a full queue              0    0.0%
  two or more units busy                        0    0.0%
  MXU dispatches                               12
  DMA dispatches                               24
```

From the CLI: every test prints this after its verdict, and `run_suite.py` tabulates one
line per kernel (`--bench` for the full breakdown, `--clk-mhz` to quote the milliseconds at
a different core clock).

Three things to know before reading a number off it:

- **The keys are the board's `'T'` reply**, in wire order (`tpu_uart.TIMER_COUNTERS`), on
  every backend that reports any. So a simulated run and a hardware run are the same
  measurement, not two things that resemble each other. On `ffn`, `-b rtl` (a backdoor probe
  of `u_perf.counts`) and `-b rtl-uart` (the counters read back over the simulated link)
  both give `run = 1038`.
- **The counters overlap and do not partition the run.** `run` is the denominator; `mload`
  is a subset of `mxu`, `ovlap` of the three unit counters. The shares sum past 100% on
  purpose. `swait` and `vmm` are retired slots and always read 0.
- **The wall clock is not the measurement.** It is dominated by iverilog or by USB latency.
  `run` is the core's own busy interval, `'G'` to `done` and nothing else.

Simulation adds `mxucmd` / `dmacmd` (each queue's own `issued`, so "how many dispatches did
this shape cost the CPU" is measured rather than assumed) and `wallclk` (`host_run` to
`done`, against the counter block's own `run`). The link has no way to report those.

The ISS has no cycle model: `read_timers()` there is empty and `benchmark()` is `None`.
`'T'` is also newer than the other four commands, so a board flashed with a bitstream that
predates it warns once and reports no counters rather than failing the run.

## Shapes are a knob

A kernel's shape, its address map and its requant words all come from its
generator and reach the compiler as `-D`. The `.c` carries `#ifndef` defaults so
a bare `make` still works, but the numbers that *ran* are the generator's — which
are also the numbers the golden was computed from, so the two cannot disagree.

```bash
python accel/test/tests/matmul/generate.py  -b rtl -M 32 --ktiles 8 --ntiles 4
python accel/test/tests/ffn/generate.py     -b rtl -T 32 -d 64 -f 256
python accel/test/tests/mha/generate.py     -b rtl -T 32 -d 32 --head-dim 16
python accel/test/tests/tiled/generate.py   -b iss --depth1 2048 --vec 4000 --arena-banks 6
python accel/test/tests/spadwin/generate.py -b rtl -w 64 --row-bytes 32
```

Three pieces make that work:

- **`AddressMap`** allocates the operands instead of hardcoding them. Default
  granularity is one scratchpad bank, which is why the small kernels' tensors
  sit `0x1000` apart: a bank serves one reader per clock and a matmul reads A, B
  and C at once. It raises when a shape no longer fits rather than aliasing two
  tensors.
- **`fit_rq(accumulators)`** derives each requant word from the accumulators the
  golden just produced, so it lands the largest of them on the top of the int4
  grid. A word tuned by hand for one contraction length saturates or collapses
  at another, and an all-zero golden passes against any datapath at all. Same
  rule `export.logit_rq_word` uses for the output head. Identity passes stay
  `RQ_ONE`.
- **The build directory is keyed on the flags.** `make` compares timestamps and
  cannot see a changed `-D`, so without this a sweep silently runs the previous
  shape's image against this shape's golden. Repeat runs at one shape still hit
  the cache.

What each kernel's shape can actually be:

| kernel | free | fixed |
| --- | --- | --- |
| `matmul` | rows, contraction tiles, output tiles | — |
| `ffn` | `T`, `D`, `DFF`, all multiples of `TPU_N` | — |
| `mha` | `T`, `D`, `head_dim`, all multiples of `TPU_N` | one head, no causal mask |
| `tiled` | all seven extents, and the arena | — |
| `spadwin` | vector length, gathered row bytes (a multiple of 4) | — |
| `infer` | `L / d / d_ff / heads / T / prompt / batch / block` | via `infer_config.h`, not `-D` |

`infer` is the one that uses a generated header instead: it has a whole DRAM map
to carry, not five addresses, and a header keeps the command line short. Same
principle, different delivery.

## Running it

```bash
python accel/test/run_suite.py                      # every kernel, on the ISS
python accel/test/run_suite.py -b rtl               # ...through the whole core, ~20 s
python accel/test/run_suite.py -b rtl -k tiled -v   # one, with the simulator's output
python accel/test/run_suite.py -b rtl --bench       # ...and the full counter breakdown
python accel/test/run_suite.py -b board -p /dev/ttyUSB1

python accel/test/tests/matmul/generate.py -b rtl --ktiles 16 --ntiles 16
python accel/test/tests/infer/generate.py -b iss --synthetic --gen 3 -n 1
python accel/test/tests/infer/generate.py -b iss -n 8 --gen 4
python -m accel.test.export --model-path model/saved/int4_d128_f512_l4.pt --dump-rq
```

`infer` is not in `run_suite`'s default set: it is minutes on the ISS and hours in
simulation. Ask for it with `-k infer`, and use `--gen 3` while iterating.

The RTL block testbenches are unchanged and still live in `accel/tpu/tb`:
`make TEST=mxu`, `make all`, `make echo|mem|bram`.

## Writing a new test

1. `mkdir accel/test/tests/<name>` and put the kernel's `.c` in it. It includes `tpu.h` or
   `tpulib.h`; `accel/tpu/fw` is on the include path, and so is the test's own directory.
   **`#ifndef`-guard every shape, address and requant word** — the guarded value is the
   default for a bare `make`, and the generator's `-D` is what runs.
2. Write `generate.py`: subclass `VectorGenerator`, build an `AddressMap`, compute the
   golden stage by stage with `fit_rq` between stages, put shape + map + requant words in
   `self.defines`, and expose `program(backend, ...)` plus a `main()` built on
   `standard_parser` / `backend_from_args` / `report`. Give every shape a flag.
3. Add its watchdog to `run_suite.WATCHDOG_NS` if it runs longer than a few thousand
   clocks. A watchdog that never fires on a hang is worth nothing.

`main()` gets the counter table for free — `report(prog, args.clk_mhz)` prints it whenever
the backend produced one.

**Compute the golden, don't capture it.** `vector_generator` gives you `narrow`, `dyt`,
`rq_word` and the packing helpers; the reference should be the plainest thing that gets the
answer, in the folder next to the kernel, and should not import anything the kernel uses.

## The rules that make three backends comparable

- **The static image must be dense over every byte the kernel reads.** `iss.py` starts from
  a zeroed 512 KB `bytearray`; the board's SRAM holds whatever the last run left. A kernel
  that reads an unwritten byte passes in simulation and fails on hardware, intermittently.
  `zero_range()` is how a region the kernel reads before writing gets into the image —
  `infer`'s KV cache is the real case, ~32 KB of it.
- **DRAM carries over between cases; it is not reset.** That is the board's behaviour, so it
  is everyone's. A case's patch is written on top of the previous case's leftovers.
- **The scratchpad does not carry over on `rtl`.** Each case is one `vvp` invocation and one
  reset. The ISS and the board both keep it. No kernel currently depends on the difference,
  and one that did would be depending on something the RTL cannot show you.
- **A write outside `check_ranges` and `writable_ranges` fails the case.** On `iss` and
  `rtl` only. Most kernels declare no writable range at all, which means "this kernel writes
  its result and nothing else" — a strong claim, and the reason `tiled` catches a mis-tiled
  store rather than just a wrong number.

## The LLM path

`export.py` turns a checkpoint into the two things the device needs, and **Python owns the
addresses**:

- `infer_config.h` — shape, the whole DRAM map, and the requant table. `tests/infer/infer.c`
  includes it and computes no addresses of its own. One header is one configuration; there
  are no shape `-D` knobs left in the firmware Makefile.
- the static image — every int4 weight, the causal mask, the output head, the embedding
  table, and a zeroed KV cache.

Shape flexibility is `L / d / d_ff / heads / T / prompt / batch / block`, checked in Python
(`Shape.check`, `dram_map`) so a bad shape is a message rather than a `_Static_assert`.
Every dimension must be a whole number of `TPU_N = 8` array tiles, and the map has to fit
512 KB with the weights above it. **The block structure is not flexible** — ReLU attention,
DyT, the double residual, 14 requant sites. A structurally different model is a new kernel.

The requant derivation carries over unchanged from the retired `adder_export.py`, including
the four pinned-scale checks: `q_o`/`q_xo` on the residual's scale, `q_hr` on `q_h`,
`q_p` on `q_s`, `q_f` on `q_x1`. A vector add takes two operands at one scale and
`requant` carries one `{m0,n}` per dispatch, so a checkpoint that needs otherwise is
refused rather than exported wrong.

`tests/infer` runs either way: `--synthetic` (mixed-hash weights, a fixed requant table, no
torch) is the regression; `--model-path` is a real checkpoint, and prints the addition
accuracy next to the pass/fail. **The golden keeps no KV cache** — it recomputes the whole
prefix at every step in numpy. A reference that kept a cache would agree with a broken one.

## What this replaced

| gone | where it went |
| --- | --- |
| `accel/tpulang/fw_vectors.py` | `vector_generator.py` (helpers), `backends.ISSBackend` (co-execution), each test's `generate.py` (operands + golden) |
| `accel/tpulang/adder_export.py` | `export.py` — `derive`, the pinned-scale checks, `fixed_point`, `logit_rq_word`, `static_image` |
| `accel/tpulang/infer_export.py` | `tests/infer/generate.py` |
| `accel/tpulang/iss.py` | `accel/test/iss.py`, unchanged |
| `accel/tpu/host/tpu_uart.py` | `accel/test/tpu_uart.py`, 1091 lines to 345 — the five commands, the TX-settle fix, the frame validation, the idle probe. The byte tracer and the CLI are gone |
| `accel/tpu/host/run_adder.py`, `run_fw_matmul.py` | `TPUBackend` + `TPUProgram` |
| `accel/tpu/host/test_uart_link.py`, `uart_echo.py`, `sim_link.py` | deleted; `tb/uart_memory_cosim_tb.sv` and `make cosim` went with them |
| `tb/Makefile`'s `fw`, `fwuart`, `fwtime`, `fwvec`, `fwsweep` | `RTLBackend`. `tb/core.f` is the file list both it and the Makefile use |
| `fw/infer_rq.h`, `fw/adder_rq_ckpt.h` | the generated `infer_config.h` |
| `fw/*.c` (the kernels) | `tests/<name>/<name>.c`. `accel/tpu/fw` is now the library: `tpu.h`, `tpulib.h`, `start.S`, `memops.c`, `link.ld`, `bin2hex.py`, `mock/` |
| `model/quant.py`, `model/calibrate.py` | deleted — they described the retired ternary/int8 model and failed on a missing factory |

Two consequences worth knowing:

- **The board's echo self-test lost its driver.** `docs/uart_selftest.md` documents a
  procedure whose host script (`uart_echo.py`) is gone. `make echo` still runs the RTL half.
- **`infer.c` no longer relies on uninitialized DRAM.** The KV cache is zeroed in the static
  image. The hardware property is unchanged — the mask still makes an unwritten cache row
  exactly zero — it just is not what the test depends on any more.
