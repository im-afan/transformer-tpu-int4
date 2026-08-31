# Software rewrite — what was built

## Purpose: one way to run a firmware kernel, on three things, against a golden none of them produced

The plan is in [`sw_rewrite.md`](sw_rewrite.md). This is what it became, the four
decisions that were not in it, and where every file went.

Before: a kernel's expected answer came out of `iss.py`, the RTL was checked against that,
the board had its own script, and the export path had two more. A wrong ISS and a wrong
kernel agreed with each other. After: the golden is written by hand next to the kernel, and
the ISS, the RTL and the FPGA are three interchangeable ways of producing an answer to
compare it to.

## Flow

```
accel/test/tests/<name>/<name>.c        the kernel
accel/test/tests/<name>/generate.py     its VectorGenerator + its CLI
                                              |
              VectorGenerator.static()  ------+---- the image DRAM is loaded with, once
              VectorGenerator.cases()   ------+---- per-run patch, golden, ranges to check
                                              |
                                        TPUProgram
                                              |
              +---------------------+---------+---------+
              |                     |                   |
         ISSBackend            RTLBackend          TPUBackend
      native cc + iss.py    riscv gcc + Icarus    riscv gcc + UART
```

## The seven decisions

### 1. RTL is a third backend, not a separate Makefile

`make fw` / `make fwuart` / `make fwtime` / `make fwvec` / `make fwsweep` are gone.
`RTLBackend` builds the image, writes the vectors, invokes `iverilog`, runs `vvp`, parses
the verdict and reads the DRAM back.

- It composes an `ISSBackend` and runs it on the same kernel. That is where
  `fw_dram_exp.hex` and `fw_cmds.hex` come from, which is what makes the testbench's checks
  **RTL-vs-ISS** rather than a second opinion from the same model.
- `fw_matmul_tb.sv` and `fw_uart_tb.sv` are otherwise unchanged. Both gained one plusarg,
  `+DRAMOUT=<path>`, which `$writememh`s the final SRAM so the Python driver reads the
  simulation's own answer instead of trusting the golden.
- `uart=True` selects `fw_uart_tb.sv` — the image and the operands over the two simulated
  serial pins, ~6x slower, and the `RERUN` regression preserved.
- `tb/core.f` is the whole-core iverilog file list, shared by the backend and `tb/Makefile`.
  The Makefile keeps the block testbenches (`make TEST=mxu`, `make all`, `make echo|mem|bram`)
  and nothing else.

### 2. `load` is separate from `run`

`Backend.load(image)` persists; `Backend.run(patch, ranges)` writes only what changed.

On the board that is the difference between 35 s and 11 ms per problem: the static image is
~394 KB at `d=128 / f=512` and a prompt is 128 bytes, at 11 520 bytes/s. A 64-problem sweep
went from ~40 minutes of wire time to ~1 minute. `VectorGenerator.cases()` yields the
patches, so the same API is a one-shot unit test at N=1 and an accuracy sweep at N=256.

### 3. Python owns the shape and the DRAM map

`export.py` derives the shape from the checkpoint's state dict, computes every address, and
writes **`infer_config.h`** — shape, map, requant table, all in one generated file.
`infer.c` includes it and computes nothing.

What that removed: the `DR_ALIGN` chain in the C, the `IN_*` constants in `fw_vectors.py`,
`infer_export.layout()` and the assert tying them together, and every shape `-D` knob in
`fw/Makefile` (`M`, `KTILES`, `NTILES`, `GEN`, `BATCH`, `BLOCK`, `PHASE`, `RQ`).

Two visible consequences: `DR_LAYER0` is now wherever the activations end rather than a
hardcoded `0x20000` (at `d=64 / f=256` the whole map ends at `0x1eac0` of 512 KB), and a bad
shape is a Python message from `Shape.check` / `dram_map` rather than a `_Static_assert`.

The retired `RQ_KP` / `RQ_VP` holes are gone with it — 16 requant sites became 14. They
existed so a table emitted by `accel/tpulang` would not renumber, and `accel/tpulang` is
gone. `INFER_RQ_SITES` in the header is what catches the enum and `export.RQ_NAMES`
drifting.

### 4. The generator declares what is checked, and what may be scribbled on

```python
Case(name, patch, golden, check_ranges)
VectorGenerator.writable_ranges()   # scratch the kernel may write unchecked
```

A byte written outside `check_ranges ∪ writable_ranges` fails the case, on `iss` and `rtl`
(the board can only read ranges you name). Five of the six kernels declare **no** writable
range at all — "this kernel writes its result and nothing else" — which is a stronger claim
than the old testbench's, and it is why `tiled` catches a mis-tiled store rather than only a
wrong number.

The other half of that contract: **the static image must be dense over every byte the
kernel reads.** `iss.py` starts from a zeroed 512 KB `bytearray` and the board's SRAM holds
whatever the last run left; a kernel reading an unwritten byte passed in simulation and
failed on hardware, intermittently. `zero_range()` is how a region gets into the image, and
`infer`'s KV cache (~32 KB) is the real case.

### 5. `run_program()` carries the counters back

Every case's `Result` holds the device's own performance counters, keyed and ordered like
the board's `'T'` reply, on every backend that has them. `prog.benchmark()` aggregates;
`prog.format_benchmark()` renders; `report()` and `run_suite.py` print it.

Both whole-core testbenches gained one machine-readable `PERF` line carrying all ten
counters in `tpu_top.sv`'s `PERF_*` order — the same order the link's `'T'` reply uses. The
first cut scraped the human-readable `$display` lines instead, which was a trap:
`fw_matmul_tb`'s `SWEEP` line carries `m=`/`kt=`/`nt=`/`mxucmd=`/`dmacmd=` and matched the
same regex. `SWEEP` is gone (its consumer, `run_fw_sweep.sh`, was deleted) and `PERFCMD`
carries the two dispatch counts and the wall clocks in its place.

On `ffn`, `-b rtl` (a backdoor probe of `u_perf.counts`) and `-b rtl-uart` (the counters
read back over the simulated serial link) both report `run = 1038` — the same measurement
by two different paths, which is the point of keying them the same way.

`'T'` is newer than the other four commands, so `TPUBackend` warns once and reports no
counters rather than failing the run against an older bitstream.

### 6. A kernel's shape is its generator's, and reaches the compiler as -D

Every shape, every operand address and every requant word now comes from
`generate.py` and is handed to the compiler as `-D`; the `.c` keeps `#ifndef`
defaults for a bare `make`. The numbers that ran are the numbers the golden was
computed from, so they cannot disagree.

- **`AddressMap`** allocates the operands, at one scratchpad bank each by
  default — a bank serves one reader per clock and a matmul reads A, B and C at
  once. It raises when a shape stops fitting instead of aliasing two tensors.
- **`fit_rq(accumulators)`** derives each requant word from the accumulators the
  golden just produced. A word tuned by hand for one contraction length
  saturates or collapses at another, and an all-zero golden passes against any
  datapath at all. Identity passes stay `RQ_ONE`. `fixed_point` moved into
  `vector_generator.py` so the tests and the exporter share one definition;
  `export.rq_word` became `rq_for_scale`, which is the scale arithmetic rather
  than the two fields.
- **Build directories are keyed on a hash of the flags.** `make` compares
  timestamps and cannot see a changed `-D` — the old `tb/Makefile` worked around
  this by cleaning every time. Without it a sweep silently runs the previous
  shape's image against this shape's golden, which is exactly what the first cut
  did.

Three kernel bugs the sweep found, all of them latent before because the shape
had never moved:

- **`ffn.c` and `mha.c` only ever wrote one 8x8 output block.** The array's
  output block is `TPU_N x TPU_N` whatever the live extent, so both extents have
  to be walked; they walked neither (`ffn`) or only the columns. Fixed with the
  row/column loops `matmul.c` already had.
- **`vlen` is a 10-bit field and both kernels' relu passes exceeded it**, which
  truncates silently: `T*DFF` at `32x256` is 8192, and `8192 & 0x3FF` is 0, so
  the relu did nothing. Both now chunk at `TPU_VCHUNK_MAX`, and
  `TPU_VLEN_MAX`/`TPU_VCHUNK_MAX` moved from `tpulib.h` to `tpu.h` — the limit
  is a property of the command encoding, not of the primitive library, and a
  kernel writing raw commands has to respect it too.
- **`spadwin.c`'s table was 13 rows against a 16-entry scan**, so a winning index
  above 12 would have gathered past it. The generator sizes the table to the
  vector.

### 7. Shape-flexible, not architecture-flexible

`L / d / d_ff / heads / T / prompt / batch / block` are free. The block structure — ReLU
attention, DyT, the double residual, 14 requant sites — is not. A structurally different
model is a new kernel, not a new define.

## Files

### New — `accel/test/`

| File | What it is |
| --- | --- |
| `iss.py` | moved from `accel/tpulang/`, **unchanged** |
| `backends.py` | `Backend` + `ISSBackend` / `RTLBackend` / `TPUBackend`, and `build_firmware` |
| `tpu_uart.py` | the link driver, **1091 lines to 345** |
| `vector_generator.py` | `VectorGenerator`, `Case`, and the packing / requant / `$readmemh` helpers |
| `program.py` | `TPUProgram`, `Result`, and the CLI every test folder shares |
| `export.py` | `Shape`, `dram_map`, `derive`, `write_config`, `static_image` |
| `run_suite.py` | every kernel on one backend |
| `tests/{matmul,ffn,mha,tiled,spadwin,infer}/` | the kernel `.c` and its `generate.py` |
| `README.md` | the suite: the contract, the rules, how to add a test |

What survived the `tpu_uart.py` trim: the five commands, `_send`'s wait-for-TX-to-clear
(**the** fix for the intermittent corruption), the host-side frame validation that mirrors
the RTL's `VALIDATE` state, NAK-as-desync, `probe_idle` / `wait_until_idle` (there is no
completion signal on this link), `load_program`, `contiguous_runs`, `autodetect_port`.
What went: `UartTrace` / `_TracedSerial` / `_hexdump` (~250 lines), the argparse CLI,
`resync`.

### Deleted

| gone | why |
| --- | --- |
| `accel/tpulang/` (`fw_vectors.py`, `adder_export.py`, `infer_export.py`) | absorbed; `iss.py` moved |
| `accel/tpu/host/` (all seven files) | `TPUBackend` + `TPUProgram`, and the trimmed driver |
| `tb/uart_memory_cosim_tb.sv`, `make cosim` | its host driver is gone, and its host had *zero* baud error — it could never have seen the corruption bug |
| `tb/run_fw_sweep.sh` | shape sweeps are a loop in Python now |
| `fw/infer_rq.h`, `fw/adder_rq_ckpt.h` | the generated `infer_config.h` |
| `model/quant.py`, `model/calibrate.py` | described the retired ternary/int8 model; failed on a missing factory |
| checked-in build artifacts in `fw/` and `tb/` | `accel/test/build/`, gitignored |

### Moved

`fw/{matmul,ffn,mha,tiled,spadwin,infer}.c` → `accel/test/tests/<name>/<name>.c`.
`accel/tpu/fw/` is now the **library**: `tpu.h`, `tpulib.h`, `start.S`, `memops.c`,
`link.ld`, `bin2hex.py`, `mock/`, `Makefile`.

### Modified

- `fw/Makefile` — takes `SRC=` (any `.c`, anywhere), `BUILD=` (artifacts out of the repo
  dirs) and `EXTRA_CFLAGS=`. The `HOSTCC` / `%.trace` machinery went with it: the native
  build is `ISSBackend`'s job now.
- `tb/Makefile` — block testbenches only; `core.f` added.
- `tb/fw_matmul_tb.sv`, `tb/fw_uart_tb.sv` — `+DRAMOUT=<path>`, one `PERF` line each
  (replacing `SWEEP` in the first), and headers pointing at the new driver.
- `tests/infer/infer.c` — the shape block and the whole `DR_*` chain replaced by
  `#include "infer_config.h"`; ~40 lines out, the body untouched.
- Docs: `accel/README.md`, `accel/tpu/README.md`, `fw/README.md`, `docs/README.md`,
  `docs/fw.md`, `docs/pipeline.md` (rewritten), `docs/uart_host.md` (the corruption
  post-mortem carried over from the deleted `host/README.md`), `docs/uart_selftest.md`,
  `docs/synth.md`, `model/README.md`, `model/docs/notes.md`, the root `README.md`,
  `CLAUDE.md`. `docs/tpulang.md` became `docs/iss.md`.

## Verified

```
python accel/test/run_suite.py -b iss        5/5 pass, 0.6 s total
python accel/test/run_suite.py -b rtl        5/5 pass, 19 s total
tests/ffn/generate.py -b rtl-uart            pass, run=1038 (== -b rtl)
tests/tiled/generate.py -b rtl-uart          pass
infer.c natively + on the ISS                5610 commands, tokens out, at d=64/f=256/gen=3
infer.c cross-compiled                       11 008 bytes of the 16 KB firmware RAM
```

`cd accel/tpu/tb && make all`: every block testbench passes except two that were **already
failing on this branch before the rewrite** — `cpu_smoke` (64 errors; the committed
testbench against the committed RTL fails identically, its hand-encoded `fw_smoke.hex` has
fallen behind) and `uart_receiver_tb` (does not elaborate). Neither file was touched here.

`tests/infer/generate.py` end to end was **not** run: this machine has neither `numpy` nor
`torch`, and the integer reference needs numpy. Everything it depends on — the config
header, the cross build, the native build, co-execution, the token spill — was exercised
directly instead.

## Two consequences worth knowing

- **The board's echo self-test lost its driver.** `docs/uart_selftest.md` documents a
  procedure whose host script (`uart_echo.py`) is deleted. `make echo` still runs the RTL
  half; the board half needs a script that no longer exists.
- **`infer.c` no longer relies on uninitialized DRAM.** The KV cache is zeroed once, in the
  static image. The hardware property is unchanged — the mask still takes an unwritten cache
  row to exactly zero — it just is not what the test depends on any more.
