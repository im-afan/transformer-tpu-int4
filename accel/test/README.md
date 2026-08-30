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

| | builds | runs on | sees all of DRAM | command trace | cycle counters |
| --- | --- | --- | --- | --- | --- |
| `iss` | host `cc`, `-DTPU_TRACE` | `iss.py` | yes | yes | no |
| `rtl` | RISC-V gcc | Icarus, whole core | yes | yes | yes |
| `rtl-uart` | RISC-V gcc | Icarus, loaded over the simulated link | yes | yes | yes |
| `board` | RISC-V gcc | the Cmod A7 | no | no | yes |

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
    defines: dict                  # -D flags the kernel is compiled with
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
then per case writes the patch, runs, and compares. `read_timers()` gives the per-case
counters; `passed()` is the verdict.

## Running it

```bash
python accel/test/run_suite.py                      # every kernel, on the ISS
python accel/test/run_suite.py -b rtl               # ...through the whole core, ~20 s
python accel/test/run_suite.py -b rtl -k tiled -v   # one, with the simulator's output
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
2. Write `generate.py`: subclass `VectorGenerator`, implement `static()` and `cases()`,
   and expose `program(backend)` plus a `main()` built on `standard_parser` /
   `backend_from_args` / `report`.
3. Add its watchdog to `run_suite.WATCHDOG_NS` if it runs longer than a few thousand
   clocks. A watchdog that never fires on a hang is worth nothing.

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
