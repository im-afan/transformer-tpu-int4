# Custom TPU (FPGA)

A matrix/attention accelerator for the transformer in [`../../model`](../../model), written
in **SystemVerilog** and running on a **Digilent Cmod A7-35T** (`xc7a35t-cpg236-1`) via a
non-project Vivado batch flow.

The PyTorch model is the golden reference. Every block is validated by simulating it
against vectors produced from that reference.

## Current state

- **One command producer.** PicoRV32 firmware built out of [`fw/`](fw/README.md) pushes
  128-bit macro-ops through an MMIO aperture into per-unit queues. The scalar unit, the
  `.tpu` language and the assembler are deleted.
- **The whole model runs on it.** [`../test/tests/infer/infer.c`](../test/tests/infer/infer.c)
  is prefill + KV-cached decode with the argmax and the embedding gather on the device; the
  host tokenizes and nothing else.
- **It fits.** Last `make bit`: **17 606 LUTs of 20 800 (85%)**, 10 903 FFs, 4 RAMB36 +
  64 RAMB18, 46 DSPs, WNS **+34.98 ns**, all constraints met. That bitstream predates the
  last few RTL edits — re-synthesize before quoting it. See [`docs/synth.md`](docs/synth.md).
- Board geometry is an **8x8 array**, 64 KB scratchpad (`ADDR_W=16`), 512 KB external SRAM
  (`MEM_ADDR_W=19`), 12 MHz core clock.

### One shape, in one place

`infer.c` computes no addresses of its own. `accel/test/export.py` derives the shape from
the checkpoint, computes the whole DRAM map, and writes `infer_config.h`; the kernel
includes it. There is nothing left to keep in step by hand, and no shape `-D` knobs in
`fw/Makefile`.

## Directory layout

| Path | Contents |
| --- | --- |
| `rtl/` | Synthesizable SystemVerilog: `mxu.sv`, `vpu.sv`, `scratchpad.sv`, `dma.sv` + `sram.sv`, `cpu_subsys.sv` (PicoRV32 + AXI4-Lite), the `cmd_*.sv` queues, the UART blocks, and `tpu_top.sv`. No `rtl/luts/` — the activation ROMs went with the removed VPU ops, so the design reads no `$readmemh` file by default. |
| `fw/` | The firmware **library**: `tpu.h` (MMIO + one builder per command), `tpulib.h` (size-independent primitives), `start.S`, `memops.c`, linker script, `bin2hex.py`, Makefile. The kernels themselves live with their vectors in `../test/tests/`. Needs a RISC-V cross gcc. |
| `tb/` | Icarus testbenches, one per block (`make TEST=mxu`, `make all`), plus two that run any C kernel through the whole core: `fw_matmul_tb.sv` backdoors the image in, `fw_uart_tb.sv` loads it over the simulated serial link. Both are driven by `accel/test`'s `RTLBackend`, not by this Makefile. `core.f` is the whole-core file list they share. |
| `synth/` | Vivado non-project build (`synth/vivado/build.tcl`), per-board definitions under `synth/vivado/boards/<board>/`. Output in `synth/build/` (gitignored). |
| `constraints/` | One `.xdc` per target board. |
| `docs/` | Per-block design notes. Start at [`docs/README.md`](docs/README.md). |
| `sim/` | Placeholder for simulator artifacts (gitignored). |

Four board targets under `synth/vivado/boards/`: `cmod_a7` (the real design), `cmod_a7_mem`
and `cmod_a7_bram` (memory-path bring-up), `cmod_a7_echo` (the UART self-test image).
**Reflash `board=cmod_a7` before running anything on `-b board`** — it times out against
the echo bitstream, and `synth/build/` is not rebuilt by `mode=program`.

## The software layers

- [`fw/tpu.h`](fw/tpu.h) — the MMIO aperture and one builder per command. No abstraction:
  fields are packed exactly as `cmd_mxu.sv` / `cmd_vpu.sv` / `cmd_dma.sv` decode them.
- [`fw/tpulib.h`](fw/tpulib.h) — **size-independent primitives** over that: a matmul that
  blocks in rows, columns and the contraction and stages whatever is in DRAM; chunked
  elementwise pairs; transposes and 2-D block moves. Every primitive is self-fencing, so
  composing two is always safe.
- [`../test`](../test/README.md) — the bit-exact ISS, the three backends that run a kernel
  (native/ISS, RTL, board), the per-kernel vector generators, and the checkpoint exporter.

## Known gaps

- The inter-TPU LINK (`wrneigh`) is stubbed in `tpu_top.sv` and completes as a no-op.
- `UART_RX_TIMEOUT` is 0 in every board definition, so a corrupted frame wedges the
  receive FSM until reflash. Setting it to `20 * UART_CPB` makes a corrupted frame cost
  one legible timeout instead. That is hardening — the actual corruption bug was on the
  host and is fixed (see [`docs/uart_host.md`](docs/uart_host.md)).
- Nothing overlaps except the weight prefetch in `tpu_matmul`: every other primitive
  fences after each cross-unit dependency. `docs/scheduler_plan.md` is the sketch for
  doing better in hardware.
