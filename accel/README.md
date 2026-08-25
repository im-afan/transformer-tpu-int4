# Accelerators

Hardware/backend implementations of the operations in [`../model`](../model). The PyTorch
model is the golden reference; every backend is checked against it numerically.

| Directory | What it is | State |
| --- | --- | --- |
| [`tpu/`](tpu/README.md) | Custom SystemVerilog TPU on a Digilent Cmod A7-35T, plus its firmware, host driver and Vivado build | **live** — runs the whole model |
| [`tpulang/`](tpulang) | The TPU's software stack: bit-exact ISS, golden-vector generator, checkpoint exporters | **live** |
| `cuda/` | Hand-written CUDA MHA/GQA kernel | **legacy** — implements softmax attention; the model uses ReLU attention, and nothing in `model/` loads it |

## The TPU has one command producer

PicoRV32 firmware in [`tpu/fw/`](tpu/fw/README.md) pushes 128-bit macro-ops through an
MMIO aperture into per-unit queues. There is no assembler and no `.tpu` language.

Deleted with the scalar unit: `scalar_unit.sv`, `assembler.py`, `gen_vectors.py`,
`torch_ref.py`, `pytpu.py`, every `examples/*.tpu`, the `.tpu` testbenches and `isa.md`.
Anything describing a `.tpu` program or the scalar unit is history.

`iss.py` survived: its op bodies are still the golden numerics, now driven by command
traces instead of by an instruction decoder.

## `tpulang/` — four files

- `iss.py` — bit-exact with the RTL. `exec_command` / `run_trace` are the way in.
- `fw_vectors.py` — runs a firmware kernel's native build as a co-process, executes each
  command on the ISS as it arrives, and emits golden DRAM images + the expected command
  stream. Also the single definition of each kernel's synthetic operands.
- `adder_export.py` — real checkpoint → requant table → trace → accuracy, teacher-forced.
- `infer_export.py` — the same checkpoint *generating*, through `fw/infer.c`.

The directory name is a fossil.
