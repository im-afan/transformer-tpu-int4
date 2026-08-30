# Accelerators

Hardware/backend implementations of the operations in [`../model`](../model). The PyTorch
model is the golden reference; every backend is checked against it numerically.

| Directory | What it is | State |
| --- | --- | --- |
| [`tpu/`](tpu/README.md) | Custom SystemVerilog TPU on a Digilent Cmod A7-35T: the RTL, the firmware library, the block testbenches and the Vivado build | **live** — runs the whole model |
| [`test/`](test/README.md) | The verification suite: the bit-exact ISS, the three backends, the vector generators, the checkpoint exporter | **live** |
| `cuda/` | Hand-written CUDA MHA/GQA kernel | **legacy** — implements softmax attention; the model uses ReLU attention, and nothing in `model/` loads it |

## The TPU has one command producer

PicoRV32 firmware pushes 128-bit macro-ops through an MMIO aperture into per-unit queues.
There is no assembler and no `.tpu` language.

Deleted with the scalar unit: `scalar_unit.sv`, `assembler.py`, `gen_vectors.py`,
`torch_ref.py`, `pytpu.py`, every `examples/*.tpu`, the `.tpu` testbenches and `isa.md`.
Anything describing a `.tpu` program or the scalar unit is history.

`iss.py` survived: its op bodies are still the golden numerics, now driven by command
traces instead of by an instruction decoder. It lives in `test/`.

## `test/` — one way to run a kernel, on three things

A kernel is a `.c` file and a `generate.py` next to it. The generator writes the operands
and the expected answer; `ISSBackend`, `RTLBackend` and `TPUBackend` are three
interchangeable ways of producing an answer to compare against.

```bash
python accel/test/run_suite.py           # every kernel, on the ISS
python accel/test/run_suite.py -b rtl    # ...through the whole core in Icarus
python accel/test/run_suite.py -b board -p /dev/ttyUSB1
```

The old split — `tpulang/` for vectors and exports, `tpu/host/` for the board — is gone;
so is the fossil directory name. See [`test/README.md`](test/README.md).
