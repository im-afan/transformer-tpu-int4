# Bitnet Transformer TPU

A small decoder-only transformer that does multi-digit **addition**, plus a custom
SystemVerilog TPU that runs it end to end on an FPGA.

- Everything is **int4** — weights *and* activations, trained with QAT.
- The PyTorch model in `model/` is the golden reference. Every accelerator is checked
  against it numerically.
- The TPU runs on a **Digilent Cmod A7-35T** and generates answers autoregressively:
  prefill, then decode against a KV cache, with the argmax and embedding lookup on the
  device.

## Layout

```
model/            PyTorch golden reference
  transformer.py    architecture + named configs
  numbers_data.py   synthetic addition dataset + tokenizer
  train.py          training loop (gradient accumulation)
  quant.py          hardware-exact int8 benchmark (legacy, ternary-era)
  calibrate.py      PTQ activation calibration (legacy, ternary-era)
  make_dummy_checkpoint.py  untrained checkpoint for plumbing tests
  saved/            checkpoints (gitignored)

accel/
  cuda/             legacy CUDA MHA kernel (out of sync with the model)
  tpu/              the TPU
    rtl/ tb/          design + Icarus testbenches
    fw/               C firmware for the on-chip PicoRV32 — the only command producer
    host/             UART driver and board runners
    synth/ constraints/  Vivado build, per-board definitions
    docs/             per-block design notes
  tpulang/          ISS (bit-exact with the RTL), golden vectors, checkpoint exporters
```

## Running

Python is a package rooted at the repo, so run from the repo root with `-m`:

```bash
python -m model.train --arch int4_wide
python -m model.tests.test_inference --arch int4_wide
python -m model.make_dummy_checkpoint          # untrained .pt, for plumbing only
```

TPU stack (also from the repo root):

```bash
make -C accel/tpu/fw PROG=infer                # build a kernel (needs RISC-V gcc)
cd accel/tpu/tb && make fw FWPROG=infer GEN=3  # run it through the RTL
python accel/tpulang/infer_export.py -n 16     # score it on the ISS
python accel/tpu/host/run_adder.py -p COM5     # run it on the board
```

## Where to read next

| Doc | Covers |
| --- | --- |
| [`model/README.md`](model/README.md) | The reference model: data format, architecture, quantization |
| [`accel/README.md`](accel/README.md) | How the backends relate |
| [`accel/tpu/README.md`](accel/tpu/README.md) | The hardware: layout and current state |
| [`accel/tpu/fw/README.md`](accel/tpu/fw/README.md) | The kernels and the primitive library |
| [`accel/tpu/host/README.md`](accel/tpu/host/README.md) | The UART link and the board runners |
| [`accel/tpu/docs/README.md`](accel/tpu/docs/README.md) | Per-block microarchitecture notes |
