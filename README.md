# Int4 Transformer + TPU

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
  make_dummy_checkpoint.py  untrained checkpoint for plumbing tests
  saved/            checkpoints (gitignored)

accel/
  cuda/             legacy CUDA MHA kernel (out of sync with the model)
  tpu/              the TPU
    rtl/ tb/          design + Icarus block testbenches
    fw/               the firmware library for the on-chip PicoRV32 — the only
                      command producer: tpu.h, tpulib.h, start.S, link.ld
    synth/ constraints/  Vivado build, per-board definitions
    docs/             per-block design notes
  test/             the verification suite
    iss.py            bit-exact model of the three units
    backends.py       ISS / RTL / board — three ways to run a kernel
    export.py         checkpoint -> config header + DRAM image
    tests/<name>/     one kernel and its vectors per folder
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
python accel/test/run_suite.py                 # every kernel, on the ISS (seconds)
python accel/test/run_suite.py -b rtl          # ...through the whole core in Icarus
python accel/test/tests/infer/generate.py -b iss -n 16     # the model, on the ISS
python accel/test/tests/infer/generate.py -b board -p COM5 # ...on the board
```

`-b rtl` and `-b board` need a bare-metal RISC-V gcc; `-b rtl` also needs Icarus.
`-b iss` needs only a host C compiler.

## Where to read next

| Doc | Covers |
| --- | --- |
| [`model/README.md`](model/README.md) | The reference model: data format, architecture, quantization |
| [`accel/README.md`](accel/README.md) | How the backends relate |
| [`accel/tpu/README.md`](accel/tpu/README.md) | The hardware: layout and current state |
| [`accel/tpu/fw/README.md`](accel/tpu/fw/README.md) | The kernels and the primitive library |
| [`accel/test/README.md`](accel/test/README.md) | The verification suite: the three backends and the vector contract |
| [`accel/tpu/docs/pipeline.md`](accel/tpu/docs/pipeline.md) | Checkpoint to board, end to end |
| [`accel/tpu/docs/README.md`](accel/tpu/docs/README.md) | Per-block microarchitecture notes |
