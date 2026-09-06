# Int4 Transformer + TPU

A small decoder-only transformer that does multi-digit addition, + custom
SystemVerilog TPU that runs it end to end on an FPGA.

- Everything (weights and activations) is int4, trained with QAT.
- The TPU runs on a Digilent Cmod A7-35T and generates answers autoregressively:
  prefill then decode with a KV cache.

## Layout

```
model/            PyTorch training & architecture
  transformer.py    architecture
  numbers_data.py   addition dataset + tokenizer
  train.py          training loop
  make_dummy_checkpoint.py  untrained checkpoint for plumbing tests & benchmarking
  saved/            checkpoints

accel/
  tpu/              the TPU
    rtl/ tb/          design + iverilog module testbenches
    fw/               the firmware library for the on-chip PicoRV32, with primitives like elementwise ops and tiled matmuls 
    synth/ constraints/  Vivado build + board definitions
    docs/             per-block design notes
  test/             the verification suite
    iss.py            bit-exact model of the three units
    backends.py       ISS / RTL / board, three ways to run a kernel
    export.py         checkpoint -> config header + DRAM image
    tests/<name>/     one kernel and its vector generator 
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
python accel/test/run_suite.py                 # every kernel, on the ISS
python accel/test/run_suite.py -b rtl          # ...through the whole core in Icarus
python accel/test/tests/infer/generate.py -b iss -n 16     # the model, on the ISS
python accel/test/tests/infer/generate.py -b board -p COM5 # ...on the board
```

`-b rtl` and `-b board` require a baremetal risc-v toolchain; `-b rtl` also needs Icarus.
`-b iss` needs only a host C compiler.
