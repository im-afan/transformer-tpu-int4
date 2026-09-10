# tests/mha_prefill — the prefill on its own

## what it is
- `tests/infer/infer.c`'s layer body, with nothing around it. no decode steps,
  no output head, no argmax, no generated ids
- the counters reset at the launch and freeze at the halt, so an image that runs
  only the prefill *is* the measurement of the prefill. same trick `infer`'s
  `--phase` uses, one level down
- `--part` splits the layer itself: `attn` builds only attention, `ffn` only the
  feed-forward, `block` both. three images whose clocks subtract
- every shape is a knob: `-d`, `-f/--dff`, `-L/--layers`, `--heads`, `--prompt`,
  `--batch`, `--block`
- the weights are mixed hashes, not a checkpoint. this measures a shape, and a
  prefill's clock count is not data-dependent

## what the kernel is
- `infer_block(rows, first_pos)`, copied from `infer.c` and kept in step with
  it. Q off X, K and V spilled straight into the cache, per-head
  `relu(QK^T + mask)` then `P@V`, `A@Wo` with the double residual, then the two
  FFN matmuls with the second DyT
- the whole prompt in `PROMPT / BLOCK` passes plus a tail, exactly as `infer.c`
  chunks its prefill. a pass runs every layer before the next starts, so the
  cache a later pass attends over is complete at every depth
- the prompt reaches the CPU the same way: a DMA into the scratchpad mailbox,
  then a load through the `0x9xxx_xxxx` window per row, and one embedding gather
  per row
- the same three-rung `INFER_MM_MODE` ladder (`--mm base|dbuf|fused`) and the
  same `-DINFER_MM_WIDE=0` A/B (`--general`)

## the two halves
- `PART_ATTN` and `PART_FFN` are compile-time, and whichever half runs last
  leaves the layer's output in `X_BUF`, so a half-built image feeds itself
- a half image is not the model — the residual chain it runs is shorter, so its
  requant table and its numbers are its own. it is checked against a reference
  that skips the same half, which is what makes it a regression and not just a
  stopwatch
- `attn` + `ffn` is not exactly `block`: they are separate images and each pays
  the prompt load once. a few hundred clocks against a prefill

## the golden
- an integer recompute in numpy over the whole prompt at once, causal
- the kernel contracts over all `T` keys and the cache past the prompt is zero,
  so a key the prompt never wrote scores 0, the mask takes it to -8 and the relu
  to 0. that is why a reference over the prompt alone is exact
- checked: both KV caches over `[0, PROMPT)` for every sequence and every layer,
  plus the rows of X the last pass left. everything is already in DRAM, so the
  kernel spills nothing at the end
- **every requant word is fitted to the accumulators the reference produced**,
  per site per layer, on the first case's prompt and then frozen — the table is
  compiled into the image, so every case runs on one table. a hand-picked shift
  that suits `d=64` collapses a tensor to all-zero at `d=128`, and an all-zero
  golden passes against any datapath at all
- `ID`, `P`, `XO` and `HR` are never fitted. their word is an identity, because
  the add each one feeds takes two operands at one scale

## running it
```
python accel/test/tests/mha_prefill/generate.py -b iss
python accel/test/tests/mha_prefill/generate.py -b rtl -d 128 -f 512 -L 4 --prompt 32
python accel/test/tests/mha_prefill/generate.py -b rtl --part split
python accel/test/tests/mha_prefill/generate.py -b rtl --part attn --bench
```
- `--bench` is timing only: nothing is staged into DRAM and nothing is checked,
  so the run is the kernel and its perf counters. it needs no numpy
- it is in `run_suite.py`'s `SLOW` set, so ask for it by name
- the watchdog is computed from the shape (`watchdog_ns`), because a long Icarus
  run prints nothing until it halts and "slow" and "deadlocked" look identical
  from outside
- the image is 7 568 bytes at `d=128 / f=512 / L=4`, against `infer`'s 14 708 —
  the head, the argmax and the decode loop are what is missing

## keeping it in step
- `infer_block` here is a copy of `infer.c`'s. when one changes the other has to,
  or the numbers stop being infer's prefill
- the requant site list is `export.py`'s `RQ_NAMES`, imported rather than
  rewritten; `INFER_RQ_SITES` is what catches the enum drifting from it
