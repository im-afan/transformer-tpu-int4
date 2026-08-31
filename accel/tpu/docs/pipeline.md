# From checkpoint to board

One `.pt`, one `.c`, three backends. The same export feeds all three, so a
disagreement between them is a real disagreement and not two scripts that
drifted.

```
model/saved/*.pt
      |
      |  accel/test/export.py
      |     load_checkpoint   shape read off the state dict
      |     derive            int4 codes + one {m0,n} per requant site
      |     dram_map          every DRAM address, computed in Python
      |     write_config      -> infer_config.h
      |     static_image      -> {addr: byte}: weights, mask, head, embeddings
      v
tests/infer/infer.c  --#include-->  infer_config.h
      |
      +--> ISSBackend    native cc, commands executed on iss.py
      +--> RTLBackend    riscv gcc, the whole core in Icarus
      +--> TPUBackend    riscv gcc, the Cmod A7 over the UART
```

```bash
python -m accel.test.export --model-path model/saved/int4_d128_f512_l4.pt --dump-rq
python accel/test/tests/infer/generate.py -b iss  -n 8 --gen 4
python accel/test/tests/infer/generate.py -b rtl  --synthetic --gen 3 -n 1
python accel/test/tests/infer/generate.py -b board -p /dev/ttyUSB1 -n 8
```

## Stage 1 — export: the checkpoint becomes integers

`derive` reads the shape off the state dict, walks the modules in a fixed
order, and produces two things per layer: the six int4 weight blocks, and the
14 `{m0,n}` words the kernel compiles in.

### Where the multipliers come from

Every tensor carries a compile-time scale; integer `v` means real `v * s`. A
requant's multiplier is `s_in * s_weight / s_out`, and these are the parts that
are not free choices:

- A vector add takes two operands at one scale, so a residual add pins its
  addend's output scale. `RQ_O` and `RQ_XO` land on `s_x`, `RQ_F` on `s_x1`.
  The model expresses that by sharing the `ActQuant` instance (`q_o is q_xo is
  x_quant`), which is why `derive` asserts the pinning rather than assuming it:
  if a retrain stops sharing them, the kernel would silently add two tensors on
  different scales.
- `1/sqrt(head_dim)` is not an op; it folds into `RQ_S`.
- DyT is the `dyt` instruction, not an extra pass: with the output scale pinned
  to 1/7, the saturating narrow the residual add already needed *is* the
  hardtanh, with `alpha` folded into its multiplier.
- The `{1,0}` identities (`RQ_ID`, `RQ_P`, `RQ_XO`, `RQ_HR`) are identities by
  construction, not by tuning: their input is already on the output's grid and
  scale.
- `fixed_point` takes the largest `n` that keeps `m0` under 4096, because every
  extra shift is another bit of precision on a multiplier that is usually much
  smaller than 1.

`INFER_RQ_LOGIT` is not per-layer. The head requantizes on store, so a logit is
int4 and ties are common; the shift is chosen from the largest accumulator a
column can reach — 7 times that column's absolute weight sum — so the grid is
used and nothing clips.

### The map is Python's

`dram_map` computes every address off the shape and `write_config` emits them.
`infer.c` computes none of its own. Two things follow: a shape change is a
regenerated header rather than four files kept in step, and the layer weights
sit immediately above the activations instead of at a hardcoded `0x20000` —
at `d=64 / f=256` the whole map ends at `0x1eac0` of 512 KB.

`static_image` is read-only for the whole run. On the board it is one upload
(~394 KB at `d=128 / f=512`) and every problem after that sends only its
prompt, about 128 bytes. It also zeroes the KV cache: attention contracts over
all `T` keys to keep the block shape constant, so it reads rows no step has
written, and the board's SRAM holds whatever the last run left there.

## Stage 2 — the ISS: run it without hardware

`ISSBackend` compiles the same `.c` with the host compiler and `-DTPU_TRACE`,
which swaps `tpu.h`'s two MMIO primitives for a trace emitter, and runs the
binary as a **co-process**. `infer.c` argmaxes its own logits and the token it
picks lands in the *address* of the next DMA, so its command stream is not a
function of the program alone — the ISS answers each `SRD` out of its own
scratchpad and applies each `SWR` the same way.

Seconds per problem. This is where you iterate.

## Stage 3 — the RTL: does the hardware agree

`RTLBackend` runs an `ISSBackend` alongside itself and hands the testbench two
goldens: every DRAM byte, and the command stream the native build issued. So
the simulation checks the datapath *and* that the RISC-V build pushes the same
128-bit ops the host build did — a diverging command names the producer, a
matching trace with a wrong image names the datapath.

## Stage 4 — the board

`TPUBackend` loads the image with `'I'`, the static image with `'W'`, presses
`'G'`, waits for the core to stop NAK'ing, reads the `'T'` counters, and reads
the result with `'R'`. There is no completion signal on this link; the idle
probe is it.

`'T'` is read before the result so the number does not depend on how long the
readback took, and non-fatally: it is newer than the other four commands, so a
bitstream flashed before it existed NAKs the byte and the run reports no
counters rather than failing.

## Benchmarking

The same counters come back from `rtl` and `rtl-uart`, keyed and ordered the
same way (`tpu_uart.TIMER_COUNTERS`), so the three are one measurement rather
than three that resemble each other. `run_program()` puts them on every
`Result`; `prog.benchmark()` and `run_suite.py --bench` are the readouts. See
[`../../test/README.md`](../../test/README.md#benchmarking).

## Where a failure means what

| fails on | means |
| --- | --- |
| `iss` only | the kernel is wrong, or the golden is. Both are Python and C you can read side by side. |
| `rtl`, command trace | the RISC-V build issues different commands than the native one — an inlining or a fold that differs between compilers. |
| `rtl`, DRAM image | the datapath. The command stream was right. |
| `rtl-uart` only | the load path: `uart_interface.sv`, or the host frames. |
| `board` only | timing, the physical link, or something the simulation's zero baud error cannot see (see [`uart_selftest.md`](uart_selftest.md)). |

## Related

- [`../../test/README.md`](../../test/README.md) — the suite itself
- [`iss.md`](iss.md) — what the model does and does not model
- [`fw.md`](fw.md) — the kernels
- [`uart_host.md`](uart_host.md) — the link
