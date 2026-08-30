# `iss.py` — design notes

The bit-exact model of the three units, in `accel/test/iss.py`. It is what the
golden vectors are computed on and what `RTLBackend` checks the hardware
against; how it is driven is in [`../../test/README.md`](../../test/README.md),
and how a checkpoint reaches it is in [`pipeline.md`](pipeline.md).

## What it models

Executes the commands a firmware image pushes, against a model of the
scratchpad, DRAM and the three units, matching `accel/tpu/rtl/*.sv`. Every
dispatch is atomic — read operands, compute, write the result back — so no
cycle-level modelling is needed to reproduce the memory state a real run
leaves behind. `exec_command` and `run_trace` are the entry points.

It exists to produce the golden vectors the testbenches check the DUT
against: load the same program and input tensors, run this, and the final
DRAM image is what the hardware must reproduce byte for byte.

Numerics are bit-exact with the RTL. Everything is int4, packed two per byte,
low nibble first — the MXU's three operands and the VPU's alike:

- MXU: `C[N][N] = requant(A[N][len] @ B)`, int32 accumulate, requant to int4
  forced on store (`mxu.sv requant4`). `accumulate` adds the stored int4 back
  and re-clips, so it is a fused int4 add, not an int32 partial.
- VPU: the narrow is fused into every op (`vpu.sv narrow4`); `DYT` is the same
  fixed point with a symmetric ±7 clip, which is its hardtanh rather than an
  approximation of it. `DOT` is the one op writing int32.

DMA is a 2-D byte copy between DRAM and the scratchpad, `rows` rows of `len`
int4 with an independent row stride on each side (`dma.sv`). The two memories
are different sizes: the scratchpad is `2**addr_w` and DRAM is `2**mem_addr_w`
(the Cmod A7's 512K x 8 part), so each side of a transfer is masked in its own
space (`_a` vs `_d`). A spill is what makes a byte host-visible, so the golden
outputs are exactly the DRAM bytes it wrote (tracked in `dram_written`).

VPU opcode gaps (2, 4-9, 11-15, 17) are retired holes from the softmax/
LayerNorm/GELU datapath and the old QUANT4 slot; they are not reused, so a
stale binary decodes to an unknown op rather than to a different one.

`_d` is distinct from `_a` because the two memories are different sizes; a
DRAM address masked to 16 bits would alias into the low window instead of
reaching the rest of the chip — the one way this model could silently
disagree with the RTL.

`dyt4`'s clip is asymmetric because `hardtanh` is odd, so its floor has to be
the negative of its ceiling; int4's -8 would put the saturated end at
-8/7 = -1.143. The multiplier carries `alpha * s_in * 7`, which is what makes
the clip coincide with the hardtanh rather than merely resemble it.

`run_trace`'s WAIT is a no-op deliberately. This model has no concurrency: it
retires every command completely before looking at the next, which is the
strongest ordering any barrier placement can produce. So firmware that omits
a needed cross-unit barrier still gets correct golden images out of this —
and then diverges on the RTL, where the three queues really are independent.
Keeping the images a statement about intent, and letting the RTL run test
ordering, is what makes a mismatch mean something specific.

`exec_command` discards an unknown opcode rather than erroring, because
`cmd_{mxu,vpu,dma}.sv` pop it with a `$display` and carry on — a model that
raised instead would disagree with the hardware about a malformed stream.
