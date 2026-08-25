# TPU design docs

Per-block notes. Where one of these disagrees with the RTL, the RTL wins.

| Doc | Component |
| --- | --- |
| [scratchpad.md](scratchpad.md) | On-chip BRAM working memory: banking, ports, arbitration |
| [mxu.md](mxu.md) | Weight-stationary int4 systolic array |
| [vpu.md](vpu.md) | SIMD vector unit — the six ops it has, and what was removed |
| [dma.md](dma.md) | DMA engine and the external SRAM controller ("DRAM") |
| [macro_ops.md](macro_ops.md) | The macro-op ISA: what each command carries |
| [picorv32_migration.md](picorv32_migration.md) | The dispatch plane and the CPU producer — design record + measurements |
| [uart_host.md](uart_host.md) | The host link: frame format, the five commands, arbitration |
| [uart_selftest.md](uart_selftest.md) | The `cmod_a7_echo` bring-up image |
| [synth.md](synth.md) | Vivado build flow, Cmod A7-35T deployment, sizing |
| [comms.md](comms.md) | 2D inter-TPU link — **designed, not built** |
| [scheduler_plan.md](scheduler_plan.md) | Sketch for overlapping DMA and compute in hardware — **not built** |
| [development_notes.md](development_notes.md) | Dated working notes |

Firmware is documented in [`../fw/README.md`](../fw/README.md); the host driver in
[`../host/README.md`](../host/README.md).

## The machine in one page

```
  cpu_subsys (PicoRV32 + AXI4-Lite MMIO aperture)
     |  128-bit macro-ops pushed into per-unit queues
     +--> cmd_mxu --> mxu    weight-stationary int4 systolic array
     +--> cmd_vpu --> vpu    SIMD pointwise / narrowing
     +--> cmd_dma --> dma <--> sram_controller <--> external SRAM ("DRAM", 512 KB)
                       |
              scratchpad (BRAM, 64 KB) <-- all three units
```

### 1. Scratchpad

Banked BRAM working memory. Every unit reads and writes it; DRAM is only reachable through
the DMA. Wide enough to feed one full activation column per clock into the MXU.

### 2. MXU

Weight-stationary systolic array. Loads packed **int4** weights from the scratchpad into
per-PE registers, streams int8 activations in staggered, writes results back to the
scratchpad. `matmul_t` walks a tile grid in hardware.

### 3. VPU

SIMD, one scratchpad access feeding many ALUs. Six ops and nothing else: `DOT`, `ADD`,
`RELU`, `REQUANT`, `DYT`, `QUANT4`.

`QUANT4` is the important one: `requant`'s fixed point clipped to `[-8, 7]` and written
**four bits wide**, two elements per byte, in the MXU's packed weight encoding. That is
what lets an *activation* be a weight operand, and therefore what put `Q@K^T` and `P@V` on
the array instead of the VPU.

The unit is deliberately no larger than this — activation LUTs, broadcast/scalar ops, the
divider, reductions, `vecmatmul` and the `softmax` macro op were all removed once the model
stopped needing them. See [vpu.md](vpu.md).

### 4. DMA + SRAM

`sram.sv` moves **ranges**, not bytes: one request is a start address, a byte count and an
address stride. 1 clock/byte on fills, 2 on spills. The DMA also does a **transposing**
mode, which is how a column-major KV cache gets appended to.

### 5. Command plane

A dispatch is a 128-bit macro-op carrying its own operands and geometry, pushed into that
unit's queue. Nothing an earlier dispatch left in a register can reach it, and the units can
run at once — scratchpad access is arbitrated with real grants.

Retired opcodes are **holes, not free space**: `0x02`, `0x05`, `0x0A`–`0x0F`, `0x1B`,
`0x1E`, `0x20` are not reallocated, so a stale binary decodes to an unknown op rather than a
different one. Same for VPU op selector 13, VPU command `0x02`, and `cfg` indices 9 and
10–14. `0x22` is `quant4`; new ops go at `0x23`+.

### 6. Inter-TPU link

Designed in [comms.md](comms.md), **not built**. `tpu_top.sv` ties the `nb_*` port off, so
`wrneigh` completes as a no-op.
