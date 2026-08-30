# The macro-op ISA

The TPU has one instruction format: a **128-bit command**, four RV32 stores, uniform across
all three units. A command carries its own operands and geometry, so nothing an earlier
dispatch — or an earlier *program* — left in a register can reach it.

There is one producer: PicoRV32 firmware in [`../fw/`](../fw/README.md). `fw/tpu.h` has one
builder per command and is the authoritative encoding; `cmd_mxu.sv` / `cmd_vpu.sv` /
`cmd_dma.sv` are the decoders.

## Format

Four 32-bit words, `word0` first. The header lives in the **low** bits of `word0`, not the
high bits of the 128 — so a producer builds it with an `ori` against a small constant rather
than a shift, and the first store carries it.

| `word0` bits | Field | Meaning |
| --- | --- | --- |
| 7..0 | `op` | command selector, per-unit namespace |
| 15..8 | `flags` | `.acc` / `.rq` / `tiled` / `write` / `.t`, per op |
| 31..16 | — | the first operand field (an address, in every command that has one) |

| Command | `op` | `word0[31:16]` | `word1` | `word2` | `word3` |
| --- | --- | --- | --- | --- | --- |
| `MXU_GEOM` | `0x01` | `arow` | `wcol:crow` | `tlen(6) : ntiles(8) : ktiles(8)` | — |
| `MXU_MM` | `0x02` | `out` | `wgt:act` | `{n,m0}` | — |
| `VPU_OP` | `0x01` | `dst` (`flags[4:0]` = the `VOP_*` code) | `src1:src0` | `{n,m0} : vlen(10)` | — |
| `DMA_MOVE` | `0x01` | `spad` (`flags[0]`=write, `flags[1]`=`.t`) | `dram(19)` | `tcols:len` | `tdrow:tsrow` |

- **Commit on the last word.** Each unit's command port is a 4-word aperture; the write to
  offset `0xC` enqueues. Four stores are atomic with respect to the queue, with no separate
  trigger store, and a partial write is detectable.
- **An unknown `op` is discarded with a simulation message rather than executed**, so a
  stale command stream fails loudly instead of running the wrong instruction.
- `VPU_GEOM` (`0x02`) is retired, so the VPU has exactly one command type.

### Why the MXU needs two commands

A full `matmul_t` is `op/flags` + three 16-bit addresses + `{m0,n}` + `tlen`/`ktiles`/
`ntiles` + `arow`/`crow`/`wcol` = **146 bits**. Squeezing to 128 needs 12-bit strides and
6-bit tile counts and lands at exactly 128 with zero margin. Splitting is cheaper:

| Command | Fields | Used of 112 |
| --- | --- | --- |
| `MXU_GEOM` | `tlen`(6), `ktiles`(8), `ntiles`(8), `arow`(16), `crow`(16), `wcol`(16) | 70 |
| `MXU_MM` | `out`(16), `act`(16), `wgt`(16), `{m0,n}`(16), flags | 64 |
| `VPU_OP` | `dst`(16), `src0`(16), `src1`-or-`{m0,n}`(16), `vlen`(10) | 58 |
| `DMA_MOVE` | `spad`(16), `dram`(19), `len`(16), `tcols`(16), `tsrow`(16), `tdrow`(16) | 99 |

**Sticky geometry is queue state, not global state, and that is the actual fix.**
`MXU_GEOM` persists until the next one, but it flows through the MXU's own in-order queue,
so its scope is a position in that unit's command stream rather than a register any other
unit or any earlier program can have written. The driver caches the last geometry it emitted
and skips the command when nothing changed.

### The requant word is a literal

`{m0,n}` is 12 + 4 = 16 bits, exactly the width of the scratchpad address that used to point
at it. Same command budget, one less indirection, one less scratchpad read per dispatch — and
it deleted the block of 16 words the host used to stage per layer, plus a two-state fetch in
both `mxu.sv` and `vpu.sv`.

The consequence for a kernel: **the requant table has to be in the firmware image**, because
the device has no path by which it could fetch it from memory. See
[`../fw/README.md`](../fw/README.md).

### Uniform 128 bits, not a packed encoding

Measured at about 1% of total runtime against a tightly packed variable-width encoding, and
worth it: four unconditional stores per dispatch with no shifting or field assembly.

## Queues and synchronization

One FIFO per unit, depth 8. **A full queue stalls the store** (AXI `BVALID` withheld), so
flow control needs no software check — the CPU throttles itself and only *data* dependencies
need explicit waits.

Sync is by sequence number, not by "is the unit idle":

| Register | Meaning |
| --- | --- |
| `ISSUED[unit]` | commands accepted into the queue |
| `RETIRED[unit]` | commands completed |
| `SPACE[unit]` | free entries (advisory; the stall is the real mechanism) |

Waiting is `while (RETIRED[u] < my_seq)` — `tpu_wait(unit)` in `tpu.h`. Waiting on a
*specific* command rather than on unit idleness is what makes double-buffering expressible:
start the fill for block *n+1*, then wait only for the fill that staged block *n*.

**Every cross-unit dependency is software's responsibility.** Issue-and-wait made them
automatic; queues do not. `tpulib.h`'s primitives are all self-fencing for that reason, and
the one place two units deliberately run at once is its weight prefetch.

### `cmd_queue.sv` implementation notes

Storage is a plain 2-D array indexed by two wrapping pointers, which infers LUTRAM
(distributed) at depth 8 rather than a block RAM. `head` is a combinational read of the
storage, so a pop and the next command's decode are back to back with no bubble.

A dropped write ($display in sim) means the producer pushed through `full` instead of
stalling on it — that should never happen given the AXI-level stall above, so seeing the
message means the flow-control assumption broke somewhere.

## Address map

| Base | Size | Contents |
| --- | --- | --- |
| `0x0000_0000` | 16 KB | firmware: code, data and stack. BRAM, host-loadable over `'I'` |
| `0x8000_0000` | 16 B | MXU command port (commit on `+0xC`) |
| `0x8000_0100` | 16 B | VPU command port |
| `0x8000_0200` | 16 B | DMA command port |
| `0x8000_0300` | 64 B | `ISSUED` / `RETIRED` / `SPACE` per unit, `DONE`, perf counters |
| `0x9000_0000` | 64 KB | scratchpad, a 32-bit word window onto the `S_rw` port |

- **The scratchpad window is worth more than the instruction it replaced.** It lets a kernel
  compute requant words at run time, read back a reduction, and — with an argmax loop in C —
  end with a token id instead of 2 KB of int32 logits. `fw/infer.c` does exactly that, and
  the embedding lookup follows from it, so the host is out of the decode loop entirely.
- **A DRAM window is deliberately not mapped.** The SRAM is byte-wide at 1 clock/byte; CPU
  access to it would be a trap. DRAM stays the DMA's.

## PicoRV32 configuration

| Parameter | Setting | Why |
| --- | --- | --- |
| `BARREL_SHIFTER` | 1 | shifts drop from 4–14 clocks to 3; address math uses them constantly |
| `ENABLE_FAST_MUL` | 1 | the alternative is 40 clocks per `mul`, and tile-offset math multiplies |
| `ENABLE_DIV` | **0** | no kernel needs it — **a `div` or `rem` traps**, so build `-march=rv32ic_zmmul`, not `rv32imc` |
| `ENABLE_COUNTERS` | 1 | `rdcycle` gives firmware self-profiling for free |
| `COMPRESSED_ISA` | 1 | ~30% smaller firmware; the first thing to drop if area gets tight |
| `ENABLE_IRQ` | 0 | polling `RETIRED` is simpler and the latency does not matter |

Bus topology is `picorv32_axi` — firmware RAM, command ports, status block and the
scratchpad window are all AXI4-Lite slaves. Every instruction fetch pays the AR/R handshake,
so CPI is ~6 rather than ~4. The alternative (native core, tightly coupled RAM, a bridge on
the MMIO aperture only) is worth ~2% of runtime and is not built. Switch when `idlec` says
CPU issue is over ~10% of a run; it is 7% today.

## Retired opcodes are holes, not free space

Nothing is reallocated, so a stale binary decodes to an unknown op rather than a *different*
one.

| Namespace | Holes |
| --- | --- |
| Scalar opcodes | `0x02`, `0x05`, `0x0A`–`0x0F`, `0x1B`, `0x1E`, `0x20` |
| VPU op selector | 2, 4–9, 11–15 (13 was `vecmatmul`) |
| VPU command | `0x02` (`VPU_GEOM`) |
| `cfg` indices | 9 (`vscalar`), 10–14 (`vecmatmul` geometry) |

`cfg` 9–14 stay vacant because renumbering would silently repoint 15–17, the DMA transpose
geometry. `0x22` is `quant4`; **new ops go at `0x23`+**.

## What is built

| Piece | Where | Test |
| --- | --- | --- |
| Per-run performance counters | `rtl/perf_counters.sv` | decoded from the `'T'` reply, checked per kernel |
| MXU config strides | `mxu.sv` `a_row` / `c_row` / `w_row` | `run_suite.py -b rtl -k tiled` |
| `matmul_t` hardware tile loop | `mxu.sv` | `make fw`, `make fwsweep` |
| 128-bit commands + per-unit queues | `cmd_queue.sv`, `cmd_{mxu,vpu,dma}.sv` | `make TEST=cmd_queue` |
| PicoRV32 producer | `cpu_subsys.sv` + `rtl/vendor/picorv32.v` | `make TEST=cpu_smoke`, every `make fw` |
| CPU scratchpad window | `cpu_subsys.sv` | `run_suite.py -b rtl -k spadwin` |

Not built, and not planned:

- **`softmax`** was built, validated and then **removed** with `EXP`, `REDUCEMAX`,
  `REDUCESUM`, `SCALAR_DIV`, both activation ROMs and `cfg vscalar`. The model is ReLU
  attention.
- **`layernorm`** and its `rsqrt` LUT are dropped. DyT replaced LayerNorm and needs no macro
  op at all — it is `requant` with a symmetric clip, fused into a narrow the residual add
  already required.
- **`vecmatmul`** was built, validated, measured and **removed**. It was 88.3% of all VPU
  time, so packing K and V to int4 moved both attention matmuls onto the array; once
  `Model.fc` went int4 the op had no caller. `VOP_DOT`, the datapath it wrapped, is untouched.
- **The inter-TPU LINK.** `wrneigh` is a completing no-op. See [comms.md](comms.md).

## What the tile loop bought

Measured instruction counts, same problem:

| Kernel | Software loop | Macro-op |
| --- | --- | --- |
| 8x32 @ 32x16 tiled matmul | 70 words | **25** — and that does it *twice* (int32 and requantized) |

And measured on the firmware producer (`make fwsweep`, 11 shapes):

- **The hardware tile walk makes CPU cost O(1) in the problem.** `matmul.c` issues five
  commands whatever the shape, and its `idlec` is 370–409 clocks across a **205x** range of
  array work.
- **The C loop costs ~85 clocks per dispatch** — 13 instructions, four AXI4-Lite stores, and
  every instruction fetch over the same bus. What that costs the run is
  `max(0, cpu - array)`, so it is **zero once M > ~24** on the 8x8 array.

Full tables in [picorv32_migration.md](picorv32_migration.md) §Measurements.
