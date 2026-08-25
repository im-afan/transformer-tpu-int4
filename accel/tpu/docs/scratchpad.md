# Scratchpad Memory

On-chip working memory, built from FPGA BRAM. Every unit reads and writes it; DRAM is only
reachable through the DMA. `rtl/scratchpad.sv` is the single owner of the storage.

**64 KB on the Cmod A7** (`ADDR_W = 16`).

## 1. What it holds

- Whatever the current kernel wants resident — a staged weight block, a token block's
  activations, int32 partials.
- Enough read bandwidth to feed **one full activation column per clock** into the MXU.
- The commands name scratchpad byte addresses, never DRAM addresses.

`fw/infer.c` uses it as a **staging arena plus a 320-byte mailbox**, not as a static map:
every tensor's home is DRAM, and a scratchpad copy is a compile-time promotion the kernel
can lose. Nothing asserts that an activation fits on chip.

The mailbox exists because the CPU has a window onto the scratchpad (`0x9xxx_xxxx`) and no
path to DRAM at all — that is where the head's logits and the token sequence live.

## 2. Datatypes

| Region | Element | Encoding |
| --- | --- | --- |
| Activations | int8 container | int4 values, `[-8, 7]` |
| Weights | int4 | 4-bit two's complement, **2 per byte, row-major** |
| Accumulators | int32 | MXU/VPU results before narrowing |

Narrowing (int32 -> int4-in-int8, or -> packed int4) happens on the store path out of the
MXU/VPU. The scratchpad stores whatever width the writer produces.

## 3. Ports

| Port | Width | Driven by |
| --- | --- | --- |
| `A_rd` | `ROWS` int8 | `mxu.sv` activation feed — one column per clock |
| `W_rd` | `COLS` int4 | `mxu.sv` weight load |
| `C_rw` | `COLS` int32 | `mxu.sv` result store + accumulate readback, per-byte strobes |
| `V_rw` | `VPU_BYTES` | `vpu.sv` SIMD read/modify/write |
| `S_rw` | 32 bit | the CPU's scratchpad window (`cpu_subsys.sv`) |
| `DMA` | bus width | `dma.sv`, DRAM <-> scratchpad |

**Contract.** Byte-addressed; each port gathers/scatters its own byte width. Reads are
synchronous — `*_rdata` valid the cycle after `*_re`. Writes are single-cycle under a
per-byte strobe. A read and a write to the same byte in one cycle returns the **old** value
(read-first). A window running off the top wraps mod `2**ADDR_W`.

**One read and one write per cycle.** The six read ports mux onto a single banked-BRAM read
port and the four write ports onto a single write port. A read and a write proceed together
— they are the two ports of the same BRAM — but two simultaneous reads, or two simultaneous
writes, are not supported.

## 4. Arbitration is real now

The old exclusivity invariant held only while the scalar unit was issue-and-wait: a
dispatch parked it until the unit reported done, so the MXU, VPU and DMA could not overlap.
**Per-unit command queues exist precisely to break that.**

So the priority mux genuinely arbitrates, and every requester that can lose takes a grant
back (`V_rgnt`/`V_wgnt`, `s_rgnt`/`s_wgnt`, `dma_rgnt`/`dma_wgnt`). Both chains are ordered
so the requester that *cannot* stall is first:

```
reads    A > W > C > V > s > DMA
writes   C > V > s > DMA
```

All three MXU ports are at the top because its writeback drains `result_buf` with no
handshake at all.

- The VPU freezes its FSM for a clock when denied.
- The CPU's S port re-presents.
- The DMA parks the byte in a skid buffer and pauses the SRAM read stream
  (`sram.sv`'s `dout_ready`).

**If you make two units run at once, check every path into the scratchpad takes its grant
back.** A denied requester that ignores it loses the access silently.

One visible consequence: `*_rdata` are slices of a single shared output register, so a read
on any port updates all of them. Each consumer samples one cycle after its own enable with
nothing else reading, so behaviour is unchanged — but the ports are no longer independently
held.

## 5. Banking

Storage is split into `NBANK` one-byte-wide banks (`NBANK` = the widest port rounded up to
a power of two). Byte address `a` lives in bank `a[OFF_W-1:0]` at row `a[ADDR_W-1:OFF_W]`.

An unaligned `NBANK`-byte window then takes exactly one byte from every bank: bank `b`
needs row `row0` when `b >= off` and `row0 + 1` when `b < off`, so the per-bank address is
a **1-bit adjustment** rather than an arbitrary index. The gathered bytes come back rotated
by `off`, undone by one barrel rotate on the read path (mirrored on the write path).

That replaces `NBANK` 64Ki-to-1 byte multiplexers with `NBANK` dual-port BRAMs plus two
log2(`NBANK`)-stage rotate networks — the difference between "does not fit on any part" and
"fits". A flat byte array with six read ports synthesized to ~524 288 FFs against the
part's 41 600.

Post-route on the Cmod A7: **2413 LUTs, 64 RAMB18, 9 FFs.** The rotate networks are the
LUT cost and are the thing to attack if area gets tight.

### `MEM_STYLE`

Banking is unconditional; `MEM_STYLE` only picks the primitive each bank is built from,
with identical functional and timing behaviour either way.

| `MEM_STYLE` | `ram_style` | Primitive | Use when |
| --- | --- | --- | --- |
| `"BRAM"` (default) | `block` | FPGA block RAM | anything board-sized |
| `"REG"` | `registers` | flip-flops | shallow unit-test depths only |

At `ADDR_W=16`, `"REG"` is the flip-flop explosion the banking exists to avoid. It is there
so `scratchpad_tb.sv` can cross-check the two elaborations.

`make TEST=scratchpad sim` instantiates both side by side against a byte-array reference —
strobed writes, cross-port coherence, an unaligned sweep over every byte offset in a bank
row, address wrap, back-to-back reads on different ports, read-first.

## 6. Model sizing, for reference

Per-layer weight footprint at `d=128`, `f=512`, int4:

| Tensor | Shape | Packed |
| --- | --- | --- |
| Wq/Wk/Wv/Wo | 128 x 128 | 8 KB each |
| FFN `W1` | 128 x 512 | 32 KB |
| FFN `W2` | 512 x 128 | 32 KB |
| **per layer** | | **96 KB** |
| **4 layers** | | **384 KB** |

All of it lives in the 512 KB DRAM and streams per layer. The scratchpad holds one staged
block at a time, which is why the weight prefetch in `tpulib.h` is where the DMA time goes.

## 7. Open questions

- Whether the double-buffering should move into hardware (see
  [scheduler_plan.md](scheduler_plan.md)) rather than being a firmware pipeline.
- Whether to synthesize as **true** dual-port BRAM so the DMA and a compute unit stop
  contending at all.
