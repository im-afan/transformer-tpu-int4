# Communication Interface

Inter-TPU link that lets one device write into a neighbour's DRAM, for multi-device
inference.

> **Not built.** `tpu_top.sv` ties the `nb_*` port off, so `wrneigh` completes as a no-op.
> There is no command for it in the macro-op ISA either. Everything below is design intent.

## Purpose

Partition a model across several TPUs and move activations between them without host
involvement: a device produces a tile, then pushes it directly into the DRAM of the
neighbour that owns the next stage — a one-sided **remote write**.

## Topology

2D mesh, four neighbours (N/E/S/W), addressed by **direction** rather than a global address.
Edge devices mark unconnected directions absent in a config register, and a write to one
faults. Multi-hop routing is out of scope.

## The transaction

`WriteNeighbor(neighbor, my_addr, neighbor_addr)` — a one-sided put. The initiator reads
`len` bytes from its own DRAM and streams them to the target's DRAM. The target is not
interrupted per word; it polls a doorbell.

```
initiator                        target
read my_addr (DMA) --+
                     +--> link (header + payload)
                     |    header = {dir, neighbor_addr, len}
                     +------------------> RX FSM --> DMA write --> doorbell
```

Suggested flit header: `dst` (2 b), `addr` (32 b), `len` (16 b), `seq` (8 b), then payload.

## Link layer

- **Physical:** one link per edge, width a parameter. PHY choice is board-dependent and
  deferred.
- **Flow control:** credit-based — the receiver advertises free RX-buffer credits, so a slow
  receiver back-pressures rather than dropping data.
- **Ordering / integrity:** per-link sequence numbers plus a per-flit CRC; a failed CRC
  retransmits that flit. Writes from one sender to one neighbour are delivered in order.
- **Clock domain:** its own, with an async FIFO at each end. This is the one place in the
  design that would have a CDC — the DMA and SRAM share a clock deliberately.

## Interaction with memory

`WriteNeighbor` targets **DRAM**, not the scratchpad. A scale-out step looks like:

1. Device A finishes its layers; results are in the scratchpad.
2. A spills the boundary activations to its own DRAM.
3. `WriteNeighbor` pushes them into B's DRAM.
4. B sees the doorbell, fills them into its scratchpad, continues.

Because the write is one-sided, B needs a synchronization signal: a per-neighbour
**doorbell register**, incremented on completion, that B polls or blocks on.

## Interface

| Signal | Dir | Width | Meaning |
| --- | --- | --- | --- |
| `nb_start` | in | 1 | begin a WriteNeighbor |
| `nb_dir` | in | 2 | target direction |
| `src_addr` | in | 32 | local DRAM source |
| `dst_addr` | in | 32 | neighbour DRAM destination |
| `len` | in | 16 | bytes |
| `nb_done` | out | 1 | local send complete |
| `doorbell[4]` | out | — | per-direction inbound-completion flags |
| `link_absent` | out | 4 | which directions are unconnected |

## Open questions

- Link PHY choice and width — gated on a target board with spare pins.
- Whether a lightweight multi-hop router is needed for larger meshes.
- Whether per-flit CRC + retransmit is enough, or an end-to-end ack per `WriteNeighbor`.
- Whether the doorbell should interrupt the CPU or stay poll-only.
