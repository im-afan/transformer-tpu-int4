# UART Host Interface

A serial link that lets a host PC read and write the external SRAM ("DRAM"), load a firmware
image, start the core and read its performance counters.

Device side is [`rtl/uart_interface.sv`](../rtl/uart_interface.sv) plus `uart_receiver.sv` /
`uart_transmitter.sv`. Host side is [`../host/README.md`](../host/README.md).

- **Host is the sole master.** The FPGA never initiates a frame; it only answers.
- **Byte-serial**, because the link is: at 115200 baud a byte is ~87 us, thousands of core
  clocks, so the memory is idle between every byte.
- No CRC, no flow control beyond the natural UART back-pressure, no multi-bank addressing.

## Physical layer

| Property | Value |
| --- | --- |
| Encoding | UART 8N1 (1 start, 8 data LSB-first, 1 stop) |
| Baud | 115200 |
| `UART_CPB` | clocks per bit, **104** on the Cmod A7 (12 MHz core). Derived in `board.tcl` from `clk_mhz`, so the two cannot drift |
| Throughput | ~11.5 KB/s — a full 512 KB SRAM preload is ~45 s |

## Address and data sizes

The SRAM is 512K x 8 (ISSI IS61/64WV5128): a **19-bit** byte address, 8-bit data.

- **Address** is 3 bytes, big-endian; `addr[18:0]` is used.
- **Length** is 2 bytes, big-endian, **in bytes**; `1 <= len <= 65535`.

Big-endian so a frame reads left-to-right the way the address is written.

## Frame formats

```
byte:   0     1     2     3     4     5     6 .. 6+len-1
      +-----+-----+-----+-----+-----+-----+-------------------+
write | CMD | A2  | A1  | A0  | L1  | L0  | data[0 .. len-1]  |
      +-----+-----+-----+-----+-----+-----+-------------------+

byte:   0     1     2     3     4     5
      +-----+-----+-----+-----+-----+-----+
read  | CMD | A2  | A1  | A0  | L1  | L0  |
      +-----+-----+-----+-----+-----+-----+
```

The 6-byte header is fixed-length, so the FSM always knows how many bytes to expect before
the command byte matters — the only branch is whether a data phase follows.

| CMD | Value | Frame | Reply |
| --- | --- | --- | --- |
| `R` read SRAM | 0x52 | header | exactly `len` data bytes, **nothing else** |
| `W` write SRAM | 0x57 | header + payload | `ACK` / `NAK` |
| `I` load image | 0x49 | header + payload | `ACK` / `NAK` |
| `G` go | 0x47 | `CMD A2 A1 A0` — no length, no payload | `ACK` / `NAK` |
| `T` counters | 0x54 | `CMD` alone | `TIMER_WORDS` 32-bit words, no status byte |

| Status | Value | Meaning |
| --- | --- | --- |
| `ACK` | 0x06 | completed, all bytes committed |
| `NAK` | 0x15 | rejected: bad `CMD`, `len == 0`, or a range error |

Header-less read replies mean the host, which already knows `len`, can simply `read(len)`.
A rejected read returns a single `NAK` and no data, which the host detects by the short
reply plus its own timeout.

### `I` — loading a firmware image

`I` addresses instruction **word** indices, not bytes, and its length must be a multiple of
4 (words packed MSB first). Bit 12 of the word address (`FW_BASE`) is what selects the
PicoRV32's firmware RAM.

### `G` — releasing the core

`G` releases the CPU from reset. It always starts at firmware address 0 —
`PROGADDR_RESET` is fixed in the RTL, so the rest of the address is ignored.

### `T` — the performance counters

`TIMER_WORDS` 32-bit words, MSB first within and across words, **no status byte**. Every
counter shares one window — reset when the core starts, frozen when it stops — so they
describe the last run, or the current one so far. Each saturates at `0xFFFFFFFF` rather
than wrapping (358 s at 12 MHz).

- **Word 0 is always the run length**, which keeps the reply prefix-compatible with the
  single-word one this command used to give, and keeps images with different counter sets
  mutually intelligible.
- `TIMER_WORDS` is 1 in the bring-up images (`cmod_a7_mem`, `cmod_a7_bram`, which have no
  core) and `NPERF` in `tpu_top`, currently **10**: run, mxu, mload, vpu, dma, swait, vmm,
  idlec, qfull, ovlap.
- The counters **overlap rather than partition** the run: `mload` is a sub-state of `mxu`,
  and `ovlap` of the three unit counters.
- Two are **retired slots that always read 0** — `swait` (the deleted scalar unit's
  issue-and-wait stall) and `vmm` (the removed `vecmatmul` op). They are kept because word
  order *is* the protocol: renumbering would silently repoint every counter a host reads by
  index. `tpu_top.sv`'s `PERF_*` indices define the wire order; `host/tpu_uart.py`'s
  `TIMER_COUNTERS` decodes it.

`T` is the only command **not** subject to the arbitration rule below: it reads a counter,
contends over nothing, and cannot corrupt a run. That is its point — a rising count is the
only live evidence this link can give that a long program is still progressing. It is not a
completion signal: the count also sits still if the run never started.

## Device FSM

One command channel, one op in flight.

```
IDLE
  on a valid byte -> latch CMD
    'R' / 'W' / 'I' -> RX_ADDR -> RX_LEN -> validate
    'G'             -> RX_ADDR -> release the core -> ACK
    'T'             -> sample the counters -> TMR_TX -> IDLE
    else            -> NAK, back to IDLE

RX_ADDR   3 bytes -> addr[23:0]
RX_LEN    2 bytes -> len[15:0]
          validate: addr[23:19] == 0, addr + len <= 2^19, len != 0
          fail -> NAK, IDLE

'W': WR_RX -> WR_MEM  (loop len times) -> ACK -> IDLE
'R': RD_MEM -> RD_TX  (loop len times) -> IDLE
```

No pipelining: a UART byte takes ~1040 core clocks while a one-byte SRAM range takes 1–2,
so the UART is hundreds of times slower and the SRAM is never the bottleneck. One holding
byte in each direction suffices.

## SRAM arbitration

The `sram_controller` user port is shared between the DMA and the UART host, so exactly one
may drive it at a time.

**The UART host only touches SRAM while the core is idle.** A command arriving mid-run is
NAK'd and touches nothing. `T` is exempt.

This keeps the link a pure bring-up/debug path and avoids needing a coherency story between
host writes and in-flight compute.

## Framing, errors, resync

There is **no delimiter byte** — framing is the fixed 6-byte header plus the known `len`.

- **Unknown `CMD`** -> `NAK` and back to `IDLE`, so a stray byte costs one bad command
  rather than permanent desync.
- **Inter-byte timeout.** `UART_RX_TIMEOUT` clocks; if the FSM is mid-frame and no byte
  arrives for that long, it aborts and returns to `IDLE`. The host resynchronizes by
  pausing longer than that before a fresh command.
- **`UART_RX_TIMEOUT` is 0 (disabled) on every board today.** That is what turns one
  corrupted byte into a permanently wedged link — the FSM sits mid-frame forever and only a
  reflash clears it. `20 * UART_CPB` makes a corrupted frame cost one legible timeout
  instead. The measured floor is one byte time; at ~2 bit times the abort fires mid-frame
  during ordinary streaming and every command NAKs.

**Client-side validation is load-bearing.** The FSM validates right after the header and,
on failure, NAKs and returns to `IDLE` *without* consuming the data phase — so payload
bytes already in flight get re-decoded as command bytes. `tpu_uart.py` applies the RTL's
exact rules before sending, so an invalid frame never goes out.

## Example exchanges

```
write 3 bytes AA BB CC to 0x00100
  host -> dev:  57  00 01 00  00 03  AA BB CC
  dev  -> host: 06                                (ACK)

read 4 bytes from 0x00100
  host -> dev:  52  00 01 00  00 04
  dev  -> host: AA BB CC DD                       (4 data bytes, no framing)

bad command byte
  host -> dev:  99 ...
  dev  -> host: 15                                (NAK, FSM back to IDLE)
```

## Testing

- `tb/uart_receiver_tb.sv`, `tb/uart_transmitter_tb.sv` — the blocks alone.
- `tb/uart_interface_tb.sv` — the command FSM: round-trip, `len == 1`, a burst crossing a
  256-byte boundary, `len == 0` and out-of-range NAKs, unknown command, mid-frame abort,
  back-to-back commands.
- `tb/uart_memory_cosim_tb.sv` (`make cosim`) — the real host driver against the RTL.
- `tb/fw_uart_tb.sv` (`make fwuart`) — a whole kernel over the simulated link.

**A co-simulation blind spot worth knowing:** `uart_memory_cosim_tb.sv`'s host clocks every
bit for exactly `CPB` clocks, so it has *zero* baud error, where a real 115200 host against
a 12 MHz CPB=104 device runs at 104.1667. `make cosim` cannot see anything analogue or
fractional. See the corruption post-mortem in [`../host/README.md`](../host/README.md).

## Open questions

- **Read error signalling.** A header-less read reply gives no in-band error channel. If
  that proves fragile, prefix read replies with a status byte and make them framed.
- **Baud.** 115200 makes a full-SRAM preload ~45 s. `UART_CPB` is a parameter, so 921600
  (~6 s) is a one-line change once the link is proven.
- **Cross-command auto-increment** to shave header bytes on large sequential transfers,
  versus keeping every command self-contained (the current choice).
