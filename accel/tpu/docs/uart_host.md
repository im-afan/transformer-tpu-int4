# UART Host Interface

A serial link that lets a host PC read and write the external SRAM ("DRAM"), load a firmware
image, start the core and read its performance counters.

Device side is [`rtl/uart_interface.sv`](../rtl/uart_interface.sv) plus `uart_receiver.sv` /
`uart_transmitter.sv`. Host side is `accel/test/tpu_uart.py`.

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

### Start-bit re-validation (`rtl/uart_receiver.sv`)

`uart_rx` is an asynchronous pin with no glitch filter, so a brief low during a stop bit or
the idle gap looks exactly like the start of a byte. Taken at face value it shifts the whole
frame by up to one bit period, silently rewriting the byte (a one-bit shift turns `0x40`
into `0x80`), and on a command length field that makes `uart_interface` read or write far
past the end of the frame.

A real start bit stays low for a full bit period; a glitch does not. The receiver holds the
line low all the way to the bit centre — the first sample that comes back high drops it back
to `IDLE` to re-arm rather than waiting out the half bit, so a real start bit a few clocks
behind the glitch is still caught, and `data`/`valid` are left untouched by a rejected
glitch. Aborting cannot lose a byte: `IDLE` re-arms on the same condition, so a bounce on a
genuine falling edge just re-enters `START` a clock or two later, well inside the sampling
margin.

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
- `uart_interface`'s `cycle_count` port is sampled once, at the clock the `T` command byte
  is decoded, so the bytes that go out are one coherent reading rather than values that
  moved between them — ratios between counters in one reply stay meaningful. Wire order is
  high word first.

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

### Why the receive holding register is unconditional

`rtl/uart_interface.sv`'s single-clock `rx_strobe` used to be consumed directly inside the
FSM's state case, which made every state that is not `IDLE`/`RX_ADDR`/`RX_LEN`/`WR_RX`/
`IMEM_RX` a window where an arriving byte was destroyed with no error, no retry and no
trace. The worst window, `SEND_STATUS`/`SEND_STATUS_WAIT` (~10 `CLK_PER_BIT` clocks), sits
exactly at the command turnaround — the device starts the ACK about one bit before the host
has finished the payload's stop bit, opening straight into where the host's next command
byte lands.

One dropped byte is not one bad command: the FSM re-enters `IDLE` a byte out of phase, so
every subsequent payload byte decodes as an unknown command and draws a NAK, and with
`RX_TIMEOUT = 0` (every board today) nothing ever recovers.

The fix is to capture `rx_hold` every cycle in every state, ahead of the FSM, and have the
FSM read the register instead of the wire — the "single holding byte" behaviour this
section already specifies.

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
- `tb/fw_uart_tb.sv` — a whole kernel over the simulated link:
  `python accel/test/run_suite.py -b rtl-uart`.

`tb/uart_memory_cosim_tb.sv` and `make cosim` are gone, with the host scripts that drove
them. **The blind spot that removal costs nothing:** its host clocked every bit for exactly
`CPB` clocks, so it had *zero* baud error, where a real 115200 host against a 12 MHz
CPB=104 device runs at 104.1667. It could never see anything analogue or fractional — and
the intermittent corruption bug, when it came, was exactly that. `fw_uart_tb.sv` has the
same property; the board is the only thing that does not.

## Generating on the board

`TPUBackend` (`accel/test/backends.py`) is the whole host now: load the image with `'I'`,
the static image with `'W'`, `'G'`, wait for idle, `'R'`. `tests/infer/generate.py -b board`
is the model generating, problem by problem — the device is handed a prompt and nothing
else, prefills it, then decodes its own output token by token against a KV cache. A single
wrong digit derails everything after it, which is the property teacher forcing hides.

**What the device owns.** `cpu_subsys.sv` maps the scratchpad at `0x9xxx_xxxx`: the head
writes its logits there, the CPU argmaxes them and DMAs the embedding gather itself. The
host only tokenizes. The whole autoregressive loop closes on the device: one `'G'` per
problem, a finished sequence comes back out of `DR_TOKENS`.

**The requant table is compiled into the image.** The device has no path to read scales out
of memory, so `export.py` writes them into `infer_config.h` and the kernel is rebuilt
against it before anything is loaded. An image built against a different header carries a
different table and scores nothing.

**Wire traffic.** Weights, causal mask, output head, embedding table and the zeroed KV
cache (~394 KB at `d=128`) go down once, with the firmware image — about 35 s at 115200.
Per problem after that: 128 bytes of prompt out, 128 bytes of generated ids back. Decode
bound, not link bound, which is the whole reason `Backend.load` is separate from
`Backend.run`.

**Nothing is cleared between problems**, on the board or on the ISS. The causal mask makes
the previous problem's leftover KV cache harmless — a masked score is at most -1 before
ReLU. Running problems back to back is the test, not a shortcut. (The cache is zeroed
*once*, in the static image, so that the first problem sees the same memory on every
backend; after that nobody touches it.)

**Benchmarking a phase in isolation.** The counters reset at `'G'` and freeze at the halt,
so measuring prefill and decode separately needs two images — `INFER_PREFILL` and
`INFER_DECODE` in the config header, set by `tests/infer/generate.py --phase`.
`--phase split` builds and runs both in turn and prints a per-token cost for each. The
decode-only build scores nothing — the ids it generates are noise, but its clocks are the
same clocks; the prefill-only build still produces the first token and is checked on it.

`BATCH = B` runs B independent sequences per `'G'`, sharing one weight stream for the
projections, `Wo` and both FFN matmuls (`B*rows` rows), while attention stays per sequence
with its own KV cache. B is bounded by DRAM, and `dram_map` fails loudly in Python when it
does not fit.

## Two things to know

**Client-side validation is load-bearing.** The FSM validates a frame right after the
6-byte header and, on failure, NAKs and returns to IDLE *without* consuming the data phase —
so payload bytes already in flight get re-decoded as commands. The driver applies the RTL's
exact rules before sending, so an invalid frame never goes out; a NAK on a write is
therefore treated as a desync, not a routine rejection.

**The core has priority.** Any command arriving while the core runs is NAK'd and touches
nothing. `T` is the exception. There is no `busy`/`done` status command, so `go()` returns
once the launch is ACK'd and completion is inferred out-of-band.

## Solved: intermittent UART corruption

**The fault was on the host, not the FPGA.** Reading the serial port while the USB-serial
bridge is still transmitting corrupts the byte in flight: the host's IN requests disturb the
FT2232H's transmit bit timing by roughly half a bit, so the device decodes `sent[k]` **or**
`sent[k-1]` for every bit `k` — usually `value << 1`.

Fixed in `tpu_uart.py`: `TPUUart._send` waits for a frame to clear the wire before anything
reads (`TX_SETTLE_S` / `TX_RATE_SLACK`). All commands route through it.

The old code issued `ser.write(frame)` then read immediately, so the read landed on the
*first few bytes of every command* — the header, where a corrupted length does the most
damage. A 16-bit length arriving one bit left-shifted is the whole original symptom.

**How it was pinned down**, on the echo image, 64-byte bursts, 6400 bytes per arm:

| host behaviour | corrupted | positions in the burst |
| --- | --- | --- |
| read immediately after write | 72/6400 | 1–6 |
| sleep past the whole burst, then read | **0/6400** | — |
| sleep over only the *first half*, then read | 77/6400 | **33–46** |

The third row is the proof: delaying the read moves the damage to where the read begins.

**Ruled out — do not re-litigate.** Anything in the RTL (the receiver decodes a bit-exact
stream perfectly, all 256 byte values); `uart_interface`'s `SEND_STATUS` blind window;
external SRAM and its pins (`cmod_a7_bram` reproduces at the same rate); byte value and
predecessor; position in the burst per se; baud mismatch and sampling phase (swept ±3%,
flat); long low runs on the line; the activity LED's load step; `reset_input_buffer()`;
metastability, crosstalk, timing closure.

**The simulation blind spot.** Every simulated host — the retired `uart_memory_cosim_tb`
and `fw_uart_tb.sv` alike — clocks each bit for exactly `CPB` clocks, so it has *zero*
baud error, where a real 115200 host against a 12 MHz CPB=104 device runs at 104.1667.
No simulation could ever have seen this. Driving the same RTL at a fractional bit period
is still clean, which is what exonerated the RTL.

## Not covered

- The device's `RX_TIMEOUT` mid-frame abort is `0` (disabled) on every board, so a device
  stuck mid-frame needs a reset. `20 * UART_CPB` in `boards/*/board.tcl` would make a
  corrupted frame cost one legible timeout instead. That is hardening, not the fix.
- A rejected read with `len == 1` is undetectable — a lone `NAK` (0x15) is indistinguishable
  from one data byte of value 0x15.
- No CRC and no `SYNC` preamble.

## Open questions

- **Read error signalling.** A header-less read reply gives no in-band error channel. If
  that proves fragile, prefix read replies with a status byte and make them framed.
- **Baud.** 115200 makes a full-SRAM preload ~45 s. `UART_CPB` is a parameter, so 921600
  (~6 s) is a one-line change once the link is proven.
- **Cross-command auto-increment** to shave header bytes on large sequential transfers,
  versus keeping every command self-contained (the current choice).
