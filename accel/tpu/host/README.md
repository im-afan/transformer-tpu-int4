# Host tools

Host-PC side of the TPU's serial link. Device side is
[`rtl/uart_interface.sv`](../rtl/uart_interface.sv); the protocol is
[`docs/uart_host.md`](../docs/uart_host.md).

| File | What it does |
| --- | --- |
| `tpu_uart.py` | The driver for all five commands. Needs `pyserial`, imports nothing else from the repo, so it runs standalone. |
| `run_adder.py` | **The model generating on the board**, problem by problem: prompt in, generated token ids out. Builds `fw/infer.c` against the checkpoint's requant table first. |
| `run_fw_matmul.py` | One kernel vs. the ISS: loads a firmware image, stages operands, runs, checks the result. |
| `test_uart_link.py` | Self-checking tests for the link and SRAM alone — no program, no toolchain. |
| `uart_echo.py` | One layer below that: 64-byte block loopback against the `cmod_a7_echo` bitstream. |

## Commands

| CMD | Method | Frame | Reply |
| --- | --- | --- | --- |
| `R` 0x52 | `read_mem(addr, len)` | `CMD A2 A1 A0 L1 L0` | `len` data bytes |
| `W` 0x57 | `write_mem(addr, data)` | `CMD A2 A1 A0 L1 L0` + `data[len]` | ACK / NAK |
| `I` 0x49 | `load_program(waddr, words)` | `CMD A2 A1 A0 L1 L0` + `data[len]` | ACK / NAK |
| `G` 0x47 | `go(pc)` | `CMD A2 A1 A0` | ACK / NAK |
| `T` 0x54 | `read_counters()` | `CMD` | 10 x 4 counter bytes |

- Address is 3 bytes big-endian; length is 2 bytes big-endian **in bytes**.
- `R`/`W` address 19-bit SRAM bytes. `I` addresses instruction *word* indices and its
  length must be a multiple of 4 (words packed MSB first). `G` has no length or payload.
- `G` releases the PicoRV32 from reset. It always starts at firmware address 0 —
  `PROGADDR_RESET` is fixed in the RTL, so the rest of the address is ignored.
- Transfers longer than the 16-bit length field are split automatically.

### `T` — the performance counters

Ten 32-bit words, counter 0 first, MSB first within and across words. Every counter
measures the same window: reset at the start of a run, frozen at the halt.

| # | Name | Counts clocks where... |
| --- | --- | --- |
| 0 | `run` | the core is busy — the denominator for the rest |
| 1 | `mxu` | the MXU is busy |
| 2 | `mload` | the MXU is loading weights |
| 3 | `vpu` | the VPU is busy |
| 4 | `dma` | the DMA is busy |
| 5 | `swait` | *retired* — tied low |
| 6 | `vmm` | *retired* — tied low |
| 7 | `idlec` | **no** unit is busy — what instruction issue costs |
| 8 | `qfull` | a push hit a full queue (structurally 0: both producers gate on `!full`) |
| 9 | `ovlap` | two or more units are busy at once |

Slots 5 and 6 are kept rather than reused so the counters after them do not shift.
`0xFFFFFFFF` is a counter saturating, not wrapping. Divide by the device clock (12 MHz on
the Cmod A7) for seconds.

`T` is the **only** command answered while the core is running — it touches no memory.
A count that keeps rising means the core is still working; a count that has stopped does
not by itself mean the run finished, since it also never moves if the run never started.

## Library

```python
from tpu_uart import TPUUart

with TPUUart("COM5") as tpu:                  # 115200 8N1
    tpu.write_mem(0x1000, input_tensor)       # preload DRAM
    tpu.load_program(0, packed_words)         # firmware image
    tpu.go(0)                                 # release the core
    result = tpu.read_mem(0x2000, 256)        # once it has halted
```

Failures raise: `ValueError` for a frame the device would reject (caught before anything is
sent), `NakError` when the device rejects one anyway, `ReplyTimeout` when the expected bytes
never arrive.

## CLI

```bash
python accel/tpu/host/tpu_uart.py -p COM5 write 0x1000 --hex deadbeef
python accel/tpu/host/tpu_uart.py -p COM5 write 0x1000 --file acts.bin
python accel/tpu/host/tpu_uart.py -p COM5 read  0x1000 64          # hex dump
python accel/tpu/host/tpu_uart.py -p COM5 read  0x1000 64 -o out.bin
python accel/tpu/host/tpu_uart.py -p COM5 load  ../fw/infer.hex --go
python accel/tpu/host/tpu_uart.py -p COM5 go 0
python accel/tpu/host/tpu_uart.py -p COM5 timer
```

`load` takes the `$readmemh` format `bin2hex.py` emits — one 32-bit hex word per line.

### Seeing the bytes

Both link failure modes (a desync, an unexpected NAK) are byte-level, so the driver can log
every byte in both directions: `-T/--trace` for a live hexdump on stderr, `--trace-file
FILE` to capture it.

```
   0.0013  TX      10B  write_mem[0x00001000+4]
           0000  57 00 10 00 00 04 de ad be ef                     |W.........|
   0.0498  RX       1B  write_mem[0x00001000+4]
           0000  06                                                |.|
```

`DROP` rows are bytes found sitting in the input buffer and thrown away — the tell-tale of
a previous command desyncing. From Python, `trace=` also takes a file object or a
`callable(line)`; `tpu.dump_trace()` prints a quietly-recorded trace only on failure.

---

## Running the model on the board

`run_adder.py` hands the device a **prompt** and nothing else. The device prefills it, then
decodes its own output token by token against a KV cache, and what comes back is a sequence
it chose. A single wrong digit derails everything after it — the property teacher forcing
hides.

```bash
python accel/tpu/host/run_adder.py --dry-run -n 4       # ISS, no board
python accel/tpu/host/run_adder.py -p COM5 -n 64
python accel/tpu/host/run_adder.py -p COM5 -n 4 --compare-iss
python accel/tpu/host/run_adder.py -p COM5 -n 8 --gen 3          # 3 tokens, quick
python accel/tpu/host/run_adder.py -p COM5 -n 1 --split          # prefill vs decode
python accel/tpu/host/run_adder.py -p COM5 -n 8 --phase decode   # benchmark
python accel/tpu/host/run_adder.py -p COM5 -n 8 --batch 2 --phase decode
python accel/tpu/host/run_adder.py -p COM5 -n 64 --no-show       # progress only
```

- **The whole autoregressive loop is on the device.** `cpu_subsys.sv` maps the scratchpad
  at `0x9xxx_xxxx`, so the head writes its 13 logits there, the CPU reads them back and
  argmaxes, and the embedding "gather" is a DMA at `DR_EMBED + token*D`. The host tokenizes
  and nothing else.
- **The requant table is compiled into the image.** The 16 `{m0,n}` words per layer are
  literals in the macro-ops, so a checkpoint's scales reach the board only through the
  build. The script derives the table, writes `fw/adder_rq_ckpt.h` and rebuilds
  `fw/infer.hex` before loading anything. `--fw` takes a prebuilt image instead — and then
  it is on you that the two agree: an image carrying the synthetic `infer_rq.h` runs
  perfectly and generates noise. `--gen`, `--batch`, `--block` and `--phase` are all
  compile-time constants too.
- **Traffic.** Weights, causal mask, output head and embedding table are ~390 KB at
  `d=128`, identical for every problem, and go down once with the image. Per problem the
  host sends 128 bytes of token ids and reads 128 back.
- **The scratchpad is not cleared between problems.** The previous problem's KV cache is
  still there; the causal mask is what makes it harmless (a masked score is at most -1
  before ReLU). Running back to back is the test, not a shortcut.
- **Counters are read after every problem** (`--no-timing` to skip), so each row carries
  that problem's core run time and the summary reports mean/min/max plus unit occupancy.
- **Benchmarking one half.** The counters cannot be read mid-run, so measuring the prefill
  and the decode separately means two images. `--phase prefill|decode` builds one;
  `--split` does both plus the whole generation and tables the three.

Everything that is not the transport is shared with the simulated path
(`infer_export.static_image` and its codecs), so a disagreement between `--dry-run` and a
board run localizes to the hardware.

## Running one kernel

`run_fw_matmul.py` answers the narrower question: does the board compute what the ISS
computes, on a synthetic image?

```bash
python accel/tpu/host/run_fw_matmul.py -p COM5
python accel/tpu/host/run_fw_matmul.py --dry-run          # operands + reference, no board
python accel/tpu/host/run_fw_matmul.py -p COM5 --fw ../fw/foo.hex
make -C accel/tpu/fw run PORT=COM5                        # build then run
```

1. `I` the firmware into the CPU's RAM, `W` the operands into DRAM.
2. `G` releases the core; the firmware pushes its own commands and raises `done`.
3. Wait for idle, `T` the counters, `R` the result back, compare against a plain-Python
   matmul.

`--m` / `--ktiles` / `--ntiles` set the shape (they must match what the image was built
with). The simulation counterpart is `cd accel/tpu/tb && make fw`.

**How the runners wait.** There is no status command, but everything except `T` is NAK'd
while the core runs — so they probe with a harmless 2-byte read: NAK means "still running",
data means "idle, results readable". `--run-timeout` bounds the wait; on the Cmod A7 the
same state is on `led[1]`.

## Testing the link

`test_uart_link.py` answers "is the link and the SRAM good?" separately from "is the core
computing the right answer?" — run it first when a board misbehaves.

```bash
python accel/tpu/host/test_uart_link.py -p COM5              # ~10 s
python accel/tpu/host/test_uart_link.py -p COM5 --slow       # + the >64 KiB transfer
python accel/tpu/host/test_uart_link.py -p COM5 --only sram_roundtrip
python accel/tpu/host/test_uart_link.py --offline            # driver checks, no board
```

| Test | What it catches |
| --- | --- |
| `sram_roundtrip` | `W` then `R` a block, six data patterns — the core check |
| `sram_isolation` | one command's write landing in another's region |
| `sram_address_bus` | shorted/open address lines |
| `sram_short_frames` | off-by-one in the FSM's `idx + 1 == len` termination |
| `link_rejects_bad_command` | unknown command byte → NAK, and the FSM returns to IDLE |
| `link_rejects_out_of_range` | the device's VALIDATE rules, host check bypassed |
| `link_timer_command` | `T` replies, and holds still while the core is idle |
| `sram_long_transfer` | (`--slow`) the driver's >64 KiB frame splitting |
| `host_validation` | (offline) host frame rules still a superset of the RTL's |
| `frame_encoding` | (offline) header/word endianness |

Two assumptions: **it overwrites all of SRAM**, and **the core must be idle** (a running
program NAKs everything). It checks the second up front rather than reporting 20 bogus
failures.

Not covered: anything needing a program to be running — that is `run_adder.py`'s job on
hardware and `make fwuart` in simulation.

## Testing the UART alone

`uart_echo.py` talks to a **different bitstream** — `cmod_a7_echo`, which contains the two
UART blocks, a 64-byte register file and nothing else. Details in
[`docs/uart_selftest.md`](../docs/uart_selftest.md).

```bash
python accel/tpu/host/uart_echo.py -p COM5                # 30-second run
python accel/tpu/host/uart_echo.py -p COM5 --minutes 30   # soak
python accel/tpu/host/uart_echo.py -p COM5 --baud 117000  # sampling-margin check
python accel/tpu/host/uart_echo.py --offline
```

On a mismatch it reports whether the received stream is the sent stream shifted by a whole
**bit** (mis-framed byte), a whole **byte** (lost or invented frame), or neither (a single
mis-sampled bit) — three different bugs a byte diff renders identically.

This image has no SRAM and no core, so **reflash `cmod_a7` before running anything else.**

---

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

**The simulation blind spot.** `tb/uart_memory_cosim_tb.sv`'s host clocks every bit for
exactly `CPB` clocks, so it has *zero* baud error — a real 115200 host against a 12 MHz
CPB=104 device runs at 104.1667. `make cosim` could never see this. Driving the same RTL at
a fractional bit period is still clean, which is what exonerated the RTL.

## Not covered

- `resync()` idles the link long enough to trip the device's `RX_TIMEOUT` mid-frame abort.
  That parameter is `0` (disabled) on every board, so a device stuck mid-frame needs a reset.
- A rejected read with `len == 1` is undetectable — a lone `NAK` (0x15) is indistinguishable
  from one data byte of value 0x15.
- No CRC and no `SYNC` preamble.
