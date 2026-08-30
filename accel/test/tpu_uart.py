#!/usr/bin/env python3
"""tpu_uart.py — the UART command link to the board (rtl/uart_interface.sv).

Five commands, nothing else. The protocol, the device behaviours that shape
this driver, and why `_send` waits before anything reads: accel/tpu/docs/uart_host.md.
"""
from __future__ import annotations

import re
import time
from typing import Iterable, Sequence

CMD_READ = 0x52   # 'R'  read  SRAM     CMD A2 A1 A0 L1 L0        -> len bytes
CMD_WRITE = 0x57  # 'W'  write SRAM     CMD A2 A1 A0 L1 L0 + data -> ACK/NAK
CMD_IMEM = 0x49   # 'I'  write IMEM     CMD A2 A1 A0 L1 L0 + data -> ACK/NAK
CMD_GO = 0x47     # 'G'  run            CMD A2 A1 A0              -> ACK/NAK
CMD_TIMER = 0x54  # 'T'  read counters  CMD                       -> 40 bytes

STAT_ACK = 0x06
STAT_NAK = 0x15

# One 32-bit word per counter, MSB first. This IS the wire order; `swait` and
# `vmm` are retired slots that read 0 and are kept so nothing after them shifts.
TIMER_COUNTERS = ("run", "mxu", "mload", "vpu", "dma", "swait", "vmm",
                  "idlec", "qfull", "ovlap")
TIMER_BYTES = len(TIMER_COUNTERS) * 4

DEFAULT_BAUD = 115200

MEM_ADDR_W = 19            # external SRAM byte address
IMEM_AW = 10               # scalar unit instruction memory word address
FW_AW = 12                 # firmware RAM word address (16 KB)
FW_BASE = 1 << FW_AW       # OR into an 'I'/'G' address to reach the CPU
HOST_AW = FW_AW + 1
MEM_LIMIT = 1 << MEM_ADDR_W
IMEM_LIMIT = 1 << IMEM_AW
HOST_LIMIT = 1 << HOST_AW

MAX_LEN = 0xFFFF
MAX_IMEM_LEN = MAX_LEN & ~0x3
ADDR_LIMIT = 1 << 24

# Reading the port while the USB-serial bridge is still shifting a frame out
# corrupts the byte in flight. `_send` waits out the computed wire time plus a
# fixed pad for the driver-to-wire start-up skew and 2% for the rate error
# between the bridge's divider and the device's.
TX_SETTLE_S = 0.005
TX_RATE_SLACK = 1.02

# Idle probe: two bytes, not one — a 1-byte read cannot tell a NAK (0x15) from
# a data byte of 0x15. A NAK'd read goes silent after that byte, so a busy probe
# costs a whole timeout; keep this one short.
PROBE_ADDR, PROBE_LEN, PROBE_TIMEOUT = 0x0, 2, 0.3


class ProtocolError(Exception):
    """The device answered, but not the way the protocol says it should."""


class NakError(ProtocolError):
    """The device rejected the command (bad frame, or the core was busy)."""


class ReplyTimeout(ProtocolError):
    """Too few bytes came back. `partial` tells a lone NAK from a quiet link."""

    def __init__(self, message: str, partial: bytes = b""):
        super().__init__(message)
        self.partial = partial


# ---- frame validation, mirroring the RTL's VALIDATE state --------------------
# A rejected command does not consume its data phase, so the device would decode
# the payload as fresh command bytes. Every frame is checked here, before a byte
# goes out, rather than letting the device reject it.

def _check_mem(addr: int, length: int) -> None:
    if not 0 <= addr < ADDR_LIMIT:
        raise ValueError(f"address {addr:#x} does not fit the 24-bit field")
    if addr >= MEM_LIMIT:
        raise ValueError(f"address {addr:#x} outside the {MEM_ADDR_W}-bit SRAM space")
    if not 1 <= length <= MAX_LEN:
        raise ValueError(f"length {length} outside 1..{MAX_LEN}")
    if addr + length > MEM_LIMIT:
        raise ValueError(f"range {addr:#x}+{length} runs past SRAM ({MEM_LIMIT:#x})")


def _producer(word_addr: int) -> tuple:
    """(name, base, words) for a word address in the 'I'/'G' space."""
    if word_addr & FW_BASE:
        return "firmware RAM", FW_BASE, 1 << FW_AW
    return "IMEM", 0, IMEM_LIMIT


def _check_imem(word_addr: int, length: int) -> None:
    if not 0 <= word_addr < HOST_LIMIT:
        raise ValueError(f"word address {word_addr:#x} outside the "
                         f"{HOST_AW}-bit 'I' space")
    if not 1 <= length <= MAX_LEN:
        raise ValueError(f"length {length} outside 1..{MAX_LEN}")
    if length % 4:
        raise ValueError(f"IMEM payload {length} bytes is not a multiple of 4")
    name, base, words = _producer(word_addr)
    if word_addr - base + length // 4 > words:
        raise ValueError(f"{length // 4} words at {word_addr:#x} run past the "
                         f"end of {name} ({words} words)")


def _check_pc(pc: int) -> None:
    if not 0 <= pc < HOST_LIMIT:
        raise ValueError(f"boot PC {pc:#x} outside the {HOST_AW}-bit 'G' space")


def _header(cmd: int, addr: int, length: int | None = None) -> bytes:
    frame = bytes([cmd]) + addr.to_bytes(3, "big")
    return frame + length.to_bytes(2, "big") if length is not None else frame


def pack_words(words: Iterable[int]) -> bytes:
    out = bytearray()
    for i, w in enumerate(words):
        if not 0 <= w <= 0xFFFFFFFF:
            raise ValueError(f"instruction word {i} ({w:#x}) is not 32 bits")
        out += w.to_bytes(4, "big")
    return bytes(out)


class TPUUart:
    """A connection to the board's command interface."""

    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = 2.0):
        try:
            import serial
        except ImportError as exc:
            raise ImportError("pyserial is required: pip install pyserial") from exc

        self.port, self.baud, self.timeout = port, baud, timeout
        self.ser = serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1,
                                 timeout=timeout)

    def close(self) -> None:
        self.ser.close()

    def __enter__(self) -> "TPUUart":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _byte_time(self, n: int) -> float:
        return 10.0 * n / self.baud            # 8N1

    def _send(self, frame: bytes) -> None:
        """Put `frame` on the wire and wait until it has actually left.

        Nothing may read the port before this returns. flush() is not enough:
        on Windows it returns once the bytes reach the driver, not the wire, and
        no API reports the shift register — so the wire time is computed and
        waited out. See docs/uart_host.md for the measurement this comes from.
        """
        self.ser.write(frame)
        self.ser.flush()
        time.sleep(self._byte_time(len(frame)) * TX_RATE_SLACK + TX_SETTLE_S)

    def _recv(self, n: int, timeout: float | None = None) -> bytes:
        if timeout is not None:
            self.ser.timeout = timeout
        try:
            data = self.ser.read(n)
        finally:
            self.ser.timeout = self.timeout
        if len(data) != n:
            raise ReplyTimeout(f"expected {n} byte(s), got {len(data)}: "
                               f"{data.hex(' ') or '<nothing>'}", partial=data)
        return data

    def _status(self) -> None:
        st = self._recv(1)[0]
        if st == STAT_ACK:
            return
        if st == STAT_NAK:
            raise NakError("device sent NAK — the core was busy, or host and "
                           "device are out of sync")
        raise ProtocolError(f"expected ACK/NAK, got {st:#04x}")

    def read_mem(self, addr: int, length: int) -> bytes:
        if length <= 0:
            raise ValueError(f"length {length} must be >= 1")
        out = bytearray()
        for off in range(0, length, MAX_LEN):
            out += self._read_chunk(addr + off, min(MAX_LEN, length - off))
        return bytes(out)

    def _read_chunk(self, addr: int, length: int) -> bytes:
        _check_mem(addr, length)
        self.ser.reset_input_buffer()
        self._send(_header(CMD_READ, addr, length))
        try:
            return self._recv(length, timeout=self.timeout + self._byte_time(length))
        except ReplyTimeout as exc:
            # A rejected read answers with a lone NAK and no data.
            if length > 1 and exc.partial == bytes([STAT_NAK]):
                raise NakError("device rejected the read command") from exc
            raise

    def write_mem(self, addr: int, data: bytes) -> None:
        if not data:
            raise ValueError("nothing to write (len == 0 is rejected)")
        for off in range(0, len(data), MAX_LEN):
            self._payload_cmd(CMD_WRITE, addr + off, data[off:off + MAX_LEN],
                              _check_mem)

    def load_program(self, word_addr: int, words: Sequence[int] | bytes) -> None:
        payload = words if isinstance(words, (bytes, bytearray)) else pack_words(words)
        if not payload:
            raise ValueError("nothing to load (len == 0 is rejected)")
        for off in range(0, len(payload), MAX_IMEM_LEN):
            self._payload_cmd(CMD_IMEM, word_addr + off // 4,
                              bytes(payload[off:off + MAX_IMEM_LEN]), _check_imem)

    def _payload_cmd(self, cmd: int, addr: int, payload: bytes, check) -> None:
        check(addr, len(payload))
        self.ser.reset_input_buffer()
        # One write() so the device sees the header and data as one burst.
        self._send(_header(cmd, addr, len(payload)) + payload)
        try:
            self._status()
        except ProtocolError:
            # The NAK came before the data phase, so the device decoded the
            # payload as command bytes. Clear the wreckage before propagating.
            self.ser.reset_input_buffer()
            time.sleep(self._byte_time(4))
            self.ser.reset_input_buffer()
            raise

    def go(self, pc: int = 0) -> None:
        """Start a producer. Returns on the ACK; the run itself is asynchronous
        and this link has no completion signal — see `wait_until_idle`."""
        _check_pc(pc)
        self.ser.reset_input_buffer()
        self._send(_header(CMD_GO, pc))
        self._status()

    def read_counters(self) -> dict:
        """The per-run performance counters, in core clocks.

        All ten measure the same window: reset at 'G', frozen at the halt. They
        overlap and do not partition the run — `mload` is a subset of `mxu`.
        0xFFFFFFFF is a saturated counter, not a wrap.
        """
        self.ser.reset_input_buffer()
        self._send(bytes([CMD_TIMER]))
        raw = self._recv(TIMER_BYTES)
        return {name: int.from_bytes(raw[i * 4:(i + 1) * 4], "big")
                for i, name in enumerate(TIMER_COUNTERS)}


# ---- helpers ----------------------------------------------------------------
_HEX_WORD = re.compile(r"^[0-9a-fA-F]{1,8}$")


def parse_hex_program(text: str) -> list:
    """A $readmemh-style firmware image (one 32-bit hex word per line)."""
    words: list = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//")[0].strip()
        for tok in line.split():
            if not _HEX_WORD.match(tok):
                raise ValueError(f"line {lineno}: {tok!r} is not a 32-bit hex word")
            words.append(int(tok, 16))
    return words


def contiguous_runs(image: dict) -> list:
    """A sparse {addr: byte} image collapsed into [(addr, bytes), ...], so one
    UART command covers each dense span instead of each byte."""
    runs: list = []
    start, buf = None, bytearray()
    for addr in sorted(image):
        if start is not None and addr != start + len(buf):
            runs.append((start, bytes(buf)))
            start, buf = None, bytearray()
        if start is None:
            start = addr
        buf.append(image[addr] & 0xFF)
    if start is not None:
        runs.append((start, bytes(buf)))
    return runs


def autodetect_port() -> str:
    """The Cmod A7's FT2232 UART channel."""
    from serial.tools import list_ports

    cands = [p for p in list_ports.comports() if (p.vid, p.pid) == (0x0403, 0x6010)]
    if not cands:
        cands = [p for p in list_ports.comports() if p.vid == 0x0403]
    if not cands:
        raise SystemExit("no FTDI serial port found — pass --port explicitly")
    if len(cands) > 1:
        names = ", ".join(f"{p.device} ({p.description})" for p in cands)
        raise SystemExit(f"several candidate ports, pass --port: {names}")
    return cands[0].device


def probe_idle(uart: TPUUart) -> bool:
    """True if the core is idle — i.e. it answers a read instead of NAK'ing it.
    There is no status command; this is the only completion signal there is."""
    saved, uart.timeout = uart.timeout, PROBE_TIMEOUT
    try:
        uart.read_mem(PROBE_ADDR, PROBE_LEN)
        return True
    except NakError:
        return False
    finally:
        uart.timeout = saved


def wait_until_idle(uart: TPUUart, limit: float, interval: float = 0.05) -> float:
    """Block until the core stops NAK'ing commands. Returns seconds waited."""
    t0 = time.monotonic()
    while True:
        if probe_idle(uart):
            return time.monotonic() - t0
        if time.monotonic() - t0 >= limit:
            raise SystemExit(f"core still busy {limit:.1f}s after 'G' — it never "
                             f"halted (check led[1]/done), or the link desynced")
        time.sleep(interval)


def format_counters(ctr: dict, indent: str = "  ") -> str:
    """The counters as a bottleneck breakdown. Percentages are of `run` and
    deliberately sum past 100%: the counters overlap."""
    labels = {"mxu": "MXU busy", "mload": "  of which weight load",
              "vpu": "VPU busy", "dma": "DMA busy",
              "idlec": "no unit busy (issue overhead)", "ovlap": "two+ units busy"}
    run = ctr.get("run", 0)
    lines = [f"{indent}{'run':<35} {run:>10} clocks"]
    for key, label in labels.items():
        val = ctr.get(key, 0)
        pct = f"{100.0 * val / run:5.1f}%" if run else "    - "
        lines.append(f"{indent}{label:<35} {val:>10}  {pct}")
    if any(v == 0xFFFFFFFF for v in ctr.values()):
        lines.append(f"{indent}[SATURATED — at least one counter pinned]")
    return "\n".join(lines)
