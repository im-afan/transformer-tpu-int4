#!/usr/bin/env python3
"""program.py — a kernel, a backend and a generator, run against each other.

    TPUProgram(source, backend, generator).run_program()

builds the kernel for the backend, loads the generator's static image once,
then runs each case: write the patch, run, read the checked ranges, compare.
See accel/test/README.md.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_generator import VectorGenerator            # noqa: E402


@dataclass
class Result:
    """One case's verdict."""

    name: str
    ok: bool = True
    mismatches: list = field(default_factory=list)   # (addr, got, want)
    strays: list = field(default_factory=list)       # (addr, got, was)
    n_checked: int = 0
    n_cmds: int | None = None
    counters: dict | None = None

    def summary(self) -> str:
        if self.ok:
            extra = f", {self.n_cmds} commands" if self.n_cmds is not None else ""
            return f"{self.name}: {self.n_checked} bytes match{extra}"
        parts = []
        if self.mismatches:
            parts.append(f"{len(self.mismatches)} of {self.n_checked} bytes wrong")
        if self.strays:
            parts.append(f"{len(self.strays)} stray writes")
        return f"{self.name}: " + ", ".join(parts)

    def detail(self, limit: int = 8) -> str:
        lines = []
        for addr, got, want in self.mismatches[:limit]:
            lines.append(f"    [0x{addr:05x}] = 0x{got:02x}, expected 0x{want:02x}")
        for addr, got, was in self.strays[:limit]:
            lines.append(f"    [0x{addr:05x}] = 0x{got:02x}, untouched byte was "
                         f"0x{was:02x}")
        return "\n".join(lines)


def _in_ranges(addr: int, ranges: list) -> bool:
    return any(base <= addr < base + length for base, length in ranges)


class TPUProgram:
    """A kernel bound to a backend and a set of vectors."""

    def __init__(self, source: str, backend, generator: VectorGenerator,
                 name: str | None = None, include_dirs=()):
        self.source = os.path.abspath(source)
        self.backend = backend
        self.generator = generator
        self.name = name or os.path.splitext(os.path.basename(source))[0]
        self.include_dirs = list(include_dirs)
        self.results: list = []

    def run_program(self, limit: int | None = None, verbose: bool = True) -> list:
        """Build, load, and run every case. Returns one `Result` per case."""
        gen = self.generator
        self.backend.build(self.source, dict(gen.defines), self.include_dirs)

        static = gen.static()
        self.backend.load(static)
        # What DRAM held going in, so a byte that changed can be told from one
        # that was already like that.
        seen = dict(static)

        writable = gen.writable_ranges()
        self.results = []
        for i, case in enumerate(gen.cases()):
            if limit is not None and i >= limit:
                break
            seen.update(case.patch)
            run = self.backend.run(case.patch, case.check_ranges)
            res = Result(name=case.name,
                         n_cmds=len(run.cmds) if run.cmds is not None else None,
                         counters=run.counters)

            for addr, want in sorted(case.golden.items()):
                res.n_checked += 1
                got = run.dram.get(addr)
                if got != want:
                    res.mismatches.append((addr, got if got is not None else -1, want))

            # A byte written outside the checked result and outside what the
            # kernel declared it may scribble on. Only the backends that can see
            # all of DRAM can say this; the board cannot.
            if run.written is not None:
                for addr in sorted(run.written):
                    if _in_ranges(addr, case.check_ranges) or _in_ranges(addr, writable):
                        continue
                    if addr in case.golden:
                        continue
                    res.strays.append((addr, self._peek(run, addr),
                                       seen.get(addr, 0)))

            res.ok = not res.mismatches and not res.strays
            self.results.append(res)
            seen.update(case.golden)
            if verbose:
                mark = "ok  " if res.ok else "FAIL"
                print(f"  {mark} {res.summary()}")
                if not res.ok:
                    print(res.detail())
        return self.results

    def _peek(self, run, addr: int) -> int:
        return run.dram.get(addr, -1)

    def read_timers(self) -> list:
        """Per-case performance counters, for the backends that have them."""
        return [r.counters for r in self.results if r.counters]

    def passed(self) -> bool:
        return bool(self.results) and all(r.ok for r in self.results)


# =============================================================================
# The CLI every test folder shares.
# =============================================================================
def standard_parser(description: str):
    import argparse

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-b", "--backend", default="iss",
                    choices=("iss", "rtl", "rtl-uart", "board"),
                    help="iss: the native build against the model. rtl: the "
                         "RISC-V image through the whole core in Icarus, "
                         "checked against the ISS. rtl-uart: the same, loaded "
                         "over the simulated serial link. board: the FPGA.")
    ap.add_argument("-n", "--cases", type=int, default=None,
                    help="stop after this many cases")
    ap.add_argument("-p", "--port", default=None, help="serial port (board only)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show the simulator's own output")
    return ap


def backend_from_args(args, watchdog_ns: int = 2_000_000):
    from backends import ISSBackend, RTLBackend, TPUBackend

    if args.backend == "iss":
        return ISSBackend()
    if args.backend == "rtl":
        return RTLBackend(watchdog_ns=watchdog_ns, quiet=not args.verbose)
    if args.backend == "rtl-uart":
        return RTLBackend(uart=True, watchdog_ns=watchdog_ns,
                          quiet=not args.verbose)
    return TPUBackend(port=args.port, quiet=not args.verbose)


def report(program: "TPUProgram") -> int:
    """Print the verdict; return a process exit code."""
    n = len(program.results)
    bad = [r for r in program.results if not r.ok]
    print()
    if not bad:
        print(f"{program.name}: {n} case(s) PASSED on {program.backend.name}")
        return 0
    print(f"{program.name}: {len(bad)} of {n} case(s) FAILED on "
          f"{program.backend.name}")
    return 1
