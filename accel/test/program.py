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

    # ---- benchmarking -------------------------------------------------------
    def read_timers(self) -> list:
        """One counter dict per case, for the backends that have them.

        `rtl`, `rtl-uart` and `board` do; the ISS has no cycle model and returns
        an empty list. The keys and their order are the board's 'T' reply
        (`tpu_uart.TIMER_COUNTERS`), so a simulated run and a hardware run are
        directly comparable. The simulation adds `mxucmd` / `dmacmd` / `wallclk`,
        which the link has no way to report.

        Every counter measures the same window — reset at 'G', frozen at the
        halt — and they **overlap** rather than partitioning it: `mload` is a
        subset of `mxu`, `ovlap` of the three unit counters. `run` is the
        denominator. `swait` and `vmm` are retired slots and always read 0.
        """
        return [r.counters for r in self.results if r.counters]

    def timer_totals(self) -> dict:
        """The per-case counters summed. Empty if the backend has none."""
        totals: dict = {}
        for ctr in self.read_timers():
            for key, val in ctr.items():
                totals[key] = totals.get(key, 0) + val
        return totals

    def benchmark(self, clk_mhz: float = 12.0) -> dict | None:
        """A summary of what the run cost, or None if nothing was timed.

        `clk_mhz` only converts clocks to milliseconds — the clocks themselves
        are the measurement, and they are the same clocks in simulation and on
        the board. 12 MHz is the Cmod A7's core clock.
        """
        per_case = self.read_timers()
        if not per_case:
            return None
        runs = [c.get("run", 0) for c in per_case]
        totals = self.timer_totals()
        total_run = sum(runs) or 1
        return {
            "cases": len(per_case),
            "clk_mhz": clk_mhz,
            "clocks": totals.get("run", 0),
            "ms": totals.get("run", 0) / (clk_mhz * 1e3),
            "per_case": per_case,
            "run_min": min(runs), "run_max": max(runs),
            "run_mean": sum(runs) / len(runs),
            "totals": totals,
            # Fraction of the run each counter was active. Sums past 1.0 on
            # purpose: they overlap.
            "share": {k: v / total_run for k, v in totals.items()
                      if k not in ("run", "mxucmd", "dmacmd", "wallclk")},
        }

    def format_benchmark(self, clk_mhz: float = 12.0, indent: str = "  ") -> str:
        """`benchmark()` as a table. Empty string when nothing was timed."""
        bench = self.benchmark(clk_mhz)
        if bench is None:
            return ""
        labels = {"mxu": "MXU busy", "mload": "  of which weight load",
                  "vpu": "VPU busy", "dma": "DMA busy",
                  "idlec": "no unit busy (issue overhead)",
                  "qfull": "producer stalled on a full queue",
                  "ovlap": "two or more units busy"}
        lines = [f"{indent}{'run':<34} {bench['clocks']:>12} clocks  "
                 f"{bench['ms']:.3f} ms @ {clk_mhz:g} MHz"]
        for key, label in labels.items():
            if key not in bench["totals"]:
                continue
            lines.append(f"{indent}{label:<34} {bench['totals'][key]:>12}  "
                         f"{100 * bench['share'][key]:5.1f}%")
        for key, label in (("mxucmd", "MXU dispatches"),
                           ("dmacmd", "DMA dispatches")):
            if key in bench["totals"]:
                lines.append(f"{indent}{label:<34} {bench['totals'][key]:>12}")
        if bench["cases"] > 1:
            lines.append(f"{indent}{'per case':<34} "
                         f"{bench['run_mean']:>12.0f} mean, "
                         f"{bench['run_min']} min, {bench['run_max']} max")
        if any(v == 0xFFFFFFFF for v in bench["totals"].values()):
            lines.append(f"{indent}[SATURATED — at least one counter pinned]")
        return "\n".join(lines)

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
    ap.add_argument("--clk-mhz", type=float, default=12.0,
                    help="core clock the benchmark's milliseconds are quoted "
                         "at (default 12, the Cmod A7's)")
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


def report(program: "TPUProgram", clk_mhz: float = 12.0) -> int:
    """Print the verdict and, where the backend timed the run, what it cost.
    Returns a process exit code."""
    n = len(program.results)
    bad = [r for r in program.results if not r.ok]
    print()
    bench = program.format_benchmark(clk_mhz)
    if bench:
        print(f"{program.name} on {program.backend.name}, {n} case(s):")
        print(bench)
        print()
    if not bad:
        print(f"{program.name}: {n} case(s) PASSED on {program.backend.name}")
        return 0
    print(f"{program.name}: {len(bad)} of {n} case(s) FAILED on "
          f"{program.backend.name}")
    return 1
