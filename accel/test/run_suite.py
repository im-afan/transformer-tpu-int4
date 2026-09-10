#!/usr/bin/env python3
"""run_suite.py — every kernel in tests/, on one backend.

    python accel/test/run_suite.py                 # the ISS, all of them
    python accel/test/run_suite.py -b rtl -k tiled # one, through the RTL
    python accel/test/run_suite.py -b board -p COM5

Each tests/<name>/generate.py exposes `program(backend, ...)`; this imports them
and runs each in turn. `infer` is not in the default set — it is minutes on the
ISS and hours in simulation — so ask for it by name.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "tests")
sys.path.insert(0, HERE)

from program import backend_from_args                    # noqa: E402

# Ordered cheapest first, so a broken dispatch plane fails in seconds rather
# than after the block loops have run.
DEFAULT_ORDER = ["matmul", "ffn", "mha", "spadwin", "dma_roundtrip",
                 "tiled_simple", "wide", "fused", "argmax", "tiled"]
SLOW = {"infer", "mha_prefill"}

WATCHDOG_NS = {"matmul": 2_000_000, "ffn": 2_000_000, "mha": 2_000_000,
               "spadwin": 4_000_000, "dma_roundtrip": 20_000_000,
               "tiled_simple": 20_000_000,
               "argmax": 20_000_000,
               "wide": 60_000_000,
               "fused": 60_000_000,
               "tiled": 60_000_000,
               "mha_prefill": 200_000_000,
               "infer": 1_000_000_000}


def discover() -> list:
    found = [d for d in sorted(os.listdir(TESTS))
             if os.path.isfile(os.path.join(TESTS, d, "generate.py"))]
    ordered = [n for n in DEFAULT_ORDER if n in found]
    return ordered + [n for n in found if n not in ordered]


def load(name: str):
    path = os.path.join(TESTS, name, "generate.py")
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    # Registered before it runs: a dataclass in the module resolves its own
    # annotations through sys.modules, and an unregistered one raises.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-b", "--backend", default="iss",
                    choices=("iss", "rtl", "rtl-uart", "board"))
    ap.add_argument("-k", "--kernel", action="append", default=None,
                    help="run only these (repeatable); the default set skips "
                         f"{', '.join(sorted(SLOW))}")
    ap.add_argument("-p", "--port", default=None, help="serial port (board only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--cases", type=int, default=None)
    ap.add_argument("--clk-mhz", type=float, default=12.0,
                    help="core clock the benchmark's milliseconds are quoted at "
                         "(default 12, the Cmod A7's)")
    ap.add_argument("--bench", action="store_true",
                    help="print each kernel's full counter breakdown, not just "
                         "its clocks")
    args = ap.parse_args()

    names = args.kernel or [n for n in discover() if n not in SLOW]
    failed, rows = [], []

    for name in names:
        print(f"==== {name} on {args.backend} ====")
        backend = backend_from_args(args, watchdog_ns=WATCHDOG_NS.get(name, 2_000_000))
        mod = load(name)
        t0 = time.monotonic()
        try:
            prog = mod.program(backend)
            prog.run_program(limit=args.cases)
            ok = prog.passed()
            bench = prog.benchmark(args.clk_mhz)
            if args.bench and bench:
                print(prog.format_benchmark(args.clk_mhz))
        finally:
            backend.close()
        rows.append((name, time.monotonic() - t0, bench))
        if not ok:
            failed.append(name)
        print()

    # The device's own counters, not the wall clock: the wall clock is dominated
    # by iverilog or by USB latency and says nothing about the hardware.
    timed = [r for r in rows if r[2]]
    print("==== summary ====")
    if timed:
        print(f"  {'':4} {'kernel':<10} {'wall':>7}  {'clocks':>12} "
              f"{'ms @ ' + format(args.clk_mhz, 'g') + ' MHz':>14}  "
              f"{'mxu':>6} {'dma':>6} {'idle':>6}")
    for name, secs, bench in rows:
        mark = "FAIL" if name in failed else "ok  "
        if not bench:
            print(f"  {mark} {name:<10} {secs:6.1f}s")
            continue
        share = bench["share"]
        print(f"  {mark} {name:<10} {secs:6.1f}s  {bench['clocks']:>12} "
              f"{bench['ms']:>14.3f}  "
              f"{100 * share.get('mxu', 0):5.1f}% {100 * share.get('dma', 0):5.1f}% "
              f"{100 * share.get('idlec', 0):5.1f}%")
    if timed and not args.bench:
        print("  (--bench for the full counter breakdown; the shares overlap "
              "and do not sum to 100%)")
    if failed:
        print(f"\n{len(failed)} of {len(names)} kernels FAILED: "
              f"{', '.join(failed)}")
        return 1
    print(f"\nall {len(names)} kernels passed on {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
