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
DEFAULT_ORDER = ["matmul", "ffn", "mha", "spadwin", "tiled"]
SLOW = {"infer"}

WATCHDOG_NS = {"matmul": 2_000_000, "ffn": 2_000_000, "mha": 2_000_000,
               "spadwin": 4_000_000, "tiled": 60_000_000,
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
    args = ap.parse_args()

    names = args.kernel or [n for n in discover() if n not in SLOW]
    failed, timings = [], []

    for name in names:
        print(f"==== {name} on {args.backend} ====")
        backend = backend_from_args(args, watchdog_ns=WATCHDOG_NS.get(name, 2_000_000))
        mod = load(name)
        t0 = time.monotonic()
        try:
            prog = mod.program(backend)
            prog.run_program(limit=args.cases)
            ok = prog.passed()
        finally:
            backend.close()
        timings.append((name, time.monotonic() - t0))
        if not ok:
            failed.append(name)
        print()

    print("==== summary ====")
    for name, secs in timings:
        mark = "FAIL" if name in failed else "ok  "
        print(f"  {mark} {name:<10} {secs:6.1f}s")
    if failed:
        print(f"\n{len(failed)} of {len(names)} kernels FAILED: "
              f"{', '.join(failed)}")
        return 1
    print(f"\nall {len(names)} kernels passed on {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
