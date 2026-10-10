"""Run PAL's native correctness suite in independent safe and fast processes.

Usage (from repository root):
    python Testing/run_native_audit.py
    python Testing/run_native_audit.py --mode safe
    python Testing/run_native_audit.py --mode fast -- -k diffusion

Requires an installed/importable PAL package and pytest. This script never
changes PAL's implementation or suppresses failing test exit statuses.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("safe", "fast", "both"), default="both")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Seconds per mode (default: 300)")
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    tests = Path(__file__).resolve().parent
    root = tests.parent
    modes = ("safe", "fast") if args.mode == "both" else (args.mode,)
    results = {}
    for mode in modes:
        env = os.environ.copy()
        env["PAL_TEST_MODE"] = mode
        cmd = [sys.executable, "-m", "pytest", str(tests), "-q", "--tb=short", *rest]
        print(f"\n=== PAL native audit: {mode} ===", flush=True)
        start = time.monotonic()
        try:
            result = subprocess.run(cmd, cwd=root, env=env, timeout=args.timeout,
                                    check=False)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = 124
            print(f"TIMEOUT after {args.timeout:g}s in {mode} mode", flush=True)
        elapsed = time.monotonic() - start
        results[mode] = (status, elapsed)
        print(f"=== {mode}: exit={status}, elapsed={elapsed:.2f}s ===", flush=True)
    print("\nPAL native audit summary:")
    for mode, (status, elapsed) in results.items():
        print(f"  {mode:4s} exit={status:3d} elapsed={elapsed:.2f}s")
    return 0 if all(status == 0 for status, _ in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
