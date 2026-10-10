"""Verify a built PAL wheel includes the native library needed at runtime.

Build a wheel first, then:
    python Testing/check_wheel_contents.py dist/pythonautomatalibrary-*.whl

Use a shell to expand the glob; this script takes exactly one wheel path.
The wheel must include the platform-specific shared library; C source is
optional when a binary is included. No PAL implementation files are modified.
"""
import argparse
from pathlib import Path
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as wheel:
        names = set(wheel.namelist())
    expected = {"PythonAutomataLibrary/libpal_native.so",
                "PythonAutomataLibrary/libpal_native.dylib",
                "PythonAutomataLibrary/pal_native.dll"}
    present = sorted(expected.intersection(names))
    print(f"Wheel: {args.wheel}")
    print(f"Native libraries included: {present or 'NONE'}")
    if not present:
        print("FAIL: wheel omits PAL's required native shared library", file=sys.stderr)
        return 1
    if args.wheel.name.endswith("-none-any.whl"):
        print("FAIL: platform-dependent native library packaged as universal wheel",
              file=sys.stderr)
        return 1
    print("PASS: native library present with platform-specific wheel tag")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
