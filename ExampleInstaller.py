"""Copy PAL's bundled examples to a user-editable directory."""
import argparse
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Copy bundled PAL examples into an editable directory.")
    parser.add_argument("path", nargs="?", default=".", help="Parent directory for PAL-Examples (default: current directory)")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "Examples"
    parent = Path(args.path).expanduser().resolve()
    if not parent.is_dir():
        print(f"Parent directory does not exist: {parent}", file=sys.stderr)
        return 1
    target = parent / "PAL-Examples"
    if not source.is_dir():
        print("PAL examples are missing from this installation.", file=sys.stderr)
        return 1
    if target.exists():
        print(f"Destination already exists: {target}\nNothing was overwritten.", file=sys.stderr)
        return 1
    shutil.copytree(source, target)
    print(f"Copied PAL examples to {target}")
    print("Run an example with: python PAL-Examples/Agents/<example>.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
