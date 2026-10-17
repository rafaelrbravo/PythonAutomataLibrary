"""Copy PAL's bundled examples to a user-editable directory."""
from pathlib import Path
import shutil
import sys


def main():
    source = Path(__file__).resolve().parent / "Examples"
    target = Path.cwd() / "PAL-Examples"
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
