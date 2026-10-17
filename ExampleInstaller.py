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
    docs = Path(__file__).resolve().parent / "Documentation"
    pdfs = sorted(docs.glob("*.pdf")) if docs.is_dir() else []
    if not pdfs:
        print("PAL documentation PDFs are missing from this installation.", file=sys.stderr)
        return 1
    shutil.copytree(source, target)
    docs_target = target / "Documentation"
    docs_target.mkdir()
    for pdf in pdfs:
        shutil.copy2(pdf, docs_target / pdf.name)
    print(f"Copied PAL examples and documentation PDFs to {target}")
    print("Run an example with: python PAL-Examples/Agents/<example>.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
