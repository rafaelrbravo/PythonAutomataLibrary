"""Set the build-time package version from a release tag without committing changes."""
import os
from pathlib import Path
import re
import sys

tag = os.environ.get("RELEASE_TAG", "")
if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
    raise SystemExit(f"Expected final release tag vMAJOR.MINOR.PATCH, got {tag!r}")
version = tag[1:]
path = Path("pyproject.toml")
contents = path.read_text(encoding="utf-8")
updated, count = re.subn(r'(?m)^(version\s*=\s*")[^"]+(")$', lambda m: m.group(1) + version + m.group(2), contents)
if count != 1:
    raise SystemExit(f"Expected exactly one project version in {path}; found {count}")
path.write_text(updated, encoding="utf-8")
print(f"Building PAL {version} from tag {tag}")
