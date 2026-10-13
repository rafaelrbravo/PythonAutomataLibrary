"""Verify PAL is importable and its bundled native core is operational."""
from pathlib import Path
import ctypes
import sys

import PythonAutomataLibrary as pal

package_dir = Path(pal.__file__).resolve().parent
native_name = ("pal_native.dll" if sys.platform == "win32" else
               "libpal_native.dylib" if sys.platform == "darwin" else
               "libpal_native.so")
native_path = package_dir / native_name
assert native_path.is_file(), f"Missing native PAL binary: {native_path}"
ctypes.CDLL(str(native_path))
assert hasattr(pal, "AgentGrid"), "PAL public API missing AgentGrid"
assert hasattr(pal, "PDEgrid"), "PAL public API missing PDEgrid"
assert hasattr(pal, "PopGrid"), "PAL public API missing PopGrid"
print(f"PAL package: {package_dir}")
print(f"Native library loaded: {native_path}")
