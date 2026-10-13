"""Verify PAL is importable and its bundled native core is operational."""
from pathlib import Path
import ctypes
import sys

import PythonAutomataLibrary as pal

package_dir = Path(pal.__file__).resolve().parent
assert "site-packages" in package_dir.parts or "dist-packages" in package_dir.parts, f"PAL imported from non-installed location: {package_dir}"
native_name = ("pal_native.dll" if sys.platform == "win32" else
               "libpal_native.dylib" if sys.platform == "darwin" else
               "libpal_native.so")
native_path = package_dir / native_name
assert native_path.is_file(), f"Missing native PAL binary: {native_path}"
lib = ctypes.CDLL(str(native_path))
lib.pal_rgb.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.pal_rgb.restype = ctypes.c_uint32
assert lib.pal_rgb(12, 34, 56) == 0x0C2238, 'Native PAL function returned wrong value'
assert hasattr(pal, "AgentGrid"), "PAL public API missing AgentGrid"
assert hasattr(pal, "PDEgrid"), "PAL public API missing PDEgrid"
assert hasattr(pal, "PopGrid"), "PAL public API missing PopGrid"
print(f"PAL package: {package_dir}")
print(f"Native library loaded: {native_path}")
