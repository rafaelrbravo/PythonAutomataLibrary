"""Build PAL's platform-native library into binary wheels."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
from setuptools import setup, Distribution
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent

class BuildWithNative(build_py):
    def run(self):
        super().run()
        source = ROOT / "pal_native.c"
        destination = Path(self.build_lib) / "PythonAutomataLibrary"
        destination.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            output = destination / "pal_native.dll"
            # Use the Visual Studio environment provided by cibuildwheel.
            compiler = shlex.split(os.environ.get("CC", "cl"))
            subprocess.check_call([*compiler, "/O2", "/LD", str(source), "/link", "/OUT:" + str(output)], cwd=ROOT)
        elif sys.platform == "darwin":
            output = destination / "libpal_native.dylib"
            compiler = shlex.split(os.environ.get("CC", "cc"))
            flags = shlex.split(os.environ.get("CFLAGS", ""))
            subprocess.check_call([*compiler, *flags, "-O3", "-dynamiclib", str(source), "-o", str(output)], cwd=ROOT)
        else:
            output = destination / "libpal_native.so"
            compiler = shlex.split(os.environ.get("CC", "cc"))
            flags = shlex.split(os.environ.get("CFLAGS", ""))
            subprocess.check_call([*compiler, *flags, "-O3", "-std=c11", "-shared", "-fPIC", str(source), "-o", str(output), "-lm"], cwd=ROOT)

class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True

class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

setup(distclass=NativeDistribution, cmdclass={"build_py": BuildWithNative, "bdist_wheel": PlatformWheel})
