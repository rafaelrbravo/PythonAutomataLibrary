"""Build PAL's platform-native library into binary wheels."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from setuptools import setup, Distribution
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent

class BuildWithNative(build_py):
    def run(self):
        super().run()
        examples_source = ROOT / "Examples"
        if examples_source.is_dir():
            shutil.copytree(examples_source, Path(self.build_lib) / "PythonAutomataLibrary" / "Examples", dirs_exist_ok=True)
        docs_source = ROOT / "Documentation"
        docs_target = Path(self.build_lib) / "PythonAutomataLibrary" / "Documentation"
        if docs_source.is_dir():
            docs_target.mkdir(parents=True, exist_ok=True)
            for pdf in docs_source.glob("*.pdf"):
                shutil.copy2(pdf, docs_target / pdf.name)
        source = ROOT / "pal_native.c"
        destination = Path(self.build_lib) / "PythonAutomataLibrary"
        destination.mkdir(parents=True, exist_ok=True)
        # A source checkout can contain binaries from other operating systems.
        # Never bundle those into a platform-specific wheel.
        native_names = ("pal_native.dll", "libpal_native.dylib", "libpal_native.so")
        selected = ("pal_native.dll" if sys.platform == "win32" else
                    "libpal_native.dylib" if sys.platform == "darwin" else
                    "libpal_native.so")
        for name in native_names:
            if name != selected:
                (destination / name).unlink(missing_ok=True)
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

class SourceWithoutNativeBinaries(sdist):
    def get_file_list(self):
        super().get_file_list()
        forbidden_suffixes = (".dll", ".so", ".dylib", ".pyd")
        self.filelist.files = [name for name in self.filelist.files if not name.lower().endswith(forbidden_suffixes)]

class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True

class PlatformWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        # PAL loads a standalone C shared library via ctypes; it does not
        # use the CPython extension ABI. Keep the platform-specific tag,
        # but allow the same wheel to install on any supported Python 3.
        _, _, platform = super().get_tag()
        return "py3", "none", platform

setup(distclass=NativeDistribution, cmdclass={"build_py": BuildWithNative, "bdist_wheel": PlatformWheel, "sdist": SourceWithoutNativeBinaries})
