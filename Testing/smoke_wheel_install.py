"""Install a PAL wheel into an isolated target and check native-backed import.

Usage: python Testing/smoke_wheel_install.py path/to/wheel.whl

Requires pip and the package's runtime dependencies in the invoking interpreter.
The wheel is installed with --no-deps to a temporary directory, and a fresh
Python process imports PAL from that target. Failure propagates as nonzero.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        parser.error(f'wheel does not exist: {wheel}')
    with tempfile.TemporaryDirectory(prefix='pal-wheel-') as tmp:
        target = Path(tmp) / 'site'
        install = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir',
             '--target', str(target), str(wheel)], check=False)
        if install.returncode:
            return install.returncode
        env = os.environ.copy()
        env['PYTHONPATH'] = str(target)
        # Run outside the repository and verify the imported module is from target.
        code = (
            'import pathlib, PythonAutomataLibrary as pal; '
            'p=pathlib.Path(pal.__file__).resolve(); '
            f't=pathlib.Path({str(target)!r}).resolve(); '
            'assert p.is_relative_to(t), (p,t); '
            'print("PASS: wheel-installed PAL imports from", p)'
        )
        result = subprocess.run([sys.executable, '-c', code], cwd=tmp,
                                env=env, check=False)
        return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
