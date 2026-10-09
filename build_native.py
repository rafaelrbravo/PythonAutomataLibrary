import os
import sys
import subprocess
import shutil

here=os.path.dirname(os.path.abspath(__file__))
src=os.path.join(here,"pal_native.c")

if sys.platform.startswith("win"):
    out=os.path.join(here,"pal_native.dll")
    gcc=shutil.which("gcc") or shutil.which("clang")
    if gcc:
        cmd=[gcc,"-O3","-shared","-std=c11",src,"-o",out]
    else:
        cl=shutil.which("cl")
        if not cl:
            raise SystemExit("Need gcc/clang (MinGW) or MSVC cl on PATH")
        cmd=[cl,"/O2","/LD",src,"/Fe:"+out]
elif sys.platform=="darwin":
    out=os.path.join(here,"libpal_native.dylib")
    cmd=[shutil.which("cc") or "cc","-O3","-dynamiclib",src,"-o",out]
else:
    out=os.path.join(here,"libpal_native.so")
    cmd=[shutil.which("cc") or "cc","-O3","-shared","-fPIC","-std=c11",src,"-o",out,"-lm"]

print(" ".join(cmd))
subprocess.check_call(cmd,cwd=here)
print(out)
