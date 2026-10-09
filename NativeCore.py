"""Thin Numba handles for PAL's precompiled native core.

Design rule: this file contains API glue, not simulation algorithms.
The stable work lives in pal_native.c; user model code sees tiny jitclasses.
"""

import ctypes
import os
import sys
import operator
import ast
import copy
import inspect
import textwrap
from typing import Protocol

import numba as nb
import numpy as np
from numba import float32, int32, uint64
from numba.experimental import jitclass
from numba import types
from numba.extending import overload, typeof_impl, register_model, models, unbox, NativeValue, type_callable, lower_builtin, overload_method, make_attribute_wrapper, intrinsic
from llvmlite import binding as llvm
from llvmlite import ir as lir
from numba.core import cgutils


def _PALDiagnostic(message,sourceLine):
    if sourceLine>=0: return message+" at source line "+str(sourceLine)
    return message

class _PALMethodDiagnosticTransformer(ast.NodeTransformer):
    def __init__(self,methodNames,diagnostics):
        self.methodNames=methodNames; self.diagnostics=diagnostics
    def visit_FunctionDef(self,node):
        # The extracted tree contains only this method.
        if node.name!='__init__':
            node.args.args.append(ast.arg(arg='_palLine'))
            node.args.defaults.append(ast.Constant(-1))
        self.generic_visit(node)
        return node
    def visit_Raise(self,node):
        node=self.generic_visit(node)
        if not self.diagnostics or node.exc is None: return node
        if isinstance(node.exc,ast.Call) and len(node.exc.args)==1 and not node.exc.keywords:
            arg=node.exc.args[0]
            node.exc.args[0]=ast.IfExp(ast.Compare(ast.Name('_palLine',ast.Load()),[ast.GtE()],[ast.Constant(0)]),ast.BinOp(ast.BinOp(arg,ast.Add(),ast.Constant(' at source line ')),ast.Add(),ast.Call(ast.Name('str',ast.Load()),[ast.Name('_palLine',ast.Load())],[])),arg)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='self' and node.func.attr in self.methodNames and node.func.attr!='__init__':
            if not any(k.arg=='_palLine' for k in node.keywords): node.keywords.append(ast.keyword('_palLine',ast.Name('_palLine',ast.Load())))
        return node

def _InstrumentPALClassMethods(cls,diagnostics):
    methods={name for name,value in cls.__dict__.items() if inspect.isfunction(value)}
    for name in methods:
        # Python/Numba operator dunders have fixed calling conventions. Never add
        # PAL's diagnostic argument to them; source-aware AST helpers handle
        # diagnostics for user indexing instead.
        if name in {'__init__','__getitem__','__setitem__','__len__'}: continue
        fn=getattr(cls,name)
        try:
            lines,start=inspect.getsourcelines(fn)
            tree=ast.parse(textwrap.dedent(''.join(lines)))
        except (OSError,TypeError,SyntaxError): continue
        tree=_PALMethodDiagnosticTransformer(methods,diagnostics).visit(tree);ast.fix_missing_locations(tree)
        env=dict(fn.__globals__);env['_PALDiagnostic']=_PALDiagnostic
        exec(compile(tree,inspect.getsourcefile(fn) or '<PAL>', 'exec'),env)
        out=env[name];out.__defaults__=fn.__defaults__+( -1,) if fn.__defaults__ else (-1,)
        out.__annotations__=fn.__annotations__;out.__module__=fn.__module__;out.__qualname__=fn.__qualname__
        setattr(cls,name,out)
    return cls

def _jitclass(spec):
    """Create a jitclass whose Numba type name is stable across processes.

    Numba normally includes id(ClassType) in this name. That makes otherwise
    valid disk-cache entries miss on every fresh Python process. PAL's handle
    layouts are fixed, so using the class name + field layout is the stable
    identity we actually want.
    """
    def decorate(cls):
        if cls.__name__.endswith("SafePython"):
            cls=_InstrumentPALClassMethods(cls,True)
        elif cls.__name__.endswith("FastPython"):
            cls=_InstrumentPALClassMethods(cls,False)
        compiled=jitclass(spec)(cls)
        name=compiled.class_type.name
        if "#" in name and "<" in name:
            compiled.class_type.name=name.split("#",1)[0]+"<"+name.split("<",1)[1]
        return compiled
    return decorate



# PAL visualization window placement. Reservations happen synchronously in
# Start...Window() call order in the parent process, so renderer startup timing
# cannot shuffle the initial window order.
_WINDOW_PLACEMENT_LOCK=__import__("threading").Lock()
_WINDOW_PLACEMENT_NEXT_TOKEN=0
_WINDOW_PLACEMENT_WINDOWS=[]
_WINDOW_PLACEMENT_LEFT=32
_WINDOW_PLACEMENT_TOP=32
_WINDOW_PLACEMENT_X_PAD=12
_WINDOW_PLACEMENT_Y_PAD=32
_WINDOW_PLACEMENT_CASCADE=32

# Windows start asynchronously. Start...Window() registers its initialization
# event here; AwaitWindows() snapshots whatever is still initializing and waits
# for those windows together before model execution continues.
_WINDOW_INIT_LOCK=__import__("threading").Lock()
_WINDOW_INITS=[]

# Safe-mode visualization crash handling. PAL windows render in child processes,
# so an uncaught model exception can leave the last completed frame visible while
# the parent waits for the user to close the windows.
_CRASH_WINDOW_LOCK=__import__("threading").Lock()
_CRASH_WINDOWS=[]
_CRASH_HOOK_INSTALLED=False
_CRASH_PREV_EXCEPTHOOK=None

def _CrashWindowHook(excType,excValue,traceback):
    # Preserve normal Python traceback behavior first. Do not turn Ctrl+C or
    # explicit interpreter exits into a wait-for-window-close state.
    _CRASH_PREV_EXCEPTHOOK(excType,excValue,traceback)
    if not issubclass(excType,Exception): return
    import time as _time
    with _CRASH_WINDOW_LOCK:
        windows=tuple(_CRASH_WINDOWS)
    if not windows: return

    # Keep every visualization on its last completed frame after a model crash.
    # The first window the user closes dismisses the whole crashed run. Snapshot
    # the group here so cleanup/unregistration cannot change which windows belong
    # to this crash while we are waiting.
    while True:
        openWindows=[]
        for window in windows:
            try:
                if window.IsOpen(): openWindows.append(window)
            except Exception:
                pass
        if not openWindows: return
        if len(openWindows)<len(windows):
            for window in openWindows:
                try:
                    window._Close()
                except Exception:
                    pass
            return
        _time.sleep(.05)

def _RegisterCrashWindow(window):
    global _CRASH_HOOK_INSTALLED,_CRASH_PREV_EXCEPTHOOK
    with _CRASH_WINDOW_LOCK:
        _CRASH_WINDOWS.append(window)
        if not _CRASH_HOOK_INSTALLED:
            _CRASH_PREV_EXCEPTHOOK=sys.excepthook
            sys.excepthook=_CrashWindowHook
            _CRASH_HOOK_INSTALLED=True

def _UnregisterCrashWindow(window):
    with _CRASH_WINDOW_LOCK:
        _CRASH_WINDOWS[:]=[entry for entry in _CRASH_WINDOWS if entry is not window]

def _RegisterWindowInit(readyEvent,process,name):
    with _WINDOW_INIT_LOCK:
        _WINDOW_INITS.append((readyEvent,process,name))

def AwaitWindows(timeout=30):
    import time as _time
    with _WINDOW_INIT_LOCK:
        _WINDOW_INITS[:]=[entry for entry in _WINDOW_INITS if not entry[0].is_set()]
        pending=tuple(_WINDOW_INITS)
    if not pending: return
    end=_time.time()+timeout
    for readyEvent,process,name in pending:
        while not readyEvent.is_set():
            if not process.is_alive():
                raise RuntimeError(f"{name} closed before initialization completed")
            remaining=end-_time.time()
            if remaining<=0:
                raise TimeoutError("timed out waiting for PAL window initialization")
            readyEvent.wait(min(.05,remaining))
    with _WINDOW_INIT_LOCK:
        completed={id(entry[0]) for entry in pending}
        _WINDOW_INITS[:]=[entry for entry in _WINDOW_INITS if id(entry[0]) not in completed and not entry[0].is_set()]

def _ReserveWindow(width,height):
    global _WINDOW_PLACEMENT_NEXT_TOKEN
    width=max(1,int(width)); height=max(1,int(height))
    with _WINDOW_PLACEMENT_LOCK:
        token=_WINDOW_PLACEMENT_NEXT_TOKEN
        _WINDOW_PLACEMENT_NEXT_TOKEN+=1
        _WINDOW_PLACEMENT_WINDOWS.append((token,width,height))
        return token,tuple(_WINDOW_PLACEMENT_WINDOWS)


def _ScreenSize():
    """Best-effort usable screen size for deterministic parent-side placement."""
    try:
        import tkinter as tk
        root=tk.Tk()
        root.withdraw()
        width=root.winfo_screenwidth()
        height=root.winfo_screenheight()
        root.destroy()
        return int(width),int(height)
    except Exception:
        return 1920,1080

def _ReserveWindowPosition(width,height):
    token,windows=_ReserveWindow(width,height)
    screenWidth,screenHeight=_ScreenSize()
    x,y=_WindowPosition(token,windows,screenWidth,screenHeight)
    return token,x,y

def _ReleaseWindow(token):
    with _WINDOW_PLACEMENT_LOCK:
        for i,(t,_,_) in enumerate(_WINDOW_PLACEMENT_WINDOWS):
            if t==token:
                _WINDOW_PLACEMENT_WINDOWS.pop(i)
                return

def _WindowPosition(token,windows,screenWidth,screenHeight):
    screenWidth=max(1,int(screenWidth)); screenHeight=max(1,int(screenHeight))
    x=_WINDOW_PLACEMENT_LEFT; y=_WINDOW_PLACEMENT_TOP; rowHeight=0
    for t,width,height in windows:
        outerWidth=width+_WINDOW_PLACEMENT_X_PAD
        outerHeight=height+_WINDOW_PLACEMENT_Y_PAD
        if x>_WINDOW_PLACEMENT_LEFT and x+outerWidth>screenWidth:
            x=_WINDOW_PLACEMENT_LEFT; y+=rowHeight; rowHeight=0
        if y+outerHeight>screenHeight:
            step=t*_WINDOW_PLACEMENT_CASCADE
            maxX=max(_WINDOW_PLACEMENT_LEFT,screenWidth-min(width,screenWidth))
            maxY=max(_WINDOW_PLACEMENT_TOP,screenHeight-min(height,screenHeight))
            px=_WINDOW_PLACEMENT_LEFT+(step % max(1,maxX-_WINDOW_PLACEMENT_LEFT+1))
            py=_WINDOW_PLACEMENT_TOP+(step % max(1,maxY-_WINDOW_PLACEMENT_TOP+1))
        else:
            px=x; py=y
        if t==token: return int(px),int(py)
        x+=outerWidth
        rowHeight=max(rowHeight,outerHeight)
    return _WINDOW_PLACEMENT_LEFT,_WINDOW_PLACEMENT_TOP

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_NAME = (
    "pal_native.dll" if sys.platform.startswith("win") else
    "libpal_native.dylib" if sys.platform == "darwin" else
    "libpal_native.so"
)
_LIB = ctypes.CDLL(os.path.join(_HERE, _LIBRARY_NAME))

_PTR = ctypes.c_uint64
_I32 = ctypes.c_int32
_I64 = ctypes.c_int64
_F32 = ctypes.c_float
_I32_PTR = ctypes.POINTER(_I32)
_I64_PTR = ctypes.POINTER(_I64)
_F32_PTR = ctypes.POINTER(_F32)
_U32 = ctypes.c_uint32
_U8 = ctypes.c_uint8
_F64 = ctypes.c_double
_SIZE = ctypes.c_size_t
_U8_PTR = ctypes.POINTER(ctypes.c_uint8)




def _BCArray(bc, expected):
    if bc is None: return np.empty(0,dtype=np.float32)
    if np.isscalar(bc): return np.full(expected,bc,dtype=np.float32)
    return bc

@overload(_BCArray)
def _BCArrayOverload(bc, expected):
    if isinstance(bc, types.NoneType):
        def impl(bc, expected): return np.empty(0,dtype=np.float32)
        return impl
    if isinstance(bc, types.Number):
        def impl(bc, expected):
            out=np.empty(expected,dtype=np.float32)
            out[:]=bc
            return out
        return impl
    if isinstance(bc, types.Array):
        def impl(bc, expected): return bc
        return impl

def _NumbaCType(cType):
    if cType is None: return types.void
    if cType is _PTR: return types.uint64
    if cType is _I32: return types.int32
    if cType is _I64: return types.int64
    if cType is _F32: return types.float32
    if cType is _F64: return types.float64
    if cType is _U32: return types.uint32
    if cType is _U8: return types.uint8
    if cType is _SIZE: return types.uintp
    if cType is _I32_PTR: return types.CPointer(types.int32)
    if cType is _I64_PTR: return types.CPointer(types.int64)
    if cType is _F32_PTR: return types.CPointer(types.float32)
    if cType is _U8_PTR: return types.CPointer(types.uint8)
    raise TypeError("unsupported PAL native C type: %r" % (cType,))


def _bind(name, result, *args):
    """Bind one C symbol for both ordinary Python and cacheable Numba calls."""
    cFn=getattr(_LIB,name)
    cFn.restype=result
    cFn.argtypes=list(args)

    # Register the exported name with LLVM. Numba can then emit a symbolic call
    # instead of embedding the process-local ctypes function address.
    address=ctypes.cast(cFn,ctypes.c_void_p).value
    if not address: raise RuntimeError("unable to resolve PAL native symbol: "+name)
    llvm.add_symbol(name,address)
    signature=_NumbaCType(result)(*[_NumbaCType(arg) for arg in args])
    external=types.ExternalFunction(name,signature)

    def fn(*callArgs):
        return cFn(*callArgs)

    @overload(fn)
    def _overload(*callArgs):
        def impl(*callArgs):
            return external(*callArgs)
        return impl

    return fn


# General utility C API
_rgb = _bind("pal_rgb", _U32, _I32, _I32, _I32)
_get_red = _bind("pal_get_red", _U8, _U32)
_get_green = _bind("pal_get_green", _U8, _U32)
_get_blue = _bind("pal_get_blue", _U8, _U32)

RGB = _rgb
GetRed = _get_red
GetGreen = _get_green
GetBlue = _get_blue

@nb.njit(cache=True, inline="always")
def ColorScale(colors, value):
    nColors=len(colors)
    if nColors==0: return np.uint32(0)
    if nColors==1: return np.uint32(colors[0])
    value=max(0.0,min(1.0,value))
    scaled=value*(nColors-1)
    lo=int(np.floor(scaled))
    hi=int(np.ceil(scaled))
    t=scaled-lo
    c0=np.uint32(colors[lo])
    c1=np.uint32(colors[hi])
    r0=(c0>>np.uint32(16))&np.uint32(255)
    g0=(c0>>np.uint32(8))&np.uint32(255)
    b0=c0&np.uint32(255)
    r1=(c1>>np.uint32(16))&np.uint32(255)
    g1=(c1>>np.uint32(8))&np.uint32(255)
    b1=c1&np.uint32(255)
    r=np.uint32(int(np.int64(r0)+(np.int64(r1)-np.int64(r0))*t+0.5))
    g=np.uint32(int(np.int64(g0)+(np.int64(g1)-np.int64(g0))*t+0.5))
    b=np.uint32(int(np.int64(b0)+(np.int64(b1)-np.int64(b0))*t+0.5))
    return (r<<np.uint32(16))|(g<<np.uint32(8))|b

# RNG + Multinomial C API
_seed = _bind("pal_seed", None, _PTR)
_random = _bind("pal_random", _F64)
_rand_int = _bind("pal_rand_int", _I64, _I64)
_multi_new = _bind("pal_multi_new", _PTR, _PTR)
_multi_free = _bind("pal_multi_free", None, _PTR)
_multi_binomial = _bind("pal_multi_binomial", _I64, _PTR, _I64, _F64)
_multi_setup = _bind("pal_multi_setup", _I32, _PTR, _I64)
_multi_sample = _bind("pal_multi_sample", _I64, _PTR, _F64)

# IList C API
_q_new = _bind("pal_q_new", _PTR, _I32)
_q_free = _bind("pal_q_free", None, _PTR)
_q_len = _bind("pal_q_len", _I32, _PTR)
_q_data = _bind("pal_q_data", _I32_PTR, _PTR)
_q_get = _bind("pal_q_get", _I32, _PTR, _I32)
_q_get_safe = _bind("pal_q_get_safe", _I64, _PTR, _I32)
_q_copy = _bind("pal_q_copy", _I32, _PTR, _I32_PTR)
_q_set = _bind("pal_q_set", None, _PTR, _I32, _I32)
_q_add = _bind("pal_q_add", _I32, _PTR, _I32)
_q_clear = _bind("pal_q_clear", None, _PTR)
_q_random = _bind("pal_q_random", _I32, _PTR)
_q_shuffle = _bind("pal_q_shuffle", None, _PTR)

# Generic Grid C API
_grid_new = _bind("pal_grid_new", _PTR, _I32_PTR, _I32, _I32, _SIZE)
_grid_free = _bind("pal_grid_free", None, _PTR)
_grid_data = _bind("pal_grid_data", _PTR, _PTR)
_grid_len = _bind("pal_grid_len", _I32, _PTR)
_grid_dim = _bind("pal_grid_dim", _I32, _PTR, _I32)

# AgentGrid C API
_ag_new = _bind("pal_ag_new", _PTR, _I32_PTR, _I32, _I32, _I32)
_ag_free = _bind("pal_ag_free", None, _PTR)
_ag_dim = _bind("pal_ag_dim", _I32, _PTR, _I32)
_ag_pop = _bind("pal_ag_pop", _I32, _PTR)
_ag_len = _bind("pal_ag_len", _I32, _PTR)
_ag_toi = _bind("pal_ag_toi", _I32, _PTR, _I32, _I32, _I32)
_ag_itox = _bind("pal_ag_itox", _I32, _PTR, _I32)
_ag_itoy = _bind("pal_ag_itoy", _I32, _PTR, _I32)
_ag_itoz = _bind("pal_ag_itoz", _I32, _PTR, _I32)
_ag_all_copy = _bind("pal_ag_all_copy", _I32, _PTR, _I32_PTR, _I32)
_ag_alive = _bind("pal_ag_alive", _I32, _PTR, _I32)
_ag_alive_safe = _bind("pal_ag_alive_safe", _I32, _PTR, _I32)
_ag_nagents = _bind("pal_ag_nagents", _I32, _PTR)
_ag_grid_data = _bind("pal_ag_grid_data", _I32_PTR, _PTR)
_ag_int_data = _bind("pal_ag_int_data", _I32_PTR, _PTR)
_ag_float_data = _bind("pal_ag_float_data", _F32_PTR, _PTR)
_ag_nfloatprops = _bind("pal_ag_nfloatprops", _I32, _PTR)
_ag_stackable = _bind("pal_ag_stackable", _I32, _PTR)
_ag_wrap = _bind("pal_ag_wrap", _I32, _PTR, _I32)
_ag_generation = _bind("pal_ag_generation", _PTR, _PTR)
_ag_new_i = _bind("pal_ag_new_i", _I32, _PTR, _I32)
_ag_new_sq = _bind("pal_ag_new_sq", _I32, _PTR, _I32, _I32, _I32)
_ag_new_pt = _bind("pal_ag_new_pt", _I32, _PTR, _F32, _F32, _F32)
_ag_dispose = _bind("pal_ag_dispose", None, _PTR, _I32)
_ag_i = _bind("pal_ag_i", _I32, _PTR, _I32)
_ag_xsq = _bind("pal_ag_xsq", _I32, _PTR, _I32)
_ag_ysq = _bind("pal_ag_ysq", _I32, _PTR, _I32)
_ag_zsq = _bind("pal_ag_zsq", _I32, _PTR, _I32)
_ag_x = _bind("pal_ag_x", _F32, _PTR, _I32)
_ag_y = _bind("pal_ag_y", _F32, _PTR, _I32)
_ag_z = _bind("pal_ag_z", _F32, _PTR, _I32)
_ag_getp = _bind("pal_ag_getp", _F32, _PTR, _I32, _I32)
_ag_setp = _bind("pal_ag_setp", None, _PTR, _I32, _I32, _F32)
_ag_all_id = _bind("pal_ag_all_id", _I32, _PTR, _I32)
_ag_count_i = _bind("pal_ag_count_i", _I32, _PTR, _I32)
_ag_last_i = _bind("pal_ag_last_i", _I32, _PTR, _I32)
_ag_move_i = _bind("pal_ag_move_i", _I32, _PTR, _I32, _I32)
_ag_move_pt = _bind("pal_ag_move_pt", _I32, _PTR, _I32, _F32, _F32, _F32)
_ag_add_i_q = _bind("pal_ag_add_i_q", _I32, _PTR, _PTR, _I32)
_ag_add_q = _bind("pal_ag_add_q", _I32, _PTR, _PTR, _I32, _I32, _I32)
_ag_add_radius_q = _bind("pal_ag_add_radius_q", _I32, _PTR, _PTR, _F64, _F64, _F64, _F64, _I32)
_ag_new_i_fast = _bind("pal_ag_new_i_", _I32, _PTR, _I32)
_ag_new_sq_fast = _bind("pal_ag_new_sq_", _I32, _PTR, _I32, _I32, _I32)
_ag_new_pt_fast = _bind("pal_ag_new_pt_", _I32, _PTR, _F32, _F32, _F32)
_ag_dispose_fast = _bind("pal_ag_dispose_", None, _PTR, _I32)
_ag_move_i_fast = _bind("pal_ag_move_i_", _I32, _PTR, _I32, _I32)
_ag_move_pt_fast = _bind("pal_ag_move_pt_", _I32, _PTR, _I32, _F32, _F32, _F32)
_ag_getp_fast = _bind("pal_ag_getp_", _F32, _PTR, _I32, _I32)
_ag_setp_fast = _bind("pal_ag_setp_", None, _PTR, _I32, _I32, _F32)
_ag_count_i_fast = _bind("pal_ag_count_i_", _I32, _PTR, _I32)
_ag_counts_linear = _bind("pal_ag_counts_linear", None, _PTR, _I32, _I32, _I32_PTR)
_ag_counts_region = _bind("pal_ag_counts_region", None, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _I32_PTR)
_ag_last_i_fast = _bind("pal_ag_last_i_", _I32, _PTR, _I32)
_ag_inwrap_sq_x = _bind("pal_ag_inwrap_sq_x", _I32, _PTR, _I32)
_ag_inwrap_sq_y = _bind("pal_ag_inwrap_sq_y", _I32, _PTR, _I32)
_ag_inwrap_sq_z = _bind("pal_ag_inwrap_sq_z", _I32, _PTR, _I32)
_ag_inwrap_x = _bind("pal_ag_inwrap_x", _F64, _PTR, _F64)
_ag_inwrap_y = _bind("pal_ag_inwrap_y", _F64, _PTR, _F64)
_ag_inwrap_z = _bind("pal_ag_inwrap_z", _F64, _PTR, _F64)
_ag_dispwrap_x = _bind("pal_ag_dispwrap_x", _F64, _PTR, _F64, _F64)
_ag_dispwrap_y = _bind("pal_ag_dispwrap_y", _F64, _PTR, _F64, _F64)
_ag_dispwrap_z = _bind("pal_ag_dispwrap_z", _F64, _PTR, _F64, _F64)
_ag_i_safe = _bind("pal_ag_i_safe", _I32, _PTR, _I32)
_ag_xsq_safe = _bind("pal_ag_xsq_safe", _I32, _PTR, _I32)
_ag_ysq_safe = _bind("pal_ag_ysq_safe", _I32, _PTR, _I32)
_ag_zsq_safe = _bind("pal_ag_zsq_safe", _I32, _PTR, _I32)
_ag_x_safe = _bind("pal_ag_x_safe", _F32, _PTR, _I32)
_ag_y_safe = _bind("pal_ag_y_safe", _F32, _PTR, _I32)
_ag_z_safe = _bind("pal_ag_z_safe", _F32, _PTR, _I32)
_ag_getp_safe = _bind("pal_ag_getp_safe", _F32, _PTR, _I32, _I32)
_ag_setp_safe = _bind("pal_ag_setp_safe", _I32, _PTR, _I32, _I32, _F32)
_ag_inwrap_safe = _bind("pal_ag_inwrap_safe", _F64, _PTR, _F64, _I32)
_ag_dispwrap_safe = _bind("pal_ag_dispwrap_safe", _F64, _PTR, _F64, _F64, _I32)
_ag_inwrap_sq_safe = _bind("pal_ag_inwrap_sq_safe", _I32, _PTR, _I32, _I32)
_ag_toi_safe = _bind("pal_ag_toi_safe", _I32, _PTR, _I32, _I32, _I32)
_ag_itox_safe = _bind("pal_ag_itox_safe", _I32, _PTR, _I32)
_ag_itoy_safe = _bind("pal_ag_itoy_safe", _I32, _PTR, _I32)
_ag_itoz_safe = _bind("pal_ag_itoz_safe", _I32, _PTR, _I32)

# PopGrid C API
_pg_new = _bind("pal_pg_new", _PTR, _I32_PTR, _I32, _I64)
_pg_free = _bind("pal_pg_free", None, _PTR)
_pg_dim = _bind("pal_pg_dim", _I32, _PTR, _I32)
_pg_len = _bind("pal_pg_len", _I32, _PTR)
_pg_toi = _bind("pal_pg_toi", _I32, _PTR, _I32, _I32, _I32)
_pg_itox = _bind("pal_pg_itox", _I32, _PTR, _I32)
_pg_itoy = _bind("pal_pg_itoy", _I32, _PTR, _I32)
_pg_itoz = _bind("pal_pg_itoz", _I32, _PTR, _I32)
_pg_toi_safe = _bind("pal_pg_toi_safe", _I32, _PTR, _I32, _I32, _I32)
_pg_itox_safe = _bind("pal_pg_itox_safe", _I32, _PTR, _I32)
_pg_itoy_safe = _bind("pal_pg_itoy_safe", _I32, _PTR, _I32)
_pg_itoz_safe = _bind("pal_pg_itoz_safe", _I32, _PTR, _I32)
_pg_geti = _bind("pal_pg_geti", _I64, _PTR, _I32)
_pg_get = _bind("pal_pg_get", _I64, _PTR, _I32, _I32, _I32)
_pg_seti = _bind("pal_pg_seti", _I32, _PTR, _I32, _I64)
_pg_addi = _bind("pal_pg_addi", _I32, _PTR, _I32, _I64)
_pg_set = _bind("pal_pg_set", _I32, _PTR, _I32, _I32, _I32, _I64)
_pg_add = _bind("pal_pg_add", _I32, _PTR, _I32, _I32, _I32, _I64)
_pg_update = _bind("pal_pg_update", _I32, _PTR)
_pg_seti_fast = _bind("pal_pg_seti_", _I32, _PTR, _I32, _I64)
_pg_addi_fast = _bind("pal_pg_addi_", _I32, _PTR, _I32, _I64)
_pg_set_fast = _bind("pal_pg_set_", _I32, _PTR, _I32, _I32, _I32, _I64)
_pg_add_fast = _bind("pal_pg_add_", _I32, _PTR, _I32, _I32, _I32, _I64)
_pg_update_fast = _bind("pal_pg_update_", _I32, _PTR)
_pg_pop = _bind("pal_pg_pop", _I64, _PTR)
_pg_clear = _bind("pal_pg_clear", None, _PTR)
_pg_clear_value = _bind("pal_pg_clear_value", _I32, _PTR, _I64)
_pg_clear_value_fast = _bind("pal_pg_clear_value_", None, _PTR, _I64)
_pg_inwrap_x = _bind("pal_pg_inwrap_x", _I32, _PTR, _I32)
_pg_inwrap_y = _bind("pal_pg_inwrap_y", _I32, _PTR, _I32)
_pg_inwrap_z = _bind("pal_pg_inwrap_z", _I32, _PTR, _I32)
_pg_copy = _bind("pal_pg_copy", None, _PTR, _I64_PTR)
_pg_all_count = _bind("pal_pg_all_count", _I32, _PTR)
_pg_all_copy = _bind("pal_pg_all_copy", _I32, _PTR, _I32_PTR)
_pg_linear_get = _bind("pal_pg_linear_get", None, _PTR, _I32, _I32, _I64_PTR)
_pg_region_get = _bind("pal_pg_region_get", None, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _I64_PTR)
_pg_linear_set_scalar = _bind("pal_pg_linear_set_scalar", _I32, _PTR, _I32, _I32, _I64)
_pg_linear_set_scalar_fast = _bind("pal_pg_linear_set_scalar_", _I32, _PTR, _I32, _I32, _I64)
_pg_linear_set_array = _bind("pal_pg_linear_set_array", _I32, _PTR, _I32, _I32, _I64_PTR)
_pg_linear_set_array_fast = _bind("pal_pg_linear_set_array_", _I32, _PTR, _I32, _I32, _I64_PTR)
_pg_region_set_scalar = _bind("pal_pg_region_set_scalar", _I32, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _I64)
_pg_region_set_scalar_fast = _bind("pal_pg_region_set_scalar_", _I32, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _I64)
_pg_region_set_array = _bind("pal_pg_region_set_array", _I32, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _I64_PTR)
_pg_region_set_array_fast = _bind("pal_pg_region_set_array_", _I32, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _I64_PTR)

# PDEgrid C API
_pd_new = _bind("pal_pd_new", _PTR, _I32_PTR, _I32)
_pd_free = _bind("pal_pd_free", None, _PTR)
_pd_dim = _bind("pal_pd_dim", _I32, _PTR, _I32)
_pd_len = _bind("pal_pd_len", _I32, _PTR)
_pd_toi = _bind("pal_pd_toi", _I32, _PTR, _I32, _I32, _I32)
_pd_itox = _bind("pal_pd_itox", _I32, _PTR, _I32)
_pd_itoy = _bind("pal_pd_itoy", _I32, _PTR, _I32)
_pd_itoz = _bind("pal_pd_itoz", _I32, _PTR, _I32)
_pd_toi_safe = _bind("pal_pd_toi_safe", _I32, _PTR, _I32, _I32, _I32)
_pd_itox_safe = _bind("pal_pd_itox_safe", _I32, _PTR, _I32)
_pd_itoy_safe = _bind("pal_pd_itoy_safe", _I32, _PTR, _I32)
_pd_itoz_safe = _bind("pal_pd_itoz_safe", _I32, _PTR, _I32)
_pd_geti = _bind("pal_pd_geti", _F32, _PTR, _I32)
_pd_get = _bind("pal_pd_get", _F32, _PTR, _I32, _I32, _I32)
_pd_seti = _bind("pal_pd_seti", _I32, _PTR, _I32, _F32)
_pd_geti_fast = _bind("pal_pd_geti_", _F32, _PTR, _I32)
_pd_seti_fast = _bind("pal_pd_seti_", None, _PTR, _I32, _F32)
_pd_addi = _bind("pal_pd_addi", None, _PTR, _I32, _F32)
_pd_set = _bind("pal_pd_set", None, _PTR, _I32, _I32, _I32, _F32)
_pd_add = _bind("pal_pd_add", None, _PTR, _I32, _I32, _I32, _F32)
_pd_update = _bind("pal_pd_update", None, _PTR)
_pd_steps = _bind("pal_pd_steps", None, _PTR, _F32, _F32, _F32, _F32)
_pd_voxel = _bind("pal_pd_voxel", _F32, _PTR)
_pd_clear = _bind("pal_pd_clear", None, _PTR, _F32)
_pd_diffusion = _bind("pal_pd_diffusion", _I32, _PTR, _F32)
_pd_inwrap_x = _bind("pal_pd_inwrap_x", _I32, _PTR, _I32)
_pd_inwrap_y = _bind("pal_pd_inwrap_y", _I32, _PTR, _I32)
_pd_inwrap_z = _bind("pal_pd_inwrap_z", _I32, _PTR, _I32)
_pd_iswrap = _bind("pal_pd_iswrap", _I32, _PTR, _I32)
_pd_dx = _bind("pal_pd_dx", _F32, _PTR)
_pd_dy = _bind("pal_pd_dy", _F32, _PTR)
_pd_dz = _bind("pal_pd_dz", _F32, _PTR)
_pd_dt = _bind("pal_pd_dt", _F32, _PTR)
_pd_copy = _bind("pal_pd_copy", None, _PTR, _PTR)
_pd_linear_get = _bind("pal_pd_linear_get", None, _PTR, _I32, _I32, _F32_PTR)
_pd_region_get = _bind("pal_pd_region_get", None, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _F32_PTR)
_pd_linear_set_scalar = _bind("pal_pd_linear_set_scalar", None, _PTR, _I32, _I32, _F32)
_pd_linear_set_array = _bind("pal_pd_linear_set_array", None, _PTR, _I32, _I32, _F32_PTR)
_pd_region_set_scalar = _bind("pal_pd_region_set_scalar", None, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _F32)
_pd_region_set_array = _bind("pal_pd_region_set_array", None, _PTR, _I32, _I32, _I32, _I32, _I32, _I32, _F32_PTR)
_pd_bc_scalar = _bind("pal_pd_bc_scalar", _PTR, _PTR, _I32, _F32)
_pd_diffusion_bc = _bind("pal_pd_diffusion_bc", _I32, _PTR, _F32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_diffusion_bc_fast = _bind("pal_pd_diffusion_bc_fast", _I32, _PTR, _F32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_diffusion_mask = _bind("pal_pd_diffusion_mask", _I32, _PTR, _F32, _PTR)
_pd_diffusion_mask_fast = _bind("pal_pd_diffusion_mask_fast", _I32, _PTR, _F32, _PTR)
_pd_diffusion_field = _bind("pal_pd_diffusion_field", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_diffusion_field_fast = _bind("pal_pd_diffusion_field_fast", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_diffusion_interfaces = _bind("pal_pd_diffusion_interfaces", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_diffusion_interfaces_fast = _bind("pal_pd_diffusion_interfaces_fast", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_diffusion_radial_circle = _bind("pal_pd_diffusion_radial_circle", _I32, _PTR, _F32, _PTR)
_pd_diffusion_radial_circle_fast = _bind("pal_pd_diffusion_radial_circle_fast", _I32, _PTR, _F32, _PTR)
_pd_diffusion_radial_sphere = _bind("pal_pd_diffusion_radial_sphere", _I32, _PTR, _F32, _PTR)
_pd_diffusion_radial_sphere_fast = _bind("pal_pd_diffusion_radial_sphere_fast", _I32, _PTR, _F32, _PTR)
_pd_diffusion_adi = _bind("pal_pd_diffusion_adi", _I32, _PTR, _F32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_advection = _bind("pal_pd_advection", _I32, _PTR, _F32, _F32, _F32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_advection_fast = _bind("pal_pd_advection_fast", _I32, _PTR, _F32, _F32, _F32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_advection_field = _bind("pal_pd_advection_field", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_advection_field_fast = _bind("pal_pd_advection_field_fast", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_advection_interfaces = _bind("pal_pd_advection_interfaces", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)
_pd_advection_interfaces_fast = _bind("pal_pd_advection_interfaces_fast", _I32, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR, _PTR)



def _BCIsScalar(bc): return np.isscalar(bc)
@overload(_BCIsScalar)
def _BCIsScalarOverload(bc):
    if isinstance(bc, types.NoneType):
        def impl(bc): return False
        return impl
    if isinstance(bc, types.Number):
        def impl(bc): return True
        return impl
    if isinstance(bc, types.Array):
        def impl(bc): return False
        return impl

def _CheckBCValue(bc, gridPtr, dimension, axis):
    if bc is None: return
    if _pd_iswrap(gridPtr,axis): raise ValueError("boundary conditions cannot be specified on a wrapped axis")
    if np.isscalar(bc):
        if not np.isfinite(bc): raise ValueError("boundary values must be finite")
        return
    expected=1
    for d in range(dimension):
        if d != axis: expected *= _pd_dim(gridPtr,d)
    if bc.size != expected: raise ValueError("boundary array size does not match grid face")
    if bc.dtype != np.float32: raise ValueError("boundary arrays must use dtype float32")
    if not bc.flags.c_contiguous: raise ValueError("boundary arrays must be C-contiguous")
    for v in bc.flat:
        if not np.isfinite(v): raise ValueError("boundary values must be finite")

@overload(_CheckBCValue)
def _CheckBCValueOverload(bc, gridPtr, dimension, axis):
    if isinstance(bc, types.NoneType):
        def impl(bc, gridPtr, dimension, axis): return
        return impl
    if isinstance(bc, types.Number):
        def impl(bc, gridPtr, dimension, axis):
            if _pd_iswrap(gridPtr,axis): raise ValueError("boundary conditions cannot be specified on a wrapped axis")
            if not np.isfinite(bc): raise ValueError("boundary values must be finite")
        return impl
    if isinstance(bc, types.Array):
        if bc.dtype != types.float32 or bc.layout != "C":
            def impl(bc, gridPtr, dimension, axis): raise ValueError("boundary arrays must be C-contiguous float32 arrays")
            return impl
        def impl(bc, gridPtr, dimension, axis):
            if _pd_iswrap(gridPtr,axis): raise ValueError("boundary conditions cannot be specified on a wrapped axis")
            expected=1
            for d in range(dimension):
                if d != axis: expected *= _pd_dim(gridPtr,d)
            if bc.size != expected: raise ValueError("boundary array size does not match grid face")
            for v in bc.flat:
                if not np.isfinite(v): raise ValueError("boundary values must be finite")
        return impl

def _BCPointer(bc, gridPtr, slot):
    if bc is None: return 0
    if np.isscalar(bc): return _pd_bc_scalar(gridPtr,slot,bc)
    return bc.ctypes.data

@overload(_BCPointer)
def _BCPointerOverload(bc, gridPtr, slot):
    if isinstance(bc, types.NoneType):
        def impl(bc, gridPtr, slot): return 0
        return impl
    if isinstance(bc, types.Number):
        def impl(bc, gridPtr, slot): return _pd_bc_scalar(gridPtr,slot,bc)
        return impl
    if isinstance(bc, types.Array):
        def impl(bc, gridPtr, slot): return bc.ctypes.data
        return impl

# Native pickle snapshot API
_q_snapshot_size = _bind("pal_q_snapshot_size", _SIZE, _PTR)
_q_snapshot = _bind("pal_q_snapshot", _I32, _PTR, _U8_PTR, _SIZE)
_q_restore = _bind("pal_q_restore", _PTR, _U8_PTR, _SIZE)
_multi_snapshot_size = _bind("pal_multi_snapshot_size", _SIZE, _PTR)
_multi_snapshot = _bind("pal_multi_snapshot", _I32, _PTR, _U8_PTR, _SIZE)
_multi_restore = _bind("pal_multi_restore", _PTR, _U8_PTR, _SIZE)
_ag_snapshot_size = _bind("pal_ag_snapshot_size", _SIZE, _PTR)
_ag_snapshot = _bind("pal_ag_snapshot", _I32, _PTR, _U8_PTR, _SIZE)
_ag_restore = _bind("pal_ag_restore", _PTR, _U8_PTR, _SIZE)
_pg_snapshot_size = _bind("pal_pg_snapshot_size", _SIZE, _PTR)
_pg_snapshot = _bind("pal_pg_snapshot", _I32, _PTR, _U8_PTR, _SIZE)
_pg_restore = _bind("pal_pg_restore", _PTR, _U8_PTR, _SIZE)
_pd_snapshot_size = _bind("pal_pd_snapshot_size", _SIZE, _PTR)
_pd_snapshot = _bind("pal_pd_snapshot", _I32, _PTR, _U8_PTR, _SIZE)
_pd_restore = _bind("pal_pd_restore", _PTR, _U8_PTR, _SIZE)


# Public typing interfaces. These exist for autocomplete/type checking only;
# New* factories return the concrete safe/fast jitclass implementations below.
class _MultinomialProtocol(Protocol):
    def Binomial(self, n: int, p: float) -> int: ...
    def Setup(self, n: int): ...
    def Sample(self, p: float) -> int: ...


class _IListProtocol(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> int: ...
    def Append(self, i: int): ...
    def Clear(self): ...
    def Random(self) -> int: ...
    def Shuffle(self): ...
    def All(self) -> np.ndarray: ...
    def Iter(self) -> np.ndarray: ...


class _AgentGridProtocol(Protocol):
    def __len__(self) -> int: ...
    @property
    def xDim(self) -> int: ...
    @property
    def yDim(self) -> int: ...
    @property
    def zDim(self) -> int: ...
    @property
    def nDims(self) -> int: ...
    @property
    def wrapX(self) -> bool: ...
    @property
    def wrapY(self) -> bool: ...
    @property
    def wrapZ(self) -> bool: ...
    def Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None): ...
    def Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False): ...
    def GetPop(self) -> int: ...
    def Alive(self, agent: int) -> bool: ...
    def NewAgentSQ(self, x: int, y: int = -1, z: int = -1) -> int: ...
    def NewAgent(self, x: float, y: float = -1.0, z: float = -1.0) -> int: ...
    def Dispose(self, agent: int): ...
    def I(self, agent: int) -> int: ...
    def XSQ(self, agent: int) -> int: ...
    def YSQ(self, agent: int) -> int: ...
    def ZSQ(self, agent: int) -> int: ...
    def X(self, agent: int) -> float: ...
    def Y(self, agent: int) -> float: ...
    def Z(self, agent: int) -> float: ...
    def __getitem__(self, key) -> float: ...
    def __setitem__(self, key, value: float): ...
    def ToI(self, x: int, y: int = -1, z: int = -1) -> int: ...
    def ItoX(self, i: int) -> int: ...
    def ItoY(self, i: int) -> int: ...
    def ItoZ(self, i: int) -> int: ...
    def InWrapSQX(self, value: int) -> int: ...
    def InWrapSQY(self, value: int) -> int: ...
    def InWrapSQZ(self, value: int) -> int: ...
    def InWrapX(self, value: float) -> float: ...
    def InWrapY(self, value: float) -> float: ...
    def InWrapZ(self, value: float) -> float: ...
    def DispWrapX(self, x1: float, x2: float) -> float: ...
    def DispWrapY(self, y1: float, y2: float) -> float: ...
    def DispWrapZ(self, z1: float, z2: float) -> float: ...
    @property
    def counts(self) -> object: ...
    def LastAgent(self, x: int, y: int = -1, z: int = -1) -> int: ...
    def MoveSQ(self, agent: int, x: int, y: int = -1, z: int = -1): ...
    def Move(self, agent: int, x: float, y: float = -1.0, z: float = -1.0): ...
    def AgentsAt(self, x: int, y: int = -1, z: int = -1): ...
    def All(self, shuffle: bool = False) -> np.ndarray: ...


class _GridProtocol(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value): ...
    @property
    def xDim(self) -> int: ...
    @property
    def yDim(self) -> int: ...
    @property
    def zDim(self) -> int: ...
    @property
    def nDims(self) -> int: ...
    @property
    def wrapX(self) -> bool: ...
    @property
    def wrapY(self) -> bool: ...
    @property
    def wrapZ(self) -> bool: ...
    def Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None): ...
    def Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False): ...
    def ToI(self, x: int, y: int = -1, z: int = -1) -> int: ...
    def ItoX(self, i: int) -> int: ...
    def ItoY(self, i: int) -> int: ...
    def ItoZ(self, i: int) -> int: ...


class _PopGridProtocol(Protocol):
    def __len__(self) -> int: ...
    @property
    def xDim(self) -> int: ...
    @property
    def yDim(self) -> int: ...
    @property
    def zDim(self) -> int: ...
    @property
    def nDims(self) -> int: ...
    @property
    def wrapX(self) -> bool: ...
    @property
    def wrapY(self) -> bool: ...
    @property
    def wrapZ(self) -> bool: ...
    def Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None): ...
    def Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False): ...
    def ToI(self, x: int, y: int = -1, z: int = -1) -> int: ...
    def ItoX(self, i: int) -> int: ...
    def ItoY(self, i: int) -> int: ...
    def ItoZ(self, i: int) -> int: ...
    def InWrapX(self, value: int) -> int: ...
    def InWrapY(self, value: int) -> int: ...
    def InWrapZ(self, value: int) -> int: ...
    def __getitem__(self, key) -> int: ...
    def __setitem__(self, key, value: int): ...
    def Add(self, value: int, x: int, y: int = -1, z: int = -1): ...
    def Update(self): ...
    def Reset(self): ...
    def GetPop(self) -> int: ...
    def All(self) -> np.ndarray: ...


class _PDEgridProtocol(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> float: ...
    @property
    def xDim(self) -> int: ...
    @property
    def yDim(self) -> int: ...
    @property
    def zDim(self) -> int: ...
    @property
    def nDims(self) -> int: ...
    @property
    def wrapX(self) -> bool: ...
    @property
    def wrapY(self) -> bool: ...
    @property
    def wrapZ(self) -> bool: ...
    def Box(self, x1: int, x2: int, y1: int = None, y2: int = None, z1: int = None, z2: int = None): ...
    def Hood(self, hood, x: int, y: int = None, z: int = None, *, unroll: bool = False): ...
    def ToI(self, x: int, y: int = -1, z: int = -1) -> int: ...
    def ItoX(self, i: int) -> int: ...
    def ItoY(self, i: int) -> int: ...
    def ItoZ(self, i: int) -> int: ...
    def InWrapX(self, value: int) -> int: ...
    def InWrapY(self, value: int) -> int: ...
    def InWrapZ(self, value: int) -> int: ...
    def __getitem__(self, key) -> float: ...
    def __setitem__(self, key, value: float): ...
    def Add(self, value: float, x: int, y: int = -1, z: int = -1): ...
    def Update(self): ...
    def Reset(self): ...
    def SetTimeSpaceStep(self, dt: float, dx: float, dy: float = 1.0, dz: float = 1.0): ...
    def Dx(self) -> float: ...
    def Dy(self) -> float: ...
    def Dz(self) -> float: ...
    def Dt(self) -> float: ...
    def Diffusion(self, rateConstant: float, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None): ...
    def DiffusionMask(self, rateConstant: float, mask=None): ...
    def DiffusionField(self, rateConstants, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None): ...
    def DiffusionInterfaces(self, rateConstantsX, rateConstantsY=None, rateConstantsZ=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None): ...
    def DiffusionADI(self, rateConstant: float, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None): ...
    def DiffusionRadialCircle(self, rateConstant: float, outerBC=None): ...
    def DiffusionRadialSphere(self, rateConstant: float, outerBC=None): ...
    def Advection(self, vx: float, vy: float = 0.0, vz: float = 0.0, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None): ...
    def AdvectionField(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None): ...
    def AdvectionInterfaces(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None): ...



class _MultinomialSafePython:
    def __init__(self, ptr):
        self._ptr = ptr

    def Binomial(self, n, p):
        if not np.isfinite(n) or n!=int(n) or n<0 or n>np.iinfo(np.int64).max: raise ValueError("n must be a nonnegative int64 integer")
        out = _multi_binomial(self._ptr, n, p)
        if out < 0: raise ValueError("invalid binomial arguments")
        return out

    def Setup(self, n):
        if not np.isfinite(n) or n!=int(n) or n<0 or n>np.iinfo(np.int64).max: raise ValueError("n must be a nonnegative int64 integer")
        if not _multi_setup(self._ptr, n): raise ValueError("invalid multinomial n")
        return self

    def Sample(self, p):
        out = _multi_sample(self._ptr, p)
        if out < 0: raise ValueError("p exceeds remaining multinomial probability")
        return out


class _MultinomialFastPython:
    def __init__(self, ptr):
        self._ptr = ptr

    def Binomial(self, n, p):
        return _multi_binomial(self._ptr, n, p)

    def Setup(self, n):
        _multi_setup(self._ptr, n)
        return self

    def Sample(self, p):
        return _multi_sample(self._ptr, p)


_MultinomialSafe=_jitclass([("_ptr", uint64)])(_MultinomialSafePython)
_MultinomialFast=_jitclass([("_ptr", uint64)])(_MultinomialFastPython)


def _MultinomialSampleAt(multi,p,sourceLine):
    return multi.Sample(p)

@overload(_MultinomialSampleAt,inline="never",prefer_literal=True)
def _olMultinomialSampleAt(multi,p,sourceLine):
    if multi==_MultinomialSafe.class_type.instance_type:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        message="Multinomial.Sample failed at source line "+str(line) if line>=0 else "Multinomial.Sample failed"
        def impl(multi,p,sourceLine):
            out=_multi_sample(multi._ptr,p)
            if out<0:
                raise ValueError(message)
            return out
        return impl
    if multi==_MultinomialFast.class_type.instance_type:
        def impl(multi,p,sourceLine):
            return _multi_sample(multi._ptr,p)
        return impl
    def impl(multi,p,sourceLine):
        return multi.Sample(p)
    return impl


@intrinsic
def _IListHasCapacity(typingctx, handle):
    sig=types.boolean(types.uint64)
    def codegen(context,builder,signature,args):
        i32=lir.IntType(32); st=lir.LiteralStructType([i32,i32,i32.as_pointer()])
        ptr=builder.inttoptr(args[0],st.as_pointer())
        length=builder.load(builder.gep(ptr,[lir.Constant(i32,0),lir.Constant(i32,0)]))
        capacity=builder.load(builder.gep(ptr,[lir.Constant(i32,0),lir.Constant(i32,1)]))
        return builder.icmp_signed('<',length,capacity)
    return sig,codegen

@intrinsic
def _IListAppendNoGrow(typingctx, handle, value):
    sig=types.void(types.uint64,types.int32)
    def codegen(context,builder,signature,args):
        i32=lir.IntType(32); st=lir.LiteralStructType([i32,i32,i32.as_pointer()])
        ptr=builder.inttoptr(args[0],st.as_pointer())
        lenp=builder.gep(ptr,[lir.Constant(i32,0),lir.Constant(i32,0)])
        length=builder.load(lenp)
        items=builder.load(builder.gep(ptr,[lir.Constant(i32,0),lir.Constant(i32,2)]))
        builder.store(args[1],builder.gep(items,[length]))
        builder.store(builder.add(length,lir.Constant(i32,1)),lenp)
    return sig,codegen

def _IListAppendInline(q,i): return q.Append(i)
def _IListAppendAt(q,i,sourceLine): return q.Append(i)

@overload(_IListAppendAt,inline="never",prefer_literal=True)
def _olIListAppendAt(q,i,sourceLine):
    if q==_IListSafe.class_type.instance_type:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        message="IList values must be nonnegative int32 indices at source line "+str(line) if line>=0 else "IList values must be nonnegative int32 indices"
        def impl(q,i,sourceLine):
            if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise ValueError(message)
            value=int(i)
            if _IListHasCapacity(q._ptr): _IListAppendNoGrow(q._ptr,value)
            elif not _q_add(q._ptr,value): raise MemoryError("IList allocation failed")
        return impl
    if q==_IListFast.class_type.instance_type:
        def impl(q,i,sourceLine):
            if _IListHasCapacity(q._ptr): _IListAppendNoGrow(q._ptr,i)
            elif not _q_add(q._ptr,i): raise MemoryError("IList allocation failed")
        return impl

class _IListSafePython:
    """A tiny native handle. Query storage itself lives in C."""

    def __init__(self, ptr, dimension):
        self._ptr = ptr
        self._dimension = dimension

    def __len__(self):
        return _q_len(self._ptr)

    def __getitem__(self, index):
        if not np.isfinite(index) or index!=int(index) or index<np.iinfo(np.int32).min or index>np.iinfo(np.int32).max: raise IndexError("IList index must be an int32 integer")
        out=_q_get_safe(self._ptr,index)
        if out==np.iinfo(np.int64).min: raise IndexError("IList index out of bounds")
        return int(out)

    def Append(self, i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise ValueError("IList values must be nonnegative int32 indices")
        if not _q_add(self._ptr, i):
            raise MemoryError("IList allocation failed")
        return self

    def Clear(self):
        _q_clear(self._ptr)
        return self

    def Random(self):
        out=_q_random(self._ptr)
        if out<0: raise ValueError("cannot sample an empty IList")
        return out

    def Shuffle(self):
        _q_shuffle(self._ptr)
        return self

    def All(self):
        out = np.empty(len(self), dtype=np.int32)
        _q_copy(self._ptr, out.ctypes)
        return out

    def Iter(self):
        return nb.carray(_q_data(self._ptr), len(self))

_IListSafe=_jitclass([("_ptr", uint64), ("_dimension", int32)])(_IListSafePython)

class _IListFastPython:
    """A tiny native handle. Query storage itself lives in C."""

    def __init__(self, ptr, dimension):
        self._ptr = ptr
        self._dimension = dimension

    def __len__(self):
        return _q_len(self._ptr)

    def __getitem__(self, index):
        return _q_get(self._ptr, index)

    def Append(self, i):
        if not _q_add(self._ptr, i):
            raise MemoryError("IList allocation failed")
        return self

    def Clear(self):
        _q_clear(self._ptr)
        return self

    def Random(self):
        return _q_random(self._ptr)

    def Shuffle(self):
        _q_shuffle(self._ptr)
        return self

    def All(self):
        out = np.empty(len(self), dtype=np.int32)
        _q_copy(self._ptr, out.ctypes)
        return out

    def Iter(self):
        return nb.carray(_q_data(self._ptr), len(self))

_IListFast=_jitclass([("_ptr", uint64), ("_dimension", int32)])(_IListFastPython)




def _SliceBounds(s,n):
    if s.step not in (None,1): raise ValueError("PAL slices currently require step=1")
    if s.start is not None and (not np.isfinite(s.start) or s.start!=int(s.start)): raise ValueError("slice bounds must be finite integers")
    if s.stop is not None and (not np.isfinite(s.stop) or s.stop!=int(s.stop)): raise ValueError("slice bounds must be finite integers")
    start=0 if s.start is None else int(s.start)
    stop=n if s.stop is None else int(s.stop)
    if start<0: start+=n
    if stop<0: stop+=n
    start=max(0,min(n,start)); stop=max(0,min(n,stop))
    if stop<start: stop=start
    return int(start),int(stop)

@overload(_SliceBounds)
def _olSliceBounds(s,n):
    if isinstance(s,types.SliceType):
        def impl(s,n):
            step=1 if s.step is None else s.step
            if step!=1: raise ValueError("PAL slices currently require step=1")
            if s.start is not None and (not np.isfinite(s.start) or s.start!=int(s.start)): raise ValueError("slice bounds must be finite integers")
            if s.stop is not None and (not np.isfinite(s.stop) or s.stop!=int(s.stop)): raise ValueError("slice bounds must be finite integers")
            start=0 if s.start is None else int(s.start)
            stop=n if s.stop is None else int(s.stop)
            if start<0: start+=n
            if stop<0: stop+=n
            start=max(0,min(n,start)); stop=max(0,min(n,stop))
            if stop<start: stop=start
            return int(start),int(stop)
        return impl

def _AxisBounds(k,n,safe):
    if isinstance(k,slice): return _SliceBounds(k,n)
    if safe and (not np.isfinite(k) or k!=int(k)): raise ValueError("index must be a finite integer")
    i=int(k)
    if safe and (i<0 or i>=n): raise IndexError("index out of bounds")
    return i,i+1

@overload(_AxisBounds)
def _olAxisBounds(k,n,safe):
    if isinstance(k,types.SliceType):
        def impl(k,n,safe): return _SliceBounds(k,n)
    else:
        def impl(k,n,safe):
            if safe and (not np.isfinite(k) or k!=int(k)): raise ValueError("index must be a finite integer")
            i=int(k)
            if safe and (i<0 or i>=n): raise IndexError("index out of bounds")
            return i,i+1
    return impl

def _RegionBounds(grid,key,safe):
    # Python path. Tuple slices use spatial dimensions; one slice uses linear indexing.
    if isinstance(key,slice):
        a,b=_SliceBounds(key,len(grid)); return (a,b,0,1,0,1),(b-a,)
    if not isinstance(key,tuple): return None,None
    if len(key)!=grid._dimension: raise IndexError("slice tuple must match grid dimension")
    dims=(grid.xDim, grid.yDim if grid._dimension>1 else 1, grid.zDim if grid._dimension>2 else 1)
    bounds=[]; shape=[]
    for d,k in enumerate(key):
        a,b=_AxisBounds(k,dims[d],safe); bounds.extend((a,b))
        if isinstance(k,slice): shape.append(b-a)
    while len(bounds)<6: bounds.extend((0,1))
    return tuple(bounds),tuple(shape)

def _PDEGetSlice(grid,key,safe):
    bounds,shape=_RegionBounds(grid,key,safe)
    out=np.empty(int(np.prod(shape)),dtype=np.float32)
    if isinstance(key,slice): _pd_linear_get(grid._ptr,bounds[0],bounds[1],out.ctypes)
    else: _pd_region_get(grid._ptr,*bounds,out.ctypes)
    return out.reshape(shape)

def _PopGetSlice(grid,key,safe):
    bounds,shape=_RegionBounds(grid,key,safe)
    out=np.empty(int(np.prod(shape)),dtype=np.int64)
    if isinstance(key,slice): _pg_linear_get(grid._ptr,bounds[0],bounds[1],out.ctypes)
    else: _pg_region_get(grid._ptr,*bounds,out.ctypes)
    return out.reshape(shape)

def _CountsGetSlice(grid,key,safe):
    bounds,shape=_RegionBounds(grid,key,safe)
    out=np.empty(int(np.prod(shape)),dtype=np.int32)
    if isinstance(key,slice): _ag_counts_linear(grid._ptr,bounds[0],bounds[1],out.ctypes)
    else: _ag_counts_region(grid._ptr,*bounds,out.ctypes)
    return out.reshape(shape)

def _slice_shape_impl(grid,key,mask):
    pass

# Numba overload factories: key type fixes output dimensionality at compile time.
def _make_getslice_overload(dtype,linearFn,regionFn):
    def ol(grid,key,safe):
        if isinstance(key,types.SliceType):
            def impl(grid,key,safe):
                a,b=_SliceBounds(key,len(grid)); out=np.empty(b-a,dtype=dtype); linearFn(grid._ptr,a,b,out.ctypes); return out
            return impl
        if isinstance(key,types.BaseTuple):
            kt=key.types; mask=tuple(isinstance(t,types.SliceType) for t in kt); nd=sum(mask)
            if nd==0: return None
            if len(kt)==1:
                def impl(grid,key,safe):
                    a0,b0=_AxisBounds(key[0],grid.xDim,safe); out=np.empty(b0-a0,dtype=dtype); regionFn(grid._ptr,a0,b0,0,1,0,1,out.ctypes); return out
                return impl
            if len(kt)==2:
                if mask==(True,True):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe); a1,b1=_AxisBounds(key[1],grid.yDim,safe); out=np.empty((b0-a0,b1-a1),dtype=dtype); regionFn(grid._ptr,a0,b0,a1,b1,0,1,out.ctypes); return out
                    return impl
                if mask==(True,False):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe); a1,b1=_AxisBounds(key[1],grid.yDim,safe); out=np.empty(b0-a0,dtype=dtype); regionFn(grid._ptr,a0,b0,a1,b1,0,1,out.ctypes); return out
                    return impl
                if mask==(False,True):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe); a1,b1=_AxisBounds(key[1],grid.yDim,safe); out=np.empty(b1-a1,dtype=dtype); regionFn(grid._ptr,a0,b0,a1,b1,0,1,out.ctypes); return out
                    return impl
            if len(kt)==3:
                # Eight masks; all-scalar handled elsewhere.
                if nd==3:
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);out=np.empty((b0-a0,b1-a1,b2-a2),dtype=dtype);regionFn(grid._ptr,a0,b0,a1,b1,a2,b2,out.ctypes);return out
                    return impl
                if mask==(True,True,False):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);out=np.empty((b0-a0,b1-a1),dtype=dtype);regionFn(grid._ptr,a0,b0,a1,b1,a2,b2,out.ctypes);return out
                    return impl
                if mask==(True,False,True):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);out=np.empty((b0-a0,b2-a2),dtype=dtype);regionFn(grid._ptr,a0,b0,a1,b1,a2,b2,out.ctypes);return out
                    return impl
                if mask==(False,True,True):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);out=np.empty((b1-a1,b2-a2),dtype=dtype);regionFn(grid._ptr,a0,b0,a1,b1,a2,b2,out.ctypes);return out
                    return impl
                if mask==(True,False,False):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);out=np.empty(b0-a0,dtype=dtype);regionFn(grid._ptr,a0,b0,a1,b1,a2,b2,out.ctypes);return out
                    return impl
                if mask==(False,True,False):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);out=np.empty(b1-a1,dtype=dtype);regionFn(grid._ptr,a0,b0,a1,b1,a2,b2,out.ctypes);return out
                    return impl
                if mask==(False,False,True):
                    def impl(grid,key,safe):
                        a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);out=np.empty(b2-a2,dtype=dtype);regionFn(grid._ptr,a0,b0,a1,b1,a2,b2,out.ctypes);return out
                    return impl
        return None
    return ol

@overload(_PDEGetSlice)
def _olPDEGetSlice(grid,key,safe): return _make_getslice_overload(np.float32,_pd_linear_get,_pd_region_get)(grid,key,safe)
@overload(_PopGetSlice)
def _olPopGetSlice(grid,key,safe): return _make_getslice_overload(np.int64,_pg_linear_get,_pg_region_get)(grid,key,safe)
@overload(_CountsGetSlice)
def _olCountsGetSlice(grid,key,safe): return _make_getslice_overload(np.int32,_ag_counts_linear,_ag_counts_region)(grid,key,safe)



def _PDEItem(grid,key,safe):
    if isinstance(key,slice) or isinstance(key,tuple) and any(isinstance(k,slice) for k in key): return _PDEGetSlice(grid,key,safe)
    if isinstance(key,tuple):
        if len(key)==2: return _pd_geti(grid._ptr,grid.ToI(key[0],key[1]))
        if len(key)==3: return _pd_geti(grid._ptr,grid.ToI(key[0],key[1],key[2]))
        raise IndexError("PDEgrid expects i, x,y, x,y,z, or slices")
    if safe: grid._CheckI(key)
    return _pd_geti(grid._ptr,key)

@overload(_PDEItem)
def _olPDEItem(grid,key,safe):
    if isinstance(key,types.SliceType) or isinstance(key,types.BaseTuple) and any(isinstance(t,types.SliceType) for t in key.types):
        def impl(grid,key,safe): return _PDEGetSlice(grid,key,safe)
        return impl
    if isinstance(key,types.BaseTuple):
        if len(key.types)==2:
            def impl(grid,key,safe): return _pd_geti(grid._ptr,grid.ToI(key[0],key[1]))
        else:
            def impl(grid,key,safe): return _pd_geti(grid._ptr,grid.ToI(key[0],key[1],key[2]))
        return impl
    # safe is a compile-time literal when called by the concrete safe/fast
    # jitclasses. Specialize here so the fast class never has to type-check
    # safe-only methods such as _CheckI.
    if isinstance(safe,types.BooleanLiteral) and not safe.literal_value:
        def impl(grid,key,safe): return _pd_geti_fast(grid._ptr,key)
        return impl
    def impl(grid,key,safe):
        if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError("index must be a valid integer")
        v=_pd_geti(grid._ptr,key)
        if np.isnan(v): raise IndexError("index out of bounds")
        return v
    return impl


def _PDEItemAt(grid,key,sourceLine): return grid[key]
@overload(_PDEItemAt,inline="never",prefer_literal=True)
def _olPDEItemAt(grid,key,sourceLine):
    safeType=_PDEgridSafe.class_type.instance_type
    fastType=_PDEgridFast.class_type.instance_type
    if grid==safeType:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        suffix=" at source line "+str(line) if line>=0 else ""
        badShape="PDEgrid expects i, x,y, x,y,z, or slices"+suffix
        badIndex="index must be a valid integer"+suffix
        outBounds="index out of bounds"+suffix
        if isinstance(key,types.BaseTuple):
            if len(key.types)==2:
                def impl(grid,key,sourceLine): return _pd_geti(grid._ptr,grid.ToI(key[0],key[1],_palLine=sourceLine))
            elif len(key.types)==3:
                def impl(grid,key,sourceLine): return _pd_geti(grid._ptr,grid.ToI(key[0],key[1],key[2],_palLine=sourceLine))
            else:
                def impl(grid,key,sourceLine): raise IndexError(badShape)
            return impl
        if isinstance(key,types.SliceType):
            def impl(grid,key,sourceLine): return _PDEGetSlice(grid,key,True)
            return impl
        def impl(grid,key,sourceLine):
            if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError(badIndex)
            v=_pd_geti(grid._ptr,key)
            if np.isnan(v): raise IndexError(outBounds)
            return v
        return impl
    if grid==fastType:
        if isinstance(key,types.BaseTuple) and any(isinstance(t,types.SliceType) for t in key.types):
            def impl(grid,key,sourceLine): return _PDEGetSlice(grid,key,False)
        elif isinstance(key,types.BaseTuple):
            if len(key.types)==2:
                def impl(grid,key,sourceLine): return _pd_geti_fast(grid._ptr,grid.ToI(key[0],key[1]))
            else:
                def impl(grid,key,sourceLine): return _pd_geti_fast(grid._ptr,grid.ToI(key[0],key[1],key[2]))
        elif isinstance(key,types.SliceType):
            def impl(grid,key,sourceLine): return _PDEGetSlice(grid,key,False)
        else:
            def impl(grid,key,sourceLine): return _pd_geti_fast(grid._ptr,key)
        return impl

def _PopItemAt(grid,key,sourceLine): return grid[key]
@overload(_PopItemAt,inline="never",prefer_literal=True)
def _olPopItemAt(grid,key,sourceLine):
    safeType=_PopGridSafe.class_type.instance_type
    fastType=_PopGridFast.class_type.instance_type
    if grid==safeType:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        suffix=" at source line "+str(line) if line>=0 else ""
        badShape="PopGrid expects i, x,y, x,y,z, or slices"+suffix
        badIndex="index must be a valid integer"+suffix
        outBounds="index out of bounds"+suffix
        if isinstance(key,types.BaseTuple):
            if len(key.types)==2:
                def impl(grid,key,sourceLine): return _pg_geti(grid._ptr,grid.ToI(key[0],key[1],_palLine=sourceLine))
            elif len(key.types)==3:
                def impl(grid,key,sourceLine): return _pg_geti(grid._ptr,grid.ToI(key[0],key[1],key[2],_palLine=sourceLine))
            else:
                def impl(grid,key,sourceLine): raise IndexError(badShape)
            return impl
        if isinstance(key,types.SliceType):
            def impl(grid,key,sourceLine): return _PopGetSlice(grid,key,True)
            return impl
        def impl(grid,key,sourceLine):
            if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError(badIndex)
            v=_pg_geti(grid._ptr,key)
            if v<0: raise IndexError(outBounds)
            return v
        return impl
    if grid==fastType:
        def impl(grid,key,sourceLine): return _PopItem(grid,key,False)
        return impl

def _PopItem(grid,key,safe):
    if isinstance(key,slice) or isinstance(key,tuple) and any(isinstance(k,slice) for k in key): return _PopGetSlice(grid,key,safe)
    if isinstance(key,tuple):
        if len(key)==2: return _pg_geti(grid._ptr,grid.ToI(key[0],key[1]))
        if len(key)==3: return _pg_geti(grid._ptr,grid.ToI(key[0],key[1],key[2]))
        raise IndexError("PopGrid expects i, x,y, x,y,z, or slices")
    if safe: grid._CheckI(key)
    return _pg_geti(grid._ptr,key)

@overload(_PopItem)
def _olPopItem(grid,key,safe):
    if isinstance(key,types.SliceType) or isinstance(key,types.BaseTuple) and any(isinstance(t,types.SliceType) for t in key.types):
        def impl(grid,key,safe): return _PopGetSlice(grid,key,safe)
        return impl
    if isinstance(key,types.BaseTuple):
        if len(key.types)==2:
            def impl(grid,key,safe): return _pg_geti(grid._ptr,grid.ToI(key[0],key[1]))
        else:
            def impl(grid,key,safe): return _pg_geti(grid._ptr,grid.ToI(key[0],key[1],key[2]))
        return impl
    # Do not leave the safe check behind a runtime branch: Numba still types
    # both branches, and _PopGridFast intentionally has no _CheckI method.
    if isinstance(safe,types.BooleanLiteral) and not safe.literal_value:
        def impl(grid,key,safe): return _pg_geti(grid._ptr,key)
        return impl
    def impl(grid,key,safe):
        if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError("index must be a valid integer")
        v=_pg_geti(grid._ptr,key)
        if v<0: raise IndexError("index out of bounds")
        return v
    return impl

def _CountsItem(grid,key,safe):
    if isinstance(key,slice) or isinstance(key,tuple) and any(isinstance(k,slice) for k in key): return _CountsGetSlice(grid,key,safe)
    if isinstance(key,tuple):
        if safe:
            if len(key)!=grid._dimension: raise IndexError("AgentGrid counts coordinate dimensionality mismatch")
            dims=(grid.xDim, grid.yDim if grid._dimension>1 else 1, grid.zDim if grid._dimension>2 else 1)
            for d,v in enumerate(key):
                if not np.isfinite(v) or v!=int(v) or v<0 or v>=dims[d]: raise IndexError("AgentGrid counts coordinate out of bounds")
        if len(key)==2: return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,key[0],key[1],-1))
        if len(key)==3: return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,key[0],key[1],key[2]))
        if len(key)==1: return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,key[0],-1,-1))
        raise IndexError("AgentGrid counts expects i or grid coordinates")
    if safe and (not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid)): raise IndexError("AgentGrid counts index out of bounds")
    return _ag_count_i(grid._ptr,key)

@overload(_CountsItem)
def _olCountsItem(grid,key,safe):
    if isinstance(key,types.SliceType) or isinstance(key,types.BaseTuple) and any(isinstance(t,types.SliceType) for t in key.types):
        def impl(grid,key,safe): return _CountsGetSlice(grid,key,safe)
        return impl
    safeMode=not (isinstance(safe,types.BooleanLiteral) and not safe.literal_value)
    if isinstance(key,types.BaseTuple):
        n=len(key.types)
        if safeMode:
            if n==1:
                def impl(grid,key,safe):
                    x=key[0]
                    if grid._dimension!=1 or not np.isfinite(x) or x!=int(x) or x<0 or x>=grid.xDim: raise IndexError("AgentGrid counts coordinate out of bounds")
                    return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,x,-1,-1))
            elif n==2:
                def impl(grid,key,safe):
                    x,y=key
                    if grid._dimension!=2 or not np.isfinite(x) or x!=int(x) or not np.isfinite(y) or y!=int(y) or x<0 or x>=grid.xDim or y<0 or y>=grid.yDim: raise IndexError("AgentGrid counts coordinate out of bounds")
                    return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,x,y,-1))
            elif n==3:
                def impl(grid,key,safe):
                    x,y,z=key
                    if grid._dimension!=3 or not np.isfinite(x) or x!=int(x) or not np.isfinite(y) or y!=int(y) or not np.isfinite(z) or z!=int(z) or x<0 or x>=grid.xDim or y<0 or y>=grid.yDim or z<0 or z>=grid.zDim: raise IndexError("AgentGrid counts coordinate out of bounds")
                    return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,x,y,z))
            else:
                return None
        else:
            if n==1:
                def impl(grid,key,safe): return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,key[0],-1,-1))
            elif n==2:
                def impl(grid,key,safe): return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,key[0],key[1],-1))
            elif n==3:
                def impl(grid,key,safe): return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,key[0],key[1],key[2]))
            else:
                return None
        return impl
    if safeMode:
        def impl(grid,key,safe):
            if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError("AgentGrid counts index out of bounds")
            return _ag_count_i(grid._ptr,key)
    else:
        def impl(grid,key,safe): return _ag_count_i(grid._ptr,key)
    return impl


def _AgentCountsItemAt(grid,key,sourceLine): return _CountsItem(grid,key,True)
@overload(_AgentCountsItemAt,inline="never",prefer_literal=True)
def _olAgentCountsItemAt(grid,key,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    badCoord="AgentGrid counts coordinate out of bounds"+suffix
    badIndex="AgentGrid counts index out of bounds"+suffix
    if isinstance(key,types.BaseTuple):
        n=len(key.types)
        if n==1:
            def impl(grid,key,sourceLine):
                x=key[0]
                if grid._dimension!=1 or not np.isfinite(x) or x!=int(x) or x<0 or x>=grid.xDim: raise IndexError(badCoord)
                return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,x,-1,-1))
        elif n==2:
            def impl(grid,key,sourceLine):
                x,y=key
                if grid._dimension!=2 or not np.isfinite(x) or x!=int(x) or not np.isfinite(y) or y!=int(y) or x<0 or x>=grid.xDim or y<0 or y>=grid.yDim: raise IndexError(badCoord)
                return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,x,y,-1))
        elif n==3:
            def impl(grid,key,sourceLine):
                x,y,z=key
                if grid._dimension!=3 or not np.isfinite(x) or x!=int(x) or not np.isfinite(y) or y!=int(y) or not np.isfinite(z) or z!=int(z) or x<0 or x>=grid.xDim or y<0 or y>=grid.yDim or z<0 or z>=grid.zDim: raise IndexError(badCoord)
                return _ag_count_i(grid._ptr,_ag_toi(grid._ptr,x,y,z))
        else: return None
        return impl
    if isinstance(key,types.SliceType):
        def impl(grid,key,sourceLine): return _CountsGetSlice(grid,key,True)
        return impl
    def impl(grid,key,sourceLine):
        if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError(badIndex)
        return _ag_count_i(grid._ptr,key)
    return impl

def _SetRegionPython(grid,key,value,safe,isPop):
    bounds,shape=_RegionBounds(grid,key,safe)
    scalar=np.isscalar(value)
    if isPop:
        if scalar:
            if safe and (not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max): raise ValueError("PopGrid values must be nonnegative int64 integers")
            fn=_pg_linear_set_scalar if isinstance(key,slice) and safe else _pg_linear_set_scalar_fast if isinstance(key,slice) else _pg_region_set_scalar if safe else _pg_region_set_scalar_fast
            ok=fn(grid._ptr,bounds[0],bounds[1],value) if isinstance(key,slice) else fn(grid._ptr,*bounds,value)
        else:
            raw=np.asarray(value)
            if safe and (np.any(~np.isfinite(raw)) or np.any(raw!=np.floor(raw)) or np.any(raw<0) or np.any(raw>np.iinfo(np.int64).max)): raise ValueError("PopGrid values must be nonnegative int64 integers")
            arr=np.asarray(value,dtype=np.int64)
            if arr.shape!=shape: raise ValueError("assigned array shape must match slice shape")
            flat=np.ascontiguousarray(arr).reshape(-1)
            fn=_pg_linear_set_array if isinstance(key,slice) and safe else _pg_linear_set_array_fast if isinstance(key,slice) else _pg_region_set_array if safe else _pg_region_set_array_fast
            ok=fn(grid._ptr,bounds[0],bounds[1],flat.ctypes) if isinstance(key,slice) else fn(grid._ptr,*bounds,flat.ctypes)
        if safe and not ok: raise ValueError("PopGrid slice assignment value is outside capacity or population overflowed")
    else:
        if scalar:
            if safe and not np.isfinite(value): raise ValueError("value must be finite")
            if isinstance(key,slice): _pd_linear_set_scalar(grid._ptr,bounds[0],bounds[1],value)
            else: _pd_region_set_scalar(grid._ptr,*bounds,value)
        else:
            arr=np.asarray(value,dtype=np.float32)
            if arr.shape!=shape: raise ValueError("assigned array shape must match slice shape")
            if safe and not np.all(np.isfinite(arr)): raise ValueError("values must be finite")
            flat=np.ascontiguousarray(arr).reshape(-1)
            if isinstance(key,slice): _pd_linear_set_array(grid._ptr,bounds[0],bounds[1],flat.ctypes)
            else: _pd_region_set_array(grid._ptr,*bounds,flat.ctypes)



def _Bounds6(grid,key,safe):
    b,shape=_RegionBounds(grid,key,safe); return b

@overload(_Bounds6)
def _olBounds6(grid,key,safe):
    if isinstance(key,types.SliceType):
        def impl(grid,key,safe):
            a,b=_SliceBounds(key,len(grid)); return a,b,0,1,0,1
        return impl
    if isinstance(key,types.BaseTuple):
        n=len(key.types)
        if n==1:
            def impl(grid,key,safe):
                a0,b0=_AxisBounds(key[0],grid.xDim,safe);return a0,b0,0,1,0,1
        elif n==2:
            def impl(grid,key,safe):
                a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);return a0,b0,a1,b1,0,1
        else:
            def impl(grid,key,safe):
                a0,b0=_AxisBounds(key[0],grid.xDim,safe);a1,b1=_AxisBounds(key[1],grid.yDim,safe);a2,b2=_AxisBounds(key[2],grid.zDim,safe);return a0,b0,a1,b1,a2,b2
        return impl

def _PDESetSlice(grid,key,value,safe): _SetRegionPython(grid,key,value,safe,False)
def _PopSetSlice(grid,key,value,safe): _SetRegionPython(grid,key,value,safe,True)

@overload(_PDESetSlice)
def _olPDESetSlice(grid,key,value,safe):
    linear=isinstance(key,types.SliceType)
    if isinstance(value,types.Number):
        if linear:
            def impl(grid,key,value,safe):
                if safe and not np.isfinite(value): raise ValueError("value must be finite")
                a,b,_,_,_,_=_Bounds6(grid,key,safe);_pd_linear_set_scalar(grid._ptr,a,b,value)
        else:
            def impl(grid,key,value,safe):
                if safe and not np.isfinite(value): raise ValueError("value must be finite")
                a,b,c,d,e,f=_Bounds6(grid,key,safe);_pd_region_set_scalar(grid._ptr,a,b,c,d,e,f,value)
        return impl
    if isinstance(value,types.Array):
        if linear:
            def impl(grid,key,value,safe):
                a,b,_,_,_,_=_Bounds6(grid,key,safe)
                if value.size!=b-a: raise ValueError("assigned array shape must match slice shape")
                arr=value.astype(np.float32).reshape(value.size)
                if safe and not np.all(np.isfinite(arr)): raise ValueError("values must be finite")
                _pd_linear_set_array(grid._ptr,a,b,arr.ctypes)
        else:
            def impl(grid,key,value,safe):
                a,b,c,d,e,f=_Bounds6(grid,key,safe);n=(b-a)*(d-c)*(f-e)
                if value.size!=n: raise ValueError("assigned array shape must match slice shape")
                arr=value.astype(np.float32).reshape(value.size)
                if safe and not np.all(np.isfinite(arr)): raise ValueError("values must be finite")
                _pd_region_set_array(grid._ptr,a,b,c,d,e,f,arr.ctypes)
        return impl

@overload(_PopSetSlice)
def _olPopSetSlice(grid,key,value,safe):
    linear=isinstance(key,types.SliceType)
    if isinstance(value,types.Number):
        if linear:
            def impl(grid,key,value,safe):
                if safe and (not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max): raise ValueError("PopGrid values must be nonnegative int64 integers")
                a,b,_,_,_,_=_Bounds6(grid,key,safe);ok=_pg_linear_set_scalar(grid._ptr,a,b,value) if safe else _pg_linear_set_scalar_fast(grid._ptr,a,b,value)
                if safe and not ok: raise ValueError("PopGrid slice assignment value is outside capacity or population overflowed")
        else:
            def impl(grid,key,value,safe):
                if safe and (not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max): raise ValueError("PopGrid values must be nonnegative int64 integers")
                a,b,c,d,e,f=_Bounds6(grid,key,safe);ok=_pg_region_set_scalar(grid._ptr,a,b,c,d,e,f,value) if safe else _pg_region_set_scalar_fast(grid._ptr,a,b,c,d,e,f,value)
                if safe and not ok: raise ValueError("PopGrid slice assignment value is outside capacity or population overflowed")
        return impl
    if isinstance(value,types.Array):
        if linear:
            def impl(grid,key,value,safe):
                a,b,_,_,_,_=_Bounds6(grid,key,safe)
                if value.size!=b-a: raise ValueError("assigned array shape must match slice shape")
                if safe and (np.any(~np.isfinite(value)) or np.any(value!=np.floor(value)) or np.any(value<0) or np.any(value>np.iinfo(np.int64).max)): raise ValueError("PopGrid values must be nonnegative int64 integers")
                arr=value.astype(np.int64).reshape(value.size);ok=_pg_linear_set_array(grid._ptr,a,b,arr.ctypes) if safe else _pg_linear_set_array_fast(grid._ptr,a,b,arr.ctypes)
                if safe and not ok: raise ValueError("PopGrid slice assignment value is outside capacity or population overflowed")
        else:
            def impl(grid,key,value,safe):
                a,b,c,d,e,f=_Bounds6(grid,key,safe);n=(b-a)*(d-c)*(f-e)
                if value.size!=n: raise ValueError("assigned array shape must match slice shape")
                if safe and (np.any(~np.isfinite(value)) or np.any(value!=np.floor(value)) or np.any(value<0) or np.any(value>np.iinfo(np.int64).max)): raise ValueError("PopGrid values must be nonnegative int64 integers")
                arr=value.astype(np.int64).reshape(value.size);ok=_pg_region_set_array(grid._ptr,a,b,c,d,e,f,arr.ctypes) if safe else _pg_region_set_array_fast(grid._ptr,a,b,c,d,e,f,arr.ctypes)
                if safe and not ok: raise ValueError("PopGrid slice assignment value is outside capacity or population overflowed")
        return impl



def _PDESetItem(grid,key,value,safe):
    if isinstance(key,slice) or isinstance(key,tuple) and any(isinstance(k,slice) for k in key): return _PDESetSlice(grid,key,value,safe)
    if safe and not np.isfinite(value): raise ValueError("value must be finite")
    if isinstance(key,tuple):
        if len(key)==2: i=grid.ToI(key[0],key[1])
        elif len(key)==3: i=grid.ToI(key[0],key[1],key[2])
        else: raise IndexError("PDEgrid expects i, x,y, x,y,z, or slices")
    else:
        if safe: grid._CheckI(key)
        i=key
    _pd_seti(grid._ptr,i,value)

@overload(_PDESetItem)
def _olPDESetItem(grid,key,value,safe):
    if isinstance(key,types.SliceType) or isinstance(key,types.BaseTuple) and any(isinstance(t,types.SliceType) for t in key.types):
        def impl(grid,key,value,safe): _PDESetSlice(grid,key,value,safe)
        return impl
    if isinstance(key,types.BaseTuple):
        if len(key.types)==2:
            def impl(grid,key,value,safe):
                if safe and not np.isfinite(value): raise ValueError("value must be finite")
                _pd_seti(grid._ptr,grid.ToI(key[0],key[1]),value)
        else:
            def impl(grid,key,value,safe):
                if safe and not np.isfinite(value): raise ValueError("value must be finite")
                _pd_seti(grid._ptr,grid.ToI(key[0],key[1],key[2]),value)
        return impl
    if isinstance(safe,types.BooleanLiteral) and not safe.literal_value:
        def impl(grid,key,value,safe): _pd_seti_fast(grid._ptr,key,value)
        return impl
    def impl(grid,key,value,safe):
        if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError("index must be a valid integer")
        if not np.isfinite(value): raise ValueError("value must be finite")
        if not _pd_seti(grid._ptr,key,value): raise ValueError("invalid index or non-finite value")
    return impl


def _PDESetItemAt(grid,key,value,sourceLine): grid[key]=value
@overload(_PDESetItemAt,inline="never",prefer_literal=True)
def _olPDESetItemAt(grid,key,value,sourceLine):
    if grid==_PDEgridSafe.class_type.instance_type:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        suffix=" at source line "+str(line) if line>=0 else ""
        badShape="PDEgrid expects i, x,y, x,y,z, or slices"+suffix
        badIndex="index must be a valid integer"+suffix
        badValue="value must be finite"+suffix
        badSet="invalid index or non-finite value"+suffix
        if isinstance(key,types.BaseTuple):
            if len(key.types)==2:
                def impl(grid,key,value,sourceLine):
                    if not np.isfinite(value): raise ValueError(badValue)
                    i=grid.ToI(key[0],key[1],_palLine=sourceLine)
                    if not _pd_seti(grid._ptr,i,value): raise ValueError(badSet)
            elif len(key.types)==3:
                def impl(grid,key,value,sourceLine):
                    if not np.isfinite(value): raise ValueError(badValue)
                    i=grid.ToI(key[0],key[1],key[2],_palLine=sourceLine)
                    if not _pd_seti(grid._ptr,i,value): raise ValueError(badSet)
            else:
                def impl(grid,key,value,sourceLine): raise IndexError(badShape)
            return impl
        if isinstance(key,types.SliceType):
            def impl(grid,key,value,sourceLine): _PDESetItem(grid,key,value,True)
            return impl
        def impl(grid,key,value,sourceLine):
            if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError(badIndex)
            if not np.isfinite(value): raise ValueError(badValue)
            if not _pd_seti(grid._ptr,key,value): raise ValueError(badSet)
        return impl
    if grid==_PDEgridFast.class_type.instance_type:
        if isinstance(key,types.BaseTuple) and any(isinstance(t,types.SliceType) for t in key.types):
            def impl(grid,key,value,sourceLine): _PDESetSlice(grid,key,value,False)
        elif isinstance(key,types.BaseTuple):
            if len(key.types)==2:
                def impl(grid,key,value,sourceLine): _pd_seti_fast(grid._ptr,grid.ToI(key[0],key[1]),value)
            elif len(key.types)==3:
                def impl(grid,key,value,sourceLine): _pd_seti_fast(grid._ptr,grid.ToI(key[0],key[1],key[2]),value)
            else:
                def impl(grid,key,value,sourceLine): _PDESetItem(grid,key,value,False)
        elif isinstance(key,types.SliceType):
            def impl(grid,key,value,sourceLine): _PDESetSlice(grid,key,value,False)
        else:
            def impl(grid,key,value,sourceLine): _pd_seti_fast(grid._ptr,key,value)
        return impl

def _PopSetItemAt(grid,key,value,sourceLine): grid[key]=value
@overload(_PopSetItemAt,inline="never",prefer_literal=True)
def _olPopSetItemAt(grid,key,value,sourceLine):
    if grid==_PopGridSafe.class_type.instance_type:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        suffix=" at source line "+str(line) if line>=0 else ""
        badShape="PopGrid expects i, x,y, x,y,z, or slices"+suffix
        badIndex="index must be a valid integer"+suffix
        badValue="population must be a nonnegative int64 integer"+suffix
        badSet="invalid index or population outside capacity"+suffix
        if isinstance(key,types.BaseTuple):
            if len(key.types)==2:
                def impl(grid,key,value,sourceLine):
                    if not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max: raise ValueError(badValue)
                    i=grid.ToI(key[0],key[1],_palLine=sourceLine)
                    if not _pg_seti(grid._ptr,i,value): raise ValueError(badSet)
            elif len(key.types)==3:
                def impl(grid,key,value,sourceLine):
                    if not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max: raise ValueError(badValue)
                    i=grid.ToI(key[0],key[1],key[2],_palLine=sourceLine)
                    if not _pg_seti(grid._ptr,i,value): raise ValueError(badSet)
            else:
                def impl(grid,key,value,sourceLine): raise IndexError(badShape)
            return impl
        if isinstance(key,types.SliceType):
            def impl(grid,key,value,sourceLine): _PopSetItem(grid,key,value,True)
            return impl
        def impl(grid,key,value,sourceLine):
            if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError(badIndex)
            if not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max: raise ValueError(badValue)
            if not _pg_seti(grid._ptr,key,value): raise ValueError(badSet)
        return impl
    if grid==_PopGridFast.class_type.instance_type:
        def impl(grid,key,value,sourceLine): _PopSetItem(grid,key,value,False)
        return impl

def _PopSetItem(grid,key,value,safe):
    if isinstance(key,slice) or isinstance(key,tuple) and any(isinstance(k,slice) for k in key): return _PopSetSlice(grid,key,value,safe)
    if isinstance(key,tuple):
        if len(key)==2: i=grid.ToI(key[0],key[1])
        elif len(key)==3: i=grid.ToI(key[0],key[1],key[2])
        else: raise IndexError("PopGrid expects i, x,y, x,y,z, or slices")
    else:
        if safe: grid._CheckI(key)
        i=key
    ok=_pg_seti(grid._ptr,i,value) if safe else _pg_seti_fast(grid._ptr,i,value)
    if safe and not ok: raise ValueError("population must be between 0 and capacity")

@overload(_PopSetItem)
def _olPopSetItem(grid,key,value,safe):
    if isinstance(key,types.SliceType) or isinstance(key,types.BaseTuple) and any(isinstance(t,types.SliceType) for t in key.types):
        def impl(grid,key,value,safe): _PopSetSlice(grid,key,value,safe)
        return impl
    if isinstance(key,types.BaseTuple):
        if len(key.types)==2:
            def impl(grid,key,value,safe):
                if safe and (not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max): raise ValueError("population must be a nonnegative int64 integer")
                i=grid.ToI(key[0],key[1]);ok=_pg_seti(grid._ptr,i,value) if safe else _pg_seti_fast(grid._ptr,i,value)
                if safe and not ok: raise ValueError("population must be between 0 and capacity")
        else:
            def impl(grid,key,value,safe):
                if safe and (not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max): raise ValueError("population must be a nonnegative int64 integer")
                i=grid.ToI(key[0],key[1],key[2]);ok=_pg_seti(grid._ptr,i,value) if safe else _pg_seti_fast(grid._ptr,i,value)
                if safe and not ok: raise ValueError("population must be between 0 and capacity")
        return impl
    if isinstance(safe,types.BooleanLiteral) and not safe.literal_value:
        def impl(grid,key,value,safe): _pg_seti_fast(grid._ptr,key,value)
        return impl
    def impl(grid,key,value,safe):
        if not np.isfinite(key) or key!=int(key) or key<0 or key>=len(grid): raise IndexError("index must be a valid integer")
        if not np.isfinite(value) or value!=int(value) or value<0 or value>np.iinfo(np.int64).max: raise ValueError("population must be a nonnegative int64 integer")
        ok=_pg_seti(grid._ptr,key,value)
        if not ok: raise ValueError("invalid index or population outside capacity")
    return impl


class _AgentGridSafePython:
    """Tiny JIT-compatible handle for the native AgentGrid."""

    def __init__(self, ptr, dimension, nProps, xDim, yDim, zDim, length, wrapX, wrapY, wrapZ):
        self._ptr=ptr; self._dimension=dimension; self._nProps=nProps; self._xDim=xDim; self._yDim=yDim; self._zDim=zDim; self._length=length; self._wrapX=wrapX; self._wrapY=wrapY; self._wrapZ=wrapZ

    def __len__(self): return self._length

    @property
    def xDim(self):
        if self._dimension<1: raise ValueError("xDim requires a spatial grid")
        return self._xDim
    @property
    def yDim(self):
        if self._dimension<2: raise ValueError("yDim requires a 2D or 3D grid")
        return self._yDim
    @property
    def zDim(self):
        if self._dimension<3: raise ValueError("zDim requires a 3D grid")
        return self._zDim
    @property
    def nDims(self): return self._dimension
    @property
    def wrapX(self): return self._wrapX
    @property
    def wrapY(self): return self._wrapY
    @property
    def wrapZ(self): return self._wrapZ
    def GetPop(self): return _ag_pop(self._ptr)

    def Alive(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_alive_safe(self._ptr,agent)
        if v<0: raise ValueError("invalid agent id")
        return v!=0

    def _CheckAlive(self, agent):
        if not self.Alive(agent): raise ValueError("agent is not alive")

    def _NewAgentI(self, i):
        if self._dimension==0: raise ValueError("NewAgentI requires a spatial AgentGrid")
        if not np.isfinite(i) or i!=int(i): raise ValueError("index must be a finite integer")
        if i < 0 or i >= len(self): raise IndexError("index out of bounds")
        agent = _ag_new_i(self._ptr, i)
        if agent < 0: raise ValueError("unable to place agent")
        return agent

    def _CheckCoords(self,x,y,z):
        if not np.isfinite(x) or x!=int(x): raise ValueError("coordinates must be finite integers")
        if self._dimension==1:
            if y!=-1 or z!=-1: raise ValueError("1D AgentGrid requires x only")
        elif self._dimension==2:
            if not np.isfinite(y) or y!=int(y) or z!=-1: raise ValueError("2D AgentGrid requires x,y")
        elif self._dimension==3:
            if not np.isfinite(y) or not np.isfinite(z) or y!=int(y) or z!=int(z): raise ValueError("3D AgentGrid requires x,y,z")
    def _CheckPoint(self,x,y,z):
        if not np.isfinite(x): raise ValueError("coordinates must be finite")
        if self._dimension==1:
            if y!=-1.0 or z!=-1.0: raise ValueError("1D AgentGrid requires x only")
        elif self._dimension==2:
            if not np.isfinite(y) or z!=-1.0: raise ValueError("2D AgentGrid requires x,y")
        elif self._dimension==3:
            if not np.isfinite(y) or not np.isfinite(z): raise ValueError("3D AgentGrid requires x,y,z")

    def NewAgentSQ(self, x, y=-1, z=-1):
        if self._dimension==0: raise ValueError("NewAgentSQ requires a spatial AgentGrid")
        if y==-1 and z==-1: return self._NewAgentI(x)
        self._CheckCoords(x,y,z)
        agent = _ag_new_sq(self._ptr, x, y, z)
        if agent < 0: raise ValueError("unable to place agent")
        return agent

    def NewAgent(self, x=-1.0, y=-1.0, z=-1.0):
        if self._dimension == 0:
            agent=_ag_new_i(self._ptr,-1)
        else:
            self._CheckPoint(x,y,z)
            agent=_ag_new_pt(self._ptr,x,y,z)
        if agent < 0: raise ValueError("unable to place agent")
        return agent

    def Dispose(self, agent):
        self._CheckAlive(agent)
        _ag_dispose(self._ptr, agent)

    def I(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_i_safe(self._ptr,agent)
        if v==-2147483648: raise ValueError("invalid or dead agent")
        return v
    def XSQ(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_xsq_safe(self._ptr,agent)
        if v==-2147483648: raise ValueError("invalid agent or dimension")
        return v
    def YSQ(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_ysq_safe(self._ptr,agent)
        if v==-2147483648: raise ValueError("invalid agent or dimension")
        return v
    def ZSQ(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_zsq_safe(self._ptr,agent)
        if v==-2147483648: raise ValueError("invalid agent or dimension")
        return v
    def X(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_x_safe(self._ptr,agent)
        if np.isnan(v): raise ValueError("invalid agent or dimension")
        return v
    def Y(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_y_safe(self._ptr,agent)
        if np.isnan(v): raise ValueError("invalid agent or dimension")
        return v
    def Z(self, agent):
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        v=_ag_z_safe(self._ptr,agent)
        if np.isnan(v): raise ValueError("invalid agent or dimension")
        return v

    def __getitem__(self,key):
        agent,propId=key
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        if not np.isfinite(propId) or propId!=int(propId) or propId<0 or propId>np.iinfo(np.int32).max: raise IndexError("invalid property id")
        v=_ag_getp_safe(self._ptr,agent,propId)
        if np.isnan(v): raise ValueError("invalid agent or property")
        return v

    def __setitem__(self,key,value):
        agent,propId=key
        if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError("invalid agent id")
        if not np.isfinite(propId) or propId!=int(propId) or propId<0 or propId>np.iinfo(np.int32).max: raise IndexError("invalid property id")
        if not np.isfinite(value): raise ValueError("property value must be finite")
        if not _ag_setp_safe(self._ptr,agent,propId,value): raise ValueError("invalid agent, property, or value")

    def ToI(self,x,y=-1,z=-1):
        self._CheckCoords(x,y,z)
        if x<0 or x>=self.xDim or self._dimension>1 and (y<0 or y>=self.yDim) or self._dimension>2 and (z<0 or z>=self.zDim): raise IndexError("coordinate out of bounds")
        v=_ag_toi_safe(self._ptr,x,y,z)
        if v==-2147483648: raise IndexError("invalid coordinates")
        return v
    def ItoX(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_ag_itox_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v
    def ItoY(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_ag_itoy_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v
    def ItoZ(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_ag_itoz_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v

    def InWrapSQX(self,value):
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("square coordinate must be an int32 integer")
        v=_ag_inwrap_sq_safe(self._ptr,value,0)
        if v==-2147483648: raise ValueError("invalid dimension")
        return v
    def InWrapSQY(self,value):
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("square coordinate must be an int32 integer")
        v=_ag_inwrap_sq_safe(self._ptr,value,1)
        if v==-2147483648: raise ValueError("invalid dimension")
        return v
    def InWrapSQZ(self,value):
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("square coordinate must be an int32 integer")
        v=_ag_inwrap_sq_safe(self._ptr,value,2)
        if v==-2147483648: raise ValueError("invalid dimension")
        return v
    def InWrapX(self,value):
        v=_ag_inwrap_safe(self._ptr,value,0)
        if np.isnan(v): raise ValueError("invalid coordinate or dimension")
        return v
    def InWrapY(self,value):
        v=_ag_inwrap_safe(self._ptr,value,1)
        if np.isnan(v): raise ValueError("invalid coordinate or dimension")
        return v
    def InWrapZ(self,value):
        v=_ag_inwrap_safe(self._ptr,value,2)
        if np.isnan(v): raise ValueError("invalid coordinate or dimension")
        return v
    def DispWrapX(self,x1,x2):
        v=_ag_dispwrap_safe(self._ptr,x1,x2,0)
        if np.isnan(v): raise ValueError("invalid coordinates or dimension")
        return v
    def DispWrapY(self,y1,y2):
        v=_ag_dispwrap_safe(self._ptr,y1,y2,1)
        if np.isnan(v): raise ValueError("invalid coordinates or dimension")
        return v
    def DispWrapZ(self,z1,z2):
        v=_ag_dispwrap_safe(self._ptr,z1,z2,2)
        if np.isnan(v): raise ValueError("invalid coordinates or dimension")
        return v

    def _GetLastI(self,i):
        if self._dimension==0: raise ValueError("GetLastI requires a spatial AgentGrid")
        if not np.isfinite(i) or i!=int(i) or i<0 or i>=len(self): raise IndexError("index out of bounds")
        return _ag_last_i(self._ptr,i)
    def LastAgent(self,x,y=-1,z=-1):
        return self._GetLastI(x) if y==-1 and z==-1 else self._GetLastI(self.ToI(x,y,z))

    def _MoveI(self, agent, i):
        if self._dimension==0: raise ValueError("MoveI requires a spatial AgentGrid")
        self._CheckAlive(agent)
        if not np.isfinite(i) or i!=int(i) or i<0 or i>=len(self): raise IndexError("index out of bounds")
        ok = _ag_move_i(self._ptr, agent, i)
        if not ok: raise ValueError("unable to move agent")

    def MoveSQ(self, agent, x, y=-1, z=-1):
        self._MoveI(agent, x) if y==-1 and z==-1 else self._MoveI(agent, self.ToI(x, y, z))

    def Move(self,agent,x,y=-1.0,z=-1.0):
        if self._dimension==0: raise ValueError("Move requires a spatial AgentGrid")
        self._CheckAlive(agent); self._CheckPoint(x,y,z)
        ok = _ag_move_pt(self._ptr,agent,x,y,z)
        if not ok: raise ValueError("unable to move agent")

    def AgentsAt(self, x, y=-1, z=-1):
        raise RuntimeError("AgentsAt is direct-iteration syntax and must be used in a for loop inside @pal.njit")

    def All(self, shuffle=False):
        out = np.empty(self.GetPop(), dtype=np.int32)
        _ag_all_copy(self._ptr, out.ctypes, int(shuffle))
        return out



_AgentGridSafe=_jitclass([("_ptr",uint64),("_dimension",int32),("_nProps",int32),("_xDim",int32),("_yDim",int32),("_zDim",int32),("_length",int32),("_wrapX",types.boolean),("_wrapY",types.boolean),("_wrapZ",types.boolean)])(_AgentGridSafePython)

class _AgentGridFastPython:
    """Tiny JIT-compatible handle for the native AgentGrid."""

    def __init__(self, ptr, dimension, xDim, yDim, zDim, length, wrapX, wrapY, wrapZ):
        self._ptr=ptr; self._dimension=dimension; self._xDim=xDim; self._yDim=yDim; self._zDim=zDim; self._length=length; self._wrapX=wrapX; self._wrapY=wrapY; self._wrapZ=wrapZ

    def __len__(self): return self._length

    @property
    def xDim(self): return self._xDim
    @property
    def yDim(self): return self._yDim
    @property
    def zDim(self): return self._zDim
    @property
    def nDims(self): return self._dimension
    @property
    def wrapX(self): return self._wrapX
    @property
    def wrapY(self): return self._wrapY
    @property
    def wrapZ(self): return self._wrapZ
    def GetPop(self): return _ag_pop(self._ptr)

    def Alive(self, agent):
        return _ag_alive(self._ptr, agent) != 0

    def _NewAgentI(self, i):
        agent = _ag_new_i_fast(self._ptr, i)
        return agent

    def NewAgentSQ(self, x, y=-1, z=-1):
        if y==-1 and z==-1: return self._NewAgentI(x)
        return _ag_new_sq_fast(self._ptr, x, y, z)

    def NewAgent(self, x=-1.0, y=-1.0, z=-1.0):
        if self._dimension==0: return _ag_new_i_fast(self._ptr,-1)
        return _ag_new_pt_fast(self._ptr, x, y, z)

    def Dispose(self, agent):
        _ag_dispose_fast(self._ptr, agent)

    def I(self, agent): return _ag_i(self._ptr, agent)
    def XSQ(self, agent): return _ag_xsq(self._ptr, agent)
    def YSQ(self, agent): return _ag_ysq(self._ptr, agent)
    def ZSQ(self, agent): return _ag_zsq(self._ptr, agent)
    def X(self, agent): return _ag_x(self._ptr, agent)
    def Y(self, agent): return _ag_y(self._ptr, agent)
    def Z(self, agent): return _ag_z(self._ptr, agent)

    def __getitem__(self,key):
        agent,propId=key
        return _ag_getp_fast(self._ptr,agent,propId)

    def __setitem__(self,key,value):
        agent,propId=key
        _ag_setp_fast(self._ptr,agent,propId,value)

    def ToI(self, x, y=-1, z=-1): return _ag_toi(self._ptr, x, y, z)
    def ItoX(self, i): return _ag_itox(self._ptr, i)
    def ItoY(self, i): return _ag_itoy(self._ptr, i)
    def ItoZ(self, i): return _ag_itoz(self._ptr, i)

    def InWrapSQX(self, value): return _ag_inwrap_sq_x(self._ptr, value)
    def InWrapSQY(self, value): return _ag_inwrap_sq_y(self._ptr, value)
    def InWrapSQZ(self, value): return _ag_inwrap_sq_z(self._ptr, value)
    def InWrapX(self, value): return _ag_inwrap_x(self._ptr, value)
    def InWrapY(self, value): return _ag_inwrap_y(self._ptr, value)
    def InWrapZ(self, value): return _ag_inwrap_z(self._ptr, value)
    def DispWrapX(self, x1, x2): return _ag_dispwrap_x(self._ptr, x1, x2)
    def DispWrapY(self, y1, y2): return _ag_dispwrap_y(self._ptr, y1, y2)
    def DispWrapZ(self, z1, z2): return _ag_dispwrap_z(self._ptr, z1, z2)

    def _GetLastI(self, i): return _ag_last_i_fast(self._ptr, i)
    def LastAgent(self, x, y=-1, z=-1): return self._GetLastI(x) if y==-1 and z==-1 else self._GetLastI(self.ToI(x, y, z))

    def _MoveI(self, agent, i):
        ok = _ag_move_i_fast(self._ptr, agent, i)

    def MoveSQ(self, agent, x, y=-1, z=-1):
        self._MoveI(agent, x) if y==-1 and z==-1 else self._MoveI(agent, self.ToI(x, y, z))

    def Move(self, agent, x, y=-1.0, z=-1.0):
        _ag_move_pt_fast(self._ptr, agent, x, y, z)

    def AgentsAt(self, x, y=-1, z=-1):
        raise RuntimeError("AgentsAt is direct-iteration syntax and must be used in a for loop inside @pal.njit")

    def All(self, shuffle=False):
        out = np.empty(self.GetPop(), dtype=np.int32)
        _ag_all_copy(self._ptr, out.ctypes, int(shuffle))
        return out



_AgentGridFast=_jitclass([("_ptr",uint64),("_dimension",int32),("_xDim",int32),("_yDim",int32),("_zDim",int32),("_length",int32),("_wrapX",types.boolean),("_wrapY",types.boolean),("_wrapZ",types.boolean)])(_AgentGridFastPython)


def _AgentNewSQTrustedAt(grid,x,y,z,sourceLine): return grid.NewAgentSQ(x,y,z)
@overload(_AgentNewSQTrustedAt,inline="never",prefer_literal=True)
def _olAgentNewSQTrustedAt(grid,x,y,z,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    mUnable="unable to place agent"+suffix
    if grid==_AgentGridSafe.class_type.instance_type:
        def impl(grid,x,y,z,sourceLine):
            agent=_ag_new_sq(grid._ptr,x,y,z)
            if agent<0: raise ValueError(mUnable)
            return agent
        return impl
    if grid==_AgentGridFast.class_type.instance_type:
        def impl(grid,x,y,z,sourceLine): return _ag_new_sq_fast(grid._ptr,x,y,z)
        return impl

# Thin source-aware model-call helpers.  Keep PAL safety checks out of the
# user's Numba IR: Safe and Fast kernels compile the same small call surface,
# while Safe helpers retain the native checks and source-line diagnostics.
def _AgentNewSQAt(grid,x,y,z,sourceLine): return grid.NewAgentSQ(x,y,z)
@overload(_AgentNewSQAt,inline="never",prefer_literal=True)
def _olAgentNewSQAt(grid,x,y,z,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    mSpatial="NewAgentSQ requires a spatial AgentGrid"+suffix; mIndexFinite="index must be a finite integer"+suffix; mIndexBounds="index out of bounds"+suffix; mCoords="coordinates must be finite integers"+suffix; m1="1D AgentGrid requires x only"+suffix; m2="2D AgentGrid requires x,y"+suffix; m3="3D AgentGrid requires x,y,z"+suffix; mUnable="unable to place agent"+suffix
    if grid==_AgentGridSafe.class_type.instance_type:
        def impl(grid,x,y,z,sourceLine):
            if grid._dimension==0: raise ValueError(mSpatial)
            if y==-1 and z==-1:
                if not np.isfinite(x) or x!=int(x): raise ValueError(mIndexFinite)
                if x<0 or x>=grid._length: raise IndexError(mIndexBounds)
                agent=_ag_new_i(grid._ptr,x)
            else:
                if not np.isfinite(x) or x!=int(x): raise ValueError(mCoords)
                if grid._dimension==1:
                    if y!=-1 or z!=-1: raise ValueError(m1)
                elif grid._dimension==2:
                    if not np.isfinite(y) or y!=int(y) or z!=-1: raise ValueError(m2)
                elif grid._dimension==3:
                    if not np.isfinite(y) or not np.isfinite(z) or y!=int(y) or z!=int(z): raise ValueError(m3)
                agent=_ag_new_sq(grid._ptr,x,y,z)
            if agent<0: raise ValueError(mUnable)
            return agent
        return impl
    if grid==_AgentGridFast.class_type.instance_type:
        def impl(grid,x,y,z,sourceLine):
            if y==-1 and z==-1: return _ag_new_i_fast(grid._ptr,x)
            return _ag_new_sq_fast(grid._ptr,x,y,z)
        return impl

def _AgentDisposeAt(grid,agent,sourceLine): return grid.Dispose(agent)
@overload(_AgentDisposeAt,inline="never",prefer_literal=True)
def _olAgentDisposeAt(grid,agent,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    mInvalid="invalid agent id"+suffix; mDead="agent is not alive"+suffix
    if grid==_AgentGridSafe.class_type.instance_type:
        def impl(grid,agent,sourceLine):
            if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError(mInvalid)
            alive=_ag_alive_safe(grid._ptr,agent)
            if alive<0: raise ValueError(mInvalid)
            if alive==0: raise ValueError(mDead)
            _ag_dispose(grid._ptr,agent)
        return impl
    if grid==_AgentGridFast.class_type.instance_type:
        def impl(grid,agent,sourceLine): _ag_dispose_fast(grid._ptr,agent)
        return impl

def _AgentCoordAt(grid,agent,axis,sourceLine): return 0
@overload(_AgentCoordAt,inline="never",prefer_literal=True)
def _olAgentCoordAt(grid,agent,axis,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    ax=axis.literal_value if isinstance(axis,types.IntegerLiteral) else None
    if ax is None: return None
    mInvalid="invalid agent id"+suffix; mDead="invalid or dead agent"+suffix; mDim="invalid agent or dimension"+suffix
    if grid==_AgentGridSafe.class_type.instance_type:
        def impl(grid,agent,axis,sourceLine):
            if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError(mInvalid)
            if ax==0: v=_ag_i_safe(grid._ptr,agent)
            elif ax==1: v=_ag_xsq_safe(grid._ptr,agent)
            elif ax==2: v=_ag_ysq_safe(grid._ptr,agent)
            else: v=_ag_zsq_safe(grid._ptr,agent)
            if v==-2147483648:
                if ax==0: raise ValueError(mDead)
                raise ValueError(mDim)
            return v
        return impl
    if grid==_AgentGridFast.class_type.instance_type:
        def impl(grid,agent,axis,sourceLine):
            if ax==0: return _ag_i(grid._ptr,agent)
            if ax==1: return _ag_xsq(grid._ptr,agent)
            if ax==2: return _ag_ysq(grid._ptr,agent)
            return _ag_zsq(grid._ptr,agent)
        return impl

def _AgentToIAt(grid,x,y,z,sourceLine): return grid.ToI(x,y,z)
@overload(_AgentToIAt,inline="never",prefer_literal=True)
def _olAgentToIAt(grid,x,y,z,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    mCoords="coordinates must be finite integers"+suffix; m1="1D AgentGrid requires x only"+suffix; m2="2D AgentGrid requires x,y"+suffix; m3="3D AgentGrid requires x,y,z"+suffix; mBounds="coordinate out of bounds"+suffix; mInvalid="invalid coordinates"+suffix
    if grid==_AgentGridSafe.class_type.instance_type:
        def impl(grid,x,y,z,sourceLine):
            if not np.isfinite(x) or x!=int(x): raise ValueError(mCoords)
            if grid._dimension==1:
                if y!=-1 or z!=-1: raise ValueError(m1)
            elif grid._dimension==2:
                if not np.isfinite(y) or y!=int(y) or z!=-1: raise ValueError(m2)
            elif grid._dimension==3:
                if not np.isfinite(y) or not np.isfinite(z) or y!=int(y) or z!=int(z): raise ValueError(m3)
            if x<0 or x>=grid._xDim or grid._dimension>1 and (y<0 or y>=grid._yDim) or grid._dimension>2 and (z<0 or z>=grid._zDim): raise IndexError(mBounds)
            v=_ag_toi_safe(grid._ptr,x,y,z)
            if v==-2147483648: raise IndexError(mInvalid)
            return v
        return impl
    if grid==_AgentGridFast.class_type.instance_type:
        def impl(grid,x,y,z,sourceLine): return _ag_toi(grid._ptr,x,y,z)
        return impl

def _AgentAllAt(grid,shuffle,sourceLine): return grid.All(shuffle)
@overload(_AgentAllAt,inline="never",prefer_literal=True)
def _olAgentAllAt(grid,shuffle,sourceLine):
    if grid==_AgentGridSafe.class_type.instance_type or grid==_AgentGridFast.class_type.instance_type:
        def impl(grid,shuffle,sourceLine):
            out=np.empty(_ag_pop(grid._ptr),dtype=np.int32)
            _ag_all_copy(grid._ptr,out.ctypes,int(shuffle))
            return out
        return impl

def _IListClearAt(q,sourceLine): return q.Clear()
@overload(_IListClearAt,inline="never",prefer_literal=True)
def _olIListClearAt(q,sourceLine):
    if q==_IListSafe.class_type.instance_type or q==_IListFast.class_type.instance_type:
        def impl(q,sourceLine):
            _q_clear(q._ptr)
            return q
        return impl

def _IListRandomAt(q,sourceLine): return q.Random()
@overload(_IListRandomAt,inline="never",prefer_literal=True)
def _olIListRandomAt(q,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    message="cannot sample an empty IList"+suffix
    if q==_IListSafe.class_type.instance_type:
        def impl(q,sourceLine):
            out=_q_random(q._ptr)
            if out<0: raise ValueError(message)
            return out
        return impl
    if q==_IListFast.class_type.instance_type:
        def impl(q,sourceLine): return _q_random(q._ptr)
        return impl

# Compiled counts access operates directly on AgentGrid.  The public
# ``grid.counts[...]`` syntax is lowered to this helper by PAL's AST pass,
# avoiding a nested counts jitclass in the AgentGrid representation.
def _AgentCountsItem(grid,key): return _CountsItem(grid,key,True)

@overload(_AgentCountsItem,inline="always")
def _olAgentCountsItem(grid,key):
    if grid==_AgentGridSafe.class_type.instance_type:
        def impl(grid,key): return _CountsItem(grid,key,True)
        return impl
    if grid==_AgentGridFast.class_type.instance_type:
        def impl(grid,key): return _CountsItem(grid,key,False)
        return impl

# Source-aware AgentGrid property access.  Do not call the jitclass dunder
# methods explicitly with the extra diagnostic argument: Numba treats
# __getitem__/__setitem__ specially and can route that call through operator
# lowering, where the extra argument produces an opaque compiler failure.
def _AgentItemAt(grid,key,sourceLine): return grid[key]
@overload(_AgentItemAt,inline="never",prefer_literal=True)
def _olAgentItemAt(grid,key,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    if not isinstance(key,types.BaseTuple) or len(key.types)!=2:
        return None
    safe=grid==_AgentGridSafe.class_type.instance_type
    fast=grid==_AgentGridFast.class_type.instance_type
    if not (safe or fast): return None
    if safe:
        badAgent="invalid agent id"+suffix; badProp="invalid property id"+suffix; badValue="invalid agent or property"+suffix
        def impl(grid,key,sourceLine):
            agent=key[0]; propId=key[1]
            if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError(badAgent)
            if not np.isfinite(propId) or propId!=int(propId) or propId<0 or propId>np.iinfo(np.int32).max: raise IndexError(badProp)
            v=_ag_getp_safe(grid._ptr,agent,propId)
            if np.isnan(v): raise ValueError(badValue)
            return v
    else:
        def impl(grid,key,sourceLine): return _ag_getp_fast(grid._ptr,key[0],key[1])
    return impl

def _AgentSetItemAt(grid,key,value,sourceLine): grid[key]=value
@overload(_AgentSetItemAt,inline="never",prefer_literal=True)
def _olAgentSetItemAt(grid,key,value,sourceLine):
    line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
    suffix=" at source line "+str(line) if line>=0 else ""
    if not isinstance(key,types.BaseTuple) or len(key.types)!=2:
        return None
    safe=grid==_AgentGridSafe.class_type.instance_type
    fast=grid==_AgentGridFast.class_type.instance_type
    if not (safe or fast): return None
    if safe:
        badAgent="invalid agent id"+suffix; badProp="invalid property id"+suffix; badVal="property value must be finite"+suffix; badSet="invalid agent, property, or value"+suffix
        def impl(grid,key,value,sourceLine):
            agent=key[0]; propId=key[1]
            if not np.isfinite(agent) or agent!=int(agent) or agent<0 or agent>np.iinfo(np.int32).max: raise ValueError(badAgent)
            if not np.isfinite(propId) or propId!=int(propId) or propId<0 or propId>np.iinfo(np.int32).max: raise IndexError(badProp)
            if not np.isfinite(value): raise ValueError(badVal)
            if not _ag_setp_safe(grid._ptr,agent,propId,value): raise ValueError(badSet)
    else:
        def impl(grid,key,value,sourceLine): _ag_setp_fast(grid._ptr,key[0],key[1],value)
    return impl


class _PopGridSafePython:
    """Tiny JIT-compatible handle for the native PopGrid."""

    def __init__(self, ptr, dimension, xDim, yDim, zDim, length, wrapX, wrapY, wrapZ):
        self._ptr=ptr; self._dimension=dimension; self._xDim=xDim; self._yDim=yDim; self._zDim=zDim; self._length=length; self._wrapX=wrapX; self._wrapY=wrapY; self._wrapZ=wrapZ

    def __len__(self): return self._length
    @property
    def xDim(self): return self._xDim
    @property
    def yDim(self):
        if self._dimension<2: raise ValueError("yDim requires a 2D or 3D grid")
        return self._yDim
    @property
    def zDim(self):
        if self._dimension<3: raise ValueError("zDim requires a 3D grid")
        return self._zDim
    @property
    def nDims(self): return self._dimension
    @property
    def wrapX(self): return self._wrapX
    @property
    def wrapY(self): return self._wrapY
    @property
    def wrapZ(self): return self._wrapZ
    def _CheckI(self,i):
        if not np.isfinite(i) or i!=int(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=len(self): raise IndexError("index out of bounds")
    def _CheckCoords(self,x,y,z):
        if not np.isfinite(x) or x!=int(x): raise ValueError("coordinates must be finite integers")
        if self._dimension==1:
            if y!=-1 or z!=-1: raise ValueError("1D PopGrid requires x only")
        elif self._dimension==2:
            if not np.isfinite(y) or y!=int(y) or z!=-1: raise ValueError("2D PopGrid requires x,y")
        else:
            if not np.isfinite(y) or not np.isfinite(z) or y!=int(y) or z!=int(z): raise ValueError("3D PopGrid requires x,y,z")
        if x<0 or x>=self.xDim or self._dimension>1 and (y<0 or y>=self.yDim) or self._dimension>2 and (z<0 or z>=self.zDim): raise IndexError("coordinate out of bounds")
    def ToI(self,x,y=-1,z=-1):
        self._CheckCoords(x,y,z)
        v=_pg_toi_safe(self._ptr,x,y,z)
        if v==-2147483648: raise IndexError("invalid coordinates")
        return v
    def ItoX(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_pg_itox_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v
    def ItoY(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_pg_itoy_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v
    def ItoZ(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_pg_itoz_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v

    def InWrapX(self, value):
        if not np.isfinite(value) or value != int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("coordinate must be an int32 integer")
        return _pg_inwrap_x(self._ptr, int(value))
    def InWrapY(self, value):
        if self._dimension<2: raise ValueError("InWrapY requires a 2D or 3D PopGrid")
        if not np.isfinite(value) or value != int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("coordinate must be an int32 integer")
        return _pg_inwrap_y(self._ptr, int(value))
    def InWrapZ(self, value):
        if self._dimension<3: raise ValueError("InWrapZ requires a 3D PopGrid")
        if not np.isfinite(value) or value != int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("coordinate must be an int32 integer")
        return _pg_inwrap_z(self._ptr, int(value))

    def __getitem__(self,key): return _PopItem(self,key,True)

    def __setitem__(self,key,value): _PopSetItem(self,key,value,True)

    def _AddI(self,value,i):
        self._CheckI(i)
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int64).min or value>np.iinfo(np.int64).max: raise ValueError("delta must be an int64 integer")
        ok = _pg_addi(self._ptr,i,value)
        if not ok: raise OverflowError("PopGrid delta overflow or invalid index")

    def Add(self,value,x,y=-1,z=-1):
        if y==-1 and z==-1: return self._AddI(value,x)
        self._CheckCoords(x,y,z)
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int64).min or value>np.iinfo(np.int64).max: raise ValueError("delta must be an int64 integer")
        if not _pg_add(self._ptr,x,y,z,value): raise OverflowError("PopGrid delta overflow")

    def Update(self):
        ok = _pg_update(self._ptr)
        if not ok: raise ValueError("PopGrid update would exceed bounds")

    def Reset(self):
        _pg_clear(self._ptr)

    def GetPop(self): return _pg_pop(self._ptr)
    def All(self):
        out=np.empty(_pg_all_count(self._ptr),dtype=np.int32)
        _pg_all_copy(self._ptr,out.ctypes)
        return out

class _PopGridFastPython:
    """Tiny JIT-compatible handle for the native PopGrid."""

    def __init__(self, ptr, dimension, xDim, yDim, zDim, length, wrapX, wrapY, wrapZ):
        self._ptr=ptr; self._dimension=dimension; self._xDim=xDim; self._yDim=yDim; self._zDim=zDim; self._length=length; self._wrapX=wrapX; self._wrapY=wrapY; self._wrapZ=wrapZ

    def __len__(self): return self._length
    @property
    def xDim(self): return self._xDim
    @property
    def yDim(self): return self._yDim
    @property
    def zDim(self): return self._zDim
    @property
    def nDims(self): return self._dimension
    @property
    def wrapX(self): return self._wrapX
    @property
    def wrapY(self): return self._wrapY
    @property
    def wrapZ(self): return self._wrapZ
    def ToI(self, x, y=-1, z=-1): return _pg_toi(self._ptr, x, y, z)

    def ItoX(self, i): return _pg_itox(self._ptr, i)
    def ItoY(self, i): return _pg_itoy(self._ptr, i)
    def ItoZ(self, i): return _pg_itoz(self._ptr, i)

    def InWrapX(self, value): return _pg_inwrap_x(self._ptr, value)
    def InWrapY(self, value): return _pg_inwrap_y(self._ptr, value)
    def InWrapZ(self, value): return _pg_inwrap_z(self._ptr, value)

    def __getitem__(self,key): return _PopItem(self,key,False)

    def __setitem__(self,key,value): _PopSetItem(self,key,value,False)

    def _AddI(self, value, i):
        ok = _pg_addi_fast(self._ptr, i, value)

    def Add(self, value, x, y=-1, z=-1):
        if y==-1 and z==-1: return self._AddI(value,x)
        _pg_add_fast(self._ptr, x, y, z, value)

    def Update(self):
        ok = _pg_update_fast(self._ptr)

    def Reset(self):
        _pg_clear(self._ptr)

    def GetPop(self): return _pg_pop(self._ptr)
    def All(self):
        out=np.empty(_pg_all_count(self._ptr),dtype=np.int32)
        _pg_all_copy(self._ptr,out.ctypes)
        return out

_PopGridSafe=_jitclass([("_ptr",uint64),("_dimension",int32),("_xDim",int32),("_yDim",int32),("_zDim",int32),("_length",int32),("_wrapX",types.boolean),("_wrapY",types.boolean),("_wrapZ",types.boolean)])(_PopGridSafePython)
_PopGridFast=_jitclass([("_ptr",uint64),("_dimension",int32),("_xDim",int32),("_yDim",int32),("_zDim",int32),("_length",int32),("_wrapX",types.boolean),("_wrapY",types.boolean),("_wrapZ",types.boolean)])(_PopGridFastPython)


class _PDEgridSafePython:
    """Tiny JIT-compatible handle for the native PDEgrid."""

    def __init__(self, ptr, dimension, xDim, yDim, zDim, length, wrapX, wrapY, wrapZ):
        self._ptr=ptr; self._dimension=dimension; self._xDim=xDim; self._yDim=yDim; self._zDim=zDim; self._length=length; self._wrapX=wrapX; self._wrapY=wrapY; self._wrapZ=wrapZ

    def __len__(self): return self._length
    @property
    def xDim(self): return self._xDim
    @property
    def yDim(self):
        if self._dimension<2: raise ValueError("yDim requires a 2D or 3D grid")
        return self._yDim
    @property
    def zDim(self):
        if self._dimension<3: raise ValueError("zDim requires a 3D grid")
        return self._zDim
    @property
    def nDims(self): return self._dimension
    @property
    def wrapX(self): return self._wrapX
    @property
    def wrapY(self): return self._wrapY
    @property
    def wrapZ(self): return self._wrapZ
    def _CheckI(self,i):
        if not np.isfinite(i) or i!=int(i): raise ValueError("index must be a finite integer")
        if i<0 or i>=len(self): raise IndexError("index out of bounds")
    def _CheckCoords(self,x,y,z):
        if not np.isfinite(x) or x!=int(x): raise ValueError("coordinates must be finite integers")
        if self._dimension==1:
            if y!=-1 or z!=-1: raise ValueError("1D PDEgrid requires x only")
        elif self._dimension==2:
            if not np.isfinite(y) or y!=int(y) or z!=-1: raise ValueError("2D PDEgrid requires x,y")
        else:
            if not np.isfinite(y) or not np.isfinite(z) or y!=int(y) or z!=int(z): raise ValueError("3D PDEgrid requires x,y,z")
        if x<0 or x>=self.xDim or self._dimension>1 and (y<0 or y>=self.yDim) or self._dimension>2 and (z<0 or z>=self.zDim): raise IndexError("coordinate out of bounds")
    def ToI(self,x,y=-1,z=-1):
        self._CheckCoords(x,y,z)
        v=_pd_toi_safe(self._ptr,x,y,z)
        if v==-2147483648: raise IndexError("invalid coordinates")
        return v
    def ItoX(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_pd_itox_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v
    def ItoY(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_pd_itoy_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v
    def ItoZ(self,i):
        if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError("index must be a nonnegative int32 integer")
        v=_pd_itoz_safe(self._ptr,i)
        if v==-2147483648: raise IndexError("invalid index or dimension")
        return v

    def __getitem__(self,key): return _PDEItem(self,key,True)

    def __setitem__(self,key,value): _PDESetItem(self,key,value,True)

    def _AddI(self,value,i):
        if not np.isfinite(value): raise ValueError("value must be finite")
        self._CheckI(i); _pd_addi(self._ptr,i,value)

    def Add(self,value,x,y=-1,z=-1):
        if y==-1 and z==-1: return self._AddI(value,x)
        if not np.isfinite(value): raise ValueError("value must be finite")
        self._CheckCoords(x,y,z); _pd_add(self._ptr,x,y,z,value)
    def Update(self): _pd_update(self._ptr)

    def Reset(self):
        _pd_clear(self._ptr,0.0)

    def SetTimeSpaceStep(self, dt, dx, dy=1.0, dz=1.0):
        if not np.isfinite(dt) or not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(dz) or dt<=0 or dx<=0 or dy<=0 or dz<=0:
            raise ValueError("time and space steps must be finite and positive")
        _pd_steps(self._ptr, dt, dx, dy, dz)



    def InWrapX(self,value):
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("index must be an int32 integer")
        return _pd_inwrap_x(self._ptr,value)
    def InWrapY(self,value):
        if self._dimension<2: raise ValueError("InWrapY requires a 2D or 3D PDEgrid")
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("index must be an int32 integer")
        return _pd_inwrap_y(self._ptr,value)
    def InWrapZ(self,value):
        if self._dimension<3: raise ValueError("InWrapZ requires a 3D PDEgrid")
        if not np.isfinite(value) or value!=int(value) or value<np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("index must be an int32 integer")
        return _pd_inwrap_z(self._ptr,value)
    def Dx(self): return _pd_dx(self._ptr)
    def Dy(self):
        if self._dimension<2: raise ValueError("Dy requires a 2D or 3D grid")
        return _pd_dy(self._ptr)
    def Dz(self):
        if self._dimension<3: raise ValueError("Dz requires a 3D grid")
        return _pd_dz(self._ptr)
    def Dt(self): return _pd_dt(self._ptr)
    def _CheckBC(self, bc, axis):
        _CheckBCValue(bc,self._ptr,self._dimension,axis)
    def _CheckBCs(self,x0,x1,y0,y1,z0,z1):
        self._CheckBC(x0,0); self._CheckBC(x1,0)
        if self._dimension>1: self._CheckBC(y0,1); self._CheckBC(y1,1)
        elif y0 is not None or y1 is not None: raise ValueError("1D grids do not have Y boundary conditions")
        if self._dimension>2: self._CheckBC(z0,2); self._CheckBC(z1,2)
        elif z0 is not None or z1 is not None: raise ValueError("grid does not have Z boundary conditions")

    def Diffusion(self, rateConstant, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("diffusion rate must be finite and nonnegative")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        ok=_pd_diffusion_bc(self._ptr,rateConstant,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        if ok==0: raise ValueError("explicit diffusion stability condition violated")
    def DiffusionMask(self, rateConstant, mask):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("diffusion rate must be finite and nonnegative")
        if mask.size != len(self): raise ValueError("mask shape must match grid dimensions")
        maskF=np.asarray(mask,dtype=np.float32).reshape(len(self))
        ok=_pd_diffusion_mask(self._ptr,rateConstant,maskF.ctypes.data)
        if ok<0: raise ValueError("mask values must be finite")
        if ok==0: raise ValueError("explicit diffusion stability condition violated")
    def DiffusionField(self, rateConstants, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        if rateConstants.size != len(self): raise ValueError("rateConstants shape must match grid dimensions")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        ratesF=np.asarray(rateConstants,dtype=np.float32).reshape(len(self))
        ok=_pd_diffusion_field(self._ptr,ratesF.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        if ok<0: raise ValueError("rate constants must be finite and nonnegative")
        if ok==0: raise ValueError("explicit diffusion stability condition violated")
    def DiffusionInterfaces(self, rateConstantsX, rateConstantsY=None, rateConstantsZ=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        if rateConstantsX.size != len(self): raise ValueError("rateConstantsX shape must match grid dimensions")
        if self._dimension==1 and (rateConstantsY is not None or rateConstantsZ is not None): raise ValueError("1D DiffusionInterfaces takes x rates only")
        if self._dimension==2 and rateConstantsZ is not None: raise ValueError("2D DiffusionInterfaces takes x and y rates only")
        if self._dimension>1 and (rateConstantsY is None or rateConstantsY.size!=len(self)): raise ValueError("rateConstantsY shape must match grid dimensions")
        if self._dimension>2 and (rateConstantsZ is None or rateConstantsZ.size!=len(self)): raise ValueError("rateConstantsZ shape must match grid dimensions")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        rx=np.asarray(rateConstantsX,dtype=np.float32).reshape(len(self)); ry=np.empty(0,dtype=np.float32) if rateConstantsY is None else np.asarray(rateConstantsY,dtype=np.float32).reshape(len(self)); rz=np.empty(0,dtype=np.float32) if rateConstantsZ is None else np.asarray(rateConstantsZ,dtype=np.float32).reshape(len(self))
        ok=_pd_diffusion_interfaces(self._ptr,rx.ctypes.data,0 if ry.size==0 else ry.ctypes.data,0 if rz.size==0 else rz.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        if ok<0: raise ValueError("rate constants must be finite and nonnegative")
        if ok==0: raise ValueError("explicit diffusion stability condition violated")
    def DiffusionADI(self, rateConstant, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("diffusion rate must be finite and nonnegative")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        ok=_pd_diffusion_adi(self._ptr,rateConstant,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        if ok<0: raise MemoryError("ADI scratch allocation failed")
    def DiffusionRadialCircle(self, rateConstant, outerBC=None):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("diffusion rate must be finite and nonnegative")
        self._CheckBC(outerBC,0)
        ok=_pd_diffusion_radial_circle(self._ptr,rateConstant,_BCPointer(outerBC,self._ptr,1))
        if ok<0: raise ValueError("DiffusionRadialCircle requires a non-wrapped 1D grid with at least 2 points")
        if ok==0: raise ValueError("radial circle diffusion stability condition violated")
    def DiffusionRadialSphere(self, rateConstant, outerBC=None):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("diffusion rate must be finite and nonnegative")
        self._CheckBC(outerBC,0)
        ok=_pd_diffusion_radial_sphere(self._ptr,rateConstant,_BCPointer(outerBC,self._ptr,1))
        if ok<0: raise ValueError("DiffusionRadialSphere requires a non-wrapped 1D grid with at least 2 points")
        if ok==0: raise ValueError("radial sphere diffusion stability condition violated")

    def Advection(self, vx, vy=0.0, vz=0.0, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        if not np.isfinite(vx) or not np.isfinite(vy) or not np.isfinite(vz): raise ValueError("advection velocities must be finite")
        if self._dimension==1 and (vy!=0 or vz!=0): raise ValueError("1D Advection accepts vx only")
        if self._dimension==2 and vz!=0: raise ValueError("2D Advection accepts vx and vy only")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        ok=_pd_advection(self._ptr,vx,vy,vz,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        if ok==0: raise ValueError("advection CFL condition violated")
    def AdvectionField(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        self._AdvectionArrays(xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC,False)
    def AdvectionInterfaces(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        self._AdvectionArrays(xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC,True)
    def _AdvectionArrays(self,xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC,interfaces):
        if xVels.size!=len(self): raise ValueError("x velocity shape must match grid dimensions")
        if self._dimension==1 and (yVels is not None or zVels is not None): raise ValueError("1D advection takes x velocities only")
        if self._dimension==2 and zVels is not None: raise ValueError("2D advection takes x and y velocities only")
        if self._dimension>1 and (yVels is None or yVels.size!=len(self)): raise ValueError("y velocity shape must match grid dimensions")
        if self._dimension>2 and (zVels is None or zVels.size!=len(self)): raise ValueError("z velocity shape must match grid dimensions")
        self._CheckBCs(xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC)
        xv=np.asarray(xVels,dtype=np.float32).reshape(len(self)); yv=np.empty(0,dtype=np.float32) if yVels is None else np.asarray(yVels,dtype=np.float32).reshape(len(self)); zv=np.empty(0,dtype=np.float32) if zVels is None else np.asarray(zVels,dtype=np.float32).reshape(len(self))
        if interfaces: ok=_pd_advection_interfaces(self._ptr,xv.ctypes.data,0 if yv.size==0 else yv.ctypes.data,0 if zv.size==0 else zv.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        else: ok=_pd_advection_field(self._ptr,xv.ctypes.data,0 if yv.size==0 else yv.ctypes.data,0 if zv.size==0 else zv.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        if ok<0: raise ValueError("advection velocities must be finite")
        if ok==0: raise ValueError("advection CFL condition violated")

_PDEgridSafe=_jitclass([("_ptr",uint64),("_dimension",int32),("_xDim",int32),("_yDim",int32),("_zDim",int32),("_length",int32),("_wrapX",types.boolean),("_wrapY",types.boolean),("_wrapZ",types.boolean)])(_PDEgridSafePython)

class _PDEgridFastPython:
    """Tiny JIT-compatible handle for the native PDEgrid."""

    def __init__(self, ptr, dimension, xDim, yDim, zDim, length, wrapX, wrapY, wrapZ):
        self._ptr=ptr; self._dimension=dimension; self._xDim=xDim; self._yDim=yDim; self._zDim=zDim; self._length=length; self._wrapX=wrapX; self._wrapY=wrapY; self._wrapZ=wrapZ

    def __len__(self): return self._length
    @property
    def xDim(self): return self._xDim
    @property
    def yDim(self): return self._yDim
    @property
    def zDim(self): return self._zDim
    @property
    def nDims(self): return self._dimension
    @property
    def wrapX(self): return self._wrapX
    @property
    def wrapY(self): return self._wrapY
    @property
    def wrapZ(self): return self._wrapZ
    def ToI(self, x, y=-1, z=-1): return _pd_toi(self._ptr, x, y, z)

    def ItoX(self, i): return _pd_itox(self._ptr, i)
    def ItoY(self, i): return _pd_itoy(self._ptr, i)
    def ItoZ(self, i): return _pd_itoz(self._ptr, i)

    def __getitem__(self,key): return _PDEItem(self,key,False)

    def __setitem__(self,key,value): _PDESetItem(self,key,value,False)

    def _AddI(self, value, i):
        _pd_addi(self._ptr, i, value)

    def Add(self, value, x, y=-1, z=-1):
        if y==-1 and z==-1: return self._AddI(value,x)
        _pd_add(self._ptr, x, y, z, value)
    def Update(self): _pd_update(self._ptr)

    def Reset(self):
        _pd_clear(self._ptr,0.0)

    def SetTimeSpaceStep(self, dt, dx, dy=1.0, dz=1.0):
        _pd_steps(self._ptr, dt, dx, dy, dz)



    def InWrapX(self, value): return _pd_inwrap_x(self._ptr, value)
    def InWrapY(self, value): return _pd_inwrap_y(self._ptr, value)
    def InWrapZ(self, value): return _pd_inwrap_z(self._ptr, value)
    def Dx(self): return _pd_dx(self._ptr)
    def Dy(self): return _pd_dy(self._ptr)
    def Dz(self): return _pd_dz(self._ptr)
    def Dt(self): return _pd_dt(self._ptr)
    def _CheckBC(self, bc, axis):
        _CheckBCValue(bc,self._ptr,self._dimension,axis)
    def _CheckBCs(self,x0,x1,y0,y1,z0,z1):
        self._CheckBC(x0,0); self._CheckBC(x1,0)
        if self._dimension>1: self._CheckBC(y0,1); self._CheckBC(y1,1)
        elif y0 is not None or y1 is not None: raise ValueError("1D grids do not have Y boundary conditions")
        if self._dimension>2: self._CheckBC(z0,2); self._CheckBC(z1,2)
        elif z0 is not None or z1 is not None: raise ValueError("grid does not have Z boundary conditions")

    def Diffusion(self, rateConstant, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        _pd_diffusion_bc_fast(self._ptr,rateConstant,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
    def DiffusionMask(self, rateConstant, mask):
        if not np.isfinite(rateConstant) or rateConstant<0: raise ValueError("diffusion rate must be finite and nonnegative")
        maskF=np.asarray(mask,dtype=np.float32).reshape(len(self))
        _pd_diffusion_mask_fast(self._ptr,rateConstant,maskF.ctypes.data)
    def DiffusionField(self, rateConstants, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        ratesF=np.asarray(rateConstants,dtype=np.float32).reshape(len(self))
        _pd_diffusion_field_fast(self._ptr,ratesF.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
    def DiffusionInterfaces(self, rateConstantsX, rateConstantsY=None, rateConstantsZ=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        rx=np.asarray(rateConstantsX,dtype=np.float32).reshape(len(self)); ry=np.empty(0,dtype=np.float32) if rateConstantsY is None else np.asarray(rateConstantsY,dtype=np.float32).reshape(len(self)); rz=np.empty(0,dtype=np.float32) if rateConstantsZ is None else np.asarray(rateConstantsZ,dtype=np.float32).reshape(len(self))
        _pd_diffusion_interfaces_fast(self._ptr,rx.ctypes.data,0 if ry.size==0 else ry.ctypes.data,0 if rz.size==0 else rz.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
    def DiffusionADI(self, rateConstant, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        ok=_pd_diffusion_adi(self._ptr,rateConstant,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        if ok<0: raise MemoryError("ADI scratch allocation failed")
    def DiffusionRadialCircle(self, rateConstant, outerBC=None):
        ok=_pd_diffusion_radial_circle_fast(self._ptr,rateConstant,_BCPointer(outerBC,self._ptr,1))
    def DiffusionRadialSphere(self, rateConstant, outerBC=None):
        ok=_pd_diffusion_radial_sphere_fast(self._ptr,rateConstant,_BCPointer(outerBC,self._ptr,1))

    def Advection(self, vx, vy=0.0, vz=0.0, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        _pd_advection_fast(self._ptr,vx,vy,vz,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
    def AdvectionField(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        self._AdvectionArrays(xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC,False)
    def AdvectionInterfaces(self, xVels, yVels=None, zVels=None, xMinBC=None, xMaxBC=None, yMinBC=None, yMaxBC=None, zMinBC=None, zMaxBC=None):
        self._AdvectionArrays(xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC,True)
    def _AdvectionArrays(self,xVels,yVels,zVels,xMinBC,xMaxBC,yMinBC,yMaxBC,zMinBC,zMaxBC,interfaces):
        xv=np.asarray(xVels,dtype=np.float32).reshape(len(self)); yv=np.empty(0,dtype=np.float32) if yVels is None else np.asarray(yVels,dtype=np.float32).reshape(len(self)); zv=np.empty(0,dtype=np.float32) if zVels is None else np.asarray(zVels,dtype=np.float32).reshape(len(self))
        if interfaces: _pd_advection_interfaces_fast(self._ptr,xv.ctypes.data,0 if yv.size==0 else yv.ctypes.data,0 if zv.size==0 else zv.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))
        else: _pd_advection_field_fast(self._ptr,xv.ctypes.data,0 if yv.size==0 else yv.ctypes.data,0 if zv.size==0 else zv.ctypes.data,_BCPointer(xMinBC,self._ptr,0),_BCPointer(xMaxBC,self._ptr,1),_BCPointer(yMinBC,self._ptr,2),_BCPointer(yMaxBC,self._ptr,3),_BCPointer(zMinBC,self._ptr,4),_BCPointer(zMaxBC,self._ptr,5))

_PDEgridFast=_jitclass([("_ptr",uint64),("_dimension",int32),("_xDim",int32),("_yDim",int32),("_zDim",int32),("_length",int32),("_wrapX",types.boolean),("_wrapY",types.boolean),("_wrapZ",types.boolean)])(_PDEgridFastPython)



def _IListItemAt(q,i,line): return q[i]
@overload(_IListItemAt,inline="never",prefer_literal=True)
def _olIListItemAt(q,i,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""
    if q==_IListSafe.class_type.instance_type:
        def impl(q,i,line):
            if not np.isfinite(i) or i!=int(i) or i<np.iinfo(np.int32).min or i>np.iinfo(np.int32).max: raise IndexError("IList index must be an int32 integer"+sx)
            out=_q_get_safe(q._ptr,i)
            if out==np.iinfo(np.int64).min: raise IndexError("IList index out of bounds"+sx)
            return int(out)
        return impl
    if q==_IListFast.class_type.instance_type:
        def impl(q,i,line): return _q_get(q._ptr,i)
        return impl

# Universal thin model-call lowering helpers. These keep public Safe/Fast
# jitclass method bodies out of user kernels while preserving Safe validation.

def _AgentPopAt(g,line): return g.GetPop()
@overload(_AgentPopAt,inline="never",prefer_literal=True)
def _olAgentPopAt(g,line):
    if g==_AgentGridSafe.class_type.instance_type or g==_AgentGridFast.class_type.instance_type:
        def impl(g,line): return _ag_pop(g._ptr)
        return impl

def _AgentValueAt(g,a,op,line): return 0.0
@overload(_AgentValueAt,inline="never",prefer_literal=True)
def _olAgentValueAt(g,a,op,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""
    mi="invalid agent id"+sx; md="invalid agent or dimension"+sx
    if g==_AgentGridSafe.class_type.instance_type:
        def impl(g,a,op,line):
            if not np.isfinite(a) or a!=int(a) or a<0 or a>np.iinfo(np.int32).max: raise ValueError(mi)
            if o==0:
                v=_ag_alive_safe(g._ptr,a)
                if v<0: raise ValueError(mi)
                return v!=0
            if o==1: v=_ag_x_safe(g._ptr,a)
            elif o==2: v=_ag_y_safe(g._ptr,a)
            else: v=_ag_z_safe(g._ptr,a)
            if np.isnan(v): raise ValueError(md)
            return v
        return impl
    if g==_AgentGridFast.class_type.instance_type:
        def impl(g,a,op,line):
            if o==0: return _ag_alive(g._ptr,a)!=0
            if o==1: return _ag_x(g._ptr,a)
            if o==2: return _ag_y(g._ptr,a)
            return _ag_z(g._ptr,a)
        return impl

def _raise_value(message): raise ValueError(message)


def _AgentIndexAt(g,i,op,line): return 0
@overload(_AgentIndexAt,inline="never",prefer_literal=True)
def _olAgentIndexAt(g,i,op,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""
    mi="index must be a nonnegative int32 integer"+sx; mb="invalid index or dimension"+sx
    if g==_AgentGridSafe.class_type.instance_type:
        def impl(g,i,op,line):
            if not np.isfinite(i) or i!=int(i) or i<0 or i>np.iinfo(np.int32).max: raise IndexError(mi)
            if o==0: v=_ag_itox_safe(g._ptr,i)
            elif o==1: v=_ag_itoy_safe(g._ptr,i)
            else: v=_ag_itoz_safe(g._ptr,i)
            if v==-2147483648: raise IndexError(mb)
            return v
        return impl
    if g==_AgentGridFast.class_type.instance_type:
        def impl(g,i,op,line):
            if o==0:return _ag_itox(g._ptr,i)
            if o==1:return _ag_itoy(g._ptr,i)
            return _ag_itoz(g._ptr,i)
        return impl

def _AgentWrapAt(g,v,op,line): return v
@overload(_AgentWrapAt,inline="never",prefer_literal=True)
def _olAgentWrapAt(g,v,op,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""
    if g==_AgentGridSafe.class_type.instance_type:
        if o<3:
            msg="square coordinate must be an int32 integer"+sx; md="invalid dimension"+sx
            def impl(g,v,op,line):
                if not np.isfinite(v) or v!=int(v) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max: raise ValueError(msg)
                out=_ag_inwrap_sq_safe(g._ptr,v,o)
                if out==-2147483648: raise ValueError(md)
                return out
            return impl
        msg="invalid coordinate or dimension"+sx
        def impl(g,v,op,line):
            out=_ag_inwrap_safe(g._ptr,v,o-3)
            if np.isnan(out): raise ValueError(msg)
            return out
        return impl
    if g==_AgentGridFast.class_type.instance_type:
        def impl(g,v,op,line):
            if o==0:return _ag_inwrap_sq_x(g._ptr,v)
            if o==1:return _ag_inwrap_sq_y(g._ptr,v)
            if o==2:return _ag_inwrap_sq_z(g._ptr,v)
            if o==3:return _ag_inwrap_x(g._ptr,v)
            if o==4:return _ag_inwrap_y(g._ptr,v)
            return _ag_inwrap_z(g._ptr,v)
        return impl

def _AgentDispAt(g,a,b,axis,line): return 0.0
@overload(_AgentDispAt,inline="never",prefer_literal=True)
def _olAgentDispAt(g,a,b,axis,line):
    ax=axis.literal_value if isinstance(axis,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; msg="invalid coordinates or dimension"+(" at source line "+str(ln) if ln>=0 else "")
    if g==_AgentGridSafe.class_type.instance_type:
        def impl(g,a,b,axis,line):
            out=_ag_dispwrap_safe(g._ptr,a,b,ax)
            if np.isnan(out): raise ValueError(msg)
            return out
        return impl
    if g==_AgentGridFast.class_type.instance_type:
        def impl(g,a,b,axis,line):
            if ax==0:return _ag_dispwrap_x(g._ptr,a,b)
            if ax==1:return _ag_dispwrap_y(g._ptr,a,b)
            return _ag_dispwrap_z(g._ptr,a,b)
        return impl

def _AgentNewAt(g,x,y,z,line): return g.NewAgent(x,y,z)
@overload(_AgentNewAt,inline="never",prefer_literal=True)
def _olAgentNewAt(g,x,y,z,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; mc="coordinates must be finite"+sx; m1="1D AgentGrid requires x only"+sx; m2="2D AgentGrid requires x,y"+sx; m3="3D AgentGrid requires x,y,z"+sx; mu="unable to place agent"+sx
    if g==_AgentGridSafe.class_type.instance_type:
        def impl(g,x,y,z,line):
            if g._dimension==0: a=_ag_new_i(g._ptr,-1)
            else:
                if not np.isfinite(x): raise ValueError(mc)
                if g._dimension==1:
                    if y!=-1.0 or z!=-1.0: raise ValueError(m1)
                elif g._dimension==2:
                    if not np.isfinite(y) or z!=-1.0: raise ValueError(m2)
                elif not np.isfinite(y) or not np.isfinite(z): raise ValueError(m3)
                a=_ag_new_pt(g._ptr,x,y,z)
            if a<0: raise ValueError(mu)
            return a
        return impl
    if g==_AgentGridFast.class_type.instance_type:
        def impl(g,x,y,z,line): return _ag_new_i_fast(g._ptr,-1) if g._dimension==0 else _ag_new_pt_fast(g._ptr,x,y,z)
        return impl

def _AgentLastAt(g,x,y,z,line): return g.LastAgent(x,y,z)
@overload(_AgentLastAt,inline="never",prefer_literal=True)
def _olAgentLastAt(g,x,y,z,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""
    if g==_AgentGridSafe.class_type.instance_type:
        def impl(g,x,y,z,line):
            if g._dimension==0: raise ValueError("GetLastI requires a spatial AgentGrid"+sx)
            if y==-1 and z==-1: i=x
            else:
                if not np.isfinite(x) or x!=int(x) or (g._dimension>1 and (not np.isfinite(y) or y!=int(y))) or (g._dimension>2 and (not np.isfinite(z) or z!=int(z))): raise ValueError("coordinates must be finite integers"+sx)
                if x<0 or x>=g._xDim or g._dimension>1 and (y<0 or y>=g._yDim) or g._dimension>2 and (z<0 or z>=g._zDim): raise IndexError("coordinate out of bounds"+sx)
                i=_ag_toi(g._ptr,x,y,z)
            if not np.isfinite(i) or i!=int(i) or i<0 or i>=g._length: raise IndexError("index out of bounds"+sx)
            return _ag_last_i(g._ptr,i)
        return impl
    if g==_AgentGridFast.class_type.instance_type:
        def impl(g,x,y,z,line): return _ag_last_i_fast(g._ptr,x if y==-1 and z==-1 else _ag_toi(g._ptr,x,y,z))
        return impl

def _AgentMoveAt(g,a,x,y,z,point,line): return None
@overload(_AgentMoveAt,inline="never",prefer_literal=True)
def _olAgentMoveAt(g,a,x,y,z,point,line):
    pt=point.literal_value if isinstance(point,types.BooleanLiteral) else False; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; mi="invalid agent id"+sx; md="agent is not alive"+sx; mu="unable to move agent"+sx
    if g==_AgentGridSafe.class_type.instance_type:
        def impl(g,a,x,y,z,point,line):
            if not np.isfinite(a) or a!=int(a) or a<0 or a>np.iinfo(np.int32).max: raise ValueError(mi)
            alive=_ag_alive_safe(g._ptr,a)
            if alive<=0: raise ValueError(mi if alive<0 else md)
            if pt:
                if not np.isfinite(x) or (g._dimension>1 and not np.isfinite(y)) or (g._dimension>2 and not np.isfinite(z)): raise ValueError("coordinates must be finite"+sx)
                ok=_ag_move_pt(g._ptr,a,x,y,z)
            else:
                if y==-1 and z==-1:
                    if not np.isfinite(x) or x!=int(x) or x<0 or x>=g._length: raise IndexError("index out of bounds"+sx)
                    i=x
                else:
                    if not np.isfinite(x) or x!=int(x) or (g._dimension>1 and (not np.isfinite(y) or y!=int(y))) or (g._dimension>2 and (not np.isfinite(z) or z!=int(z))): raise ValueError("coordinates must be finite integers"+sx)
                    if x<0 or x>=g._xDim or g._dimension>1 and (y<0 or y>=g._yDim) or g._dimension>2 and (z<0 or z>=g._zDim): raise IndexError("coordinate out of bounds"+sx)
                    i=_ag_toi(g._ptr,x,y,z)
                ok=_ag_move_i(g._ptr,a,i)
            if ok<=0: raise ValueError(mu)
        return impl
    if g==_AgentGridFast.class_type.instance_type:
        def impl(g,a,x,y,z,point,line):
            if pt:_ag_move_pt_fast(g._ptr,a,x,y,z)
            else:_ag_move_i_fast(g._ptr,a,x if y==-1 and z==-1 else _ag_toi(g._ptr,x,y,z))
        return impl

def _IListOpAt(q,op,line): return q
@overload(_IListOpAt,inline="never",prefer_literal=True)
def _olIListOpAt(q,op,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1
    if q==_IListSafe.class_type.instance_type or q==_IListFast.class_type.instance_type:
        def impl(q,op,line):
            if o==0:_q_shuffle(q._ptr)
            return q
        return impl

def _IListArrayAt(q,copyOut,line): return q.All()
@overload(_IListArrayAt,inline="never",prefer_literal=True)
def _olIListArrayAt(q,copyOut,line):
    cp=copyOut.literal_value if isinstance(copyOut,types.BooleanLiteral) else True
    if q==_IListSafe.class_type.instance_type or q==_IListFast.class_type.instance_type:
        def impl(q,copyOut,line):
            if cp:
                out=np.empty(_q_len(q._ptr),dtype=np.int32); _q_copy(q._ptr,out.ctypes); return out
            return nb.carray(_q_data(q._ptr),_q_len(q._ptr))
        return impl

def _MultiBinomialAt(m,n,p,line): return m.Binomial(n,p)
@overload(_MultiBinomialAt,inline="never",prefer_literal=True)
def _olMultiBinomialAt(m,n,p,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; mn="n must be a nonnegative int64 integer"+sx; ma="invalid binomial arguments"+sx
    if m==_MultinomialSafe.class_type.instance_type:
        def impl(m,n,p,line):
            if not np.isfinite(n) or n!=int(n) or n<0 or n>np.iinfo(np.int64).max: raise ValueError(mn)
            out=_multi_binomial(m._ptr,n,p)
            if out<0: raise ValueError(ma)
            return out
        return impl
    if m==_MultinomialFast.class_type.instance_type:
        def impl(m,n,p,line): return _multi_binomial(m._ptr,n,p)
        return impl

def _MultiSetupAt(m,n,line): return m.Setup(n)
@overload(_MultiSetupAt,inline="never",prefer_literal=True)
def _olMultiSetupAt(m,n,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; mn="n must be a nonnegative int64 integer"+sx; ma="invalid multinomial n"+sx
    if m==_MultinomialSafe.class_type.instance_type:
        def impl(m,n,line):
            if not np.isfinite(n) or n!=int(n) or n<0 or n>np.iinfo(np.int64).max: raise ValueError(mn)
            if not _multi_setup(m._ptr,n): raise ValueError(ma)
            return m
        return impl
    if m==_MultinomialFast.class_type.instance_type:
        def impl(m,n,line): _multi_setup(m._ptr,n); return m
        return impl

def _PDEAddAt(g,v,x,y,z,line): return g.Add(v,x,y,z)
@overload(_PDEAddAt,inline="never",prefer_literal=True)
def _olPDEAddAt(g,v,x,y,z,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; mv="value must be finite"+sx; mi="index must be a finite integer"+sx; mb="index out of bounds"+sx; mc="coordinates must be finite integers"+sx; cb="coordinate out of bounds"+sx
    if g==_PDEgridSafe.class_type.instance_type:
        def impl(g,v,x,y,z,line):
            if not np.isfinite(v): raise ValueError(mv)
            if y==-1 and z==-1:
                if not np.isfinite(x) or x!=int(x): raise ValueError(mi)
                if x<0 or x>=g._length: raise IndexError(mb)
                _pd_addi(g._ptr,x,v); return
            if not np.isfinite(x) or x!=int(x) or (g._dimension>1 and (not np.isfinite(y) or y!=int(y))) or (g._dimension>2 and (not np.isfinite(z) or z!=int(z))): raise ValueError(mc)
            if x<0 or x>=g._xDim or g._dimension>1 and (y<0 or y>=g._yDim) or g._dimension>2 and (z<0 or z>=g._zDim): raise IndexError(cb)
            _pd_add(g._ptr,x,y,z,v)
        return impl
    if g==_PDEgridFast.class_type.instance_type:
        def impl(g,v,x,y,z,line):
            if y==-1 and z==-1:_pd_addi(g._ptr,x,v)
            else:_pd_add(g._ptr,x,y,z,v)
        return impl

def _PDEBasicAt(g,op,a,b,c,d,line): return 0.0
@overload(_PDEBasicAt,inline="never",prefer_literal=True)
def _olPDEBasicAt(g,op,a,b,c,d,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""
    safe=g==_PDEgridSafe.class_type.instance_type; fast=g==_PDEgridFast.class_type.instance_type
    if not (safe or fast): return None
    def impl(g,op,a,b,c,d,line):
        if o==0: _pd_update(g._ptr); return 0.0
        if o==1: _pd_clear(g._ptr,0.0); return 0.0
        if o==2:
            if safe and (not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c) or not np.isfinite(d) or a<=0 or b<=0 or c<=0 or d<=0): raise ValueError("time and space steps must be finite and positive"+sx)
            _pd_steps(g._ptr,a,b,c,d); return 0.0
        if o==3:return _pd_dx(g._ptr)
        if o==4:
            if safe and g._dimension<2: raise ValueError("Dy requires a 2D or 3D grid"+sx)
            return _pd_dy(g._ptr)
        if o==5:
            if safe and g._dimension<3: raise ValueError("Dz requires a 3D grid"+sx)
            return _pd_dz(g._ptr)
        return _pd_dt(g._ptr)
    return impl

def _PDEScalarTransportAt(g,op,a,b,c,x0,x1,y0,y1,z0,z1,line): return None
@overload(_PDEScalarTransportAt,inline="never",prefer_literal=True)
def _olPDEScalarTransportAt(g,op,a,b,c,x0,x1,y0,y1,z0,z1,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; safe=g==_PDEgridSafe.class_type.instance_type
    if safe or g==_PDEgridFast.class_type.instance_type:
        def impl(g,op,a,b,c,x0,x1,y0,y1,z0,z1,line):
            if safe:
                if o==0:
                    if not np.isfinite(a) or a<0: raise ValueError("diffusion rate must be finite and nonnegative"+sx)
                else:
                    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c): raise ValueError("advection velocities must be finite"+sx)
                    if g._dimension==1 and (b!=0 or c!=0): raise ValueError("1D Advection accepts vx only"+sx)
                    if g._dimension==2 and c!=0: raise ValueError("2D Advection accepts vx and vy only"+sx)
                _CheckBCValue(x0,g._ptr,g._dimension,0); _CheckBCValue(x1,g._ptr,g._dimension,0)
                if g._dimension>1: _CheckBCValue(y0,g._ptr,g._dimension,1); _CheckBCValue(y1,g._ptr,g._dimension,1)
                elif y0 is not None or y1 is not None: raise ValueError("1D grids do not have Y boundary conditions"+sx)
                if g._dimension>2: _CheckBCValue(z0,g._ptr,g._dimension,2); _CheckBCValue(z1,g._ptr,g._dimension,2)
                elif z0 is not None or z1 is not None: raise ValueError("grid does not have Z boundary conditions"+sx)
            if o==0:
                ok=_pd_diffusion_adi(g._ptr,a,_BCPointer(x0,g._ptr,0),_BCPointer(x1,g._ptr,1),_BCPointer(y0,g._ptr,2),_BCPointer(y1,g._ptr,3),_BCPointer(z0,g._ptr,4),_BCPointer(z1,g._ptr,5))
                if safe and ok<0: raise MemoryError("ADI scratch allocation failed"+sx)
            else:
                ok=_pd_advection(g._ptr,a,b,c,_BCPointer(x0,g._ptr,0),_BCPointer(x1,g._ptr,1),_BCPointer(y0,g._ptr,2),_BCPointer(y1,g._ptr,3),_BCPointer(z0,g._ptr,4),_BCPointer(z1,g._ptr,5)) if safe else _pd_advection_fast(g._ptr,a,b,c,_BCPointer(x0,g._ptr,0),_BCPointer(x1,g._ptr,1),_BCPointer(y0,g._ptr,2),_BCPointer(y1,g._ptr,3),_BCPointer(z0,g._ptr,4),_BCPointer(z1,g._ptr,5))
                if safe and ok==0: raise ValueError("advection CFL condition violated"+sx)
        return impl

def _PDERadialAt(g,r,bc,sphere,line): return None
@overload(_PDERadialAt,inline="never",prefer_literal=True)
def _olPDERadialAt(g,r,bc,sphere,line):
    sp=sphere.literal_value if isinstance(sphere,types.BooleanLiteral) else False; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; safe=g==_PDEgridSafe.class_type.instance_type
    if safe or g==_PDEgridFast.class_type.instance_type:
        def impl(g,r,bc,sphere,line):
            if safe:
                if not np.isfinite(r) or r<0: raise ValueError("diffusion rate must be finite and nonnegative"+sx)
                _CheckBCValue(bc,g._ptr,g._dimension,0)
                ok=_pd_diffusion_radial_sphere(g._ptr,r,_BCPointer(bc,g._ptr,1)) if sp else _pd_diffusion_radial_circle(g._ptr,r,_BCPointer(bc,g._ptr,1))
                if ok<0: raise ValueError(("DiffusionRadialSphere" if sp else "DiffusionRadialCircle")+" requires a non-wrapped 1D grid with at least 2 points"+sx)
                if ok==0: raise ValueError(("radial sphere" if sp else "radial circle")+" diffusion stability condition violated"+sx)
            elif sp:_pd_diffusion_radial_sphere_fast(g._ptr,r,_BCPointer(bc,g._ptr,1))
            else:_pd_diffusion_radial_circle_fast(g._ptr,r,_BCPointer(bc,g._ptr,1))
        return impl

def _PDEDiffusionAt(g,r,x0,x1,y0,y1,z0,z1,line): return None
@overload(_PDEDiffusionAt,inline="never",prefer_literal=True)
def _olPDEDiffusionAt(g,r,x0,x1,y0,y1,z0,z1,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; safe=g==_PDEgridSafe.class_type.instance_type
    if safe or g==_PDEgridFast.class_type.instance_type:
        def impl(g,r,x0,x1,y0,y1,z0,z1,line):
            if safe:
                if not np.isfinite(r) or r<0: raise ValueError("diffusion rate must be finite and nonnegative"+sx)
                _CheckBCValue(x0,g._ptr,g._dimension,0); _CheckBCValue(x1,g._ptr,g._dimension,0)
                if g._dimension>1: _CheckBCValue(y0,g._ptr,g._dimension,1); _CheckBCValue(y1,g._ptr,g._dimension,1)
                elif y0 is not None or y1 is not None: raise ValueError("1D grids do not have Y boundary conditions"+sx)
                if g._dimension>2: _CheckBCValue(z0,g._ptr,g._dimension,2); _CheckBCValue(z1,g._ptr,g._dimension,2)
                elif z0 is not None or z1 is not None: raise ValueError("grid does not have Z boundary conditions"+sx)
                ok=_pd_diffusion_bc(g._ptr,r,_BCPointer(x0,g._ptr,0),_BCPointer(x1,g._ptr,1),_BCPointer(y0,g._ptr,2),_BCPointer(y1,g._ptr,3),_BCPointer(z0,g._ptr,4),_BCPointer(z1,g._ptr,5))
                if ok==0: raise ValueError("explicit diffusion stability condition violated"+sx)
            else:_pd_diffusion_bc_fast(g._ptr,r,_BCPointer(x0,g._ptr,0),_BCPointer(x1,g._ptr,1),_BCPointer(y0,g._ptr,2),_BCPointer(y1,g._ptr,3),_BCPointer(z0,g._ptr,4),_BCPointer(z1,g._ptr,5))
        return impl


def _SpatialIndexAt(g,x,y,z,op,line): return 0
@overload(_SpatialIndexAt,inline="never",prefer_literal=True)
def _olSpatialIndexAt(g,x,y,z,op,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""
    isPDE=g==_PDEgridSafe.class_type.instance_type or g==_PDEgridFast.class_type.instance_type; safe=g==_PDEgridSafe.class_type.instance_type or g==_PopGridSafe.class_type.instance_type
    if not (isPDE or g==_PopGridSafe.class_type.instance_type or g==_PopGridFast.class_type.instance_type): return None
    def impl(g,x,y,z,op,line):
        if o==0:
            if safe:
                if not np.isfinite(x) or x!=int(x) or (g._dimension>1 and (not np.isfinite(y) or y!=int(y))) or (g._dimension>2 and (not np.isfinite(z) or z!=int(z))): raise ValueError("coordinates must be finite integers"+sx)
                if x<0 or x>=g._xDim or g._dimension>1 and (y<0 or y>=g._yDim) or g._dimension>2 and (z<0 or z>=g._zDim): raise IndexError("coordinate out of bounds"+sx)
            v=_pd_toi_safe(g._ptr,x,y,z) if isPDE and safe else _pd_toi(g._ptr,x,y,z) if isPDE else _pg_toi_safe(g._ptr,x,y,z) if safe else _pg_toi(g._ptr,x,y,z)
            if safe and v==-2147483648: raise IndexError("invalid coordinates"+sx)
            return v
        if safe and (not np.isfinite(x) or x!=int(x) or x<0 or x>np.iinfo(np.int32).max): raise IndexError("index must be a nonnegative int32 integer"+sx)
        if isPDE:
            v=_pd_itox_safe(g._ptr,x) if o==1 and safe else _pd_itoy_safe(g._ptr,x) if o==2 and safe else _pd_itoz_safe(g._ptr,x) if o==3 and safe else _pd_itox(g._ptr,x) if o==1 else _pd_itoy(g._ptr,x) if o==2 else _pd_itoz(g._ptr,x)
        else:
            v=_pg_itox_safe(g._ptr,x) if o==1 and safe else _pg_itoy_safe(g._ptr,x) if o==2 and safe else _pg_itoz_safe(g._ptr,x) if o==3 and safe else _pg_itox(g._ptr,x) if o==1 else _pg_itoy(g._ptr,x) if o==2 else _pg_itoz(g._ptr,x)
        if safe and v==-2147483648: raise IndexError("invalid index or dimension"+sx)
        return v
    return impl

def _SpatialWrapAt(g,v,axis,line): return 0
@overload(_SpatialWrapAt,inline="never",prefer_literal=True)
def _olSpatialWrapAt(g,v,axis,line):
    ax=axis.literal_value if isinstance(axis,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; isPDE=g==_PDEgridSafe.class_type.instance_type or g==_PDEgridFast.class_type.instance_type; safe=g==_PDEgridSafe.class_type.instance_type or g==_PopGridSafe.class_type.instance_type
    if not (isPDE or g==_PopGridSafe.class_type.instance_type or g==_PopGridFast.class_type.instance_type): return None
    def impl(g,v,axis,line):
        if safe:
            if ax==1 and g._dimension<2: raise ValueError("InWrapY requires a 2D or 3D grid"+sx)
            if ax==2 and g._dimension<3: raise ValueError("InWrapZ requires a 3D grid"+sx)
            if not np.isfinite(v) or v!=int(v) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max: raise ValueError("index must be an int32 integer"+sx)
        if isPDE:
            return _pd_inwrap_x(g._ptr,v) if ax==0 else _pd_inwrap_y(g._ptr,v) if ax==1 else _pd_inwrap_z(g._ptr,v)
        return _pg_inwrap_x(g._ptr,v) if ax==0 else _pg_inwrap_y(g._ptr,v) if ax==1 else _pg_inwrap_z(g._ptr,v)
    return impl

def _PopAddAt(g,v,x,y,z,line): return g.Add(v,x,y,z)
@overload(_PopAddAt,inline="never",prefer_literal=True)
def _olPopAddAt(g,v,x,y,z,line):
    ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; safe=g==_PopGridSafe.class_type.instance_type
    if safe or g==_PopGridFast.class_type.instance_type:
        def impl(g,v,x,y,z,line):
            if safe:
                if not np.isfinite(v) or v!=int(v) or v<np.iinfo(np.int64).min or v>np.iinfo(np.int64).max: raise ValueError("delta must be an int64 integer"+sx)
                if y==-1 and z==-1:
                    if not np.isfinite(x) or x!=int(x): raise ValueError("index must be a finite integer"+sx)
                    if x<0 or x>=g._length: raise IndexError("index out of bounds"+sx)
                    if not _pg_addi(g._ptr,x,v): raise OverflowError("PopGrid delta overflow or invalid index"+sx)
                else:
                    if not _pg_add(g._ptr,x,y,z,v): raise OverflowError("PopGrid delta overflow"+sx)
            elif y==-1 and z==-1:_pg_addi_fast(g._ptr,x,v)
            else:_pg_add_fast(g._ptr,x,y,z,v)
        return impl

def _PopBasicAt(g,op,line): return 0
@overload(_PopBasicAt,inline="never",prefer_literal=True)
def _olPopBasicAt(g,op,line):
    o=op.literal_value if isinstance(op,types.IntegerLiteral) else -1; ln=line.literal_value if isinstance(line,types.IntegerLiteral) else -1; sx=" at source line "+str(ln) if ln>=0 else ""; safe=g==_PopGridSafe.class_type.instance_type
    if safe or g==_PopGridFast.class_type.instance_type:
        def impl(g,op,line):
            if o==0:
                ok=_pg_update(g._ptr) if safe else _pg_update_fast(g._ptr)
                if safe and not ok: raise ValueError("PopGrid update would exceed bounds"+sx)
                return 0
            if o==1:_pg_clear(g._ptr); return 0
            return _pg_pop(g._ptr)
        return impl

def _PopAllAt(g,line): return g.All()
@overload(_PopAllAt,inline="never",prefer_literal=True)
def _olPopAllAt(g,line):
    if g==_PopGridSafe.class_type.instance_type or g==_PopGridFast.class_type.instance_type:
        def impl(g,line):
            out=np.empty(_pg_all_count(g._ptr),dtype=np.int32); _pg_all_copy(g._ptr,out.ctypes); return out
        return impl


def _dimensions_array(dimensions, allowZero=False):
    raw = np.asarray(dimensions)
    minDim = 0 if allowZero else 1
    if raw.ndim != 1 or len(raw) < minDim or len(raw) > 3:
        raise ValueError("dimensions must contain %d to 3 dimensions" % minDim)
    values=[]
    length=1
    for dim in raw:
        try: value=float(dim)
        except (TypeError,ValueError,OverflowError): raise ValueError("dimension sizes must be finite integers")
        if not np.isfinite(value) or value!=int(value): raise ValueError("dimension sizes must be finite integers")
        value=int(value)
        if value==0: raise ValueError("dimension sizes cannot be zero")
        if value<=np.iinfo(np.int32).min or value>np.iinfo(np.int32).max: raise ValueError("dimension size out of int32 range")
        length*=abs(value)
        if length>np.iinfo(np.int32).max: raise ValueError("grid length exceeds int32 range")
        values.append(value)
    return np.ascontiguousarray(values,dtype=np.int32)


def Seed(seed):
    try: raw=float(seed)
    except (TypeError,ValueError,OverflowError): raise ValueError("seed must be a uint64")
    if not np.isfinite(raw) or raw!=int(raw): raise ValueError("seed must be a uint64")
    seed=int(seed)
    if seed < 0 or seed > np.iinfo(np.uint64).max: raise ValueError("seed must be a uint64")
    _seed(seed)


@nb.njit(cache=True, inline="always")
def Random():
    return _random()


@nb.njit(cache=True, inline="always")
def RandInt(max):
    if not np.isfinite(max) or max!=int(max) or max<=0 or max>np.iinfo(np.int64).max: raise ValueError("max must be a positive int64")
    return _rand_int(int(max))


# Generic typed lattice Grid. Python owns only the wrapper; storage is C-owned.
# The custom Numba native type carries the raw data address by value. Consequently
# FastMode scalar [] access lowers directly to a load/store with no PAL function
# call and no jitclass method dispatch.
_GRID_DTYPES=(np.bool_,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,np.float32,np.float64)
_GRID_TYPE_CODES={np.dtype(dt):i for i,dt in enumerate(_GRID_DTYPES)}

class _GridPython:
    def __init__(self,ptr,data,dimensions,dtype,safe):
        self._ptr=int(ptr);self._data=data;self._dtype=np.dtype(dtype);self._safe=bool(safe)
        dims=np.asarray(dimensions,dtype=np.int32);self._dimension=len(dims)
        vals=np.abs(dims);self._xDim=int(vals[0]);self._yDim=int(vals[1]) if len(vals)>1 else 1;self._zDim=int(vals[2]) if len(vals)>2 else 1
        self._wrapX=bool(dims[0]<0);self._wrapY=bool(dims[1]<0) if len(dims)>1 else False;self._wrapZ=bool(dims[2]<0) if len(dims)>2 else False
        self._address=int(data.ctypes.data)
    def __len__(self):return len(self._data)
    @property
    def xDim(self): return self._xDim
    @property
    def yDim(self):
        if self._dimension<2: raise ValueError("yDim requires a 2D or 3D grid")
        return self._yDim
    @property
    def zDim(self):
        if self._dimension<3: raise ValueError("zDim requires a 3D grid")
        return self._zDim
    @property
    def nDims(self): return self._dimension
    @property
    def wrapX(self): return self._wrapX
    @property
    def wrapY(self): return self._wrapY
    @property
    def wrapZ(self): return self._wrapZ
    def _shape(self):return (self._xDim,) if self._dimension==1 else (self._xDim,self._yDim) if self._dimension==2 else (self._xDim,self._yDim,self._zDim)
    def _checkValue(self,value):
        a=np.asarray(value)
        dt=self._dtype
        if dt.kind=='b':
            if np.any((a!=0)&(a!=1)):raise ValueError("value is not representable by Grid dtype")
        elif dt.kind in 'iu':
            info=np.iinfo(dt)
            if np.any(~np.isfinite(a)) or np.any(a!=np.floor(a)) or np.any(a<info.min) or np.any(a>info.max):raise ValueError("value is not representable by Grid dtype")
        else:
            info=np.finfo(dt)
            if np.any(~np.isfinite(a)) or np.any(a<-info.max) or np.any(a>info.max):raise ValueError("value is not representable by Grid dtype")
    def __getitem__(self,key):
        if self._safe:
            if isinstance(key,tuple):
                if len(key)!=self._dimension: raise IndexError("Grid index dimensionality mismatch")
                dims=self._shape()
                for d,k in enumerate(key):
                    if not isinstance(k,slice) and (not isinstance(k,(int,np.integer)) or isinstance(k,(bool,np.bool_)) or k<0 or k>=dims[d]): raise IndexError("Grid coordinate out of bounds")
            elif not isinstance(key,slice) and (not isinstance(key,(int,np.integer)) or isinstance(key,(bool,np.bool_)) or key<0 or key>=len(self)): raise IndexError("Grid index out of bounds")
        if isinstance(key,tuple):return self._data.reshape(self._shape())[key].copy() if any(isinstance(k,slice) for k in key) else self._data.reshape(self._shape())[key]
        if isinstance(key,slice):return self._data[key].copy()
        return self._data[key]
    def __setitem__(self,key,value):
        if self._safe:
            self._checkValue(value)
            if isinstance(key,tuple):
                if len(key)!=self._dimension: raise IndexError("Grid index dimensionality mismatch")
                dims=self._shape()
                for d,k in enumerate(key):
                    if not isinstance(k,slice) and (not isinstance(k,(int,np.integer)) or isinstance(k,(bool,np.bool_)) or k<0 or k>=dims[d]): raise IndexError("Grid coordinate out of bounds")
            elif not isinstance(key,slice) and (not isinstance(key,(int,np.integer)) or isinstance(key,(bool,np.bool_)) or key<0 or key>=len(self)): raise IndexError("Grid index out of bounds")
        if isinstance(key,tuple):self._data.reshape(self._shape())[key]=value
        else:self._data[key]=value
    def ToI(self,x,y=-1,z=-1):
        if self._safe:
            vals=(x,) if self._dimension==1 else (x,y) if self._dimension==2 else (x,y,z)
            dims=(self._xDim,self._yDim,self._zDim)
            for d,v in enumerate(vals):
                if not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)): raise ValueError("coordinates must be integers")
                if v<0 or v>=dims[d]: raise IndexError("coordinate out of bounds")
            if self._dimension==1 and (y!=-1 or z!=-1) or self._dimension==2 and z!=-1: raise ValueError("coordinate dimensionality mismatch")
        if self._dimension==1:return int(x)
        if self._dimension==2:return int(x)*self._yDim+int(y)
        return (int(x)*self._yDim+int(y))*self._zDim+int(z)
    def ItoX(self,i):
        if self._safe and (not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) or i<0 or i>=len(self)): raise IndexError("index out of bounds")
        return int(i) if self._dimension==1 else int(i)//self._yDim if self._dimension==2 else int(i)//(self._yDim*self._zDim)
    def ItoY(self,i):
        if self._dimension<2:raise ValueError("ItoY requires a 2D or 3D grid")
        if self._safe and (not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) or i<0 or i>=len(self)): raise IndexError("index out of bounds")
        return int(i)%self._yDim if self._dimension==2 else (int(i)//self._zDim)%self._yDim
    def ItoZ(self,i):
        if self._dimension<3:raise ValueError("ItoZ requires a 3D grid")
        if self._safe and (not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) or i<0 or i>=len(self)): raise IndexError("index out of bounds")
        return int(i)%self._zDim
    def InWrapX(self,v):
        if self._safe and (not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max): raise ValueError("coordinate must be an int32 integer")
        return v if 0<=v<self._xDim else v%self._xDim if self._wrapX else -1
    def InWrapY(self,v):
        if self._dimension<2 and self._safe: raise ValueError("InWrapY requires a 2D or 3D Grid")
        if self._safe and (not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max): raise ValueError("coordinate must be an int32 integer")
        return v if 0<=v<self._yDim else v%self._yDim if self._wrapY else -1
    def InWrapZ(self,v):
        if self._dimension<3 and self._safe: raise ValueError("InWrapZ requires a 3D Grid")
        if self._safe and (not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max): raise ValueError("coordinate must be an int32 integer")
        return v if 0<=v<self._zDim else v%self._zDim if self._wrapZ else -1
    def DispWrapX(self,a,b):
        d=b-a
        if self._wrapX:
            h=self._xDim/2
            if d>h:d-=self._xDim
            elif d<-h:d+=self._xDim
        return d
    def DispWrapY(self,a,b):
        d=b-a
        if self._wrapY:
            h=self._yDim/2
            if d>h:d-=self._yDim
            elif d<-h:d+=self._yDim
        return d
    def DispWrapZ(self,a,b):
        d=b-a
        if self._wrapZ:
            h=self._zDim/2
            if d>h:d-=self._zDim
            elif d<-h:d+=self._zDim
        return d
class _GridNumbaType(types.Type):
    def __init__(self,dtype,safe,dims,wrap):
        self.dtype=dtype;self.safe=bool(safe);self.dims=tuple(int(v) for v in dims);self.wrap=tuple(bool(v) for v in wrap);self.dimension=len(self.dims)
        super().__init__(name=f"Grid[{dtype},{'safe' if safe else 'fast'},dims={self.dims},wrap={self.wrap}]")
    def __hash__(self):return hash((type(self),self.dtype,self.safe,self.dims,self.wrap))
    def __eq__(self,o):return isinstance(o,_GridNumbaType) and self.dtype==o.dtype and self.safe==o.safe and self.dims==o.dims and self.wrap==o.wrap

@typeof_impl.register(_GridPython)
def _typeofGrid(val,c):
    dims=(val._xDim,) if val._dimension==1 else (val._xDim,val._yDim) if val._dimension==2 else (val._xDim,val._yDim,val._zDim)
    wrap=(val._wrapX,) if val._dimension==1 else (val._wrapX,val._wrapY) if val._dimension==2 else (val._wrapX,val._wrapY,val._wrapZ)
    return _GridNumbaType(nb.from_dtype(val._dtype),val._safe,dims,wrap)

@register_model(_GridNumbaType)
class _GridModel(models.StructModel):
    def __init__(self,dmm,fe):
        super().__init__(dmm,fe,[("ptr",types.uintp),("address",types.uintp),("dimension",types.int32),("xDim",types.int32),("yDim",types.int32),("zDim",types.int32),("wrapX",types.boolean),("wrapY",types.boolean),("wrapZ",types.boolean)])
for _native,_py in (("ptr","_ptr"),("address","_address"),("dimension","_dimension"),("xDim","_xDim"),("yDim","_yDim"),("zDim","_zDim"),("wrapX","_wrapX"),("wrapY","_wrapY"),("wrapZ","_wrapZ")):
    make_attribute_wrapper(_GridNumbaType,_native,_py)
for _native,_py in (("dimension","nDims"),("xDim","xDim"),("yDim","yDim"),("zDim","zDim"),("wrapX","wrapX"),("wrapY","wrapY"),("wrapZ","wrapZ")):
    make_attribute_wrapper(_GridNumbaType,_native,_py)

@unbox(_GridNumbaType)
def _unboxGrid(typ,obj,c):
    p=cgutils.create_struct_proxy(typ)(c.context,c.builder)
    for native,py,nt in (("ptr","_ptr",types.uintp),("address","_address",types.uintp),("dimension","_dimension",types.int32),("xDim","_xDim",types.int32),("yDim","_yDim",types.int32),("zDim","_zDim",types.int32),("wrapX","_wrapX",types.boolean),("wrapY","_wrapY",types.boolean),("wrapZ","_wrapZ",types.boolean)):
        o=c.pyapi.object_getattr_string(obj,py);setattr(p,native,c.unbox(nt,o).value);c.pyapi.decref(o)
    return NativeValue(p._getvalue())

@nb.njit(cache=True,inline="always")
def _GridSliceLen(start,stop,step):
    if step>0:
        return 0 if start>=stop else (stop-start+step-1)//step
    step=-step
    return 0 if start<=stop else (start-stop+step-1)//step

@overload(operator.getitem)
def _olGridSliceGet(g,key):
    if not isinstance(g,_GridNumbaType): return None
    npdt=np.dtype(str(g.dtype))
    if isinstance(key,types.SliceType):
        def impl(g,key):
            a,b,c=key.indices(len(g)); n=_GridSliceLen(a,b,c)
            out=np.empty(n,dtype=npdt)
            for oi in range(n): out[oi]=g[a+oi*c]
            return out
        return impl
    if isinstance(key,types.BaseTuple):
        kt=key.types
        if len(kt)==2:
            s0=isinstance(kt[0],types.SliceType); s1=isinstance(kt[1],types.SliceType)
            if s0 and s1:
                def impl(g,key):
                    a0,b0,c0=key[0].indices(g._xDim); a1,b1,c1=key[1].indices(g._yDim)
                    n0=_GridSliceLen(a0,b0,c0); n1=_GridSliceLen(a1,b1,c1); out=np.empty((n0,n1),dtype=npdt)
                    for o0 in range(n0):
                        for o1 in range(n1): out[o0,o1]=g[a0+o0*c0,a1+o1*c1]
                    return out
                return impl
            if s0 and isinstance(kt[1],types.Integer):
                def impl(g,key):
                    a,b,c=key[0].indices(g._xDim); n=_GridSliceLen(a,b,c); out=np.empty(n,dtype=npdt); y=key[1]
                    for oi in range(n): out[oi]=g[a+oi*c,y]
                    return out
                return impl
            if isinstance(kt[0],types.Integer) and s1:
                def impl(g,key):
                    a,b,c=key[1].indices(g._yDim); n=_GridSliceLen(a,b,c); out=np.empty(n,dtype=npdt); x=key[0]
                    for oi in range(n): out[oi]=g[x,a+oi*c]
                    return out
                return impl
        if len(kt)==3:
            s0=isinstance(kt[0],types.SliceType); s1=isinstance(kt[1],types.SliceType); s2=isinstance(kt[2],types.SliceType)
            if s0 and s1 and s2:
                def impl(g,key):
                    a0,b0,c0=key[0].indices(g._xDim); a1,b1,c1=key[1].indices(g._yDim); a2,b2,c2=key[2].indices(g._zDim)
                    n0=_GridSliceLen(a0,b0,c0); n1=_GridSliceLen(a1,b1,c1); n2=_GridSliceLen(a2,b2,c2); out=np.empty((n0,n1,n2),dtype=npdt)
                    for o0 in range(n0):
                        for o1 in range(n1):
                            for o2 in range(n2): out[o0,o1,o2]=g[a0+o0*c0,a1+o1*c1,a2+o2*c2]
                    return out
                return impl
            # Mixed 3D slices: build explicit 1D/2D cases.
            if s0 and s1 and isinstance(kt[2],types.Integer):
                def impl(g,key):
                    a0,b0,c0=key[0].indices(g._xDim); a1,b1,c1=key[1].indices(g._yDim); z=key[2]
                    n0=_GridSliceLen(a0,b0,c0); n1=_GridSliceLen(a1,b1,c1); out=np.empty((n0,n1),dtype=npdt)
                    for o0 in range(n0):
                        for o1 in range(n1): out[o0,o1]=g[a0+o0*c0,a1+o1*c1,z]
                    return out
                return impl
            if s0 and isinstance(kt[1],types.Integer) and s2:
                def impl(g,key):
                    a0,b0,c0=key[0].indices(g._xDim); y=key[1]; a2,b2,c2=key[2].indices(g._zDim)
                    n0=_GridSliceLen(a0,b0,c0); n2=_GridSliceLen(a2,b2,c2); out=np.empty((n0,n2),dtype=npdt)
                    for o0 in range(n0):
                        for o2 in range(n2): out[o0,o2]=g[a0+o0*c0,y,a2+o2*c2]
                    return out
                return impl
            if isinstance(kt[0],types.Integer) and s1 and s2:
                def impl(g,key):
                    x=key[0]; a1,b1,c1=key[1].indices(g._yDim); a2,b2,c2=key[2].indices(g._zDim)
                    n1=_GridSliceLen(a1,b1,c1); n2=_GridSliceLen(a2,b2,c2); out=np.empty((n1,n2),dtype=npdt)
                    for o1 in range(n1):
                        for o2 in range(n2): out[o1,o2]=g[x,a1+o1*c1,a2+o2*c2]
                    return out
                return impl
            if s0 and isinstance(kt[1],types.Integer) and isinstance(kt[2],types.Integer):
                def impl(g,key):
                    a,b,c=key[0].indices(g._xDim); n=_GridSliceLen(a,b,c); out=np.empty(n,dtype=npdt); y=key[1];z=key[2]
                    for oi in range(n): out[oi]=g[a+oi*c,y,z]
                    return out
                return impl
            if isinstance(kt[0],types.Integer) and s1 and isinstance(kt[2],types.Integer):
                def impl(g,key):
                    a,b,c=key[1].indices(g._yDim); n=_GridSliceLen(a,b,c); out=np.empty(n,dtype=npdt); x=key[0];z=key[2]
                    for oi in range(n): out[oi]=g[x,a+oi*c,z]
                    return out
                return impl
            if isinstance(kt[0],types.Integer) and isinstance(kt[1],types.Integer) and s2:
                def impl(g,key):
                    a,b,c=key[2].indices(g._zDim); n=_GridSliceLen(a,b,c); out=np.empty(n,dtype=npdt); x=key[0];y=key[1]
                    for oi in range(n): out[oi]=g[x,y,a+oi*c]
                    return out
                return impl

@overload(operator.setitem)
def _olGridSliceSet(g,key,value):
    if not isinstance(g,_GridNumbaType): return None
    scalar=isinstance(value,types.Number) or isinstance(value,types.Boolean)
    array=isinstance(value,types.Array)
    if isinstance(key,types.SliceType):
        if scalar:
            def impl(g,key,value):
                a,b,c=key.indices(len(g)); n=_GridSliceLen(a,b,c)
                for oi in range(n): g[a+oi*c]=value
            return impl
        if array:
            def impl(g,key,value):
                a,b,c=key.indices(len(g)); n=_GridSliceLen(a,b,c)
                if value.size!=n: raise ValueError("slice assignment size mismatch")
                for oi in range(n): g[a+oi*c]=value[oi]
            return impl
    if isinstance(key,types.BaseTuple):
        kt=key.types
        if len(kt)==2 and isinstance(kt[0],types.SliceType) and isinstance(kt[1],types.Integer):
            if scalar:
                def impl(g,key,value):
                    a,b,c=key[0].indices(g._xDim);n=_GridSliceLen(a,b,c);y=key[1]
                    for oi in range(n):g[a+oi*c,y]=value
                return impl
            if array:
                def impl(g,key,value):
                    a,b,c=key[0].indices(g._xDim);n=_GridSliceLen(a,b,c);y=key[1]
                    if value.size!=n:raise ValueError("slice assignment size mismatch")
                    for oi in range(n):g[a+oi*c,y]=value[oi]
                return impl
        if len(kt)==2 and isinstance(kt[0],types.Integer) and isinstance(kt[1],types.SliceType):
            if scalar:
                def impl(g,key,value):
                    x=key[0];a,b,c=key[1].indices(g._yDim);n=_GridSliceLen(a,b,c)
                    for oi in range(n):g[x,a+oi*c]=value
                return impl
            if array:
                def impl(g,key,value):
                    x=key[0];a,b,c=key[1].indices(g._yDim);n=_GridSliceLen(a,b,c)
                    if value.size!=n:raise ValueError("slice assignment size mismatch")
                    for oi in range(n):g[x,a+oi*c]=value[oi]
                return impl
        if len(kt)==2 and all(isinstance(t,types.SliceType) for t in kt):
            if scalar:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);a1,b1,c1=key[1].indices(g._yDim);n0=_GridSliceLen(a0,b0,c0);n1=_GridSliceLen(a1,b1,c1)
                    for o0 in range(n0):
                        for o1 in range(n1):g[a0+o0*c0,a1+o1*c1]=value
                return impl
            if array:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);a1,b1,c1=key[1].indices(g._yDim);n0=_GridSliceLen(a0,b0,c0);n1=_GridSliceLen(a1,b1,c1)
                    if value.shape!=(n0,n1):raise ValueError("slice assignment shape mismatch")
                    for o0 in range(n0):
                        for o1 in range(n1):g[a0+o0*c0,a1+o1*c1]=value[o0,o1]
                return impl
        if len(kt)==3 and isinstance(kt[0],types.SliceType) and isinstance(kt[1],types.SliceType) and isinstance(kt[2],types.Integer):
            if scalar:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);a1,b1,c1=key[1].indices(g._yDim);z=key[2];n0=_GridSliceLen(a0,b0,c0);n1=_GridSliceLen(a1,b1,c1)
                    for o0 in range(n0):
                        for o1 in range(n1):g[a0+o0*c0,a1+o1*c1,z]=value
                return impl
            if array:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);a1,b1,c1=key[1].indices(g._yDim);z=key[2];n0=_GridSliceLen(a0,b0,c0);n1=_GridSliceLen(a1,b1,c1)
                    if value.shape!=(n0,n1):raise ValueError("slice assignment shape mismatch")
                    for o0 in range(n0):
                        for o1 in range(n1):g[a0+o0*c0,a1+o1*c1,z]=value[o0,o1]
                return impl
        if len(kt)==3 and isinstance(kt[0],types.SliceType) and isinstance(kt[1],types.Integer) and isinstance(kt[2],types.SliceType):
            if scalar:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);y=key[1];a2,b2,c2=key[2].indices(g._zDim);n0=_GridSliceLen(a0,b0,c0);n2=_GridSliceLen(a2,b2,c2)
                    for o0 in range(n0):
                        for o2 in range(n2):g[a0+o0*c0,y,a2+o2*c2]=value
                return impl
            if array:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);y=key[1];a2,b2,c2=key[2].indices(g._zDim);n0=_GridSliceLen(a0,b0,c0);n2=_GridSliceLen(a2,b2,c2)
                    if value.shape!=(n0,n2):raise ValueError("slice assignment shape mismatch")
                    for o0 in range(n0):
                        for o2 in range(n2):g[a0+o0*c0,y,a2+o2*c2]=value[o0,o2]
                return impl
        if len(kt)==3 and isinstance(kt[0],types.Integer) and isinstance(kt[1],types.SliceType) and isinstance(kt[2],types.SliceType):
            if scalar:
                def impl(g,key,value):
                    x=key[0];a1,b1,c1=key[1].indices(g._yDim);a2,b2,c2=key[2].indices(g._zDim);n1=_GridSliceLen(a1,b1,c1);n2=_GridSliceLen(a2,b2,c2)
                    for o1 in range(n1):
                        for o2 in range(n2):g[x,a1+o1*c1,a2+o2*c2]=value
                return impl
            if array:
                def impl(g,key,value):
                    x=key[0];a1,b1,c1=key[1].indices(g._yDim);a2,b2,c2=key[2].indices(g._zDim);n1=_GridSliceLen(a1,b1,c1);n2=_GridSliceLen(a2,b2,c2)
                    if value.shape!=(n1,n2):raise ValueError("slice assignment shape mismatch")
                    for o1 in range(n1):
                        for o2 in range(n2):g[x,a1+o1*c1,a2+o2*c2]=value[o1,o2]
                return impl
        if len(kt)==3 and isinstance(kt[0],types.SliceType) and isinstance(kt[1],types.Integer) and isinstance(kt[2],types.Integer):
            if scalar:
                def impl(g,key,value):
                    a,b,c=key[0].indices(g._xDim);n=_GridSliceLen(a,b,c);y=key[1];z=key[2]
                    for oi in range(n):g[a+oi*c,y,z]=value
                return impl
            if array:
                def impl(g,key,value):
                    a,b,c=key[0].indices(g._xDim);n=_GridSliceLen(a,b,c);y=key[1];z=key[2]
                    if value.size!=n:raise ValueError("slice assignment size mismatch")
                    for oi in range(n):g[a+oi*c,y,z]=value[oi]
                return impl
        if len(kt)==3 and isinstance(kt[0],types.Integer) and isinstance(kt[1],types.SliceType) and isinstance(kt[2],types.Integer):
            if scalar:
                def impl(g,key,value):
                    x=key[0];a,b,c=key[1].indices(g._yDim);n=_GridSliceLen(a,b,c);z=key[2]
                    for oi in range(n):g[x,a+oi*c,z]=value
                return impl
            if array:
                def impl(g,key,value):
                    x=key[0];a,b,c=key[1].indices(g._yDim);n=_GridSliceLen(a,b,c);z=key[2]
                    if value.size!=n:raise ValueError("slice assignment size mismatch")
                    for oi in range(n):g[x,a+oi*c,z]=value[oi]
                return impl
        if len(kt)==3 and isinstance(kt[0],types.Integer) and isinstance(kt[1],types.Integer) and isinstance(kt[2],types.SliceType):
            if scalar:
                def impl(g,key,value):
                    x=key[0];y=key[1];a,b,c=key[2].indices(g._zDim);n=_GridSliceLen(a,b,c)
                    for oi in range(n):g[x,y,a+oi*c]=value
                return impl
            if array:
                def impl(g,key,value):
                    x=key[0];y=key[1];a,b,c=key[2].indices(g._zDim);n=_GridSliceLen(a,b,c)
                    if value.size!=n:raise ValueError("slice assignment size mismatch")
                    for oi in range(n):g[x,y,a+oi*c]=value[oi]
                return impl
        if len(kt)==3 and all(isinstance(t,types.SliceType) for t in kt):
            if scalar:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);a1,b1,c1=key[1].indices(g._yDim);a2,b2,c2=key[2].indices(g._zDim);n0=_GridSliceLen(a0,b0,c0);n1=_GridSliceLen(a1,b1,c1);n2=_GridSliceLen(a2,b2,c2)
                    for o0 in range(n0):
                        for o1 in range(n1):
                            for o2 in range(n2):g[a0+o0*c0,a1+o1*c1,a2+o2*c2]=value
                return impl
            if array:
                def impl(g,key,value):
                    a0,b0,c0=key[0].indices(g._xDim);a1,b1,c1=key[1].indices(g._yDim);a2,b2,c2=key[2].indices(g._zDim);n0=_GridSliceLen(a0,b0,c0);n1=_GridSliceLen(a1,b1,c1);n2=_GridSliceLen(a2,b2,c2)
                    if value.shape!=(n0,n1,n2):raise ValueError("slice assignment shape mismatch")
                    for o0 in range(n0):
                        for o1 in range(n1):
                            for o2 in range(n2):g[a0+o0*c0,a1+o1*c1,a2+o2*c2]=value[o0,o1,o2]
                return impl

@type_callable(operator.getitem)
def _typeGridGet(context):
    def typer(g,key):
        if not isinstance(g,_GridNumbaType):return None
        if isinstance(key,types.Integer):return g.dtype
        if isinstance(key,types.BaseTuple) and len(key.types) in (2,3) and all(isinstance(t,types.Integer) for t in key.types):return g.dtype
    return typer

def _GridNativeIndex(context,builder,gtyp,gval,keytyp,keyval):
    p=cgutils.create_struct_proxy(gtyp)(context,builder,value=gval)
    # Fast Grid indexing intentionally lowers to a raw load/store. Safe Grid
    # indexing must prove the complete address is in-bounds before the GEP;
    # otherwise one bad index becomes arbitrary native memory access.
    if isinstance(keytyp,types.Integer):
        if gtyp.safe:
            i64=context.cast(builder,keyval,keytyp,types.int64)
            n=context.get_constant(types.int64,int(np.prod(gtyp.dims,dtype=np.int64)))
            bad=builder.or_(builder.icmp_signed('<',i64,context.get_constant(types.int64,0)),builder.icmp_signed('>=',i64,n))
            with builder.if_then(bad,likely=False): context.call_conv.return_user_exc(builder,IndexError,("Grid index out of bounds",))
            return p,context.cast(builder,i64,types.int64,keytyp)
        return p,keyval
    vals=cgutils.unpack_tuple(builder,keyval,len(keytyp.types))
    if gtyp.safe:
        expected=len(keytyp.types)
        if expected!=gtyp.dimension:
            context.call_conv.return_user_exc(builder,IndexError,("Grid index dimensionality mismatch",))
        checked=[]
        for j,v in enumerate(vals):
            vt=keytyp.types[j]; vi=context.cast(builder,v,vt,types.int64); dj=context.get_constant(types.int64,gtyp.dims[j])
            bad=builder.or_(builder.icmp_signed('<',vi,context.get_constant(types.int64,0)),builder.icmp_signed('>=',vi,dj))
            with builder.if_then(bad,likely=False): context.call_conv.return_user_exc(builder,IndexError,("Grid coordinate out of bounds",))
            checked.append(context.cast(builder,vi,types.int64,vt))
        vals=checked
    ity=keytyp.types[0]; ydim=context.get_constant(ity,gtyp.dims[1] if gtyp.dimension>1 else 1); zdim=context.get_constant(ity,gtyp.dims[2] if gtyp.dimension>2 else 1)
    if len(vals)==2:return p,builder.add(builder.mul(vals[0],ydim),vals[1])
    return p,builder.add(builder.mul(builder.add(builder.mul(vals[0],ydim),vals[1]),zdim),vals[2])

def _GridNativeIndexAt(context,builder,gtyp,gval,keytyp,keyval,sourceLine):
    p=cgutils.create_struct_proxy(gtyp)(context,builder,value=gval)
    suffix=sourceLine
    if isinstance(keytyp,types.Integer):
        i64=context.cast(builder,keyval,keytyp,types.int64)
        if gtyp.safe:
            n=context.get_constant(types.int64,int(np.prod(gtyp.dims,dtype=np.int64)))
            bad=builder.or_(builder.icmp_signed('<',i64,context.get_constant(types.int64,0)),builder.icmp_signed('>=',i64,n))
            with builder.if_then(bad,likely=False): context.call_conv.return_user_exc(builder,IndexError,("Grid index out of bounds"+suffix,))
        return p,context.cast(builder,i64,types.int64,keytyp)
    vals=cgutils.unpack_tuple(builder,keyval,len(keytyp.types))
    if gtyp.safe:
        expected=len(keytyp.types)
        if expected!=gtyp.dimension: context.call_conv.return_user_exc(builder,IndexError,("Grid index dimensionality mismatch"+suffix,))
        checked=[]
        for j,v in enumerate(vals):
            vt=keytyp.types[j];vi=context.cast(builder,v,vt,types.int64);dj=context.get_constant(types.int64,gtyp.dims[j])
            bad=builder.or_(builder.icmp_signed('<',vi,context.get_constant(types.int64,0)),builder.icmp_signed('>=',vi,dj))
            with builder.if_then(bad,likely=False): context.call_conv.return_user_exc(builder,IndexError,("Grid coordinate out of bounds"+suffix,))
            checked.append(context.cast(builder,vi,types.int64,vt))
        vals=checked
    ity=keytyp.types[0];ydim=context.get_constant(ity,gtyp.dims[1] if gtyp.dimension>1 else 1);zdim=context.get_constant(ity,gtyp.dims[2] if gtyp.dimension>2 else 1)
    if len(vals)==2:return p,builder.add(builder.mul(vals[0],ydim),vals[1])
    return p,builder.add(builder.mul(builder.add(builder.mul(vals[0],ydim),vals[1]),zdim),vals[2])

@intrinsic
def _GridGetUnchecked(typingctx,g,i):
    if not isinstance(g,_GridNumbaType) or not isinstance(i,types.Integer): return None
    sig=g.dtype(g,i)
    def codegen(context,builder,signature,args):
        p=cgutils.create_struct_proxy(signature.args[0])(context,builder,value=args[0]); ii=context.cast(builder,args[1],signature.args[1],types.int64)
        pty=context.get_value_type(signature.return_type).as_pointer();base=builder.inttoptr(p.address,pty);return builder.load(builder.gep(base,[ii]))
    return sig,codegen

@intrinsic
def _GridSetUnchecked(typingctx,g,i,value):
    if not isinstance(g,_GridNumbaType) or not isinstance(i,types.Integer): return None
    sig=types.void(g,i,value)
    def codegen(context,builder,signature,args):
        gtyp,ityp,vtyp=signature.args;p=cgutils.create_struct_proxy(gtyp)(context,builder,value=args[0]);ii=context.cast(builder,args[1],ityp,types.int64)
        pty=context.get_value_type(gtyp.dtype).as_pointer();base=builder.inttoptr(p.address,pty);val=context.cast(builder,args[2],vtyp,gtyp.dtype);builder.store(val,builder.gep(base,[ii]));return context.get_dummy_value()
    return sig,codegen

def _GridGetAt(g,key,sourceLine): return g[key]
@overload(_GridGetAt,inline="never",prefer_literal=True)
def _olGridGetAt(g,key,sourceLine):
    if not isinstance(g,_GridNumbaType): return None
    suffix=sourceLine.literal_value if isinstance(sourceLine,types.StringLiteral) else ""
    badI="Grid index out of bounds"+suffix; badD="Grid index dimensionality mismatch"+suffix; badC="Grid coordinate out of bounds"+suffix
    if isinstance(key,types.Integer):
        n=int(np.prod(g.dims,dtype=np.int64))
        if g.safe:
            def impl(g,key,sourceLine):
                if key<0 or key>=n: raise IndexError(badI)
                return _GridGetUnchecked(g,key)
        else:
            def impl(g,key,sourceLine): return _GridGetUnchecked(g,key)
        return impl
    if isinstance(key,types.BaseTuple) and all(isinstance(t,types.Integer) for t in key.types):
        if len(key.types)!=g.dimension:
            def impl(g,key,sourceLine): raise IndexError(badD)
            return impl
        dims=g.dims
        if len(key.types)==1:
            if g.safe:
                def impl(g,key,sourceLine):
                    x=key[0]
                    if x<0 or x>=dims[0]: raise IndexError(badC)
                    return _GridGetUnchecked(g,x)
            else:
                def impl(g,key,sourceLine): return _GridGetUnchecked(g,key[0])
        elif len(key.types)==2:
            if g.safe:
                def impl(g,key,sourceLine):
                    x,y=key
                    if x<0 or x>=dims[0] or y<0 or y>=dims[1]: raise IndexError(badC)
                    return _GridGetUnchecked(g,x*dims[1]+y)
            else:
                def impl(g,key,sourceLine): return _GridGetUnchecked(g,key[0]*dims[1]+key[1])
        else:
            if g.safe:
                def impl(g,key,sourceLine):
                    x,y,z=key
                    if x<0 or x>=dims[0] or y<0 or y>=dims[1] or z<0 or z>=dims[2]: raise IndexError(badC)
                    return _GridGetUnchecked(g,(x*dims[1]+y)*dims[2]+z)
            else:
                def impl(g,key,sourceLine): return _GridGetUnchecked(g,(key[0]*dims[1]+key[1])*dims[2]+key[2])
        return impl

def _GridSetAt(g,key,value,sourceLine): g[key]=value
@overload(_GridSetAt,inline="never",prefer_literal=True)
def _olGridSetAt(g,key,value,sourceLine):
    if not isinstance(g,_GridNumbaType): return None
    suffix=sourceLine.literal_value if isinstance(sourceLine,types.StringLiteral) else ""
    badI="Grid index out of bounds"+suffix; badD="Grid index dimensionality mismatch"+suffix; badC="Grid coordinate out of bounds"+suffix
    if isinstance(key,types.Integer):
        n=int(np.prod(g.dims,dtype=np.int64))
        if g.safe:
            def impl(g,key,value,sourceLine):
                if key<0 or key>=n: raise IndexError(badI)
                _GridSetUnchecked(g,key,value)
        else:
            def impl(g,key,value,sourceLine): _GridSetUnchecked(g,key,value)
        return impl
    if isinstance(key,types.BaseTuple) and all(isinstance(t,types.Integer) for t in key.types):
        if len(key.types)!=g.dimension:
            def impl(g,key,value,sourceLine): raise IndexError(badD)
            return impl
        dims=g.dims
        if len(key.types)==1:
            if g.safe:
                def impl(g,key,value,sourceLine):
                    x=key[0]
                    if x<0 or x>=dims[0]: raise IndexError(badC)
                    _GridSetUnchecked(g,x,value)
            else:
                def impl(g,key,value,sourceLine): _GridSetUnchecked(g,key[0],value)
        elif len(key.types)==2:
            if g.safe:
                def impl(g,key,value,sourceLine):
                    x,y=key
                    if x<0 or x>=dims[0] or y<0 or y>=dims[1]: raise IndexError(badC)
                    _GridSetUnchecked(g,x*dims[1]+y,value)
            else:
                def impl(g,key,value,sourceLine): _GridSetUnchecked(g,key[0]*dims[1]+key[1],value)
        else:
            if g.safe:
                def impl(g,key,value,sourceLine):
                    x,y,z=key
                    if x<0 or x>=dims[0] or y<0 or y>=dims[1] or z<0 or z>=dims[2]: raise IndexError(badC)
                    _GridSetUnchecked(g,(x*dims[1]+y)*dims[2]+z,value)
            else:
                def impl(g,key,value,sourceLine): _GridSetUnchecked(g,(key[0]*dims[1]+key[1])*dims[2]+key[2],value)
        return impl

@lower_builtin(operator.getitem,_GridNumbaType,types.Integer)
def _lowerGridGetI(context,builder,sig,args):
    p,i=_GridNativeIndex(context,builder,sig.args[0],args[0],sig.args[1],args[1]);pty=context.get_value_type(sig.return_type).as_pointer();base=builder.inttoptr(p.address,pty);return builder.load(builder.gep(base,[i]))
@lower_builtin(operator.getitem,_GridNumbaType,types.BaseTuple)
def _lowerGridGetT(context,builder,sig,args):
    p,i=_GridNativeIndex(context,builder,sig.args[0],args[0],sig.args[1],args[1]);pty=context.get_value_type(sig.return_type).as_pointer();base=builder.inttoptr(p.address,pty);return builder.load(builder.gep(base,[i]))

@type_callable(operator.setitem)
def _typeGridSet(context):
    def typer(g,key,value):
        if isinstance(g,_GridNumbaType) and (isinstance(key,types.Integer) or isinstance(key,types.BaseTuple) and len(key.types) in (2,3) and all(isinstance(t,types.Integer) for t in key.types)):return types.none
    return typer

def _lowerGridSet(context,builder,sig,args):
    gtyp,keytyp,vtyp=sig.args;p,i=_GridNativeIndex(context,builder,gtyp,args[0],keytyp,args[1]);pty=context.get_value_type(gtyp.dtype).as_pointer();base=builder.inttoptr(p.address,pty);val=context.cast(builder,args[2],vtyp,gtyp.dtype);builder.store(val,builder.gep(base,[i]));return context.get_dummy_value()
lower_builtin(operator.setitem,_GridNumbaType,types.Integer,types.Any)(_lowerGridSet)
lower_builtin(operator.setitem,_GridNumbaType,types.BaseTuple,types.Any)(_lowerGridSet)

@overload_method(_GridNumbaType,"Add",inline="always")
def _olGridAdd(g,value,x,y=-1,z=-1):
    if g.dtype==types.boolean:
        def impl(g,value,x,y=-1,z=-1):
            if y==-1 and z==-1:g[x]=g[x] or bool(value)
            elif g._dimension==2:g[x,y]=g[x,y] or bool(value)
            else:g[x,y,z]=g[x,y,z] or bool(value)
        return impl
    def impl(g,value,x,y=-1,z=-1):
        if y==-1 and z==-1:
            g[x]=g[x]+value
        elif g._dimension==2:
            g[x,y]=g[x,y]+value
        else:
            g[x,y,z]=g[x,y,z]+value
    return impl
def _GridToIAt(g,x,y,z,sourceLine): return g.ToI(x,y,z)
@overload(_GridToIAt,inline="never",prefer_literal=True)
def _olGridToIAt(g,x,y,z,sourceLine):
    if not isinstance(g,_GridNumbaType): return None
    dim=g.dimension;dims=g.dims;suffix=sourceLine.literal_value if isinstance(sourceLine,types.StringLiteral) else ""
    badCoord="coordinates must be finite integers"+suffix;badBounds="coordinate out of bounds"+suffix;bad1="1D Grid requires x only"+suffix;bad2="2D Grid requires x,y"+suffix
    if g.safe:
        def impl(g,x,y,z,sourceLine):
            if not np.isfinite(x) or x!=int(x): raise ValueError(badCoord)
            if x<0 or x>=dims[0]: raise IndexError(badBounds)
            if dim==1:
                if y!=-1 or z!=-1: raise ValueError(bad1)
                return x
            if not np.isfinite(y) or y!=int(y): raise ValueError(badCoord)
            if y<0 or y>=dims[1]: raise IndexError(badBounds)
            if dim==2:
                if z!=-1: raise ValueError(bad2)
                return x*dims[1]+y
            if not np.isfinite(z) or z!=int(z): raise ValueError(badCoord)
            if z<0 or z>=dims[2]: raise IndexError(badBounds)
            return (x*dims[1]+y)*dims[2]+z
    elif dim==1:
        def impl(g,x,y,z,sourceLine): return x
    elif dim==2:
        def impl(g,x,y,z,sourceLine): return x*dims[1]+y
    else:
        def impl(g,x,y,z,sourceLine): return (x*dims[1]+y)*dims[2]+z
    return impl

def _GridItoAt(g,i,axis,sourceLine): return i
@overload(_GridItoAt,inline="never",prefer_literal=True)
def _olGridItoAt(g,i,axis,sourceLine):
    if not isinstance(g,_GridNumbaType): return None
    ax=axis.literal_value if isinstance(axis,types.IntegerLiteral) else -1;suffix=sourceLine.literal_value if isinstance(sourceLine,types.StringLiteral) else ""
    badI="index must be a finite integer"+suffix;badB="index out of bounds"+suffix;badY="ItoY requires a 2D or 3D grid"+suffix;badZ="ItoZ requires a 3D grid"+suffix
    if g.safe:
        def impl(g,i,axis,sourceLine):
            if ax==1 and g._dimension<2: raise ValueError(badY)
            if ax==2 and g._dimension<3: raise ValueError(badZ)
            if not np.isfinite(i) or i!=int(i): raise ValueError(badI)
            if i<0 or i>=len(g): raise IndexError(badB)
            if ax==0:return i if g._dimension==1 else i//g._yDim if g._dimension==2 else i//(g._yDim*g._zDim)
            if ax==1:return i%g._yDim if g._dimension==2 else (i//g._zDim)%g._yDim
            return i%g._zDim
    else:
        def impl(g,i,axis,sourceLine):
            if ax==0:return i if g._dimension==1 else i//g._yDim if g._dimension==2 else i//(g._yDim*g._zDim)
            if ax==1:return i%g._yDim if g._dimension==2 else (i//g._zDim)%g._yDim
            return i%g._zDim
    return impl

def _GridWrapAt(g,v,axis,sourceLine): return v
@overload(_GridWrapAt,inline="never",prefer_literal=True)
def _olGridWrapAt(g,v,axis,sourceLine):
    if not isinstance(g,_GridNumbaType): return None
    ax=axis.literal_value if isinstance(axis,types.IntegerLiteral) else -1;dim=g.dimension;n=g.dims[ax] if 0<=ax<dim else 1;wrap=g.wrap[ax] if 0<=ax<dim else False;suffix=sourceLine.literal_value if isinstance(sourceLine,types.StringLiteral) else ""
    bad="coordinate must be an int32 integer"+suffix;badY="InWrapY requires a 2D or 3D Grid"+suffix;badZ="InWrapZ requires a 3D Grid"+suffix
    if g.safe:
        def impl(g,v,axis,sourceLine):
            if ax==1 and dim<2: raise ValueError(badY)
            if ax==2 and dim<3: raise ValueError(badZ)
            if not np.isfinite(v) or v!=int(v) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max: raise ValueError(bad)
            return v if 0<=v<n else v%n if wrap else -1
    else:
        def impl(g,v,axis,sourceLine): return v if 0<=v<n else v%n if wrap else -1
    return impl

@overload_method(_GridNumbaType,"ToI",inline="always")
def _olGridToI(g,x,y=-1,z=-1):
    dim=g.dimension;dims=g.dims
    if g.safe:
        def impl(g,x,y=-1,z=-1):
            if not np.isfinite(x) or x!=int(x): raise ValueError("coordinates must be finite integers")
            if x<0 or x>=dims[0]: raise IndexError("coordinate out of bounds")
            if dim==1:
                if y!=-1 or z!=-1: raise ValueError("1D Grid requires x only")
                return x
            if not np.isfinite(y) or y!=int(y): raise ValueError("coordinates must be finite integers")
            if y<0 or y>=dims[1]: raise IndexError("coordinate out of bounds")
            if dim==2:
                if z!=-1: raise ValueError("2D Grid requires x,y")
                return x*dims[1]+y
            if not np.isfinite(z) or z!=int(z): raise ValueError("coordinates must be finite integers")
            if z<0 or z>=dims[2]: raise IndexError("coordinate out of bounds")
            return (x*dims[1]+y)*dims[2]+z
    else:
        if dim==1:
            def impl(g,x,y=-1,z=-1): return x
        elif dim==2:
            def impl(g,x,y=-1,z=-1): return x*dims[1]+y
        else:
            def impl(g,x,y=-1,z=-1): return (x*dims[1]+y)*dims[2]+z
    return impl
@overload_method(_GridNumbaType,"ItoX",inline="always")
def _olGridItoX(g,i):
    if g.safe:
        def impl(g,i):
            if not np.isfinite(i) or i!=int(i): raise ValueError("index must be a finite integer")
            if i<0 or i>=len(g): raise IndexError("index out of bounds")
            return i if g._dimension==1 else i//g._yDim if g._dimension==2 else i//(g._yDim*g._zDim)
    else:
        def impl(g,i): return i if g._dimension==1 else i//g._yDim if g._dimension==2 else i//(g._yDim*g._zDim)
    return impl
@overload_method(_GridNumbaType,"ItoY",inline="always")
def _olGridItoY(g,i):
    if g.safe:
        def impl(g,i):
            if g._dimension<2: raise ValueError("ItoY requires a 2D or 3D grid")
            if not np.isfinite(i) or i!=int(i): raise ValueError("index must be a finite integer")
            if i<0 or i>=len(g): raise IndexError("index out of bounds")
            return i%g._yDim if g._dimension==2 else (i//g._zDim)%g._yDim
    else:
        def impl(g,i): return i%g._yDim if g._dimension==2 else (i//g._zDim)%g._yDim
    return impl
@overload_method(_GridNumbaType,"ItoZ",inline="always")
def _olGridItoZ(g,i):
    if g.safe:
        def impl(g,i):
            if g._dimension<3: raise ValueError("ItoZ requires a 3D grid")
            if not np.isfinite(i) or i!=int(i): raise ValueError("index must be a finite integer")
            if i<0 or i>=len(g): raise IndexError("index out of bounds")
            return i%g._zDim
    else:
        def impl(g,i): return i%g._zDim
    return impl
@overload_method(_GridNumbaType,"InWrapX",inline="always")
def _olGridWX(g,v):
    n=g.dims[0];wrap=g.wrap[0]
    if g.safe:
        def impl(g,v):
            if not np.isfinite(v) or v!=int(v) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max: raise ValueError("coordinate must be an int32 integer")
            return v if 0<=v<n else v%n if wrap else -1
    else:
        def impl(g,v):return v if 0<=v<n else v%n if wrap else -1
    return impl
@overload_method(_GridNumbaType,"InWrapY",inline="always")
def _olGridWY(g,v):
    dim=g.dimension;n=g.dims[1] if dim>1 else 1;wrap=g.wrap[1] if dim>1 else False
    if g.safe:
        def impl(g,v):
            if dim<2: raise ValueError("InWrapY requires a 2D or 3D Grid")
            if not np.isfinite(v) or v!=int(v) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max: raise ValueError("coordinate must be an int32 integer")
            return v if 0<=v<n else v%n if wrap else -1
    else:
        def impl(g,v):return v if 0<=v<n else v%n if wrap else -1
    return impl
@overload_method(_GridNumbaType,"InWrapZ",inline="always")
def _olGridWZ(g,v):
    dim=g.dimension;n=g.dims[2] if dim>2 else 1;wrap=g.wrap[2] if dim>2 else False
    if g.safe:
        def impl(g,v):
            if dim<3: raise ValueError("InWrapZ requires a 3D Grid")
            if not np.isfinite(v) or v!=int(v) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max: raise ValueError("coordinate must be an int32 integer")
            return v if 0<=v<n else v%n if wrap else -1
    else:
        def impl(g,v):return v if 0<=v<n else v%n if wrap else -1
    return impl
@overload_method(_GridNumbaType,"DispWrapX",inline="always")
def _olGridDX(g,a,b):
    def impl(g,a,b):
        d=b-a
        if g._wrapX:
            h=g._xDim/2
            if d>h:d-=g._xDim
            elif d<-h:d+=g._xDim
        return d
    return impl
@overload_method(_GridNumbaType,"DispWrapY",inline="always")
def _olGridDY(g,a,b):
    def impl(g,a,b):
        d=b-a
        if g._wrapY:
            h=g._yDim/2
            if d>h:d-=g._yDim
            elif d<-h:d+=g._yDim
        return d
    return impl
@overload_method(_GridNumbaType,"DispWrapZ",inline="always")
def _olGridDZ(g,a,b):
    def impl(g,a,b):
        d=b-a
        if g._wrapZ:
            h=g._zDim/2
            if d>h:d-=g._zDim
            elif d<-h:d+=g._zDim
        return d
    return impl
@overload(len,inline="always")
def _olGridLen(g):
    if isinstance(g,_GridNumbaType):
        def impl(g):
            if g._dimension==1:return g._xDim
            if g._dimension==2:return g._xDim*g._yDim
            return g._xDim*g._yDim*g._zDim
        return impl


def _AgentIterGeneration(grid):
    return 0

@overload(_AgentIterGeneration,inline="always")
def _olAgentIterGeneration(grid):
    safeType=_AgentGridSafe.class_type.instance_type
    if grid==safeType:
        def impl(grid): return _ag_generation(grid._ptr)
    else:
        def impl(grid): return 0
    return impl

def _AgentIterCheck(grid,generation):
    return None

@overload(_AgentIterCheck,inline="always")
def _olAgentIterCheck(grid,generation):
    safeType=_AgentGridSafe.class_type.instance_type
    if grid==safeType:
        def impl(grid,generation):
            if _ag_generation(grid._ptr)!=generation:
                raise RuntimeError("AgentGrid cannot be structurally modified during direct agent iteration")
    else:
        def impl(grid,generation): return
    return impl


def _AgentIterCheckAt(grid,generation,sourceLine): return None
@overload(_AgentIterCheckAt,inline="always",prefer_literal=True)
def _olAgentIterCheckAt(grid,generation,sourceLine):
    if grid==_AgentGridSafe.class_type.instance_type:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        message="AgentGrid cannot be structurally modified during direct agent iteration"+(" at source line "+str(line) if line>=0 else "")
        def impl(grid,generation,sourceLine):
            if _ag_generation(grid._ptr)!=generation: raise RuntimeError(message)
    else:
        def impl(grid,generation,sourceLine): return
    return impl

def _AgentRadiusWrap(value,size,wrap):
    if 0<=value<size: return value
    if not wrap: return -1
    value%=size
    if value<0: value+=size
    return value

@nb.njit(cache=True,inline="always")
def _AgentRadiusWrapJit(value,size,wrap):
    if 0<=value<size: return value
    if not wrap: return -1
    value%=size
    if value<0: value+=size
    return value

def _AgentIterValidateRadius(grid,rad,x,y,z,dim):
    return None

@overload(_AgentIterValidateRadius,inline="always")
def _olAgentIterValidateRadius(grid,rad,x,y,z,dim):
    safeType=_AgentGridSafe.class_type.instance_type
    if grid==safeType:
        def impl(grid,rad,x,y,z,dim):
            if not np.isfinite(rad) or rad<0 or not np.isfinite(x) or dim>1 and not np.isfinite(y) or dim>2 and not np.isfinite(z):
                raise ValueError("radius and coordinates must be finite and radius nonnegative")
    else:
        def impl(grid,rad,x,y,z,dim): return
    return impl


def _AgentIterValidateRadiusAt(grid,rad,x,y,z,dim,sourceLine): return None
@overload(_AgentIterValidateRadiusAt,inline="always",prefer_literal=True)
def _olAgentIterValidateRadiusAt(grid,rad,x,y,z,dim,sourceLine):
    if grid==_AgentGridSafe.class_type.instance_type:
        line=sourceLine.literal_value if isinstance(sourceLine,types.IntegerLiteral) else -1
        message="radius and coordinates must be finite and radius nonnegative"+(" at source line "+str(line) if line>=0 else "")
        def impl(grid,rad,x,y,z,dim,sourceLine):
            if not np.isfinite(rad) or rad<0 or not np.isfinite(x) or dim>1 and not np.isfinite(y) or dim>2 and not np.isfinite(z): raise ValueError(message)
    else:
        def impl(grid,rad,x,y,z,dim,sourceLine): return
    return impl

def _BoxWrapX(grid,value): return grid.InWrapSQX(value) if hasattr(grid,"InWrapSQX") else grid.InWrapX(value)
def _BoxWrapY(grid,value): return grid.InWrapSQY(value) if hasattr(grid,"InWrapSQY") else grid.InWrapY(value)
def _BoxWrapZ(grid,value): return grid.InWrapSQZ(value) if hasattr(grid,"InWrapSQZ") else grid.InWrapZ(value)

def _BoxWrapOverload(grid,axis):
    agentSafe=_AgentGridSafe.class_type.instance_type
    agentFast=_AgentGridFast.class_type.instance_type
    isAgent=grid==agentSafe or grid==agentFast
    if axis==0:
        if isAgent:
            def impl(grid,value): return grid.InWrapSQX(value)
        else:
            def impl(grid,value): return grid.InWrapX(value)
    elif axis==1:
        if isAgent:
            def impl(grid,value): return grid.InWrapSQY(value)
        else:
            def impl(grid,value): return grid.InWrapY(value)
    else:
        if isAgent:
            def impl(grid,value): return grid.InWrapSQZ(value)
        else:
            def impl(grid,value): return grid.InWrapZ(value)
    return impl

@overload(_BoxWrapX,inline="always")
def _olBoxWrapX(grid,value): return _BoxWrapOverload(grid,0)
@overload(_BoxWrapY,inline="always")
def _olBoxWrapY(grid,value): return _BoxWrapOverload(grid,1)
@overload(_BoxWrapZ,inline="always")
def _olBoxWrapZ(grid,value): return _BoxWrapOverload(grid,2)

class _HoodTrustedBodyOptimizer(ast.NodeTransformer):
    """Lower operations on PAL-generated Hood coordinates to trusted primitives."""
    def __init__(self,grid,targets,dim):
        self.grid=grid
        self.gridName=grid.id if isinstance(grid,ast.Name) else None
        self.targetNames=[t.id if isinstance(t,ast.Name) else None for t in targets]
        self.dim=dim

    def _same_grid(self,node):
        return self.gridName is not None and isinstance(node,ast.Name) and node.id==self.gridName

    def _coords(self,args):
        if len(args)<self.dim: return False
        for i in range(self.dim):
            if self.targetNames[i] is None or not isinstance(args[i],ast.Name) or args[i].id!=self.targetNames[i]: return False
        return True

    def _toi(self,args):
        vals=[copy.deepcopy(a) for a in args[:self.dim]]+[ast.Constant(-1)]*(3-self.dim)
        return ast.Call(ast.Name('_ag_toi',ast.Load()),[ast.Attribute(copy.deepcopy(self.grid),'_ptr',ast.Load()),*vals],[])

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if not isinstance(node.func,ast.Name): return node
        name=node.func.id
        # grid.counts[x,y,...] after the ordinary source-aware lowering.
        if name=='_AgentCountsItemAt' and len(node.args)>=3 and self._same_grid(node.args[0]):
            key=node.args[1]
            if isinstance(key,ast.Tuple) and self._coords(key.elts):
                return ast.copy_location(ast.Call(ast.Name('_ag_count_i',ast.Load()),[ast.Attribute(copy.deepcopy(self.grid),'_ptr',ast.Load()),self._toi(key.elts)],[]),node)
        # grid.ToI(x,y,...) after general AgentGrid call lowering.
        if name=='_AgentToIAt' and len(node.args)>=5 and self._same_grid(node.args[0]) and self._coords(node.args[1:]):
            return ast.copy_location(self._toi(node.args[1:]),node)
        # Placement at exactly the Hood-produced site: coordinates are already
        # finite, integer, dimensionality-correct, in bounds, and wrapped.
        if name=='_AgentNewSQAt' and len(node.args)>=5 and self._same_grid(node.args[0]) and self._coords(node.args[1:]):
            return ast.copy_location(ast.Call(ast.Name('_AgentNewSQTrustedAt',ast.Load()),[copy.deepcopy(a) for a in node.args],[]),node)
        return node

class _HoodExpander(ast.NodeTransformer):
    """Expand ``for i in grid.Hood(HOOD, ...)`` into explicit wrapped sites.

    HOOD must be a literal tuple-of-tuples or a module/closure constant with
    that value. Expansion happens before Numba sees the function, so traversal
    allocates nothing and arbitrary loop bodies remain ordinary model code.
    """
    def __init__(self,namespace):
        self.namespace=namespace
        self.palVars={}
        self.serial=0

    def _hood_value(self,node):
        if isinstance(node,ast.Name) and node.id in self.namespace:
            value=self.namespace[node.id]
        else:
            try: value=ast.literal_eval(node)
            except Exception as e: raise TypeError("grid.Hood hood must be a compile-time tuple of offset tuples") from e
        if not isinstance(value,tuple): raise TypeError("grid.Hood hood must be a tuple of offset tuples")
        out=[];dim=None
        for off in value:
            if not isinstance(off,tuple) or not 1<=len(off)<=3: raise TypeError("hood offsets must be 1D, 2D, or 3D tuples")
            if dim is None: dim=len(off)
            if len(off)!=dim: raise ValueError("all hood offsets must have the same dimensionality")
            vals=[]
            for v in off:
                if isinstance(v,bool) or not isinstance(v,(int,np.integer)): raise TypeError("hood offsets must be integer tuples")
                if v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max: raise ValueError("hood offsets must fit int32")
                vals.append(int(v))
            out.append(tuple(vals))
        return tuple(out),dim

    @staticmethod
    def _plus(base,delta):
        return copy.deepcopy(base) if delta==0 else ast.BinOp(left=copy.deepcopy(base),op=ast.Add(),right=ast.Constant(delta))

    @staticmethod
    def _call(name,args):
        return ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=args,keywords=[])

    @staticmethod
    def _attr(obj,name,*args):
        return ast.Call(func=ast.Attribute(value=copy.deepcopy(obj),attr=name,ctx=ast.Load()),args=[copy.deepcopy(a) for a in args],keywords=[])

    @staticmethod
    def _bind_args(node,names,defaults=()):
        """Bind an AST call to canonical parameter slots using Python call semantics."""
        if len(node.args)>len(names): raise TypeError(f"{node.func.attr} takes at most {len(names)} arguments")
        vals=[None]*len(names)
        for i,arg in enumerate(node.args): vals[i]=arg
        pos={name:i for i,name in enumerate(names)}
        for kw in node.keywords:
            if kw.arg is None: raise TypeError(f"{node.func.attr} does not accept **kwargs in pal.njit transformed calls")
            if kw.arg=='_palLine': continue
            if kw.arg not in pos: raise TypeError(f"{node.func.attr} got an unexpected keyword argument '{kw.arg}'")
            i=pos[kw.arg]
            if vals[i] is not None: raise TypeError(f"{node.func.attr} got multiple values for argument '{kw.arg}'")
            vals[i]=kw.value
        required=len(names)-len(defaults)
        for i in range(required):
            if vals[i] is None: raise TypeError(f"{node.func.attr} missing required argument '{names[i]}'")
        for i,default in enumerate(defaults,start=required):
            if vals[i] is None: vals[i]=ast.Constant(default)
        return vals

    @staticmethod
    def _hood_wrap(grid,value,axis):
        """Inline lattice wrapping directly from cached grid geometry.

        Hood expansion already runs before Numba typing, so avoid routing this
        hot primitive through the generic _BoxWrap overload/method stack.
        """
        size=ast.Attribute(copy.deepcopy(grid),'_'+axis.lower()+'Dim',ast.Load())
        wrap=ast.Attribute(copy.deepcopy(grid),'_wrap'+axis,ast.Load())
        value=copy.deepcopy(value)
        inBounds=ast.BoolOp(ast.And(),[
            ast.Compare(ast.Constant(0),[ast.LtE()],[copy.deepcopy(value)]),
            ast.Compare(copy.deepcopy(value),[ast.Lt()],[copy.deepcopy(size)]),
        ])
        wrapped=ast.BinOp(copy.deepcopy(value),ast.Mod(),copy.deepcopy(size))
        return ast.IfExp(inBounds,copy.deepcopy(value),ast.IfExp(wrap,wrapped,ast.Constant(-1)))

    def _agent_setup(self,grid,tag):
        gen=f'__pal_a{tag}_gen'; gd=f'__pal_a{tag}_grid'; ip=f'__pal_a{tag}_int'; fp=f'__pal_a{tag}_float'; nf=f'__pal_a{tag}_nfp'
        out=[
            ast.Assign([ast.Name(gen,ast.Store())],self._call('_AgentIterGeneration',[copy.deepcopy(grid)])),
            ast.Assign([ast.Name(gd,ast.Store())],self._call('_ag_grid_data',[ast.Attribute(copy.deepcopy(grid),'_ptr',ast.Load())])),
            ast.Assign([ast.Name(ip,ast.Store())],self._call('_ag_int_data',[ast.Attribute(copy.deepcopy(grid),'_ptr',ast.Load())])),
            ast.Assign([ast.Name(fp,ast.Store())],self._call('_ag_float_data',[ast.Attribute(copy.deepcopy(grid),'_ptr',ast.Load())])),
            ast.Assign([ast.Name(nf,ast.Store())],self._call('_ag_nfloatprops',[ast.Attribute(copy.deepcopy(grid),'_ptr',ast.Load())])),
        ]
        return out,gen,gd,ip,fp,nf

    def _agent_while(self,node,grid,tag,loc,targets,extra_assigns=None):
        setup,gen,gd,ip,fp,nf=self._agent_setup(grid,tag)
        cur=f'__pal_a{tag}_cur'; nxt=f'__pal_a{tag}_next'
        setup.append(ast.Assign([ast.Name(cur,ast.Store())],ast.Subscript(ast.Name(gd,ast.Load()),copy.deepcopy(loc),ast.Load())))
        body=[self._call('_AgentIterCheckAt',[copy.deepcopy(grid),ast.Name(gen,ast.Load()),ast.Constant(node.lineno)])]
        body[0]=ast.Expr(body[0])
        # Save/advance before user code so continue/dispose of current is safe.
        prevIndex=ast.BinOp(ast.BinOp(ast.Name(cur,ast.Load()),ast.Mult(),ast.Constant(5)),ast.Add(),ast.Constant(2))
        body.append(ast.Assign([ast.Name(nxt,ast.Store())],ast.Subscript(ast.Name(ip,ast.Load()),prevIndex,ast.Load())))
        for target,value in zip(targets,[ast.Name(cur,ast.Load())]+(extra_assigns or [])):
            body.append(ast.Assign([copy.deepcopy(target)],copy.deepcopy(value)))
        body.append(ast.Assign([ast.Name(cur,ast.Store())],ast.Name(nxt,ast.Load())))
        body.extend(copy.deepcopy(node.body))
        setup.append(ast.While(test=ast.Compare(ast.Name(cur,ast.Load()),[ast.NotEq()],[ast.Constant(-1)]),body=body,orelse=[]))
        setup.append(ast.Expr(self._call('_AgentIterCheckAt',[copy.deepcopy(grid),ast.Name(gen,ast.Load()),ast.Constant(node.lineno)])))
        return setup

    def _expand_agents_at(self,node,call):
        args=self._bind_args(call,('x','y','z'),(-1,-1))
        supplied=set(('x','y','z')[:len(call.args)])|{k.arg for k in call.keywords}
        if not isinstance(node.target,ast.Name): raise TypeError('AgentsAt loop target must be one agent variable')
        self.serial+=1;tag=self.serial;grid=call.func.value
        loc=copy.deepcopy(args[0]) if 'y' not in supplied and 'z' not in supplied else self._call('_AgentToIAt',[copy.deepcopy(grid),*map(copy.deepcopy,args),ast.Constant(node.lineno)])
        return self._agent_while(node,grid,tag,loc,[node.target])

    def _expand_agents_radius(self,node,call):
        args=self._bind_args(call,('rad','x','y','z','exclude'),(None,None,None))
        supplied=set(('rad','x','y','z','exclude')[:len(call.args)])|{k.arg for k in call.keywords}
        if 'x' not in supplied: raise TypeError('AgentsInRadius requires rad and x')
        if 'z' in supplied and 'y' not in supplied: raise TypeError('AgentsInRadius z requires y')
        dim=3 if 'z' in supplied else 2 if 'y' in supplied else 1
        expected=2 if dim==1 else dim+2
        if not isinstance(node.target,(ast.Tuple,ast.List)) or len(node.target.elts)!=expected:
            raise TypeError('AgentsInRadius loop target must be agent, displacement components, and distSq in 2D/3D')
        self.serial+=1;tag=self.serial;grid=call.func.value
        rad=args[0]; centers=args[1:1+dim]; exclude=ast.Constant(-1) if isinstance(args[4],ast.Constant) and args[4].value is None else copy.deepcopy(args[4])
        setup,gen,gd,ip,fp,nf=self._agent_setup(grid,tag)
        r=f'__pal_r{tag}_r';r2=f'__pal_r{tag}_r2'; ex=f'__pal_r{tag}_ex'
        cNames=[f'__pal_r{tag}_{a.lower()}c' for a in 'XYZ'[:dim]]
        setup += [ast.Assign([ast.Name(r,ast.Store())],copy.deepcopy(rad))]
        for n,v in zip(cNames,centers): setup.append(ast.Assign([ast.Name(n,ast.Store())],copy.deepcopy(v)))
        while len(cNames)<3:cNames.append(None)
        setup.append(ast.Expr(self._call('_AgentIterValidateRadiusAt',[copy.deepcopy(grid),ast.Name(r,ast.Load()),ast.Name(cNames[0],ast.Load()),ast.Name(cNames[1],ast.Load()) if cNames[1] else ast.Constant(0.0),ast.Name(cNames[2],ast.Load()) if cNames[2] else ast.Constant(0.0),ast.Constant(dim),ast.Constant(node.lineno)])))
        setup.append(ast.Assign([ast.Name(r2,ast.Store())],ast.BinOp(ast.Name(r,ast.Load()),ast.Mult(),ast.Name(r,ast.Load()))))
        setup.append(ast.Assign([ast.Name(ex,ast.Store())],copy.deepcopy(exclude)))
        coords=[]
        for d,axis in enumerate('XYZ'[:dim]):
            raw=f'__pal_r{tag}_{axis.lower()}raw'; wrapped=f'__pal_r{tag}_{axis.lower()}sq'
            center=ast.Name(cNames[d],ast.Load())
            lo=ast.Call(ast.Name('int',ast.Load()),[ast.Call(ast.Attribute(ast.Name('np',ast.Load()),'floor',ast.Load()),[ast.BinOp(copy.deepcopy(center),ast.Sub(),ast.Name(r,ast.Load()))],[])],[])
            hi=ast.BinOp(ast.Call(ast.Name('int',ast.Load()),[ast.Call(ast.Attribute(ast.Name('np',ast.Load()),'floor',ast.Load()),[ast.BinOp(copy.deepcopy(center),ast.Add(),ast.Name(r,ast.Load()))],[])],[]),ast.Add(),ast.Constant(1))
            coords.append((raw,wrapped,lo,hi,axis))
        vals=[ast.Name(w,ast.Load()) for _,w,_,_,_ in coords]
        loc=self._call('_AgentToIAt',[copy.deepcopy(grid),copy.deepcopy(vals[0]),copy.deepcopy(vals[1]) if len(vals)>1 else ast.Constant(-1),copy.deepcopy(vals[2]) if len(vals)>2 else ast.Constant(-1),ast.Constant(node.lineno)])
        cur=f'__pal_r{tag}_cur';nxt=f'__pal_r{tag}_next'; ds=f'__pal_r{tag}_ds'
        dnames=[f'__pal_r{tag}_d{a.lower()}' for a in 'XYZ'[:dim]]
        site=[ast.Assign([ast.Name(cur,ast.Store())],ast.Subscript(ast.Name(gd,ast.Load()),loc,ast.Load()))]
        wb=[ast.Expr(self._call('_AgentIterCheckAt',[copy.deepcopy(grid),ast.Name(gen,ast.Load()),ast.Constant(node.lineno)]))]
        prevIndex=ast.BinOp(ast.BinOp(ast.Name(cur,ast.Load()),ast.Mult(),ast.Constant(5)),ast.Add(),ast.Constant(2))
        wb.append(ast.Assign([ast.Name(nxt,ast.Store())],ast.Subscript(ast.Name(ip,ast.Load()),prevIndex,ast.Load())))
        # displacement = agent position - center, minimum image via grid.DispWrap*(center,agentPos)
        for d,axis in enumerate('XYZ'[:dim]):
            fidx=ast.BinOp(ast.BinOp(ast.Name(cur,ast.Load()),ast.Mult(),ast.Name(nf,ast.Load())),ast.Add(),ast.Constant(d))
            apos=ast.Subscript(ast.Name(fp,ast.Load()),fidx,ast.Load())
            disp=ast.BinOp(apos,ast.Sub(),ast.Call(ast.Attribute(ast.Name('np',ast.Load()),'float32',ast.Load()),[ast.Name(cNames[d],ast.Load())],[]))
            wb.append(ast.Assign([ast.Name(dnames[d],ast.Store())],disp))
            size=ast.Attribute(copy.deepcopy(grid),axis.lower()+'Dim',ast.Load())
            half=ast.BinOp(copy.deepcopy(size),ast.Div(),ast.Constant(2.0))
            wrapCond=ast.Attribute(copy.deepcopy(grid),'_wrap'+axis,ast.Load())
            adjust=ast.If(ast.Compare(ast.Name(dnames[d],ast.Load()),[ast.Gt()],[copy.deepcopy(half)]),[ast.AugAssign(ast.Name(dnames[d],ast.Store()),ast.Sub(),copy.deepcopy(size))],[ast.If(ast.Compare(ast.Name(dnames[d],ast.Load()),[ast.Lt()],[ast.UnaryOp(ast.USub(),copy.deepcopy(half))]),[ast.AugAssign(ast.Name(dnames[d],ast.Store()),ast.Add(),copy.deepcopy(size))],[])])
            wb.append(ast.If(wrapCond,[adjust],[]))
        sumsq=None
        for dn in dnames:
            term=ast.BinOp(ast.Name(dn,ast.Load()),ast.Mult(),ast.Name(dn,ast.Load()))
            sumsq=term if sumsq is None else ast.BinOp(sumsq,ast.Add(),term)
        wb.append(ast.Assign([ast.Name(ds,ast.Store())],sumsq))
        cond=ast.BoolOp(ast.And(),[ast.Compare(ast.Name(cur,ast.Load()),[ast.NotEq()],[ast.Name(ex,ast.Load())]),ast.Compare(ast.Name(ds,ast.Load()),[ast.LtE()],[ast.Name(r2,ast.Load())])])
        outvals=[ast.Name(cur,ast.Load())]+[ast.Name(d,ast.Load()) for d in dnames]+([] if dim==1 else [ast.Name(ds,ast.Load())])
        user=[]
        for target,value in zip(node.target.elts,outvals): user.append(ast.Assign([copy.deepcopy(target)],copy.deepcopy(value)))
        user.append(ast.Assign([ast.Name(cur,ast.Store())],ast.Name(nxt,ast.Load())))
        user.extend(copy.deepcopy(node.body))
        # If not matched, still advance.
        wb.append(ast.If(cond,user,[ast.Assign([ast.Name(cur,ast.Store())],ast.Name(nxt,ast.Load()))]))
        site.append(ast.While(ast.Compare(ast.Name(cur,ast.Load()),[ast.NotEq()],[ast.Constant(-1)]),wb,[]))
        site.append(ast.Expr(self._call('_AgentIterCheckAt',[copy.deepcopy(grid),ast.Name(gen,ast.Load()),ast.Constant(node.lineno)])))
        inner=site
        for raw,wrapped,a,b,axis in reversed(coords):
            wrapCall=self._call('_AgentRadiusWrapJit',[ast.Name(raw,ast.Load()),ast.Attribute(copy.deepcopy(grid),'_'+axis.lower()+'Dim',ast.Load()),ast.Attribute(copy.deepcopy(grid),'_wrap'+axis,ast.Load())])
            guarded=[ast.Assign([ast.Name(wrapped,ast.Store())],wrapCall),ast.If(ast.Compare(ast.Name(wrapped,ast.Load()),[ast.NotEq()],[ast.Constant(-1)]),inner,[])]
            inner=[ast.For(ast.Name(raw,ast.Store()),ast.Call(ast.Name('range',ast.Load()),[a,ast.IfExp(test=ast.Attribute(copy.deepcopy(grid),'_wrap'+axis,ast.Load()),body=ast.Call(ast.Name('min',ast.Load()),[b,ast.BinOp(copy.deepcopy(a),ast.Add(),ast.Attribute(copy.deepcopy(grid),axis.lower()+'Dim',ast.Load()))],[]),orelse=b)],[]),guarded,[])]
        return setup+inner


    @staticmethod
    def _hood_control_flow_error(body):
        """Return the first Hood-level break/continue, ignoring nested scopes/loops."""
        class Finder(ast.NodeVisitor):
            def __init__(self): self.bad=None
            def visit_Break(self,node):
                if self.bad is None: self.bad=node
            def visit_Continue(self,node):
                if self.bad is None: self.bad=node
            # break/continue below another loop belong to that loop, not Hood.
            def visit_For(self,node): return
            def visit_AsyncFor(self,node): return
            def visit_While(self,node): return
            # Nor can control flow jump out of a nested function/class scope.
            def visit_FunctionDef(self,node): return
            def visit_AsyncFunctionDef(self,node): return
            def visit_Lambda(self,node): return
            def visit_ClassDef(self,node): return
        finder=Finder()
        for stmt in body:
            finder.visit(stmt)
            if finder.bad is not None: return finder.bad
        return None

    @staticmethod
    def _hood_target(node,dim):
        if dim==1:
            if not isinstance(node.target,ast.Name):
                raise TypeError("1D grid.Hood loop target must be one coordinate variable")
            return [node.target]
        if not isinstance(node.target,(ast.Tuple,ast.List)) or len(node.target.elts)!=dim:
            raise TypeError(f"{dim}D grid.Hood loop target must contain {dim} coordinate variables")
        if any(not isinstance(t,ast.Name) for t in node.target.elts):
            raise TypeError("grid.Hood coordinate targets must be variable names")
        return list(node.target.elts)

    def _expand_hood_unrolled(self,node,grid,hood,dim,coords,targets):
        bad=self._hood_control_flow_error(node.body)
        if bad is not None:
            kind='break' if isinstance(bad,ast.Break) else 'continue'
            raise SyntaxError(f"Hood-level {kind} is not supported with unroll=True; use the default Hood loop when {kind} semantics are needed")
        self.serial+=1;tag=self.serial
        axes='XYZ'[:dim]
        unique=[sorted({off[d] for off in hood}) for d in range(dim)]
        names={};statements=[]
        for d,axis in enumerate(axes):
            for delta in unique[d]:
                name=f'__pal_h{tag}_{axis.lower()}_{"m"+str(-delta) if delta<0 else "p"+str(delta)}'
                names[d,delta]=name
                wrap=self._hood_wrap(grid,self._plus(coords[d],delta),'XYZ'[d])
                statements.append(ast.Assign([ast.Name(name,ast.Store())],wrap))
        for off in hood:
            vals=[ast.Name(names[d,off[d]],ast.Load()) for d in range(dim)]
            tests=[ast.Compare(copy.deepcopy(v),[ast.NotEq()],[ast.Constant(-1)]) for v in vals]
            cond=tests[0] if len(tests)==1 else ast.BoolOp(ast.And(),tests)
            body=[ast.Assign([copy.deepcopy(t)],copy.deepcopy(v)) for t,v in zip(targets,vals)]
            trusted=_HoodTrustedBodyOptimizer(grid,targets,dim)
            body.extend([trusted.visit(copy.deepcopy(stmt)) for stmt in node.body])
            statements.append(ast.If(cond,body,[]))
        # With Hood-level break forbidden, for...else is simply code following
        # normal exhaustion. return/raise still bypass it naturally.
        statements.extend(copy.deepcopy(node.orelse))
        return statements

    def _expand_hood_loop(self,node,grid,hood,dim,coords,targets):
        """Emit one compact offset loop; the user body appears exactly once."""
        self.serial+=1;tag=self.serial
        offName=f'__pal_h{tag}_off'
        hoodName=f'__pal_h{tag}_hood'
        # Keep large neighborhoods compact in Numba IR. A literal tuple makes
        # Numba type every offset separately; a read-only int32 array gives the
        # loop one homogeneous type regardless of neighborhood size.
        # The public/canonical Hood remains the immutable tuple-of-tuples.
        # This private array exists only as a compact lowering artifact for Numba.
        # Mark it read-only as well so even transformed-function introspection cannot
        # mutate the compiled neighborhood representation.
        hoodArray=np.asarray(hood,dtype=np.int32)
        hoodArray.flags.writeable=False
        self.namespace[hoodName]=hoodArray
        valNames=[f'__pal_h{tag}_{a.lower()}' for a in 'XYZ'[:dim]]
        body=[]
        for d,(axis,name) in enumerate(zip('XYZ'[:dim],valNames)):
            delta=ast.Subscript(ast.Name(hoodName,ast.Load()),ast.Tuple([ast.Name(offName,ast.Load()),ast.Constant(d)],ast.Load()),ast.Load())
            value=ast.BinOp(copy.deepcopy(coords[d]),ast.Add(),delta)
            body.append(ast.Assign([ast.Name(name,ast.Store())],self._hood_wrap(grid,value,axis)))
        tests=[ast.Compare(ast.Name(n,ast.Load()),[ast.NotEq()],[ast.Constant(-1)]) for n in valNames]
        cond=tests[0] if len(tests)==1 else ast.BoolOp(ast.And(),tests)
        user=[ast.Assign([copy.deepcopy(t)],ast.Name(n,ast.Load())) for t,n in zip(targets,valNames)]
        trusted=_HoodTrustedBodyOptimizer(grid,targets,dim)
        user.extend([trusted.visit(copy.deepcopy(stmt)) for stmt in node.body])
        body.append(ast.If(cond,user,[]))
        offsets=ast.Call(ast.Name('range',ast.Load()),[ast.Constant(len(hood))],[])
        return ast.For(ast.Name(offName,ast.Store()),offsets,body,copy.deepcopy(node.orelse))

    def _expand_box(self,node,call):
        args=self._bind_args(call,('x1','x2','y1','y2','z1','z2'),(None,None,None,None))
        supplied=set(('x1','x2','y1','y2','z1','z2')[:len(call.args)])|{k.arg for k in call.keywords}
        if ('y1' in supplied)!=('y2' in supplied) or ('z1' in supplied)!=('z2' in supplied) or ('z1' in supplied and 'y1' not in supplied):
            raise TypeError("grid.Box expects x1,x2[,y1,y2[,z1,z2]]")
        dim=3 if 'z1' in supplied else 2 if 'y1' in supplied else 1
        targets=self._hood_target(node,dim)
        self.serial+=1;tag=self.serial;grid=call.func.value
        bad=self._hood_control_flow_error(node.body)
        if bad is not None:
            kind='break' if isinstance(bad,ast.Break) else 'continue'
            raise SyntaxError(f"Box-level {kind} is not supported; use explicit nested range loops when {kind} semantics are needed")

        def loops(wrapped):
            body=copy.deepcopy(node.body)
            for d in range(dim-1,-1,-1):
                raw=f'__pal_b{tag}_{"xyz"[d]}raw'
                start=copy.deepcopy(args[2*d]); stop=copy.deepcopy(args[2*d+1])
                if wrapped:
                    val=f'__pal_b{tag}_{"xyz"[d]}'
                    wrapCall=self._call(('_BoxWrapX','_BoxWrapY','_BoxWrapZ')[d],[copy.deepcopy(grid),ast.Name(raw,ast.Load())])
                    assign=ast.Assign([ast.Name(val,ast.Store())],wrapCall)
                    user=[ast.Assign([copy.deepcopy(targets[d])],ast.Name(val,ast.Load()))]; user.extend(body)
                    cond=ast.Compare(ast.Name(val,ast.Load()),[ast.NotEq()],[ast.Constant(-1)])
                    loopBody=[assign,ast.If(cond,user,[])]
                else:
                    loopBody=[ast.Assign([copy.deepcopy(targets[d])],ast.Name(raw,ast.Load()))]; loopBody.extend(body)
                body=[ast.For(ast.Name(raw,ast.Store()),ast.Call(ast.Name('range',ast.Load()),[start,stop],[]),loopBody,[])]
            return body

        # If every half-open bound is already inside the grid, Box is literally
        # nested range loops. Boundary/wrap mapping is confined to the fallback.
        tests=[]
        for d,dimName in enumerate(('xDim','yDim','zDim')[:dim]):
            lo=copy.deepcopy(args[2*d]); hi=copy.deepcopy(args[2*d+1])
            tests.append(ast.Compare(lo,[ast.GtE()],[ast.Constant(0)]))
            tests.append(ast.Compare(hi,[ast.LtE()],[ast.Attribute(copy.deepcopy(grid),dimName,ast.Load())]))
        inBounds=tests[0] if len(tests)==1 else ast.BoolOp(ast.And(),tests)
        out=[ast.If(inBounds,loops(False),loops(True))]
        out.extend(copy.deepcopy(node.orelse))
        return out

    def visit_FunctionDef(self,node):
        old=self.palVars
        self.palVars={}
        for arg in node.args.args:
            ann=arg.annotation
            name=ann.attr if isinstance(ann,ast.Attribute) else ann.id if isinstance(ann,ast.Name) else None
            if name in {'Grid','AgentGrid','PopGrid','PDEgrid','IList','Multinomial'}: self.palVars[arg.arg]=name
        node=self.generic_visit(node)
        self.palVars=old
        return node

    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Name):
            sub=node.targets[0]; kind=self.palVars.get(sub.value.id)
            if kind=='Grid': return ast.copy_location(ast.Expr(self._call('_GridSetAt',[sub.value,sub.slice,node.value,ast.Constant(' at source line '+str(node.lineno))])),node)
            if kind=='PDEgrid': return ast.copy_location(ast.Expr(self._call('_PDESetItemAt',[sub.value,sub.slice,node.value,ast.Constant(node.lineno)])),node)
            if kind=='PopGrid': return ast.copy_location(ast.Expr(self._call('_PopSetItemAt',[sub.value,sub.slice,node.value,ast.Constant(node.lineno)])),node)
            if kind=='AgentGrid': return ast.copy_location(ast.Expr(self._call('_AgentSetItemAt',[sub.value,sub.slice,node.value,ast.Constant(node.lineno)])),node)
        return node

    def visit_AugAssign(self,node):
        # Augmented subscripting (for example grid[a,p] -= 1) is both a read and
        # a write. Lower it through the same source-aware helpers as ordinary
        # indexing so it never falls through to Numba's special dunder lowering.
        target=node.target
        if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name):
            kind=self.palVars.get(target.value.id)
            helperGet={'Grid':'_GridGetAt','PDEgrid':'_PDEItemAt','PopGrid':'_PopItemAt','AgentGrid':'_AgentItemAt'}.get(kind)
            helperSet={'Grid':'_GridSetAt','PDEgrid':'_PDESetItemAt','PopGrid':'_PopSetItemAt','AgentGrid':'_AgentSetItemAt'}.get(kind)
            if helperGet is not None:
                obj=self.visit(copy.deepcopy(target.value))
                key=self.visit(copy.deepcopy(target.slice))
                value=self.visit(node.value)
                if kind=='Grid':
                    lineArg=ast.Constant(' at source line '+str(node.lineno))
                else:
                    lineArg=ast.Constant(node.lineno)
                current=self._call(helperGet,[copy.deepcopy(obj),copy.deepcopy(key),copy.deepcopy(lineArg)])
                updated=ast.BinOp(current,node.op,value)
                return ast.copy_location(ast.Expr(self._call(helperSet,[obj,key,updated,lineArg])),node)
        return self.generic_visit(node)

    def visit_Subscript(self,node):
        node=self.generic_visit(node)
        # ``grid.counts[key]`` is read-only and lowers directly to AgentGrid.
        if isinstance(node.ctx,ast.Load) and isinstance(node.value,ast.Attribute) and node.value.attr=='counts':
            return ast.copy_location(self._call('_AgentCountsItemAt',[node.value.value,node.slice,ast.Constant(node.lineno)]),node)
        if isinstance(node.ctx,ast.Load) and isinstance(node.value,ast.Name):
            kind=self.palVars.get(node.value.id)
            if kind=='Grid': return ast.copy_location(self._call('_GridGetAt',[node.value,node.slice,ast.Constant(' at source line '+str(node.lineno))]),node)
            if kind=='PDEgrid': return ast.copy_location(self._call('_PDEItemAt',[node.value,node.slice,ast.Constant(node.lineno)]),node)
            if kind=='PopGrid': return ast.copy_location(self._call('_PopItemAt',[node.value,node.slice,ast.Constant(node.lineno)]),node)
            if kind=='AgentGrid': return ast.copy_location(self._call('_AgentItemAt',[node.value,node.slice,ast.Constant(node.lineno)]),node)
            if kind=='IList':
                return ast.copy_location(self._call('_IListItemAt',[node.value,node.slice,ast.Constant(node.lineno)]),node)
        return node

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='Append' and len(node.args)==1 and not node.keywords:
            return ast.copy_location(self._call('_IListAppendAt',[node.func.value,node.args[0],ast.Constant(node.lineno)]),node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='Sample' and len(node.args)==1 and not node.keywords:
            return ast.copy_location(self._call('_MultinomialSampleAt',[node.func.value,node.args[0],ast.Constant(node.lineno)]),node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name):
            obj=node.func.value; kind=self.palVars.get(obj.id); name=node.func.attr; line=ast.Constant(node.lineno)
            if kind=='AgentGrid':
                if name=='NewAgentSQ':
                    args=self._bind_args(node,('x','y','z'),(-1,-1))
                    return ast.copy_location(self._call('_AgentNewSQAt',[obj,*args,line]),node)
                if name=='Dispose' and len(node.args)==1: return ast.copy_location(self._call('_AgentDisposeAt',[obj,node.args[0],line]),node)
                if name in {'I','XSQ','YSQ','ZSQ'} and len(node.args)==1:
                    return ast.copy_location(self._call('_AgentCoordAt',[obj,node.args[0],ast.Constant({'I':0,'XSQ':1,'YSQ':2,'ZSQ':3}[name]),line]),node)
                if name in {'Alive','X','Y','Z'} and len(node.args)==1:
                    return ast.copy_location(self._call('_AgentValueAt',[obj,node.args[0],ast.Constant({'Alive':0,'X':1,'Y':2,'Z':3}[name]),line]),node)
                if name in {'ItoX','ItoY','ItoZ'} and len(node.args)==1:
                    return ast.copy_location(self._call('_AgentIndexAt',[obj,node.args[0],ast.Constant({'ItoX':0,'ItoY':1,'ItoZ':2}[name]),line]),node)
                if name in {'InWrapSQX','InWrapSQY','InWrapSQZ','InWrapX','InWrapY','InWrapZ'} and len(node.args)==1:
                    return ast.copy_location(self._call('_AgentWrapAt',[obj,node.args[0],ast.Constant({'InWrapSQX':0,'InWrapSQY':1,'InWrapSQZ':2,'InWrapX':3,'InWrapY':4,'InWrapZ':5}[name]),line]),node)
                if name in {'DispWrapX','DispWrapY','DispWrapZ'} and len(node.args)==2:
                    return ast.copy_location(self._call('_AgentDispAt',[obj,node.args[0],node.args[1],ast.Constant({'DispWrapX':0,'DispWrapY':1,'DispWrapZ':2}[name]),line]),node)
                if name=='NewAgent':
                    args=self._bind_args(node,('x','y','z'),(-1.0,-1.0))
                    return ast.copy_location(self._call('_AgentNewAt',[obj,*args,line]),node)
                if name=='LastAgent':
                    args=self._bind_args(node,('x','y','z'),(-1,-1))
                    return ast.copy_location(self._call('_AgentLastAt',[obj,*args,line]),node)
                if name in {'MoveSQ','Move'}:
                    d=-1.0 if name=='Move' else -1
                    args=self._bind_args(node,('agent','x','y','z'),(d,d))
                    return ast.copy_location(self._call('_AgentMoveAt',[obj,*args,ast.Constant(name=='Move'),line]),node)
                if name=='ToI':
                    args=self._bind_args(node,('x','y','z'),(-1,-1))
                    return ast.copy_location(self._call('_AgentToIAt',[obj,*args,line]),node)
                if name=='GetPop' and not node.args: return ast.copy_location(self._call('_AgentPopAt',[obj,line]),node)
                if name=='All':
                    shuffle=ast.Constant(False)
                    if node.args: shuffle=node.args[0]
                    for kw in node.keywords:
                        if kw.arg=='shuffle': shuffle=kw.value
                    return ast.copy_location(self._call('_AgentAllAt',[obj,shuffle,line]),node)
            if kind=='IList':
                if name=='Clear' and not node.args: return ast.copy_location(self._call('_IListClearAt',[obj,line]),node)
                if name=='Random' and not node.args: return ast.copy_location(self._call('_IListRandomAt',[obj,line]),node)
                if name=='Shuffle' and not node.args: return ast.copy_location(self._call('_IListOpAt',[obj,ast.Constant(0),line]),node)
                if name in {'All','Iter'} and not node.args: return ast.copy_location(self._call('_IListArrayAt',[obj,ast.Constant(name=='All'),line]),node)
            if kind=='Multinomial':
                if name=='Binomial' and len(node.args)==2: return ast.copy_location(self._call('_MultiBinomialAt',[obj,node.args[0],node.args[1],line]),node)
                if name=='Setup' and len(node.args)==1: return ast.copy_location(self._call('_MultiSetupAt',[obj,node.args[0],line]),node)
            if kind in {'PDEgrid','PopGrid'}:
                if name=='ToI':
                    args=self._bind_args(node,('x','y','z'),(-1,-1))
                    return ast.copy_location(self._call('_SpatialIndexAt',[obj,*args,ast.Constant(0),line]),node)
                if name in {'ItoX','ItoY','ItoZ'} and len(node.args)==1:
                    return ast.copy_location(self._call('_SpatialIndexAt',[obj,node.args[0],ast.Constant(-1),ast.Constant(-1),ast.Constant({'ItoX':1,'ItoY':2,'ItoZ':3}[name]),line]),node)
                if name in {'InWrapX','InWrapY','InWrapZ'} and len(node.args)==1:
                    return ast.copy_location(self._call('_SpatialWrapAt',[obj,node.args[0],ast.Constant({'InWrapX':0,'InWrapY':1,'InWrapZ':2}[name]),line]),node)
            if kind=='PopGrid':
                if name=='Add':
                    args=self._bind_args(node,('value','x','y','z'),(-1,-1))
                    return ast.copy_location(self._call('_PopAddAt',[obj,*args,line]),node)
                if name in {'Update','Reset','GetPop'} and not node.args: return ast.copy_location(self._call('_PopBasicAt',[obj,ast.Constant({'Update':0,'Reset':1,'GetPop':2}[name]),line]),node)
                if name=='All' and not node.args: return ast.copy_location(self._call('_PopAllAt',[obj,line]),node)
            if kind=='PDEgrid':
                if name=='Add':
                    args=self._bind_args(node,('value','x','y','z'),(-1,-1))
                    return ast.copy_location(self._call('_PDEAddAt',[obj,*args,line]),node)
                if name in {'Update','Reset'} and not node.args: return ast.copy_location(self._call('_PDEBasicAt',[obj,ast.Constant(0 if name=='Update' else 1),ast.Constant(0.0),ast.Constant(0.0),ast.Constant(0.0),ast.Constant(0.0),line]),node)
                if name=='SetTimeSpaceStep':
                    args=self._bind_args(node,('dt','dx','dy','dz'),(1.0,1.0))
                    return ast.copy_location(self._call('_PDEBasicAt',[obj,ast.Constant(2),*args,line]),node)
                if name in {'Dx','Dy','Dz','Dt'} and not node.args: return ast.copy_location(self._call('_PDEBasicAt',[obj,ast.Constant({'Dx':3,'Dy':4,'Dz':5,'Dt':6}[name]),ast.Constant(0.0),ast.Constant(0.0),ast.Constant(0.0),ast.Constant(0.0),line]),node)
                if name=='Diffusion':
                    args=self._bind_args(node,('rateConstant','xMinBC','xMaxBC','yMinBC','yMaxBC','zMinBC','zMaxBC'),(None,None,None,None,None,None))
                    return ast.copy_location(self._call('_PDEDiffusionAt',[obj,*args,line]),node)
                if name in {'DiffusionADI','Advection'}:
                    if name=='DiffusionADI':
                        args=self._bind_args(node,('rateConstant','xMinBC','xMaxBC','yMinBC','yMaxBC','zMinBC','zMaxBC'),(None,None,None,None,None,None)); abc=[args[0],ast.Constant(0.0),ast.Constant(0.0)]; vals=args[1:]; op=0
                    else:
                        args=self._bind_args(node,('vx','vy','vz','xMinBC','xMaxBC','yMinBC','yMaxBC','zMinBC','zMaxBC'),(0.0,0.0,None,None,None,None,None,None)); abc=args[:3]; vals=args[3:]; op=1
                    return ast.copy_location(self._call('_PDEScalarTransportAt',[obj,ast.Constant(op),*abc,*vals,line]),node)
                if name in {'DiffusionRadialCircle','DiffusionRadialSphere'}:
                    args=self._bind_args(node,('rateConstant','outerBC'),(None,))
                    return ast.copy_location(self._call('_PDERadialAt',[obj,*args,ast.Constant(name=='DiffusionRadialSphere'),line]),node)
        palMethods={'Binomial','Setup','Clear','Random','Shuffle','All','Iter','Alive','NewAgentSQ','NewAgent','Dispose','I','XSQ','YSQ','ZSQ','X','Y','Z','ToI','ItoX','ItoY','ItoZ','InWrapSQX','InWrapSQY','InWrapSQZ','InWrapX','InWrapY','InWrapZ','DispWrapX','DispWrapY','DispWrapZ','LastAgent','MoveSQ','Move','GetPop','Add','Update','Reset','SetTimeSpaceStep','Dx','Dy','Dz','Dt','Diffusion','DiffusionMask','DiffusionField','DiffusionInterfaces','DiffusionADI','DiffusionRadialCircle','DiffusionRadialSphere','Advection','AdvectionField','AdvectionInterfaces'}
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and self.palVars.get(node.func.value.id)=='Grid':
            suffix=ast.Constant(' at source line '+str(node.lineno)); obj=node.func.value; name=node.func.attr
            if name=='ToI':
                args=self._bind_args(node,('x','y','z'),(-1,-1))
                return ast.copy_location(self._call('_GridToIAt',[obj,*args,suffix]),node)
            if name in {'ItoX','ItoY','ItoZ'} and len(node.args)==1:
                return ast.copy_location(self._call('_GridItoAt',[obj,node.args[0],ast.Constant({'ItoX':0,'ItoY':1,'ItoZ':2}[name]),suffix]),node)
            if name in {'InWrapX','InWrapY','InWrapZ'} and len(node.args)==1:
                return ast.copy_location(self._call('_GridWrapAt',[obj,node.args[0],ast.Constant({'InWrapX':0,'InWrapY':1,'InWrapZ':2}[name]),suffix]),node)
        if isinstance(node.func,ast.Attribute) and node.func.attr in palMethods and isinstance(node.func.value,ast.Name) and self.palVars.get(node.func.value.id) not in (None,'Grid') and not any(k.arg=='_palLine' for k in node.keywords):
            node.keywords.append(ast.keyword('_palLine',ast.Constant(node.lineno)))
        return node

    def visit_For(self,node):
        node=self.generic_visit(node)
        call=node.iter
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute):
            if call.func.attr=='Box': return self._expand_box(node,call)
            if call.func.attr=='AgentsAt': return self._expand_agents_at(node,call)
            if call.func.attr=='AgentsInRadius' and isinstance(node.target,(ast.Tuple,ast.List)):
                return self._expand_agents_radius(node,call)
        if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='Hood'):
            return node
        args=self._bind_args(call,('hood','x','y','z','unroll'),(None,None,None,False))
        supplied=set(('hood','x','y','z','unroll')[:len(call.args)])|{k.arg for k in call.keywords}
        if 'x' not in supplied: raise TypeError("grid.Hood requires hood and at least x")
        if 'z' in supplied and 'y' not in supplied: raise TypeError("grid.Hood z requires y")
        unrollNode=args[4]
        if not isinstance(unrollNode,ast.Constant) or not isinstance(unrollNode.value,bool):
            raise TypeError("grid.Hood unroll must be the compile-time literal True or False")
        unroll=unrollNode.value
        hood,dim=self._hood_value(args[0])
        ncoords=3 if 'z' in supplied else 2 if 'y' in supplied else 1
        coords=args[1:1+ncoords]
        if dim is None: dim=len(coords)
        if len(coords)!=dim: raise ValueError("hood dimensionality must match the supplied coordinates")
        targets=self._hood_target(node,dim)
        grid=call.func.value
        if unroll:
            return self._expand_hood_unrolled(node,grid,hood,dim,coords,targets)
        return self._expand_hood_loop(node,grid,hood,dim,coords,targets)

def _ExpandHoods(fn):
    try:
        sourceLines,sourceStart=inspect.getsourcelines(fn)
        source=textwrap.dedent("".join(sourceLines))
    except (OSError,TypeError): return fn
    tree=ast.parse(source)
    # ast.parse() numbers this extracted function from line 1. Restore its
    # absolute position in the user's source file before PAL rewrites it so
    # Numba traceback/debug locations continue to point at model code.
    if sourceStart>1: ast.increment_lineno(tree,sourceStart-1)
    target=None
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==fn.__name__:
            target=n;break
    if target is None:return fn
    target.decorator_list=[]
    closure=inspect.getclosurevars(fn)
    namespace=dict(fn.__globals__);namespace.update(closure.globals);namespace.update(closure.nonlocals)
    tree=_HoodExpander(namespace).visit(tree);ast.fix_missing_locations(tree)
    env=dict(namespace)
    env.update({name:globals()[name] for name in ('_AgentIterGeneration','_AgentIterCheck','_AgentIterCheckAt','_AgentIterValidateRadius','_AgentIterValidateRadiusAt','_AgentRadiusWrapJit','_ag_grid_data','_ag_int_data','_ag_float_data','_ag_nfloatprops','_ag_wrap','_ag_toi','_ag_count_i','_BoxWrapX','_BoxWrapY','_BoxWrapZ','_IListAppendInline','_IListAppendAt','_MultinomialSampleAt','_AgentCountsItem','_AgentCountsItemAt','_AgentItemAt','_AgentSetItemAt','_AgentNewSQAt','_AgentNewSQTrustedAt','_AgentDisposeAt','_AgentCoordAt','_AgentToIAt','_AgentAllAt','_IListClearAt','_IListRandomAt','_IListItemAt','_IListOpAt','_IListArrayAt','_MultiBinomialAt','_MultiSetupAt','_AgentPopAt','_AgentValueAt','_AgentIndexAt','_AgentWrapAt','_AgentDispAt','_AgentNewAt','_AgentLastAt','_AgentMoveAt','_PDEAddAt','_PDEBasicAt','_PDEDiffusionAt','_PDEScalarTransportAt','_PDERadialAt','_SpatialIndexAt','_SpatialWrapAt','_PopAddAt','_PopBasicAt','_PopAllAt','_PDEItemAt','_PopItemAt','_PDESetItemAt','_PopSetItemAt','_GridGetAt','_GridSetAt','_GridToIAt','_GridItoAt','_GridWrapAt','np')})
    code=compile(tree,inspect.getsourcefile(fn) or '<pal.njit>','exec')
    exec(code,env)
    out=env[fn.__name__]
    out.__defaults__=fn.__defaults__;out.__kwdefaults__=fn.__kwdefaults__;out.__annotations__=fn.__annotations__;out.__module__=fn.__module__;out.__qualname__=fn.__qualname__
    return out

def njit(*args,**kwargs):
    """Numba njit with compile-time expansion of PAL model code."""
    if args and callable(args[0]) and len(args)==1:
        return nb.njit(_ExpandHoods(args[0]),**kwargs)
    def decorate(fn): return nb.njit(*args,**kwargs)(_ExpandHoods(fn))
    return decorate

# PAL starts in safe mode. The first performance-sensitive PAL constructor locks
# the process-wide mode so safe and fast objects cannot accidentally be mixed.
_FAST_MODE=False
_MODE_LOCKED=False

def FastMode():
    global _FAST_MODE
    if _MODE_LOCKED:
        raise RuntimeError("pal.FastMode() must be called before constructing PAL model or window objects")
    _FAST_MODE=True

def _UseFastMode():
    global _MODE_LOCKED
    _MODE_LOCKED=True
    return _FAST_MODE

def NewIList():
    fast=_UseFastMode()
    ptr=_q_new(0)
    if not ptr: raise MemoryError("unable to allocate IList")
    return _IListFast(ptr,0) if fast else _IListSafe(ptr,0)

def _HoodTuple(offsets):
    return tuple(tuple(int(v) for v in off) for off in offsets)

def MooreHood(dim,excludeCenter=False):
    if dim not in (1,2,3): raise ValueError("dim must be 1, 2, or 3")
    import itertools
    return _HoodTuple(off for off in itertools.product((-1,0,1),repeat=dim) if not excludeCenter or any(off))

def VonNeumannHood(dim,excludeCenter=False):
    if dim not in (1,2,3): raise ValueError("dim must be 1, 2, or 3")
    out=[]
    if not excludeCenter: out.append((0,)*dim)
    for d in range(dim):
        for sign in (-1,1):
            off=[0]*dim;off[d]=sign;out.append(tuple(off))
    return tuple(out)

def CircleHood(dim,rad,excludeCenter=False):
    import itertools,math
    if dim not in (1,2,3): raise ValueError("dim must be 1, 2, or 3")
    if not isinstance(rad,(int,float,np.integer,np.floating)) or not np.isfinite(rad) or rad<0: raise ValueError("radius must be finite and nonnegative")
    r=int(math.floor(rad));r2=float(rad)*float(rad)
    return _HoodTuple(off for off in itertools.product(range(-r,r+1),repeat=dim) if sum(v*v for v in off)<=r2 and (not excludeCenter or any(off)))

def NewMultinomial(other=None):
    if other is not None and not isinstance(other,(_MultinomialSafe,_MultinomialFast)):
        raise TypeError("other must be a PAL Multinomial or None")
    fast=_UseFastMode()
    if other is not None and (fast != isinstance(other,_MultinomialFast)):
        raise ValueError("Multinomial safety mode mismatch")
    otherPtr = 0 if other is None else other._ptr
    ptr = _multi_new(otherPtr)
    if not ptr: raise MemoryError("unable to allocate Multinomial")
    return _MultinomialFast(ptr) if fast else _MultinomialSafe(ptr)


def NewGrid(dimensions,dtype):
    dims=_dimensions_array(dimensions);dt=np.dtype(dtype)
    if dt not in _GRID_TYPE_CODES:raise ValueError("Grid dtype must be bool, int8/16/32/64, uint8/16/32/64, or float32/64")
    fast=_UseFastMode()
    ptr=_grid_new(dims.ctypes.data_as(_I32_PTR),len(dims),_GRID_TYPE_CODES[dt],dt.itemsize)
    if not ptr:raise MemoryError("unable to allocate Grid")
    address=_grid_data(ptr);ctype=np.ctypeslib.as_ctypes_type(dt);n=int(np.prod(np.abs(dims,dtype=np.int64)));data=np.ctypeslib.as_array((ctype*n).from_address(address))
    return _GridPython(ptr,data,dims,dt,not fast)


def _StaticGeometry(dims):
    vals=[abs(int(v)) for v in dims]; d=len(vals)
    x=vals[0] if d>0 else -1; y=vals[1] if d>1 else -1; z=vals[2] if d>2 else -1
    length=0 if d==0 else int(np.prod(np.asarray(vals,dtype=np.int64)))
    wrapX=bool(dims[0]<0) if d>0 else False; wrapY=bool(dims[1]<0) if d>1 else False; wrapZ=bool(dims[2]<0) if d>2 else False
    return x,y,z,length,wrapX,wrapY,wrapZ

def NewAgentGrid(dimensions, numAgentProps=0, isStackable=False):
    dims = _dimensions_array(dimensions,allowZero=True)
    try: props=float(numAgentProps)
    except (TypeError,ValueError,OverflowError): raise ValueError("numAgentProps must be a nonnegative int32")
    if not np.isfinite(props) or props!=int(props) or props<0 or props>np.iinfo(np.int32).max-len(dims): raise ValueError("numAgentProps is too large for AgentGrid metadata")
    numAgentProps=int(numAgentProps)
    if isStackable is not True and isStackable is not False and isStackable!=0 and isStackable!=1: raise ValueError("isStackable must be boolean")
    fast=_UseFastMode()
    ptr = _ag_new(dims.ctypes.data_as(_I32_PTR), len(dims), numAgentProps, int(isStackable))
    if not ptr: raise MemoryError("unable to allocate AgentGrid")
    x,y,z,n,wrapX,wrapY,wrapZ=_StaticGeometry(dims)
    return _AgentGridFast(ptr,len(dims),x,y,z,n,wrapX,wrapY,wrapZ) if fast else _AgentGridSafe(ptr,len(dims),numAgentProps,x,y,z,n,wrapX,wrapY,wrapZ)



def NewPopGrid(dimensions, capacity=None):
    dims = _dimensions_array(dimensions)
    if capacity is None: cap=np.iinfo(np.int64).max
    else:
        try: raw=float(capacity)
        except (TypeError,ValueError,OverflowError): raise ValueError("capacity must be a nonnegative int64")
        if not np.isfinite(raw) or raw!=int(raw): raise ValueError("capacity must be a nonnegative int64")
        cap=int(capacity)
        if cap<0 or cap>np.iinfo(np.int64).max: raise ValueError("capacity must be a nonnegative int64")
    fast=_UseFastMode()
    ptr = _pg_new(dims.ctypes.data_as(_I32_PTR), len(dims), cap)
    if not ptr: raise MemoryError("unable to allocate PopGrid")
    x,y,z,n,wrapX,wrapY,wrapZ=_StaticGeometry(dims)
    return _PopGridFast(ptr,len(dims),x,y,z,n,wrapX,wrapY,wrapZ) if fast else _PopGridSafe(ptr,len(dims),x,y,z,n,wrapX,wrapY,wrapZ)


def NewPDEgrid(dimensions):
    dims = _dimensions_array(dimensions)
    fast=_UseFastMode()
    ptr = _pd_new(dims.ctypes.data_as(_I32_PTR), len(dims))
    if not ptr: raise MemoryError("unable to allocate PDEgrid")
    x,y,z,n,wrapX,wrapY,wrapZ=_StaticGeometry(dims)
    return _PDEgridFast(ptr,len(dims),x,y,z,n,wrapX,wrapY,wrapZ) if fast else _PDEgridSafe(ptr,len(dims),x,y,z,n,wrapX,wrapY,wrapZ)



# Public instance types for annotations/autocomplete. Construction is intentionally
# separate (NewGrid, NewAgentGrid, ...) because PAL selects a safe or fast
# concrete implementation at runtime.
IList=_IListProtocol
Multinomial=_MultinomialProtocol
Grid=_GridProtocol
AgentGrid=_AgentGridProtocol
PopGrid=_PopGridProtocol
PDEgrid=_PDEgridProtocol


# Standard Python pickle support for PAL native-backed objects. The pickle
# contains native state, never process-local C pointers.
def _Snapshot(ptr,sizeFn,writeFn):
    n=sizeFn(ptr)
    buf=(ctypes.c_uint8*n)()
    if not writeFn(ptr,buf,n): raise RuntimeError("PAL snapshot failed")
    return bytes(buf)

def _BytesPtr(data):
    return (ctypes.c_uint8*len(data)).from_buffer_copy(data)

def _RestoreMultinomial(data,safe):
    b=_BytesPtr(data); ptr=_multi_restore(b,len(data))
    if not ptr: raise ValueError("invalid PAL Multinomial pickle")
    return _MultinomialSafe(ptr) if safe else _MultinomialFast(ptr)

def _RestoreIList(data,safe):
    b=_BytesPtr(data); ptr=_q_restore(b,len(data))
    if not ptr: raise ValueError("invalid PAL IList pickle")
    return _IListSafe(ptr,0) if safe else _IListFast(ptr,0)

def _SnapshotGeometry(data):
    dim=int.from_bytes(data[16:20],byteorder=sys.byteorder,signed=True)
    dims=[int.from_bytes(data[20+4*d:24+4*d],byteorder=sys.byteorder,signed=True) for d in range(3)]
    wraps=[bool(int.from_bytes(data[32+4*d:36+4*d],byteorder=sys.byteorder,signed=True)) for d in range(3)]
    length=int.from_bytes(data[44:48],byteorder=sys.byteorder,signed=True)
    x=dims[0] if dim>0 else -1; y=dims[1] if dim>1 else -1; z=dims[2] if dim>2 else -1
    return dim,x,y,z,length,wraps[0],wraps[1],wraps[2]

def _RestoreAgentGrid(data,safe):
    b=_BytesPtr(data); ptr=_ag_restore(b,len(data))
    if not ptr: raise ValueError("invalid PAL AgentGrid pickle")
    dim,x,y,z,length,wrapX,wrapY,wrapZ=_SnapshotGeometry(data)
    nProps=int.from_bytes(data[52:56],byteorder=sys.byteorder,signed=True)
    return _AgentGridSafe(ptr,dim,nProps,x,y,z,length,wrapX,wrapY,wrapZ) if safe else _AgentGridFast(ptr,dim,x,y,z,length,wrapX,wrapY,wrapZ)

def _RestorePopGrid(data,safe):
    b=_BytesPtr(data); ptr=_pg_restore(b,len(data))
    if not ptr: raise ValueError("invalid PAL PopGrid pickle")
    dim,x,y,z,length,wrapX,wrapY,wrapZ=_SnapshotGeometry(data)
    return _PopGridSafe(ptr,dim,x,y,z,length,wrapX,wrapY,wrapZ) if safe else _PopGridFast(ptr,dim,x,y,z,length,wrapX,wrapY,wrapZ)

def _RestorePDEgrid(data,safe):
    b=_BytesPtr(data); ptr=_pd_restore(b,len(data))
    if not ptr: raise ValueError("invalid PAL PDEgrid pickle")
    dim,x,y,z,length,wrapX,wrapY,wrapZ=_SnapshotGeometry(data)
    return _PDEgridSafe(ptr,dim,x,y,z,length,wrapX,wrapY,wrapZ) if safe else _PDEgridFast(ptr,dim,x,y,z,length,wrapX,wrapY,wrapZ)

def _ReduceMultinomialSafe(obj): return (_RestoreMultinomial,(_Snapshot(obj._ptr,_multi_snapshot_size,_multi_snapshot),True))
def _ReduceMultinomialFast(obj): return (_RestoreMultinomial,(_Snapshot(obj._ptr,_multi_snapshot_size,_multi_snapshot),False))
def _ReduceIListSafe(obj): return (_RestoreIList,(_Snapshot(obj._ptr,_q_snapshot_size,_q_snapshot),True))
def _ReduceIListFast(obj): return (_RestoreIList,(_Snapshot(obj._ptr,_q_snapshot_size,_q_snapshot),False))
def _ReduceAgentSafe(obj): return (_RestoreAgentGrid,(_Snapshot(obj._ptr,_ag_snapshot_size,_ag_snapshot),True))
def _ReduceAgentFast(obj): return (_RestoreAgentGrid,(_Snapshot(obj._ptr,_ag_snapshot_size,_ag_snapshot),False))
def _ReducePopSafe(obj): return (_RestorePopGrid,(_Snapshot(obj._ptr,_pg_snapshot_size,_pg_snapshot),True))
def _ReducePopFast(obj): return (_RestorePopGrid,(_Snapshot(obj._ptr,_pg_snapshot_size,_pg_snapshot),False))
def _ReducePDESafe(obj): return (_RestorePDEgrid,(_Snapshot(obj._ptr,_pd_snapshot_size,_pd_snapshot),True))
def _ReducePDEFast(obj): return (_RestorePDEgrid,(_Snapshot(obj._ptr,_pd_snapshot_size,_pd_snapshot),False))

def _RegisterPickle():
    import copyreg
    import importlib
    boxing=importlib.import_module("numba.experimental.jitclass.boxing")
    registry={
        _MultinomialSafe.class_type.instance_type:_ReduceMultinomialSafe,
        _MultinomialFast.class_type.instance_type:_ReduceMultinomialFast,
        _IListSafe.class_type.instance_type:_ReduceIListSafe,
        _IListFast.class_type.instance_type:_ReduceIListFast,
        _AgentGridSafe.class_type.instance_type:_ReduceAgentSafe,
        _AgentGridFast.class_type.instance_type:_ReduceAgentFast,
        _PopGridSafe.class_type.instance_type:_ReducePopSafe,
        _PopGridFast.class_type.instance_type:_ReducePopFast,
        _PDEgridSafe.class_type.instance_type:_ReducePDESafe,
        _PDEgridFast.class_type.instance_type:_ReducePDEFast,
    }
    previous=boxing._specialize_box
    registered=set()
    def specializeBox(typ):
        boxcls=previous(typ)
        reducer=registry.get(typ)
        if reducer is not None and boxcls not in registered:
            copyreg.pickle(boxcls,reducer)
            registered.add(boxcls)
        return boxcls
    boxing._specialize_box=specializeBox

_RegisterPickle()


# Python keyword support for boxed jitclass methods.
def _RegisterPythonKeywordBoxes(entries):
    import inspect
    import functools
    import importlib
    boxing=importlib.import_module("numba.experimental.jitclass.boxing")
    registry=getattr(boxing,"_palKeywordRegistry",None)
    if registry is None:
        registry={}
        patched=set()
        originalSpecializeBox=boxing._specialize_box

        def makeWrapper(original,signature):
            @functools.wraps(original)
            def wrapper(*args,**kwargs):
                if not kwargs:return original(*args)
                bound=signature.bind(*args,**kwargs)
                bound.apply_defaults()
                positional=[]
                for parameter in signature.parameters.values():
                    if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,inspect.Parameter.POSITIONAL_OR_KEYWORD):
                        positional.append(bound.arguments[parameter.name])
                    elif parameter.kind==inspect.Parameter.VAR_POSITIONAL:
                        positional.extend(bound.arguments[parameter.name])
                    else:
                        raise TypeError("PAL jitclass methods do not support keyword-only or **kwargs parameters")
                return original(*positional)
            wrapper.__signature__=signature
            return wrapper

        def patchBox(typ,boxcls):
            sourcecls=registry.get(typ)
            if sourcecls is None or boxcls in patched:return boxcls
            for name,func in sourcecls.__dict__.items():
                if name.startswith("_") or not callable(func) or not hasattr(boxcls,name):continue
                setattr(boxcls,name,makeWrapper(getattr(boxcls,name),inspect.signature(func)))
            patched.add(boxcls)
            return boxcls

        def specializeBox(typ):
            return patchBox(typ,originalSpecializeBox(typ))

        boxing._palKeywordRegistry=registry
        boxing._palKeywordPatched=patched
        boxing._palOriginalSpecializeBox=originalSpecializeBox
        boxing._palPatchBox=patchBox
        boxing._specialize_box=specializeBox
    registry.update(entries)


_RegisterPythonKeywordBoxes({
    _MultinomialSafe.class_type.instance_type:_MultinomialSafePython,
    _MultinomialFast.class_type.instance_type:_MultinomialFastPython,
    _IListSafe.class_type.instance_type:_IListSafePython,
    _IListFast.class_type.instance_type:_IListFastPython,
    _AgentGridSafe.class_type.instance_type:_AgentGridSafePython,
    _AgentGridFast.class_type.instance_type:_AgentGridFastPython,
    _PopGridSafe.class_type.instance_type:_PopGridSafePython,
    _PopGridFast.class_type.instance_type:_PopGridFastPython,
    _PDEgridSafe.class_type.instance_type:_PDEgridSafePython,
    _PDEgridFast.class_type.instance_type:_PDEgridFastPython,
})


# -----------------------------------------------------------------------------
# Python fallbacks for PAL direct-iteration syntax
# -----------------------------------------------------------------------------
# @pal.njit expands these loops before Numba sees them.  The boxed jitclass
# methods below provide the same public syntax in ordinary Python.  Python
# fallbacks prioritize identical semantics over hot-loop performance.

def _PythonHoodIndices(grid,hood,*coords):
    if not isinstance(hood,tuple):
        raise TypeError("hood must be a tuple of offset tuples")
    dim=len(coords)
    if dim<1 or dim>3:
        raise TypeError("Hood expects 1D, 2D, or 3D coordinates")
    axes="XYZ"[:dim]
    agentGrid=hasattr(grid,"InWrapSQX")
    for off in hood:
        if not isinstance(off,tuple) or len(off)!=dim:
            raise ValueError("hood dimensionality must match the supplied coordinates")
        vals=[]
        valid=True
        for d,axis in enumerate(axes):
            v=coords[d]+off[d]
            wrap=getattr(grid,("InWrapSQ" if agentGrid else "InWrap")+axis)
            v=wrap(v)
            if v==-1:
                valid=False;break
            vals.append(int(v))
        if valid:
            yield grid.ToI(*vals)

def _PythonBox(self,*bounds):
    if len(bounds) not in (2,4,6): raise TypeError("Box expects x1,x2[,y1,y2[,z1,z2]]")
    dim=len(bounds)//2
    if dim!=int(self._dimension): raise ValueError("Box dimensionality must match grid dimensionality")
    for v in bounds:
        if not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) or v<np.iinfo(np.int32).min or v>np.iinfo(np.int32).max:
            raise ValueError("Box bounds must be int32 integers")
    if dim==1:
        for xr in range(bounds[0],bounds[1]):
            x=_BoxWrapX(self,xr)
            if x!=-1: yield int(x)
    elif dim==2:
        for xr in range(bounds[0],bounds[1]):
            x=_BoxWrapX(self,xr)
            if x==-1: continue
            for yr in range(bounds[2],bounds[3]):
                y=_BoxWrapY(self,yr)
                if y!=-1: yield int(x),int(y)
    else:
        for xr in range(bounds[0],bounds[1]):
            x=_BoxWrapX(self,xr)
            if x==-1: continue
            for yr in range(bounds[2],bounds[3]):
                y=_BoxWrapY(self,yr)
                if y==-1: continue
                for zr in range(bounds[4],bounds[5]):
                    z=_BoxWrapZ(self,zr)
                    if z!=-1: yield int(x),int(y),int(z)

def _PythonHood(self,hood,*coords):
    return _PythonHoodIndices(self,hood,*coords)

def _PythonAgentGeneration(grid):
    # Safe AgentGrid boxes expose _nProps; fast boxes deliberately omit it.
    return int(_ag_generation(grid._ptr)) if hasattr(grid,"_nProps") else 0

def _PythonAgentCheck(grid,generation):
    if generation and int(_ag_generation(grid._ptr))!=generation:
        raise RuntimeError("AgentGrid cannot be structurally modified during direct agent iteration")

def _PythonAgentsAtIter(grid,original,*coords):
    q=NewIList()
    if len(coords)==1:
        i=coords[0]
        if i<0 or i>=len(grid): raise IndexError("index out of bounds")
        if _ag_add_i_q(grid._ptr,q._ptr,i)<0: raise ValueError("invalid location or allocation failure")
    elif len(coords)==2:
        if _ag_add_q(grid._ptr,q._ptr,coords[0],coords[1],-1)<0: raise IndexError("coordinate out of bounds")
    elif len(coords)==3:
        if _ag_add_q(grid._ptr,q._ptr,coords[0],coords[1],coords[2])<0: raise IndexError("coordinate out of bounds")
    else:
        raise TypeError("AgentsAt expects 1D, 2D, or 3D coordinates/index")
    generation=_PythonAgentGeneration(grid)
    for agent in q.Iter():
        _PythonAgentCheck(grid,generation)
        yield int(agent)
    _PythonAgentCheck(grid,generation)

def _PythonAgentsAt(self,*coords):
    return _PythonAgentsAtIter(self,None,*coords)

def _PythonRadiusIter(grid,materialized,rad,x,y=-1.0,z=-1.0,exclude=None):
    dim=int(grid._dimension)
    if dim<1:
        raise ValueError("AgentsInRadius requires a spatial AgentGrid")
    q=NewIList()
    materialized(grid,q,rad,x,y,z,exclude)
    generation=_PythonAgentGeneration(grid)
    for agent0 in q.Iter():
        _PythonAgentCheck(grid,generation)
        agent=int(agent0)
        dx=grid.DispWrapX(x,grid.X(agent))
        if dim==1:
            yield agent,dx
            continue
        dy=grid.DispWrapY(y,grid.Y(agent))
        distSq=dx*dx+dy*dy
        if dim==2:
            yield agent,dx,dy,distSq
            continue
        dz=grid.DispWrapZ(z,grid.Z(agent))
        distSq+=dz*dz
        yield agent,dx,dy,dz,distSq
    _PythonAgentCheck(grid,generation)

class _PythonAgentCountsView:
    __slots__=("_grid","_safe")
    def __init__(self,grid,safe): self._grid=grid; self._safe=safe
    @property
    def xDim(self): return self._grid.xDim
    @property
    def yDim(self): return self._grid.yDim
    @property
    def zDim(self): return self._grid.zDim
    @property
    def nDims(self): return self._grid.nDims
    @property
    def wrapX(self): return self._grid.wrapX
    @property
    def wrapY(self): return self._grid.wrapY
    @property
    def wrapZ(self): return self._grid.wrapZ
    def __len__(self):
        if self._grid.nDims==0: raise ValueError("counts requires a spatial AgentGrid")
        return len(self._grid)
    def __getitem__(self,key): return _CountsItem(self._grid,key,self._safe)

def _InstallPythonDirectIteration():
    from numba.experimental.jitclass import boxing
    agentTypes={_AgentGridSafe.class_type.instance_type,_AgentGridFast.class_type.instance_type}
    agentSafe={_AgentGridSafe.class_type.instance_type:True,_AgentGridFast.class_type.instance_type:False}
    gridTypes={_PopGridSafe.class_type.instance_type,_PopGridFast.class_type.instance_type,_PDEgridSafe.class_type.instance_type,_PDEgridFast.class_type.instance_type}
    previous=boxing._specialize_box
    patched=set()
    def specializeBox(typ):
        box=previous(typ)
        if box in patched: return box
        if typ in agentTypes:
            safe=agentSafe[typ]
            box.counts=property(lambda self,__safe=safe:_PythonAgentCountsView(self,__safe))
            def materialized(self,out,rad,x,y=-1.0,z=-1.0,exclude=None):
                excludeI=-1 if exclude is None else exclude
                if _ag_add_radius_q(self._ptr,out._ptr,rad,x,y,z,excludeI)<0:
                    raise ValueError("invalid radius or allocation failure")
                return out
            box.Hood=_PythonHood
            box.Box=_PythonBox
            box.AgentsAt=_PythonAgentsAt
            def radius(self,*args,__materialized=materialized,**kwargs):
                if not 2<=len(args)<=4:
                    raise TypeError("AgentsInRadius expects rad,x[,y[,z]]")
                if any(k!="exclude" for k in kwargs):
                    raise TypeError("AgentsInRadius only accepts exclude as a keyword")
                rad=args[0];x=args[1];y=args[2] if len(args)>2 else -1.0;z=args[3] if len(args)>3 else -1.0
                return _PythonRadiusIter(self,__materialized,rad,x,y,z,kwargs.get("exclude"))
            box.AgentsInRadius=radius
            patched.add(box)
        elif typ in gridTypes:
            box.Hood=_PythonHood
            box.Box=_PythonBox
            patched.add(box)
        return box
    boxing._specialize_box=specializeBox
    _GridPython.Hood=_PythonHood
    _GridPython.Box=_PythonBox

_InstallPythonDirectIteration()
