import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "NativeCore.py"

COMMON_SPATIAL = {
    "__len__", "xDim", "yDim", "zDim", "nDims", "wrapX", "wrapY", "wrapZ",
    "Box", "Hood", "ToI", "ItoX", "ItoY", "ItoZ", "InWrapX", "InWrapY", "InWrapZ",
}


def _protocol_methods(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == f"_{name}Protocol")
    return {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}


def test_spatial_protocols_expose_common_geometry_api():
    for name in ("Grid", "PopGrid", "PDEgrid", "AgentGrid"):
        missing = COMMON_SPATIAL - _protocol_methods(name)
        assert not missing, f"{name} Protocol is missing common spatial API: {sorted(missing)}"


PENDING_STATE = {"__getitem__", "__setitem__", "Add", "Update", "Reset"}
AGENT_POSITION = {
    "NewAgentSQ", "NewAgent", "Dispose", "I", "XSQ", "YSQ", "ZSQ", "X", "Y", "Z",
    "InWrapSQX", "InWrapSQY", "InWrapSQZ", "DispWrapX", "DispWrapY", "DispWrapZ",
    "LastAgent", "MoveSQ", "Move", "AgentsAt", "AgentsInRadius", "All",
}


def test_pending_state_protocols_expose_shared_update_api():
    for name in ("PopGrid", "PDEgrid"):
        missing = PENDING_STATE - _protocol_methods(name)
        assert not missing, f"{name} Protocol is missing pending-state API: {sorted(missing)}"


def test_agent_protocol_exposes_position_and_query_api():
    missing = AGENT_POSITION - _protocol_methods("AgentGrid")
    assert not missing, f"AgentGrid Protocol is missing position/query API: {sorted(missing)}"


VISUAL_WINDOW = {"IsOpen", "Update", "Save", "StartGif", "AddGifFrame", "StopGif", "Close"}
OPENGL_DRAW = {"Clear", "Circle", "Box", "BoxSQ", "Line"}


def _class_methods(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(classes) == 1, f"Expected exactly one {name} declaration"
    return {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}


def test_visualization_protocols_expose_output_lifecycle():
    pix_methods = _class_methods(ROOT / "PixWindow.py", "PixWindow")
    gl_methods = _class_methods(ROOT / "OpenGLWindow.py", "OpenGLWindow")
    for name, methods in (("PixWindow", pix_methods), ("OpenGLWindow", gl_methods)):
        missing = VISUAL_WINDOW - methods
        assert not missing, f"{name} is missing output lifecycle API: {sorted(missing)}"


def test_opengl_draw_exposes_compiled_primitives():
    missing = OPENGL_DRAW - _class_methods(ROOT / "OpenGLWindow.py", "OpenGLDraw")
    assert not missing, f"OpenGLDraw is missing compiled primitive API: {sorted(missing)}"
