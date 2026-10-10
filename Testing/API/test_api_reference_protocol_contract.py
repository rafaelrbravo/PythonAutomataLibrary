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
