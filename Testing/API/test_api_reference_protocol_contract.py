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
