"""Regenerate Documentation/API_REFERENCE.md from NativeCore.py without importing PAL.

Usage: python Documentation/generate_api_reference.py
       python Documentation/generate_api_reference.py --check
"""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "NativeCore.py"
TARGET = ROOT / "Documentation" / "API_REFERENCE.md"
FACTORIES = ("FastMode", "NewIList", "MooreHood", "VonNeumannHood", "CircleHood",
             "NewMultinomial", "NewGrid", "NewAgentGrid", "NewPopGrid", "NewPDEgrid")
PROTOCOLS = ("Multinomial", "IList", "AgentGrid", "Grid", "PopGrid", "PDEgrid")
VISUALIZATION = (("PixWindow.py", "PixWindow", "StartPixWindow"),
                 ("OpenGLWindow.py", "OpenGLWindow", "StartOpenGLWindow"))


def signature(node):
    """Render a function signature without its decorators or body."""
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({args}){returns}"


def render(source):
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    lines = [
        "# PAL API Reference", "",
        "Generated from public `Protocol` declarations and top-level factory signatures in `NativeCore.py`. "
        "Regenerate with `python Documentation/generate_api_reference.py`; "
        "use `--check` to detect drift. These declarations describe the annotation/autocomplete surface; "
        "consult [API Guide](API_GUIDE.md) for behavior and the implementation/tests for runtime semantics.",
        "", "## Factories and utilities", "", "```python",
    ]
    for name in FACTORIES:
        lines.append(signature(functions[name]))
    lines.extend(["```", ""])
    for name in PROTOCOLS:
        cls = classes["_" + name + "Protocol"]
        lines.extend([f"## {name}", "", "| Kind | Declaration |", "| --- | --- |"])
        for member in cls.body:
            if not isinstance(member, ast.FunctionDef):
                continue
            kind = "property" if any(isinstance(d, ast.Name) and d.id == "property"
                                     for d in member.decorator_list) else "method"
            sig = signature(member).replace("|", "\\|")
            lines.append(f"| {kind} | `{sig}` |")
        lines.append("")
    for filename, protocol, factory in VISUALIZATION:
        visual_tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        visual_functions = {n.name: n for n in visual_tree.body if isinstance(n, ast.FunctionDef)}
        visual_classes = {n.name: n for n in visual_tree.body if isinstance(n, ast.ClassDef)}
        lines.extend([f"## {protocol} visualization", "", "```python",
                      signature(visual_functions[factory]), "```", "",
                      "| Kind | Declaration |", "| --- | --- |"])
        seen = set()
        for member in visual_classes[protocol].body:
            if not isinstance(member, ast.FunctionDef) or member.name.startswith("_") or member.name in seen:
                continue
            seen.add(member.name)
            lines.append(f"| method | `{signature(member).replace('|', chr(92) + '|')}` |")
        lines.append("")
    pix_tree = ast.parse((ROOT / "PixWindow.py").read_text(encoding="utf-8"))
    pix_class = next(n for n in pix_tree.body if isinstance(n, ast.ClassDef) and n.name == "Pix")
    lines.extend(["## Pix drawing buffer", "", "| Kind | Declaration |", "| --- | --- |"])
    for member in pix_class.body:
        if isinstance(member, ast.FunctionDef) and (not member.name.startswith("_") or
                member.name in ("__setitem__", "__len__")):
            lines.append(f"| method | `{signature(member)}` |")
    lines.append("")
    gl_tree = ast.parse((ROOT / "OpenGLWindow.py").read_text(encoding="utf-8"))
    draw_classes = [n for n in ast.walk(gl_tree) if isinstance(n, ast.ClassDef) and n.name == "OpenGLDraw"]
    if len(draw_classes) != 1:
        raise ValueError("Expected exactly one OpenGLDraw declaration")
    lines.extend(["## OpenGLDraw primitive collector", "", "| Kind | Declaration |", "| --- | --- |"])
    for member in draw_classes[0].body:
        if isinstance(member, ast.FunctionDef) and (
                member.name in ("Clear", "Circle", "Box", "BoxSQ", "Line")):
            lines.append(f"| method | `{signature(member)}` |")
    lines.append("")
    lines.extend([
        "## Scope and validation", "",
        "This reference lists public Protocol signatures, not internal safe/fast jitclass methods. "
        "A signature does not capture every runtime overload, indexing form, or validation rule. "
        "See the API Guide and executable tests. Visualization constructors and window Protocol methods are included; "
        "Pix buffer and OpenGLDraw method signatures are also included; consult the API Guide for behavior.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    args = parser.parse_args()
    result = render(SOURCE.read_text(encoding="utf-8"))
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != result:
            import difflib
            current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
            diff = difflib.unified_diff(current.splitlines(True), result.splitlines(True), fromfile="committed", tofile="generated")
            print("".join(diff))
            parser.exit(1, "API reference is stale; run Documentation/generate_api_reference.py\n")
        print("API reference is current")
    else:
        TARGET.write_text(result, encoding="utf-8")
        print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
