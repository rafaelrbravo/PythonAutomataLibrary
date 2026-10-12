"""Generate Documentation/API_REFERENCE.md and API_REFERENCE.pdf from PAL source without importing PAL.

Usage: python Documentation/generators/generate_api_reference.py
       python Documentation/generators/generate_api_reference.py --check
"""
import argparse
import ast
from pathlib import Path
from generate_pdfs import render as render_pdf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "NativeCore.py"
MD_TARGET = ROOT / "Documentation" / "API_REFERENCE.md"
PDF_TARGET = ROOT / "Documentation" / "API_REFERENCE.pdf"
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
        "Regenerate with `python Documentation/generators/generate_api_reference.py`; "
        "use `--check` to detect drift. These declarations describe the annotation/autocomplete surface; "
        "consult [API Guide](API_GUIDE.pdf) for behavior and the implementation/tests for runtime semantics.",
        "", "## Factories and utilities", "",
        "| Function | Description |", "| --- | --- |",
    ]
    for name in FACTORIES:
        description = ast.get_docstring(functions[name])
        if not description:
            raise ValueError(f"Missing public API docstring: {name}")
        lines.append(f"| `{signature(functions[name])}` | {description.replace(chr(124), chr(92) + chr(124))} |")
    lines.append("")
    for name in PROTOCOLS:
        cls = classes["_" + name + "Protocol"]
        lines.extend([f"## {name}", "", "| Kind | Declaration | Description |", "| --- | --- | --- |"])
        for member in cls.body:
            if not isinstance(member, ast.FunctionDef):
                continue
            kind = "property" if any(isinstance(d, ast.Name) and d.id == "property"
                                     for d in member.decorator_list) else "method"
            sig = signature(member).replace("|", "\\|")
            description = ast.get_docstring(member)
            if not description:
                raise ValueError(f"Missing public API docstring: {name}.{member.name}")
            description = description.replace("|", "\\|").replace("\\n", " ")
            lines.append(f"| {kind} | `{sig}` | {description} |")
        lines.append("")
    for filename, protocol, factory in VISUALIZATION:
        visual_tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        visual_functions = {n.name: n for n in visual_tree.body if isinstance(n, ast.FunctionDef)}
        visual_classes = {n.name: n for n in visual_tree.body if isinstance(n, ast.ClassDef)}
        lines.extend([f"## {protocol} visualization", "", "```python",
                      signature(visual_functions[factory]), "```", "",
                      (ast.get_docstring(visual_functions[factory]) or ""), "",
                      "| Kind | Declaration | Description |", "| --- | --- | --- |"])
        seen = set()
        for member in visual_classes[protocol].body:
            if not isinstance(member, ast.FunctionDef) or member.name.startswith("_") or member.name in seen:
                continue
            seen.add(member.name)
            description = ast.get_docstring(member)
            if not description:
                raise ValueError(f"Missing visualization API docstring: {protocol}.{member.name}")
            description = description.replace("|", "\\|").replace("\\n", " ")
            lines.append(f"| method | `{signature(member).replace('|', chr(92) + '|')}` | {description} |")
        lines.append("")
    pix_tree = ast.parse((ROOT / "PixWindow.py").read_text(encoding="utf-8"))
    pix_class = next(n for n in pix_tree.body if isinstance(n, ast.ClassDef) and n.name == "Pix")
    lines.extend(["The `OpenGLWindow` table follows its source Protocol, which also contains method implementations; repeated declarations are listed once. `OpenGLDraw` is a compiled primitive collector, while the returned window handles rendering and output. Consult the [API Guide](API_GUIDE.pdf) for which object to use in Python and compiled drawing code.", ""])
    lines.extend(["## Pix drawing buffer", "", "| Kind | Declaration | Description |", "| --- | --- | --- |"])
    for member in pix_class.body:
        if isinstance(member, ast.FunctionDef) and (not member.name.startswith("_") or
                member.name in ("__setitem__", "__len__")):
            description = ast.get_docstring(member)
            if not description:
                raise ValueError(f"Missing Pix API docstring: Pix.{member.name}")
            description = description.replace("|", "\\|").replace("\\n", " ")
            lines.append(f"| method | `{signature(member)}` | {description} |")
    lines.append("")
    gl_tree = ast.parse((ROOT / "OpenGLWindow.py").read_text(encoding="utf-8"))
    draw_classes = [n for n in ast.walk(gl_tree) if isinstance(n, ast.ClassDef) and n.name == "OpenGLDraw"]
    if len(draw_classes) != 1:
        raise ValueError("Expected exactly one OpenGLDraw declaration")
    lines.extend(["## OpenGLDraw primitive collector", "", "| Kind | Declaration | Description |", "| --- | --- | --- |"])
    for member in draw_classes[0].body:
        if isinstance(member, ast.FunctionDef) and (
                member.name in ("Clear", "Circle", "Box", "BoxSQ", "Line")):
            description = ast.get_docstring(member) or "Description not yet provided."
            description = description.replace("|", "\\|").replace("\\n", " ")
            lines.append(f"| method | `{signature(member)}` | {description} |")
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
    markdown = render(SOURCE.read_text(encoding="utf-8"))
    markdown = markdown.rstrip() + "\n"
    pdf = render_pdf(markdown, "API_REFERENCE")
    if args.check:
        stale = (not MD_TARGET.exists() or MD_TARGET.read_text(encoding="utf-8") != markdown or
                 not PDF_TARGET.exists() or PDF_TARGET.read_bytes() != pdf)
        if stale:
            parser.exit(1, "Stale or missing API Reference artifacts; run Documentation/generators/generate_api_reference.py\n")
        print("API_REFERENCE.md and API_REFERENCE.pdf are current")
    else:
        MD_TARGET.write_text(markdown, encoding="utf-8")
        PDF_TARGET.write_bytes(pdf)
        print(f"Wrote {MD_TARGET.relative_to(ROOT)} and {PDF_TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
