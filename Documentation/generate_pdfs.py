"""Shared deterministic ReportLab renderer for PAL documentation generators.\n\nThis module is infrastructure; run the dedicated generate_*.py scripts to build documents.\nRequires: pip install reportlab\n"""
import argparse
import io
import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "Documentation"

INK = colors.HexColor("#172438")
ACCENT = colors.HexColor("#245e83")
LIGHT = colors.HexColor("#eef3f7")


def normalize(text):
    return (text.replace("→", "->").replace("–", "-").replace("—", "-")
            .replace("×", "x").replace("§", "section ").replace("≤", "<=")
            .replace("≥", ">=").replace("·", " / ").replace("…", "...")
            .replace("’", "'").replace("“", '"').replace("”", '"'))


def inline(text):
    """Escape markup, then support the small Markdown inline subset used here."""
    text = escape(normalize(text))
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", lambda m: m.group(1), text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.3">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text

def styles(compact=False):
    body = 8.1 if compact else 9.2
    return {
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=body,
            leading=body * 1.38, textColor=INK, spaceAfter=6 if compact else 8),
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=21,
            leading=26, textColor=INK, spaceAfter=15),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12 if compact else 14,
            leading=17, textColor=ACCENT, spaceBefore=13, spaceAfter=7, keepWithNext=True),
        "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10 if compact else 11,
            leading=14, textColor=INK, spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=6.6 if compact else 7.5,
            leading=9 if compact else 10, textColor=INK, backColor=LIGHT,
            leftIndent=8, rightIndent=5, borderPadding=5, spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7.2 if compact else 8,
            leading=10 if compact else 11, textColor=INK),
    }


def blocks(markdown, compact=False):
    st = styles(compact)
    lines = markdown.splitlines()
    story = []
    width = (A4[0] - (25 if compact else 38) * mm)
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(normalize(lines[i]).expandtabs(4))
                i += 1
            # Wrapping is visual only: source snippets remain unchanged.
            visual = []
            limit = 91 if compact else 100
            for item in code:
                visual.extend([item[j:j + limit] for j in range(0, len(item), limit)] or [""])
            story.append(Preformatted("\n".join(visual), st["code"], maxLineLength=limit))
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|\-]+\|$", lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+", c or "-") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                count = max(map(len, rows))
                data = [[Paragraph(inline(row[j] if j < len(row) else ""), st["cell"])
                         for j in range(count)] for row in rows]
                table = Table(data, colWidths=[width / count] * count, repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                    ("LINEBELOW", (0, 0), (-1, 0), .5, ACCENT),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ]))
                story.append(table)
                story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), st["title"]))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), st["h2"]))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), st["h3"]))
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
            lines[i].startswith(("#", "|", "```"))):
            paragraph.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(paragraph)), st["body"]))
    return story


def render(source, name):
    compact = name == "CHEATSHEET"
    buffer = io.BytesIO()
    margin = (12 if compact else 19) * mm
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=margin,
        rightMargin=margin, topMargin=16 * mm, bottomMargin=17 * mm,
        title="PAL " + name.replace("_", " ").title(), author="Python Automata Library",
        pageCompression=1, invariant=1)

    def footer(canvas, document):
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#687687"))
        canvas.drawString(margin, 10 * mm, "Python Automata Library")
        canvas.drawRightString(A4[0] - margin, 10 * mm, str(document.page))

    doc.build(blocks(source, compact), onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


