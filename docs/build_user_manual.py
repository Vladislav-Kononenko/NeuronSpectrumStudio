from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
SOURCE_MD = DOCS_DIR / "USER_MANUAL.md"
OUTPUT_PDF = DOCS_DIR / "NeuronSpectrum_GUI_Manual_ru.pdf"


def register_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/verdana.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("DocFont", str(path)))
            return "DocFont"
    raise FileNotFoundError("Не найден системный шрифт с поддержкой кириллицы.")


def build_styles(font_name: str):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="DocTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#143a52"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocHeading1",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#0b5d7a"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocHeading2",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#16425b"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=14,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocBullet",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=14,
            leftIndent=12,
            bulletIndent=0,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocCode",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=12,
            backColor=colors.HexColor("#f2f5f7"),
            borderPadding=6,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=5,
        )
    )
    return styles


def flush_paragraph(buffer: list[str], story: list, styles) -> None:
    if not buffer:
        return
    text = " ".join(part.strip() for part in buffer if part.strip())
    if text:
        story.append(Paragraph(text, styles["DocBody"]))
    buffer.clear()


def build_story(text: str, styles) -> list:
    story: list = []
    paragraph_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(paragraph_buffer, story, styles)
            if in_code and code_buffer:
                code_text = "<br/>".join(
                    item.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for item in code_buffer
                )
                story.append(Paragraph(code_text, styles["DocCode"]))
                code_buffer.clear()
            in_code = not in_code
            continue

        if in_code:
            code_buffer.append(line)
            continue

        if stripped.startswith("# "):
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Paragraph(stripped[2:], styles["DocTitle"]))
            story.append(Spacer(1, 4 * mm))
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Paragraph(stripped[3:], styles["DocHeading1"]))
            continue

        if stripped.startswith("### "):
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Paragraph(stripped[4:], styles["DocHeading2"]))
            continue

        if stripped.startswith("- "):
            flush_paragraph(paragraph_buffer, story, styles)
            bullet_text = stripped[2:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(bullet_text, styles["DocBullet"], bulletText="•"))
            continue

        if stripped and stripped[0].isdigit() and ". " in stripped:
            flush_paragraph(paragraph_buffer, story, styles)
            prefix, content = stripped.split(". ", 1)
            bullet_text = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(bullet_text, styles["DocBullet"], bulletText=f"{prefix}."))
            continue

        if not stripped:
            flush_paragraph(paragraph_buffer, story, styles)
            story.append(Spacer(1, 1.5 * mm))
            continue

        safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        paragraph_buffer.append(safe)

    flush_paragraph(paragraph_buffer, story, styles)
    if code_buffer:
        code_text = "<br/>".join(
            item.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for item in code_buffer
        )
        story.append(Paragraph(code_text, styles["DocCode"]))
    return story


def main() -> None:
    font_name = register_font()
    styles = build_styles(font_name)
    source_text = SOURCE_MD.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="NeuronSpectrum GUI — руководство пользователя",
        author="OpenAI Codex",
    )
    doc.build(build_story(source_text, styles))
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
