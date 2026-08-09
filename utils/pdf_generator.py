"""
utils/pdf_generator.py
Builds the recommendation PDF report: timestamp, skin tone, clothing
type/color, OpenRouter recommendations, the embedded QR code, and the
Fal.ai try-on image URL. Runs off the camera thread.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from config import PDF_DIR
from utils.logger import get_logger

logger = get_logger("pdf_generator")


@dataclass
class ReportData:
    timestamp: str
    skin_tone_hex: str
    clothing_type: str
    clothing_color_hex: str
    recommended_colors: List[str] = field(default_factory=list)
    recommended_outfits: List[str] = field(default_factory=list)
    styling_tips: List[str] = field(default_factory=list)
    fal_image_url: Optional[str] = None
    qr_code_path: Optional[Path] = None
    snapshot_path: Optional[Path] = None


def _swatch_cell(hex_color: str):
    """Return a small Table acting as a color swatch next to its hex code."""
    try:
        color = colors.HexColor(hex_color)
    except Exception:
        color = colors.grey
    t = Table([[""]], colWidths=[10 * mm], rowHeights=[6 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    return t


def generate_pdf_report(data: ReportData, filename: Optional[str] = None) -> Path:
    filename = filename or f"report_{uuid.uuid4().hex[:12]}.pdf"
    out_path = PDF_DIR / filename

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=20, spaceAfter=6,
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6)
    body = styles["BodyText"]

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
    )

    story = []
    story.append(Paragraph("Snap Retail Mirror — Style Report", title_style))
    story.append(Paragraph(f"Generated: {data.timestamp}", body))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Detected Attributes", h2))
    attr_table = Table(
        [
            ["Skin Tone", data.skin_tone_hex, _swatch_cell(data.skin_tone_hex)],
            ["Clothing Type", data.clothing_type, ""],
            ["Clothing Color", data.clothing_color_hex, _swatch_cell(data.clothing_color_hex)],
        ],
        colWidths=[45 * mm, 45 * mm, 20 * mm],
    )
    attr_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    story.append(attr_table)

    if data.recommended_colors:
        story.append(Paragraph("Recommended Colors", h2))
        for c in data.recommended_colors:
            story.append(Paragraph(f"&bull; {c}", body))

    if data.recommended_outfits:
        story.append(Paragraph("Recommended Outfits", h2))
        for o in data.recommended_outfits:
            story.append(Paragraph(f"&bull; {o}", body))

    if data.styling_tips:
        story.append(Paragraph("Styling Tips", h2))
        for tip in data.styling_tips:
            story.append(Paragraph(f"&bull; {tip}", body))

    if data.fal_image_url:
        story.append(Paragraph("Virtual Try-On", h2))
        story.append(Paragraph(data.fal_image_url, body))

    if data.snapshot_path and Path(data.snapshot_path).exists():
        try:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Captured Snapshot", h2))
            story.append(RLImage(str(data.snapshot_path), width=70 * mm, height=52.5 * mm))
        except Exception:
            logger.exception("Failed to embed snapshot image in PDF")

    if data.qr_code_path and Path(data.qr_code_path).exists():
        try:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Scan for your Virtual Try-On image", h2))
            story.append(RLImage(str(data.qr_code_path), width=35 * mm, height=35 * mm))
        except Exception:
            logger.exception("Failed to embed QR code in PDF")

    doc.build(story)
    logger.info("PDF report generated at %s", out_path)
    return out_path
