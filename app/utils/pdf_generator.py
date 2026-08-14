"""
PDF Consultation Report Generator using fpdf2.
Includes complete Unicode/Latin-1 text sanitization to eliminate encoding errors.
"""

import tempfile
import unicodedata
from datetime import datetime
from PIL import Image
from fpdf import FPDF


def sanitize_text_for_pdf(text: str) -> str:
    """
    Sanitizes UTF-8 text containing smart quotes, dashes, bullets, and emojis
    into clean, standard Latin-1 text compatible with core FPDF fonts.
    """
    if not text:
        return ""

    # Common Unicode character replacements
    replacements = {
        "\u2018": "'",   # Left single quote
        "\u2019": "'",   # Right single quote / apostrophe
        "\u201c": '"',   # Left double quote
        "\u201d": '"',   # Right double quote
        "\u2014": " - ", # Em dash
        "\u2013": " - ", # En dash
        "\u2022": "-",   # Bullet point
        "\u2026": "...", # Ellipsis
        "\u00a0": " ",   # Non-breaking space
        "\u2264": "<=",  # Less than or equal
        "\u2265": ">=",  # Greater than or equal
        "\u00b1": "+/-", # Plus-minus
        "\u00b0": " deg",# Degree symbol
        "\u2192": "->",  # Right arrow
        "\u2190": "<-",  # Left arrow
    }

    for uni_char, ascii_char in replacements.items():
        text = text.replace(uni_char, ascii_char)

    # Normalize unicode characters (NFKD)
    normalized = unicodedata.normalize("NFKD", text)

    # Encode to latin-1 and ignore any remaining unencodable symbols/emojis
    clean_bytes = normalized.encode("latin-1", "ignore")
    return clean_bytes.decode("latin-1")


class ClinicalReportPDF(FPDF):
    def header(self):
        # Header banner
        self.set_fill_color(24, 43, 73)  # Clinical Dark Navy
        self.rect(0, 0, 210, 24, "F")
        
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "MULTI-AGENT MEDICAL IMAGE ANALYZER", ln=True, align="L")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 4, "Autonomous Clinical Decision Support System", ln=True, align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-18)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(120, 120, 120)
        disclaimer = (
            "NOTICE: This document is an AI-generated clinical consultation decision-support summary. "
            "It is not an FDA/CE cleared primary diagnostic finding and requires evaluation by a licensed physician."
        )
        self.multi_cell(0, 3.5, sanitize_text_for_pdf(disclaimer), align="C")
        self.set_y(-8)
        self.cell(0, 4, f"Page {self.page_no()}", align="R")


def build_clinical_pdf(
    thread_id: str,
    modality: str,
    primary_finding: str,
    confidence: float,
    report_markdown: str,
    original_img: Image.Image,
    heatmap_img: Image.Image
) -> bytes:
    """Renders consultation details and embedded scans into a PDF byte stream."""
    pdf = ClinicalReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Metadata Grid
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(40, 6, "Session Thread ID:", border=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(55, 6, sanitize_text_for_pdf(str(thread_id)), border=0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(35, 6, "Consultation Date:", border=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M UTC"), border=0, ln=True)

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(40, 6, "Target Modality:", border=0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(55, 6, sanitize_text_for_pdf(str(modality)), border=0)

    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(35, 6, "AI Confidence:", border=0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(20, 100, 20)
    pdf.cell(0, 6, f"{confidence:.1f}%", border=0, ln=True)
    pdf.ln(3)

    # Save images to temporary files for FPDF embedding
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_orig, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_heat:
        
        orig_path = f_orig.name
        heat_path = f_heat.name
        
        original_img.convert("RGB").save(orig_path, format="JPEG", quality=90)
        heatmap_img.convert("RGB").save(heat_path, format="JPEG", quality=90)

    img_width = 85
    start_y = pdf.get_y()
    pdf.image(orig_path, x=15, y=start_y, w=img_width)
    pdf.image(heat_path, x=110, y=start_y, w=img_width)

    # Subtitles under images
    pdf.set_y(start_y + 87)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(85, 4, "Figure 1: Original Medical Radiograph", align="C")
    pdf.set_x(110)
    pdf.cell(85, 4, "Figure 2: AI Visual Attention Saliency Map", align="C", ln=True)
    pdf.ln(6)

    # Diagnostic Findings Box
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(210, 215, 225)
    pdf.rect(15, pdf.get_y(), 180, 12, "FD")
    pdf.set_xy(18, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(40, 4, "PRIMARY IMPRESSION:", border=0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(180, 30, 30)
    pdf.cell(0, 4, sanitize_text_for_pdf(str(primary_finding).upper()), border=0, ln=True)
    pdf.ln(8)

    # Structured Observations Body
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Detailed Clinical Synthesis & Observations", ln=True)
    pdf.set_draw_color(24, 43, 73)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    
    # Strip markdown syntax and sanitize all characters for PDF
    clean_report = (
        report_markdown.replace("### ", "\n")
        .replace("## ", "\n")
        .replace("**", "")
        .replace("---", "")
        .replace("📋 ", "")
        .replace("📖 ", "")
        .replace("🔬 ", "")
        .replace("📊 ", "")
        .replace("⚠️ ", "")
        .replace("💡 ", "")
        .strip()
    )

    clean_report = sanitize_text_for_pdf(clean_report)
    pdf.multi_cell(0, 4.5, clean_report)

    return bytes(pdf.output())
