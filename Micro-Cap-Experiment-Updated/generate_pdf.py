"""
generate_pdf.py

Converts a markdown report file to a formatted PDF.

Usage:
    python generate_pdf.py <input.md> <output.pdf>

Example:
    python generate_pdf.py "Weekly Deep Research (MD)/Week 27 Full.md" \
                           "Weekly Deep Research (PDF)/Week 27.pdf"

Requires: fpdf2 (pip install fpdf2)
"""

import re
import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Error: fpdf2 is not installed. Run: pip install fpdf2")
    sys.exit(1)


# ── Markdown parsing helpers ─────────────────────────────────────────────────

def strip_inline(text: str) -> str:
    """Remove inline markdown: bold, italic, inline code, links."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)        # **bold**
    text = re.sub(r"\*(.+?)\*", r"\1", text)             # *italic*
    text = re.sub(r"`(.+?)`", r"\1", text)               # `code`
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)      # [text](url)
    return text


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def is_separator_row(line: str) -> bool:
    return bool(re.match(r"^\|[\s\-:|]+\|", line.strip()))


def parse_table_row(line: str) -> list[str]:
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return [strip_inline(p) for p in parts]


# ── PDF builder ──────────────────────────────────────────────────────────────

class ReportPDF(FPDF):
    MARGIN = 15
    BODY_FONT = "Helvetica"
    MONO_FONT = "Courier"

    def header(self):
        pass  # no running header

    def footer(self):
        self.set_y(-12)
        self.set_font(self.BODY_FONT, "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)


def build_pdf(md_path: Path, pdf_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()

    pdf = ReportPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(ReportPDF.MARGIN, ReportPDF.MARGIN, ReportPDF.MARGIN)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_width = pdf.w - 2 * ReportPDF.MARGIN

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Horizontal rule ───────────────────────────────────────────────
        if re.match(r"^-{3,}$", stripped) or re.match(r"^\*{3,}$", stripped):
            pdf.ln(2)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(ReportPDF.MARGIN, pdf.get_y(), pdf.w - ReportPDF.MARGIN, pdf.get_y())
            pdf.ln(4)
            i += 1
            continue

        # ── Headings ──────────────────────────────────────────────────────
        h_match = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if h_match:
            level = len(h_match.group(1))
            text = strip_inline(h_match.group(2))
            sizes = {1: 16, 2: 13, 3: 11, 4: 10}
            pdf.ln(3 if level > 1 else 5)
            pdf.set_font(ReportPDF.BODY_FONT, "B", sizes.get(level, 10))
            pdf.multi_cell(usable_width, 6, text)
            pdf.ln(1)
            i += 1
            continue

        # ── Tables ────────────────────────────────────────────────────────
        if is_table_row(stripped):
            # Collect all table lines
            table_lines = []
            while i < len(lines) and is_table_row(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1

            rows = [parse_table_row(r) for r in table_lines
                    if not is_separator_row(r)]
            if not rows:
                continue

            cols = len(rows[0])
            col_w = usable_width / cols

            pdf.set_font(ReportPDF.MONO_FONT, "", 7)
            for row_idx, row in enumerate(rows):
                # header row gets bold background
                if row_idx == 0:
                    pdf.set_fill_color(230, 230, 230)
                    fill = True
                else:
                    pdf.set_fill_color(255, 255, 255)
                    fill = row_idx % 2 == 0

                # compute row height
                row_h = 5
                for cell in row:
                    lines_needed = max(1, len(cell) // int(col_w / 2) + 1)
                    row_h = max(row_h, lines_needed * 4)

                for cell in row:
                    pdf.multi_cell(col_w, 4, cell[:80], border=1,
                                   fill=fill, ln=3, max_line_height=4)
                pdf.ln(row_h)
            pdf.ln(2)
            continue

        # ── List items ────────────────────────────────────────────────────
        list_match = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if list_match:
            indent = len(list_match.group(1))
            text = strip_inline(list_match.group(2))
            pdf.set_font(ReportPDF.BODY_FONT, "", 9)
            x_offset = ReportPDF.MARGIN + indent * 2
            pdf.set_x(x_offset)
            bullet = "\u2022 " if indent == 0 else "\u2013 "
            pdf.multi_cell(usable_width - indent * 2, 5, bullet + text)
            i += 1
            continue

        # ── Numbered list items ───────────────────────────────────────────
        num_match = re.match(r"^\d+\.\s+(.*)", stripped)
        if num_match:
            text = strip_inline(num_match.group(1))
            pdf.set_font(ReportPDF.BODY_FONT, "", 9)
            pdf.multi_cell(usable_width, 5, f"  {text}")
            i += 1
            continue

        # ── Code blocks ───────────────────────────────────────────────────
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ```
            if code_lines:
                pdf.set_font(ReportPDF.MONO_FONT, "", 7.5)
                pdf.set_fill_color(245, 245, 245)
                for cl in code_lines:
                    pdf.multi_cell(usable_width, 4, cl[:120], fill=True)
                pdf.ln(2)
            continue

        # ── Blank line ────────────────────────────────────────────────────
        if not stripped:
            pdf.ln(3)
            i += 1
            continue

        # ── Regular paragraph ─────────────────────────────────────────────
        text = strip_inline(stripped)
        pdf.set_font(ReportPDF.BODY_FONT, "", 9)
        pdf.multi_cell(usable_width, 5, text)
        i += 1

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(pdf_path))
    print(f"PDF written to: {pdf_path}")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python generate_pdf.py <input.md> <output.pdf>")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])

    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        sys.exit(1)

    build_pdf(md_path, pdf_path)


if __name__ == "__main__":
    main()
