"""Render poster.html to a print-ready 48x36in PDF.

Run `quarto render poster.qmd` first (or `python build.py` if you don't
have Quarto installed), then run this.
"""

import os
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

base = Path(__file__).resolve().parent

# Some managed environments ship a Chromium that doesn't match the version
# Playwright expects. Point at it explicitly via CHROMIUM_PATH if so.
_chromium = os.environ.get("CHROMIUM_PATH")
_launch = {"headless": True}
if _chromium:
    _launch["executable_path"] = _chromium
html_path = base / "poster.html"
pdf_path = base / "poster_poster.pdf"

with sync_playwright() as p:
    browser = p.chromium.launch(**_launch)
    page = browser.new_page()
    page.goto(html_path.as_uri(), wait_until="networkidle")

    page.pdf(
        path=str(pdf_path),
        print_background=True,
        prefer_css_page_size=True,
        margin={"top": "0in", "right": "0in", "bottom": "0in", "left": "0in"},
    )
    browser.close()
    print(f"PDF saved: {pdf_path}")

# Crop to actual content bounds (removes white gap at bottom)
try:
    result = subprocess.run(
        ["pdfcrop", str(pdf_path), str(pdf_path)],
        capture_output=True,
        text=True,
    )
except FileNotFoundError:
    print("pdfcrop not installed — PDF left at full 48x36in (this is fine for print).")
else:
    if result.returncode == 0:
        print("PDF cropped to content bounds.")
    else:
        print(f"pdfcrop failed — PDF left at full 48x36in.\n{result.stderr}")
