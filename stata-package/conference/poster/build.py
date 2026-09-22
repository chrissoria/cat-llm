"""Build poster.html without Quarto.

`quarto render poster.qmd` is the supported path. This script is the
fallback for machines that have pandoc but not Quarto: it strips the YAML
front matter and runs the body through pandoc with the same div/span
extensions Quarto uses, producing an equivalent self-contained HTML file.

    python build.py && python render_pdf.py
"""

import subprocess
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
qmd = base / "poster.qmd"
body = base / "_poster_body.md"
out = base / "poster.html"

text = qmd.read_text()

# Drop the YAML header; everything Quarto would read from it is either
# replicated in poster.css or irrelevant to the fallback render.
if text.startswith("---"):
    text = text.split("---", 2)[2].lstrip("\n")
body.write_text(text)

try:
    import pypandoc

    pandoc = pypandoc.get_pandoc_path()
except Exception:
    pandoc = "pandoc"

cmd = [
    pandoc,
    str(body),
    "-f", "markdown+fenced_divs+bracketed_spans+raw_attribute+superscript",
    "-t", "html5",
    "--standalone",
    "--embed-resources",
    "--css", str(base / "poster.css"),
    "--metadata", "title=Coding Open-Ended Survey Text Without Leaving Stata",
    "-o", str(out),
]
result = subprocess.run(cmd, capture_output=True, text=True, cwd=base)
body.unlink(missing_ok=True)

if result.returncode != 0:
    sys.exit(f"pandoc failed:\n{result.stderr}")

print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
