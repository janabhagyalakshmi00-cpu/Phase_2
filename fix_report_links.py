from pathlib import Path
import re

report = Path("results.md")
if not report.exists():
    raise SystemExit("ERROR: results.md not found. Open PowerShell in your project folder.")

text = report.read_text(encoding="utf-8-sig")

# Convert escaped Markdown image parentheses to normal Markdown.
text = re.sub(r'\\\((results/graphs/[^)\s]+\.png)\\\)', r'(\1)', text)

report.write_text(text, encoding="utf-8")
print("Fixed escaped graph links in results.md")
