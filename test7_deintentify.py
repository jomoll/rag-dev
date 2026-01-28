import os
import re
from pathlib import Path

FOLDER = Path("/data/interagt/processed/anonymized_patient/0001095314/")  # Change this to your folder path

patterns = [
    r'Horataci',
    r'Ruhi',
    r'10.10.1945',
    r'10-okt-1945',
    r'10-oct-1945',
    r'Brunnenweg',
    r'4',
    r'85748 Garching',
    r'Garching'
]


total_md_files = 0
modified_files = 0

for path in FOLDER.rglob("*.md"):
    if not path.is_file():
        continue
    total_md_files += 1
    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    new_content = content
    for pat in patterns:
        new_content = re.sub(pat, "___", new_content)
    if new_content != content:
        modified_files += 1
        with path.open("w", encoding="utf-8") as f:
            f.write(new_content)

print(f"Processed Markdown files: {total_md_files}")
print(f"Files modified: {modified_files}")
