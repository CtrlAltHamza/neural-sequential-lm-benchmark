"""
Jane Austen Text Cleaning and Normalization Module.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Cleans Project Gutenberg noise, license headers/footers, transcriber artifacts,
and normalizes punctuation and whitespace.
"""

import re
from pathlib import Path

# Noise lines to filter out
GUTENBERG_INDICATORS = [
    "gutenberg",
    "e-text prepared",
    "transcriber's note",
    "transcribers note",
    "online distributed proofreading",
    "www.pgdp.net",
    "project gutenberg",
    "end of project gutenberg",
    "end of the project gutenberg",
    "london: printed for",
    "invented by m",
    "engraved for no.",
    "la belle assemblee",
]

def clean_austen_text(raw_filepath="data/raw/austen/austen.txt", output_filepath="data/processed/austen_clean.txt"):
    in_path = Path(raw_filepath)
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cleaned_lines = []
    removed_count = 0

    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Check if line contains Gutenberg or transcriber artifact
        if any(indicator in line_lower for indicator in GUTENBERG_INDICATORS):
            removed_count += 1
            continue

        # Remove standalone chapter/volume banners that have no sentence structure
        if re.match(r'^(volume\s+[ivxlcdm]+|chapter\s+[ivxlcdm\d]+|admonition|contents)\.?$', line_lower):
            removed_count += 1
            continue

        # Replace non-standard characters like replacement chars () or smart quotes
        line_normalized = (
            line_clean
            .replace("", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", " -- ")
            .replace("–", " - ")
            .replace("_", "")  # Gutenberg italics markers e.g. _them_ -> them
        )

        if line_normalized:
            cleaned_lines.append(line_normalized)

    full_cleaned_text = "\n".join(cleaned_lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_cleaned_text)

    print("\n--- Austen Cleaning Summary ---")
    print(f"Original Lines: {len(lines)}")
    print(f"Noise Lines Removed: {removed_count}")
    print(f"Remaining Paragraphs/Lines: {len(cleaned_lines)}")
    print(f"Saved to: {out_path.resolve()}\n")

    return full_cleaned_text

if __name__ == "__main__":
    clean_austen_text()
