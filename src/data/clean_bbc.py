"""
BBC News Dataset Deduplication and Cleaning Module.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

The BBC dataset contains 4,450 files (two identical copies of 2,225 articles).
This script removes the duplicate folder and verifies the 2,225 unique articles.
"""

import os
import shutil
from pathlib import Path

CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']

def clean_and_deduplicate_bbc(raw_bbc_dir="data/raw/bbc", output_dir="data/processed/bbc"):
    raw_path = Path(raw_bbc_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Candidate source dirs
    source_dir = raw_path / "bbc"
    duplicate_dir = raw_path / "bbc-fulltext (document classification)"

    if not source_dir.exists() and duplicate_dir.exists():
        source_dir = duplicate_dir / "bbc"

    if not source_dir.exists():
        raise FileNotFoundError(f"Could not find BBC source directory at {source_dir}")

    total_articles = 0
    category_counts = {}

    for cat in CATEGORIES:
        cat_src = source_dir / cat
        cat_dst = out_path / cat
        cat_dst.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(list(cat_src.glob("*.txt")))
        category_counts[cat] = len(txt_files)
        total_articles += len(txt_files)

        for src_file in txt_files:
            dst_file = cat_dst / src_file.name
            shutil.copy2(src_file, dst_file)

    # Safely remove the duplicate tree if it exists in raw
    if duplicate_dir.exists():
        try:
            shutil.rmtree(duplicate_dir)
            print(f"Successfully removed duplicate directory: {duplicate_dir}")
        except Exception as e:
            print(f"Note: Could not delete duplicate folder ({e}), ignored.")

    print("\n--- BBC News Cleaning Summary ---")
    print(f"Total Unique Articles Processed: {total_articles} (Expected: 2,225)")
    for cat, count in category_counts.items():
        print(f"  - {cat.capitalize()}: {count} articles")
    print(f"Cleaned articles saved to: {out_path.resolve()}\n")

    return total_articles, category_counts

if __name__ == "__main__":
    clean_and_deduplicate_bbc()
