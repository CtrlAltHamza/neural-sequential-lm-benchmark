"""
Raw dataset extraction utility for Assignment 2.
Student: Hamza Khurram (i210735)

Extracts:
1. archive (1).zip -> austen.txt into data/raw/austen/
2. archive (2).zip -> train.csv, validation.csv, test.csv into data/raw/shakespeare/
3. archive.zip -> bbc articles into data/raw/bbc/
"""

import os
import zipfile
import shutil
from pathlib import Path

def extract_datasets(base_dir="."):
    base_path = Path(base_dir).resolve()
    raw_dir = base_path / "data" / "raw"
    austen_dir = raw_dir / "austen"
    shakespeare_dir = raw_dir / "shakespeare"
    bbc_dir = raw_dir / "bbc"

    for d in [austen_dir, shakespeare_dir, bbc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Extract Austen
    zip_austen = base_path / "archive (1).zip"
    if zip_austen.exists():
        print(f"Extracting {zip_austen.name}...")
        with zipfile.ZipFile(zip_austen, 'r') as z:
            for item in z.namelist():
                if "austen.txt" in item.lower():
                    z.extract(item, austen_dir)
                    extracted_path = austen_dir / item
                    target_path = austen_dir / "austen.txt"
                    if extracted_path != target_path:
                        shutil.move(str(extracted_path), str(target_path))
                    print(f"  Extracted austen.txt ({target_path.stat().st_size} bytes)")
    else:
        print(f"Warning: {zip_austen} not found.")

    # 2. Extract Shakespeare
    zip_shakes = base_path / "archive (2).zip"
    if zip_shakes.exists():
        print(f"Extracting {zip_shakes.name}...")
        with zipfile.ZipFile(zip_shakes, 'r') as z:
            z.extractall(shakespeare_dir)
            print(f"  Extracted Shakespeare CSVs: {os.listdir(shakespeare_dir)}")
    else:
        print(f"Warning: {zip_shakes} not found.")

    # 3. Extract BBC
    zip_bbc = base_path / "archive.zip"
    if zip_bbc.exists():
        print(f"Extracting {zip_bbc.name}...")
        with zipfile.ZipFile(zip_bbc, 'r') as z:
            z.extractall(bbc_dir)
            print(f"  Extracted BBC files into {bbc_dir}")
    else:
        print(f"Warning: {zip_bbc} not found.")

    print("\nDataset extraction complete.")

if __name__ == "__main__":
    extract_datasets()
