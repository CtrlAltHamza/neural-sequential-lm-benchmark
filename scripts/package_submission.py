"""
Automated Submission Packaging Script for Assignment 2.
Student: Hamza Khurram (i210735)

Creates: i210735_Hamza_Khurram_A2.zip containing:
- notebooks/
- report.pdf
- README.txt
- README.md
- requirements.txt
- src/
- configs/
- saved_models/
- reports/
"""

import os
import zipfile
from pathlib import Path

def package_submission(zip_filename="i210735_Hamza_Khurram_A2.zip"):
    base_path = Path(".").resolve()
    zip_path = base_path / zip_filename

    # Files and folders to include
    include_paths = [
        "README.txt",
        "README.md",
        "report.pdf",
        "requirements.txt",
        "main.py",
        "configs",
        "notebooks",
        "src",
        "saved_models",
        "experiments",
        "reports"
    ]

    # Exclusions
    exclude_extensions = {".pyc", ".pyo", ".git", ".log"}
    exclude_dirs = {"__pycache__", ".venv", "venv", ".git", ".idea", ".vscode", "raw"}

    print(f"Creating submission zip: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in include_paths:
            p = base_path / item
            if not p.exists():
                print(f"  Warning: {item} not found, skipping.")
                continue

            if p.is_file():
                zipf.write(p, arcname=p.relative_to(base_path))
                print(f"  Added file: {p.relative_to(base_path)}")
            elif p.is_dir():
                for root, dirs, files in os.walk(p):
                    # Filter excluded directories in-place
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    for f in files:
                        file_path = Path(root) / f
                        if file_path.suffix.lower() in exclude_extensions:
                            continue
                        rel_path = file_path.relative_to(base_path)
                        zipf.write(file_path, arcname=rel_path)
                print(f"  Added directory: {item}/")

    print(f"\n[SUBMISSION ZIP READY] {zip_path.resolve()} ({zip_path.stat().st_size:,} bytes)\n")
    return str(zip_path.resolve())

if __name__ == "__main__":
    package_submission()
