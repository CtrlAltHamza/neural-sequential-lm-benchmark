"""
Finalize Benchmark: Serialization, PDF Report Generation, and Packaging.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)
"""

import os
import sys
import shutil
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import torch

from scripts.generate_report_pdf import generate_pdf_report
from scripts.package_submission import package_submission

def finalize():
    print("=" * 70)
    print(" FINALIZING ASSIGNMENT 2 BENCHMARK & ARTIFACTS")
    print(" Student: Hamza Khurram (i210735)")
    print("=" * 70)

    # 1. Joblib Serialization for Pretrained Models
    print("\n[STEP 1/5] Serializing Champion Models to .joblib format...")
    models_dir = Path("saved_models")
    for model_name in ["best_lstm", "best_rnn"]:
        pt_path = models_dir / f"{model_name}.pt"
        joblib_path = models_dir / f"{model_name}.joblib"
        if pt_path.exists():
            data = torch.load(pt_path, map_location="cpu")
            joblib.dump(data, joblib_path, compress=3)
            print(f"  -> Successfully serialized {pt_path.name} to {joblib_path.name} ({joblib_path.stat().st_size:,} bytes)")
        else:
            print(f"  -> WARNING: {pt_path} not found!")

    # 2. Serialize Phase 3 Results to joblib
    p3_json = Path("experiments/phase3_hyperparameters/phase3_results.json")
    p3_joblib = Path("experiments/phase3_hyperparameters/phase3_results.joblib")
    if p3_json.exists():
        p3_data = json.load(open(p3_json, "r", encoding="utf-8"))
        joblib.dump(p3_data, p3_joblib, compress=3)
        print(f"  -> Serialized {p3_json.name} to {p3_joblib.name} ({p3_joblib.stat().st_size:,} bytes)")

    # 3. Generate Formal Report PDF
    print("\n[STEP 2/5] Generating publication-quality formal PDF report...")
    report_pdf_path = generate_pdf_report(
        output_pdf="reports/report.pdf",
        stats_json="data/processed/corpus_stats.json",
        p2_json="experiments/phase2_vanishing_gradient/phase2_results.json",
        p3_json="experiments/phase3_hyperparameters/phase3_results.json",
        p5_json="experiments/phase5_perplexity_analysis/phase5_results.json",
        p6_json="experiments/phase6_adaptation/phase6_results.json",
        figures_dir="reports/figures"
    )
    # Copy to root report.pdf
    shutil.copy("reports/report.pdf", "report.pdf")
    print(f"  -> Synchronized report to root: report.pdf ({Path('report.pdf').stat().st_size:,} bytes)")

    # 4. Package Submission Archive
    print("\n[STEP 3/5] Packaging submission archive (i210735_Hamza_Khurram_A2.zip)...")
    zip_path = package_submission("i210735_Hamza_Khurram_A2.zip")

    # 5. Verify Demo Script
    print("\n[STEP 4/5] Verifying Live Teacher Demo script (scripts/demo.py)...")
    from scripts.demo import run_demo
    run_demo()

    print("\n[STEP 5/5] ALL FINALIZATION STEPS COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    finalize()
