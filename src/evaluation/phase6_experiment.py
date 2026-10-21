"""
Phase 6: Multi-Style NLP Adaptation & Generalization Gap Analysis.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Evaluates Austen+Shakespeare trained models on modern journalistic BBC News.
Calculates the Generalization Gap (Delta Perplexity) and per-category degradation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize

from src.models.rnn import SimpleRNNLM
from src.models.lstm import LSTMLM
from src.data.tokenizer import Tokenizer
from src.data.dataset import NextWordDataset
from src.evaluation.metrics import evaluate_model

CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']

def run_phase6_adaptation(
    bbc_dir: str = "data/processed/bbc",
    tokenizer_path: str = "data/processed/vocab.json",
    primary_test_results_path: str = "experiments/phase3_hyperparameters/phase3_results.json",
    best_rnn_path: str = "saved_models/best_rnn.pt",
    best_lstm_path: str = "saved_models/best_lstm.pt",
    k: int = 20,
    batch_size: int = 128,
    max_eval_samples: int = 15000,
    device: str = "cpu",
    output_dir: str = "experiments/phase6_adaptation",
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    fig_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n=======================================================")
    print(f" PHASE 6: MULTI-STYLE NLP ADAPTATION (BBC NEWS)")
    print(f" Source Domain: 18th/19th C. Literature & Drama")
    print(f" Target Domain: 21st C. Modern Journalistic Prose")
    print(f"=======================================================\n")

    tokenizer = Tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size

    # Load baseline test perplexities from Phase 3
    with open(primary_test_results_path, "r", encoding="utf-8") as f:
        phase3_meta = json.load(f)
    primary_rnn_test_ppl = phase3_meta["best_rnn"]["test_ppl"]
    primary_lstm_test_ppl = phase3_meta["best_lstm"]["test_ppl"]

    # Load Models
    rnn_ckpt = torch.load(best_rnn_path, map_location=dev)
    lstm_ckpt = torch.load(best_lstm_path, map_location=dev)

    rnn_model = SimpleRNNLM(
        vocab_size,
        embedding_dim=rnn_ckpt["embedding_dim"],
        hidden_dim=rnn_ckpt["config"]["hidden"],
        num_layers=rnn_ckpt["config"]["layers"],
        dropout=rnn_ckpt["config"]["dropout"]
    ).to(dev)
    rnn_model.load_state_dict(rnn_ckpt["state_dict"])
    rnn_model.eval()

    lstm_model = LSTMLM(
        vocab_size,
        embedding_dim=lstm_ckpt["embedding_dim"],
        hidden_dim=lstm_ckpt["config"]["hidden"],
        num_layers=lstm_ckpt["config"]["layers"],
        dropout=lstm_ckpt["config"]["dropout"]
    ).to(dev)
    lstm_model.load_state_dict(lstm_ckpt["state_dict"])
    lstm_model.eval()

    # Ingest BBC Articles by Category
    bbc_root = Path(bbc_dir)
    category_sentences = {cat: [] for cat in CATEGORIES}
    all_bbc_sentences = []

    print("[DATA] Loading cleaned BBC News articles...")
    for cat in CATEGORIES:
        cat_path = bbc_root / cat
        for txt_file in cat_path.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for line in content.splitlines():
                line_str = line.strip()
                if line_str:
                    sents = sent_tokenize(line_str)
                    for s in sents:
                        s_clean = s.strip()
                        if len(s_clean) > 2 and any(c.isalnum() for c in s_clean):
                            category_sentences[cat].append(s_clean)
                            all_bbc_sentences.append(s_clean)

    print(f"  -> Total BBC Sentences Extracted: {len(all_bbc_sentences):,}")

    # Compute BBC UNK rate
    total_tokens, unk_tokens, bbc_unk_rate = tokenizer.compute_unk_rate(all_bbc_sentences)
    print(f"[VOCABULARY DRIFT] BBC News OOV / <UNK> Rate: {unk_tokens:,} / {total_tokens:,} ({bbc_unk_rate:.2f}%)")
    print(f"  (Comparison: In-domain Primary Test <UNK> Rate was ~0.22%)\n")

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 1. Overall BBC Evaluation
    def build_dataset_from_sents(sents, max_s):
        X, y = [], []
        for s in sents:
            toks = tokenizer.sentence_to_tokens(s)
            if len(toks) <= k:
                continue
            enc = tokenizer.encode(toks)
            xs, ys = tokenizer.create_sliding_windows(enc, k)
            X.extend(xs)
            y.extend(ys)
            if len(y) >= max_s:
                X = X[:max_s]
                y = y[:max_s]
                break
        return NextWordDataset(X, y)

    overall_ds = build_dataset_from_sents(all_bbc_sentences, max_eval_samples)
    overall_loader = DataLoader(overall_ds, batch_size=batch_size, shuffle=False)

    _, rnn_bbc_ppl, rnn_bbc_acc = evaluate_model(rnn_model, overall_loader, criterion, dev)
    _, lstm_bbc_ppl, lstm_bbc_acc = evaluate_model(lstm_model, overall_loader, criterion, dev)

    delta_rnn_ppl = rnn_bbc_ppl - primary_rnn_test_ppl
    delta_lstm_ppl = lstm_bbc_ppl - primary_lstm_test_ppl
    rel_rnn_gap = (delta_rnn_ppl / primary_rnn_test_ppl) * 100.0
    rel_lstm_gap = (delta_lstm_ppl / primary_lstm_test_ppl) * 100.0

    print("=======================================================")
    print("        GENERALIZATION GAP EVALUATION RESULTS         ")
    print("=======================================================")
    print(f"Simple RNN:")
    print(f"  - Primary In-Domain Test Perplexity:  {primary_rnn_test_ppl:.2f}")
    print(f"  - BBC Out-of-Domain Test Perplexity:   {rnn_bbc_ppl:.2f}")
    print(f"  - Absolute Generalization Gap (Delta PP):  +{delta_rnn_ppl:.2f}")
    print(f"  - Relative Perplexity Increase:       +{rel_rnn_gap:.1f}%\n")

    print(f"LSTM:")
    print(f"  - Primary In-Domain Test Perplexity:  {primary_lstm_test_ppl:.2f}")
    print(f"  - BBC Out-of-Domain Test Perplexity:   {lstm_bbc_ppl:.2f}")
    print(f"  - Absolute Generalization Gap (Delta PP):  +{delta_lstm_ppl:.2f}")
    print(f"  - Relative Perplexity Increase:       +{rel_lstm_gap:.1f}%")
    print("=======================================================\n")

    # 2. Per-Category Breakdown
    category_results = {}
    print("[CATEGORY BREAKDOWN] Evaluating across 5 BBC news topics:")
    for cat in CATEGORIES:
        cat_ds = build_dataset_from_sents(category_sentences[cat], max_s=3000)
        cat_loader = DataLoader(cat_ds, batch_size=batch_size, shuffle=False)

        _, r_ppl, _ = evaluate_model(rnn_model, cat_loader, criterion, dev)
        _, l_ppl, _ = evaluate_model(lstm_model, cat_loader, criterion, dev)

        category_results[cat] = {
            "rnn_ppl": round(r_ppl, 2),
            "lstm_ppl": round(l_ppl, 2),
            "samples": len(cat_ds)
        }
        print(f"  - {cat.capitalize():<13}: RNN PP = {r_ppl:.2f} | LSTM PP = {l_ppl:.2f} ({len(cat_ds):,} samples)")

    # 3. Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left: In-Domain vs Out-of-Domain Comparison
    domains = ["In-Domain (Primary)", "Out-of-Domain (BBC News)"]
    x = np.arange(len(domains))
    rnn_bars = [primary_rnn_test_ppl, rnn_bbc_ppl]
    lstm_bars = [primary_lstm_test_ppl, lstm_bbc_ppl]

    axes[0].bar(x - 0.2, rnn_bars, width=0.4, label="Simple RNN", color="#d9534f")
    axes[0].bar(x + 0.2, lstm_bars, width=0.4, label="LSTM", color="#0275d8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domains, fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Perplexity", fontsize=12, fontweight='bold')
    axes[0].set_title("The Generalization Gap:\nIn-Domain vs. Out-of-Domain Performance", fontsize=12, fontweight='bold', pad=10)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # Right: Per Category Perplexity on BBC
    cat_names = [c.capitalize() for c in CATEGORIES]
    x_cat = np.arange(len(cat_names))
    r_cat_ppl = [category_results[c]["rnn_ppl"] for c in CATEGORIES]
    l_cat_ppl = [category_results[c]["lstm_ppl"] for c in CATEGORIES]

    axes[1].bar(x_cat - 0.2, r_cat_ppl, width=0.4, label="Simple RNN", color="#d9534f")
    axes[1].bar(x_cat + 0.2, l_cat_ppl, width=0.4, label="LSTM", color="#0275d8")
    axes[1].set_xticks(x_cat)
    axes[1].set_xticklabels(cat_names, fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Perplexity", fontsize=12, fontweight='bold')
    axes[1].set_title("BBC News Perplexity Breakdown by Category", fontsize=12, fontweight='bold', pad=10)
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    chart_path = fig_path / "phase6_domain_adaptation.png"
    plt.savefig(chart_path)
    plt.close()

    results = {
        "primary_rnn_test_ppl": primary_rnn_test_ppl,
        "primary_lstm_test_ppl": primary_lstm_test_ppl,
        "bbc_rnn_ppl": round(rnn_bbc_ppl, 2),
        "bbc_lstm_ppl": round(lstm_bbc_ppl, 2),
        "delta_rnn_ppl": round(delta_rnn_ppl, 2),
        "delta_lstm_ppl": round(delta_lstm_ppl, 2),
        "rel_rnn_gap_pct": round(rel_rnn_gap, 2),
        "rel_lstm_gap_pct": round(rel_lstm_gap, 2),
        "bbc_unk_rate_pct": round(bbc_unk_rate, 2),
        "category_results": category_results
    }

    with open(out_path / "phase6_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PHASE 6 COMPLETED] Results saved to {out_path} and plot saved to {chart_path}\n")
    return results
