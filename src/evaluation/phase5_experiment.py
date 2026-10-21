"""
Phase 5: Perplexity and Confidence Analysis.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Implements:
1. Exact sentence-level perplexity for Best RNN and Best LSTM.
2. Equal-sized tertiles stratification (Short, Medium, Long).
3. OLS Linear Regression predicting sentence log-perplexity from length & rare words share.
4. Reports R² and RMSE on 30% test split.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch

from src.models.rnn import SimpleRNNLM
from src.models.lstm import LSTMLM
from src.data.tokenizer import Tokenizer
from src.evaluation.metrics import compute_sentence_perplexity
from src.utils.seed import set_seed

def run_phase5_analysis(
    test_sentences: List[str],
    tokenizer: Tokenizer,
    best_rnn_path: str = "saved_models/best_rnn.pt",
    best_lstm_path: str = "saved_models/best_lstm.pt",
    rare_threshold: int = 50,
    regression_train_ratio: float = 0.70,
    max_eval_sentences: int = 1500,
    device: str = "cpu",
    output_dir: str = "experiments/phase5_perplexity_analysis",
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    fig_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n=======================================================")
    print(f" PHASE 5: PERPLEXITY AND CONFIDENCE ANALYSIS")
    print(f" Device: {dev} | Rare threshold: < {rare_threshold} counts")
    print(f"=======================================================\n")

    # 1. Load Models
    rnn_ckpt = torch.load(best_rnn_path, map_location=dev)
    lstm_ckpt = torch.load(best_lstm_path, map_location=dev)

    vocab_size = tokenizer.vocab_size
    k = rnn_ckpt.get("k", 20)

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

    # Identify rare tokens (frequency < 50 in train split, plus UNK)
    rare_token_ids = set()
    for token, idx in tokenizer.word2idx.items():
        count = tokenizer.word_counts.get(token, 0)
        if count < rare_threshold or idx == tokenizer.unk_id:
            rare_token_ids.add(idx)

    print(f"[VOCAB] Rare words identified (< {rare_threshold} occurrences or <UNK>): {len(rare_token_ids):,} tokens")

    # 2. Compute sentence-level perplexities
    records = []
    eval_subset = test_sentences[:max_eval_sentences]
    print(f"[ANALYSIS] Computing sentence perplexities for {len(eval_subset):,} test sentences...")

    for idx, sent in enumerate(eval_subset):
        tokens = tokenizer.sentence_to_tokens(sent)
        if len(tokens) <= 2:
            continue
        encoded = tokenizer.encode(tokens)
        predicted_tokens = encoded[1:]  # tokens from w_1 to </s>
        n_predicted = len(predicted_tokens)

        # Rare word share
        rare_count = sum(1 for tid in predicted_tokens if tid in rare_token_ids)
        rare_share = rare_count / n_predicted

        rnn_ppl, rnn_log_ppl, _ = compute_sentence_perplexity(rnn_model, encoded, k, dev)
        lstm_ppl, lstm_log_ppl, _ = compute_sentence_perplexity(lstm_model, encoded, k, dev)

        records.append({
            "sentence": sent,
            "length": n_predicted,
            "rare_share": rare_share,
            "rnn_ppl": rnn_ppl,
            "rnn_log_ppl": rnn_log_ppl,
            "lstm_ppl": lstm_ppl,
            "lstm_log_ppl": lstm_log_ppl
        })

    df = pd.DataFrame(records)
    print(f"  -> Successfully computed metrics for {len(df):,} valid sentences.")

    # 3. Tertiles Stratification (Short, Medium, Long)
    # Split into 3 equal sized groups by length
    df["tertile"] = pd.qcut(df["length"], q=3, labels=["Short", "Medium", "Long"])
    tertile_stats = df.groupby("tertile", observed=False).agg({
        "length": ["min", "max", "mean"],
        "rnn_ppl": ["mean", "median", "std"],
        "lstm_ppl": ["mean", "median", "std"]
    }).to_dict()

    print("\n[TERTILES ANALYSIS] Sentence Length Tertile Comparison:")
    for group in ["Short", "Medium", "Long"]:
        subset = df[df["tertile"] == group]
        l_min, l_max = subset["length"].min(), subset["length"].max()
        rnn_m = subset["rnn_ppl"].mean()
        lstm_m = subset["lstm_ppl"].mean()
        print(f"  - Group {group:<6} (Length {l_min}-{l_max}): RNN Mean PP = {rnn_m:.2f} | LSTM Mean PP = {lstm_m:.2f}")

    # Plot Tertiles Bar Chart
    plt.figure(figsize=(8, 5), dpi=300)
    groups = ["Short", "Medium", "Long"]
    x = np.arange(len(groups))
    rnn_means = [df[df["tertile"] == g]["rnn_ppl"].mean() for g in groups]
    lstm_means = [df[df["tertile"] == g]["lstm_ppl"].mean() for g in groups]

    plt.bar(x - 0.2, rnn_means, width=0.4, label="Simple RNN", color="#d9534f")
    plt.bar(x + 0.2, lstm_means, width=0.4, label="LSTM", color="#0275d8")
    plt.xticks(x, groups, fontsize=11, fontweight='bold')
    plt.ylabel("Mean Sentence Perplexity", fontsize=12, fontweight='bold')
    plt.title("Phase 5: Perplexity by Sentence Length Groups (Tertiles)", fontsize=13, fontweight='bold', pad=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    tertile_plot = fig_path / "phase5_sentence_length_tertiles.png"
    plt.savefig(tertile_plot)
    plt.close()

    # 4. Linear Regression Modeling
    # Fit regression predicting sentence log-perplexity from length and rare word share
    X = df[["length", "rare_share"]].values
    X_train, X_test, y_train_rnn, y_test_rnn = train_test_split(
        X, df["rnn_log_ppl"].values, train_size=regression_train_ratio, random_state=42
    )
    _, _, y_train_lstm, y_test_lstm = train_test_split(
        X, df["lstm_log_ppl"].values, train_size=regression_train_ratio, random_state=42
    )

    # Fit RNN Regression
    reg_rnn = LinearRegression()
    reg_rnn.fit(X_train, y_train_rnn)
    pred_rnn = reg_rnn.predict(X_test)
    r2_rnn = r2_score(y_test_rnn, pred_rnn)
    rmse_rnn = np.sqrt(mean_squared_error(y_test_rnn, pred_rnn))

    # Fit LSTM Regression
    reg_lstm = LinearRegression()
    reg_lstm.fit(X_train, y_train_lstm)
    pred_lstm = reg_lstm.predict(X_test)
    r2_lstm = r2_score(y_test_lstm, pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, pred_lstm))

    print(f"\n[REGRESSION RESULTS] Predicting Sentence Log-Perplexity (70% Fit / 30% Test):")
    print(f"  Simple RNN:")
    print(f"    - Formula: ln(PP) = {reg_rnn.intercept_:.3f} + ({reg_rnn.coef_[0]:.4f} * length) + ({reg_rnn.coef_[1]:.4f} * rare_share)")
    print(f"    - Test R²:   {r2_rnn:.4f}")
    print(f"    - Test RMSE: {rmse_rnn:.4f}")

    print(f"  LSTM:")
    print(f"    - Formula: ln(PP) = {reg_lstm.intercept_:.3f} + ({reg_lstm.coef_[0]:.4f} * length) + ({reg_lstm.coef_[1]:.4f} * rare_share)")
    print(f"    - Test R²:   {r2_lstm:.4f}")
    print(f"    - Test RMSE: {rmse_lstm:.4f}")

    # Plot Actual vs. Predicted Log-Perplexity
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    axes[0].scatter(y_test_rnn, pred_rnn, alpha=0.4, color="#d9534f", edgecolors='none')
    axes[0].plot([y_test_rnn.min(), y_test_rnn.max()], [y_test_rnn.min(), y_test_rnn.max()], 'k--', lw=2)
    axes[0].set_title(f"Simple RNN Fit (R² = {r2_rnn:.3f}, RMSE = {rmse_rnn:.3f})", fontweight='bold')
    axes[0].set_xlabel("Actual Log-Perplexity", fontweight='bold')
    axes[0].set_ylabel("Predicted Log-Perplexity", fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].scatter(y_test_lstm, pred_lstm, alpha=0.4, color="#0275d8", edgecolors='none')
    axes[1].plot([y_test_lstm.min(), y_test_lstm.max()], [y_test_lstm.min(), y_test_lstm.max()], 'k--', lw=2)
    axes[1].set_title(f"LSTM Fit (R² = {r2_lstm:.3f}, RMSE = {rmse_lstm:.3f})", fontweight='bold')
    axes[1].set_xlabel("Actual Log-Perplexity", fontweight='bold')
    axes[1].set_ylabel("Predicted Log-Perplexity", fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    reg_plot = fig_path / "phase5_regression_diagnostics.png"
    plt.savefig(reg_plot)
    plt.close()

    results = {
        "tertiles": {
            "short_rnn": float(df[df["tertile"] == "Short"]["rnn_ppl"].mean()),
            "short_lstm": float(df[df["tertile"] == "Short"]["lstm_ppl"].mean()),
            "medium_rnn": float(df[df["tertile"] == "Medium"]["rnn_ppl"].mean()),
            "medium_lstm": float(df[df["tertile"] == "Medium"]["lstm_ppl"].mean()),
            "long_rnn": float(df[df["tertile"] == "Long"]["rnn_ppl"].mean()),
            "long_lstm": float(df[df["tertile"] == "Long"]["lstm_ppl"].mean())
        },
        "regression_rnn": {
            "intercept": float(reg_rnn.intercept_),
            "coef_length": float(reg_rnn.coef_[0]),
            "coef_rare_share": float(reg_rnn.coef_[1]),
            "r2": float(r2_rnn),
            "rmse": float(rmse_rnn)
        },
        "regression_lstm": {
            "intercept": float(reg_lstm.intercept_),
            "coef_length": float(reg_lstm.coef_[0]),
            "coef_rare_share": float(reg_lstm.coef_[1]),
            "r2": float(r2_lstm),
            "rmse": float(rmse_lstm)
        }
    }

    with open(out_path / "phase5_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[PHASE 5 COMPLETED] Saved artifacts and plots to {out_path} and {fig_path}\n")
    return results
