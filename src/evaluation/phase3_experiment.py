"""
Phase 3: Hyperparameter & Architecture Research.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Runs at least 9 experiments (including >= 2 Simple RNNs).
Selects Best RNN and Best LSTM strictly on lowest validation perplexity.
Reports test perplexity for every run.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.rnn import SimpleRNNLM
from src.models.lstm import LSTMLM
from src.data.tokenizer import Tokenizer
from src.data.dataset import NextWordDataset
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.utils.seed import set_seed

EXPERIMENT_CONFIGS = [
    {"id": "RNN_L1_H128_D0.0",  "arch": "rnn",  "layers": 1, "hidden": 128, "dropout": 0.0},
    {"id": "RNN_L2_H256_D0.2",  "arch": "rnn",  "layers": 2, "hidden": 256, "dropout": 0.2},
    {"id": "LSTM_L1_H128_D0.0", "arch": "lstm", "layers": 1, "hidden": 128, "dropout": 0.0},
    {"id": "LSTM_L1_H256_D0.0", "arch": "lstm", "layers": 1, "hidden": 256, "dropout": 0.0},
    {"id": "LSTM_L1_H512_D0.2", "arch": "lstm", "layers": 1, "hidden": 512, "dropout": 0.2},
    {"id": "LSTM_L2_H256_D0.0", "arch": "lstm", "layers": 2, "hidden": 256, "dropout": 0.0},
    {"id": "LSTM_L2_H256_D0.2", "arch": "lstm", "layers": 2, "hidden": 256, "dropout": 0.2},
    {"id": "LSTM_L2_H512_D0.5", "arch": "lstm", "layers": 2, "hidden": 512, "dropout": 0.5},
    {"id": "LSTM_L3_H256_D0.2", "arch": "lstm", "layers": 3, "hidden": 256, "dropout": 0.2},
]

def run_phase3_experiments(
    train_sentences: List[str],
    val_sentences: List[str],
    test_sentences: List[str],
    tokenizer: Tokenizer,
    k: int = 20,
    embedding_dim: int = 128,
    lr: float = 0.001,
    batch_size: int = 128,
    epochs: int = 8,
    patience: int = 2,
    max_train_samples: int = 40000,
    max_val_samples: int = 6000,
    max_test_samples: int = 6000,
    device: str = "cpu",
    output_dir: str = "experiments/phase3_hyperparameters",
    models_dir: str = "saved_models",
    figures_dir: str = "reports/figures"
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    models_path = Path(models_dir)
    fig_path = Path(figures_dir)
    for p in [out_path, models_path, fig_path]:
        p.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n=======================================================")
    print(f" PHASE 3: HYPERPARAMETER & ARCHITECTURE SEARCH")
    print(f" Total Experiments: {len(EXPERIMENT_CONFIGS)} | Window k={k}")
    print(f" Device: {dev} | Early Stopping Patience: {patience}")
    print(f"=======================================================\n")

    vocab_size = tokenizer.vocab_size

    # Prepare datasets with fixed k=20
    def build_subset_dataset(sentences, max_s):
        X, y = [], []
        for s in sentences:
            tokens = tokenizer.sentence_to_tokens(s)
            if len(tokens) <= k:
                continue
            enc = tokenizer.encode(tokens)
            xs, ys = tokenizer.create_sliding_windows(enc, k)
            X.extend(xs)
            y.extend(ys)
            if len(y) >= max_s:
                X = X[:max_s]
                y = y[:max_s]
                break
        return NextWordDataset(X, y)

    print("[DATA] Generating sliding windows for k=20...")
    train_ds = build_subset_dataset(train_sentences, max_train_samples)
    val_ds = build_subset_dataset(val_sentences, max_val_samples)
    test_ds = build_subset_dataset(test_sentences, max_test_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,} samples\n")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    results = []

    best_rnn_val_ppl = float("inf")
    best_rnn_model = None
    best_rnn_config = None

    best_lstm_val_ppl = float("inf")
    best_lstm_model = None
    best_lstm_config = None

    for idx, cfg in enumerate(EXPERIMENT_CONFIGS, 1):
        set_seed(42)
        exp_id = cfg["id"]
        arch = cfg["arch"]
        layers = cfg["layers"]
        hidden = cfg["hidden"]
        drop = cfg["dropout"]

        print(f"--- [Run {idx}/{len(EXPERIMENT_CONFIGS)}] {exp_id} ({arch.upper()}, Layers={layers}, Hidden={hidden}, Dropout={drop}) ---")

        if arch == "rnn":
            model = SimpleRNNLM(vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden, num_layers=layers, dropout=drop)
        else:
            model = LSTMLM(vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden, num_layers=layers, dropout=drop)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(model, opt, criterion, dev, patience=patience, model_name=exp_id)
        history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=True)

        val_loss, val_ppl, val_acc = evaluate_model(model, val_loader, criterion, dev)
        test_loss, test_ppl, test_acc = evaluate_model(model, test_loader, criterion, dev)

        record = {
            "id": exp_id,
            "arch": arch,
            "layers": layers,
            "hidden": hidden,
            "dropout": drop,
            "stopped_epoch": history["stopped_epoch"],
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 2),
            "val_acc": round(val_acc, 2),
            "test_loss": round(test_loss, 4),
            "test_ppl": round(test_ppl, 2),
            "test_acc": round(test_acc, 2),
            "history": history
        }
        results.append(record)

        print(f"  ==> Result: Val PP = {val_ppl:.2f} | Val Acc = {val_acc:.2f}% | Test PP = {test_ppl:.2f} | Test Acc = {test_acc:.2f}%\n")

        # Track best models solely on lowest validation perplexity
        if arch == "rnn" and val_ppl < best_rnn_val_ppl:
            best_rnn_val_ppl = val_ppl
            best_rnn_model = model
            best_rnn_config = record

        if arch == "lstm" and val_ppl < best_lstm_val_ppl:
            best_lstm_val_ppl = val_ppl
            best_lstm_model = model
            best_lstm_config = record

    # Save best models
    torch.save({
        "config": best_rnn_config,
        "state_dict": best_rnn_model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "k": k
    }, models_path / "best_rnn.pt")
    print(f"[MODEL SELECTION] Best Simple RNN saved: {best_rnn_config['id']} (Val PP: {best_rnn_val_ppl:.2f})")

    torch.save({
        "config": best_lstm_config,
        "state_dict": best_lstm_model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "k": k
    }, models_path / "best_lstm.pt")
    print(f"[MODEL SELECTION] Best LSTM saved: {best_lstm_config['id']} (Val PP: {best_lstm_val_ppl:.2f})")

    # Save table of all 9 experiments
    summary = {
        "experiments": results,
        "best_rnn": best_rnn_config,
        "best_lstm": best_lstm_config
    }
    with open(out_path / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Bar chart comparison of Perplexities across all runs
    exp_labels = [r["id"] for r in results]
    val_ppls = [r["val_ppl"] for r in results]
    test_ppls = [r["test_ppl"] for r in results]

    x = range(len(exp_labels))
    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar([p - 0.2 for p in x], val_ppls, width=0.4, label="Validation Perplexity", color="#0275d8")
    plt.bar([p + 0.2 for p in x], test_ppls, width=0.4, label="Test Perplexity", color="#5cb85c")
    plt.xticks(x, exp_labels, rotation=35, ha='right', fontsize=9, fontweight='bold')
    plt.ylabel("Perplexity (Lower is Better)", fontsize=12, fontweight='bold')
    plt.title("Phase 3: Hyperparameter & Architecture Exploration Results", fontsize=14, fontweight='bold', pad=12)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    chart_file = fig_path / "phase3_hyperparameter_comparison.png"
    plt.savefig(chart_file)
    plt.close()
    print(f"[PHASE 3 COMPLETED] Summary chart saved to: {chart_file.resolve()}\n")

    return summary
