"""
Phase 2: The Vanishing Gradient Experiment.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Compares Simple RNN vs. LSTM across sequence lengths k in {5, 10, 20, 40, 80}.
Ensures identical target word alignment for fair perplexity comparison.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
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

def create_aligned_eval_dataset(
    sentences: List[str],
    tokenizer: Tokenizer,
    k: int,
    max_k: int = 80,
    max_samples: int = 5000
) -> NextWordDataset:
    """
    Creates an evaluation dataset where target words are strictly aligned across all k.
    Only target words at position t >= max_k are evaluated.
    For each target word w_t, the input is w_{t-k : t}.
    """
    all_X = []
    all_y = []

    for sent in sentences:
        tokens = tokenizer.sentence_to_tokens(sent)
        if len(tokens) <= max_k:
            continue
        encoded = tokenizer.encode(tokens)
        for t in range(max_k, len(encoded)):
            x_k = encoded[t - k : t]
            y_target = encoded[t]
            all_X.append(x_k)
            all_y.append(y_target)
            if len(all_y) >= max_samples:
                break
        if len(all_y) >= max_samples:
            break

    return NextWordDataset(all_X, all_y)

def run_phase2_experiments(
    train_sentences: List[str],
    val_sentences: List[str],
    tokenizer: Tokenizer,
    k_values: List[int] = [5, 10, 20, 40, 80],
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 0.001,
    batch_size: int = 128,
    epochs: int = 5,
    max_train_samples: int = 30000,
    max_eval_samples: int = 5000,
    device: str = "cpu",
    output_dir: str = "experiments/phase2_vanishing_gradient",
    figures_dir: str = "reports/figures"
) -> Dict[str, any]:
    out_path = Path(output_dir)
    fig_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"\n=======================================================")
    print(f" PHASE 2: THE VANISHING GRADIENT EXPERIMENT")
    print(f" Device: {dev} | Contexts k: {k_values}")
    print(f" Embedding: {embedding_dim} | Hidden: {hidden_dim} | Epochs: {epochs}")
    print(f"=======================================================\n")

    vocab_size = tokenizer.vocab_size
    results = {
        "k_values": k_values,
        "rnn_val_ppl": [],
        "lstm_val_ppl": [],
        "rnn_val_acc": [],
        "lstm_val_acc": [],
        "rnn_histories": {},
        "lstm_histories": {}
    }

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for k in k_values:
        print(f"\n>>> Running Experiments for Context Window k = {k} <<<")
        set_seed(42)

        # 1. Create Train Dataset for current k
        train_X, train_y = [], []
        for sent in train_sentences:
            tokens = tokenizer.sentence_to_tokens(sent)
            if len(tokens) <= k:
                continue
            enc = tokenizer.encode(tokens)
            x_s, y_s = tokenizer.create_sliding_windows(enc, k)
            train_X.extend(x_s)
            train_y.extend(y_s)
            if len(train_y) >= max_train_samples:
                train_X = train_X[:max_train_samples]
                train_y = train_y[:max_train_samples]
                break

        train_dataset = NextWordDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 2. Create Aligned Validation Dataset (strictly same target words across all k)
        val_dataset = create_aligned_eval_dataset(
            val_sentences, tokenizer, k=k, max_k=80, max_samples=max_eval_samples
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"  [k={k}] Train samples: {len(train_dataset):,} | Aligned Val samples: {len(val_dataset):,}")

        # 3. Train Simple RNN
        print(f"  --> Training Simple RNN (k={k})...")
        set_seed(42)
        rnn_model = SimpleRNNLM(vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=1)
        rnn_opt = torch.optim.Adam(rnn_model.parameters(), lr=lr)
        rnn_trainer = Trainer(rnn_model, rnn_opt, criterion, dev, patience=2, model_name=f"RNN-k{k}")
        rnn_hist = rnn_trainer.fit(train_loader, val_loader, epochs=epochs, verbose=False)

        val_loss, rnn_ppl, rnn_acc = evaluate_model(rnn_model, val_loader, criterion, dev)
        results["rnn_val_ppl"].append(rnn_ppl)
        results["rnn_val_acc"].append(rnn_acc)
        results["rnn_histories"][f"k_{k}"] = rnn_hist
        print(f"      Simple RNN (k={k}) -> Aligned Val PP: {rnn_ppl:.2f} | Acc: {rnn_acc:.2f}%")

        # 4. Train LSTM
        print(f"  --> Training LSTM (k={k})...")
        set_seed(42)
        lstm_model = LSTMLM(vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_layers=1)
        lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=lr)
        lstm_trainer = Trainer(lstm_model, lstm_opt, criterion, dev, patience=2, model_name=f"LSTM-k{k}")
        lstm_hist = lstm_trainer.fit(train_loader, val_loader, epochs=epochs, verbose=False)

        val_loss, lstm_ppl, lstm_acc = evaluate_model(lstm_model, val_loader, criterion, dev)
        results["lstm_val_ppl"].append(lstm_ppl)
        results["lstm_val_acc"].append(lstm_acc)
        results["lstm_histories"][f"k_{k}"] = lstm_hist
        print(f"      LSTM (k={k})       -> Aligned Val PP: {lstm_ppl:.2f} | Acc: {lstm_acc:.2f}%")

    # Save results
    with open(out_path / "phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot Validation Perplexity vs k
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.plot(k_values, results["rnn_val_ppl"], marker='o', linewidth=2.5, markersize=8, color='#d9534f', label='Simple RNN')
    plt.plot(k_values, results["lstm_val_ppl"], marker='s', linewidth=2.5, markersize=8, color='#0275d8', label='LSTM')
    plt.title("Phase 2: Validation Perplexity vs. Window Size (k)\n(Evaluated on Aligned Target Words)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Window Size (k)", fontsize=12, fontweight='bold')
    plt.ylabel("Validation Perplexity (Lower is Better)", fontsize=12, fontweight='bold')
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()

    plot_file = fig_path / "phase2_vanishing_gradient.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"\n[PHASE 2 COMPLETED] Plot saved to: {plot_file.resolve()}\n")

    return results
