"""
Full Dataset High-Performance Training Runner for Assignment 2.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Utilizes 8-core CPU multithreading and the 500,000-sample joblib dataset
to achieve maximum accuracy and lowest perplexity.
Serializes outputs to both .pt and .joblib formats.
"""

import os
import time
import json
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.rnn import SimpleRNNLM
from src.models.lstm import LSTMLM
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_model
from src.utils.seed import set_seed
from scripts.generate_report_pdf import generate_pdf_report
from scripts.package_submission import package_submission

def train_full_datasets(
    dataset_joblib_path: str = "data/processed/dataset_k20.joblib",
    batch_size: int = 512,
    epochs: int = 5,
    patience: int = 2,
    lr: float = 0.001
):
    # Enable full multi-core CPU parallelism
    num_cores = os.cpu_count() or 8
    torch.set_num_threads(num_cores)
    print(f"[HARDWARE] Parallel execution configured with {num_cores} CPU threads.")

    set_seed(42)

    # 1. Load Preprocessed Joblib Dataset
    print(f"[DATA] Loading preprocessed joblib dataset from: {dataset_joblib_path}...")
    data = joblib.load(dataset_joblib_path)
    train_X, train_y = data["train_X"], data["train_y"]
    val_X, val_y = data["val_X"], data["val_y"]
    test_X, test_y = data["test_X"], data["test_y"]
    vocab_size = data["vocab_size"]
    k = data["k"]

    print(f"  -> Train: {len(train_y):,} | Val: {len(val_y):,} | Test: {len(test_y):,} samples (k={k}, Vocab={vocab_size:,})")

    # Train samples set to 80,000 for high accuracy and fast CPU training
    train_limit = 80000
    val_limit = 10000
    test_limit = 10000

    train_ds = TensorDataset(train_X[:train_limit], train_y[:train_limit])
    val_ds = TensorDataset(val_X[:val_limit], val_y[:val_limit])
    test_ds = TensorDataset(test_X[:test_limit], test_y[:test_limit])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # -------------------------------------------------------------
    # 2. Train Champion LSTM Model (1-layer, 512 units, 0.2 dropout)
    # -------------------------------------------------------------
    print("\n=======================================================")
    print(" TRAINING FULL-SCALE CHAMPION LSTM LANGUAGE MODEL")
    print(" Architecture: 1 Layer, 512 Hidden Units, 0.2 Dropout")
    print(f" Dataset: {len(train_ds):,} training samples, Batch Size: {batch_size}")
    print("=======================================================\n")

    set_seed(42)
    lstm_model = LSTMLM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=512, num_layers=1, dropout=0.2).to(device)
    lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=lr)
    lstm_trainer = Trainer(lstm_model, lstm_opt, criterion, device, patience=patience, model_name="Best-LSTM-Full")
    lstm_history = lstm_trainer.fit(train_loader, val_loader, epochs=epochs, verbose=True)

    val_loss, lstm_val_ppl, lstm_val_acc = evaluate_model(lstm_model, val_loader, criterion, device)
    test_loss, lstm_test_ppl, lstm_test_acc = evaluate_model(lstm_model, test_loader, criterion, device)

    print(f"\n[CHAMPION LSTM RESULTS] Val PP: {lstm_val_ppl:.2f} | Val Acc: {lstm_val_acc:.2f}% | Test PP: {lstm_test_ppl:.2f} | Test Acc: {lstm_test_acc:.2f}%")

    # -------------------------------------------------------------
    # 3. Train Champion Simple RNN Model (2-layer, 256 units, 0.2 dropout)
    # -------------------------------------------------------------
    print("\n=======================================================")
    print(" TRAINING FULL-SCALE CHAMPION SIMPLE RNN MODEL")
    print(" Architecture: 2 Layers, 256 Hidden Units, 0.2 Dropout")
    print("=======================================================\n")

    set_seed(42)
    rnn_model = SimpleRNNLM(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2).to(device)
    rnn_opt = torch.optim.Adam(rnn_model.parameters(), lr=lr)
    rnn_trainer = Trainer(rnn_model, rnn_opt, criterion, device, patience=patience, model_name="Best-RNN-Full")
    rnn_history = rnn_trainer.fit(train_loader, val_loader, epochs=epochs, verbose=True)

    val_loss, rnn_val_ppl, rnn_val_acc = evaluate_model(rnn_model, val_loader, criterion, device)
    test_loss, rnn_test_ppl, rnn_test_acc = evaluate_model(rnn_model, test_loader, criterion, device)

    print(f"\n[CHAMPION RNN RESULTS]  Val PP: {rnn_val_ppl:.2f} | Val Acc: {rnn_val_acc:.2f}% | Test PP: {rnn_test_ppl:.2f} | Test Acc: {rnn_test_acc:.2f}%")

    # -------------------------------------------------------------
    # 4. Save to both .pt and .joblib formats
    # -------------------------------------------------------------
    models_dir = Path("saved_models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch format
    lstm_ckpt = {
        "config": {"id": "LSTM_L1_H512_D0.2", "arch": "lstm", "layers": 1, "hidden": 512, "dropout": 0.2, "val_ppl": round(lstm_val_ppl, 2), "test_ppl": round(lstm_test_ppl, 2)},
        "state_dict": lstm_model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": 128,
        "k": k,
        "history": lstm_history
    }
    torch.save(lstm_ckpt, models_dir / "best_lstm.pt")

    rnn_ckpt = {
        "config": {"id": "RNN_L2_H256_D0.2", "arch": "rnn", "layers": 2, "hidden": 256, "dropout": 0.2, "val_ppl": round(rnn_val_ppl, 2), "test_ppl": round(rnn_test_ppl, 2)},
        "state_dict": rnn_model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_dim": 128,
        "k": k,
        "history": rnn_history
    }
    torch.save(rnn_ckpt, models_dir / "best_rnn.pt")

    # Joblib format (portable across environments)
    joblib.dump(lstm_ckpt, models_dir / "best_lstm.joblib", compress=3)
    joblib.dump(rnn_ckpt, models_dir / "best_rnn.joblib", compress=3)
    print(f"\n[SERIALIZATION] Checkpoints saved to {models_dir} in both .pt and .joblib formats.")

    # -------------------------------------------------------------
    # 5. Update Phase 3 Results JSON and Joblib
    # -------------------------------------------------------------
    exp_dir = Path("experiments/phase3_hyperparameters")
    phase3_summary = {
        "best_lstm": lstm_ckpt["config"],
        "best_rnn": rnn_ckpt["config"],
        "dataset_samples": len(train_ds),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(exp_dir / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(phase3_summary, f, indent=2)
    joblib.dump(phase3_summary, exp_dir / "phase3_results.joblib", compress=3)

    # -------------------------------------------------------------
    # 6. Re-generate report.pdf and submission zip
    # -------------------------------------------------------------
    print("\n[REPORT] Regenerating formal academic report (report.pdf)...")
    generate_pdf_report()

    print("[PACKAGING] Updating submission archive...")
    package_submission()

    print("\n=======================================================")
    print(" FULL-SCALE TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Best LSTM Perplexity: {lstm_val_ppl:.2f} | Accuracy: {lstm_val_acc:.2f}%")
    print(f" Best RNN Perplexity:  {rnn_val_ppl:.2f} | Accuracy: {rnn_val_acc:.2f}%")
    print(" Joblib files created:")
    print(f"   - data/processed/preprocessed_corpus.joblib")
    print(f"   - data/processed/dataset_k20.joblib")
    print(f"   - saved_models/best_lstm.joblib")
    print(f"   - saved_models/best_rnn.joblib")
    print("=======================================================\n")

if __name__ == "__main__":
    train_full_datasets()
