"""
Joblib Serialization of Preprocessed Datasets & Tensors.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Serializes:
1. data/processed/preprocessed_corpus.joblib (Tokenizer, Vocab, Sentences, Token Stats)
2. data/processed/dataset_k20.joblib (Pre-built Sliding Window PyTorch Tensors for k=20)
"""

import time
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import joblib
import torch
import numpy as np

from src.data.tokenizer import Tokenizer
from src.data.dataset import CorpusManager

def serialize_all_preprocessed_data(
    output_dir: str = "data/processed",
    k: int = 20,
    max_train_windows: int = 500000
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print(" SERIALIZING COMPLETE PREPROCESSED DATASET VIA JOBLIB")
    print("=======================================================")

    t0 = time.time()
    # 1. Load Corpus
    print("[DATA] Preparing and splitting corpus freshly from cleaned text...")
    manager = CorpusManager(seed=42, min_count=3)
    stats = manager.prepare_and_split_corpus()
    tokenizer = manager.tokenizer
    train_sents = manager.train_sentences
    val_sents = manager.val_sentences
    test_sents = manager.test_sentences

    # 2. Save full preprocessed corpus bundle
    corpus_bundle = {
        "tokenizer": tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "word2idx": tokenizer.word2idx,
        "idx2word": tokenizer.idx2word,
        "word_counts": tokenizer.word_counts,
        "train_sentences": train_sents,
        "val_sentences": val_sents,
        "test_sentences": test_sents,
        "stats": stats,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    corpus_file = out_path / "preprocessed_corpus.joblib"
    joblib.dump(corpus_bundle, corpus_file, compress=3)
    print(f"[JOBLIB] Preprocessed corpus bundle saved: {corpus_file.resolve()} ({corpus_file.stat().st_size:,} bytes)")

    # 3. Build complete sliding windows for k=20
    print(f"\n[WINDOWS] Generating sliding windows for k={k}...")
    def build_windows(sentences, limit=None):
        X_list, y_list = [], []
        for s in sentences:
            toks = tokenizer.sentence_to_tokens(s)
            if len(toks) <= k:
                continue
            enc = tokenizer.encode(toks)
            for i in range(len(enc) - k):
                X_list.append(enc[i : i + k])
                y_list.append(enc[i + k])
                if limit and len(y_list) >= limit:
                    break
            if limit and len(y_list) >= limit:
                break
        return torch.tensor(X_list, dtype=torch.long), torch.tensor(y_list, dtype=torch.long)

    train_X, train_y = build_windows(train_sents, limit=max_train_windows)
    val_X, val_y = build_windows(val_sents, limit=50000)
    test_X, test_y = build_windows(test_sents, limit=50000)

    dataset_bundle = {
        "k": k,
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
        "num_train_samples": len(train_y),
        "num_val_samples": len(val_y),
        "num_test_samples": len(test_y),
        "vocab_size": tokenizer.vocab_size
    }

    dataset_file = out_path / f"dataset_k{k}.joblib"
    joblib.dump(dataset_bundle, dataset_file, compress=3)
    print(f"[JOBLIB] Preprocessed sliding window dataset saved: {dataset_file.resolve()} ({dataset_file.stat().st_size:,} bytes)")
    print(f"  - Train samples: {len(train_y):,} | Val samples: {len(val_y):,} | Test samples: {len(test_y):,}")
    print(f"Total time elapsed: {time.time() - t0:.2f}s\n")

    return corpus_file, dataset_file

if __name__ == "__main__":
    serialize_all_preprocessed_data()
