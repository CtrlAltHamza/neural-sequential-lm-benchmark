"""
Dataset Preparation & PyTorch DataLoader Module.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Handles:
1. Sentence extraction for Jane Austen and Tiny Shakespeare.
2. 80/10/10 train/val/test deterministic split by whole sentences.
3. Vocabulary fitting strictly on train split with <UNK> reporting.
4. PyTorch Dataset and DataLoader generation with variable window size k.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize

from src.data.tokenizer import Tokenizer, PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from src.utils.seed import set_seed

class NextWordDataset(Dataset):
    """
    PyTorch Dataset yielding (X, y) tensor pairs for next-word prediction.
    X: tensor of shape (k,) with word indices.
    y: scalar tensor with target word index.
    """
    def __init__(self, X: List[List[int]], y: List[int]):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class CorpusManager:
    def __init__(
        self,
        austen_path: str = "data/processed/austen_clean.txt",
        shakespeare_dir: str = "data/raw/shakespeare",
        processed_dir: str = "data/processed",
        seed: int = 42,
        min_count: int = 3
    ):
        self.austen_path = Path(austen_path)
        self.shakespeare_dir = Path(shakespeare_dir)
        self.processed_dir = Path(processed_dir)
        self.seed = seed
        self.min_count = min_count

        self.tokenizer: Optional[Tokenizer] = None
        self.train_sentences: List[str] = []
        self.val_sentences: List[str] = []
        self.test_sentences: List[str] = []

    def extract_all_sentences(self) -> List[str]:
        """Loads and extracts sentences from Austen and Shakespeare corpora."""
        print("[DATA] Ingesting Jane Austen text...")
        with open(self.austen_path, "r", encoding="utf-8") as f:
            austen_lines = f.readlines()
        
        austen_sents = []
        for line in austen_lines:
            line_str = line.strip()
            if line_str:
                for s in sent_tokenize(line_str):
                    s_clean = s.strip()
                    if len(s_clean) > 2 and any(c.isalnum() for c in s_clean):
                        austen_sents.append(s_clean)
        print(f"  -> Extracted {len(austen_sents):,} Austen sentences.")

        print("[DATA] Ingesting Tiny Shakespeare text...")
        shakes_parts = []
        for csv_name in ["train.csv", "validation.csv", "test.csv"]:
            csv_path = self.shakespeare_dir / csv_name
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                shakes_parts.append(df["text"].iloc[0])
        
        shakes_sents = []
        for part in shakes_parts:
            for line in part.splitlines():
                line_str = line.strip()
                if line_str:
                    for s in sent_tokenize(line_str):
                        s_clean = s.strip()
                        if len(s_clean) > 2 and any(c.isalnum() for c in s_clean):
                            shakes_sents.append(s_clean)
        print(f"  -> Extracted {len(shakes_sents):,} Shakespeare sentences.")

        all_sents = austen_sents + shakes_sents

        print(f"[DATA] Total primary corpus sentences: {len(all_sents):,}")
        return all_sents

    def prepare_and_split_corpus(self) -> Dict[str, any]:
        """
        Splits corpus into 80% train / 10% validation / 10% test by whole sentences.
        Fits tokenizer strictly on the training split.
        """
        set_seed(self.seed)
        all_sentences = self.extract_all_sentences()

        # Split 80% train, 20% temp (val + test)
        train_sents, temp_sents = train_test_split(
            all_sentences, test_size=0.20, random_state=self.seed, shuffle=True
        )
        # Split temp into 50% val (10% total) and 50% test (10% total)
        val_sents, test_sents = train_test_split(
            temp_sents, test_size=0.50, random_state=self.seed, shuffle=True
        )

        self.train_sentences = train_sents
        self.val_sentences = val_sents
        self.test_sentences = test_sents

        print(f"\n[SPLIT] Data Split Summary (Seed={self.seed}):")
        print(f"  - Train sentences:      {len(train_sents):,} ({len(train_sents)/len(all_sentences)*100:.1f}%)")
        print(f"  - Validation sentences: {len(val_sents):,} ({len(val_sents)/len(all_sentences)*100:.1f}%)")
        print(f"  - Test sentences:       {len(test_sents):,} ({len(test_sents)/len(all_sentences)*100:.1f}%)")

        # Build vocabulary strictly from the training split
        print(f"\n[VOCAB] Fitting tokenizer on training split (min_count={self.min_count})...")
        self.tokenizer = Tokenizer(min_count=self.min_count)
        self.tokenizer.fit_on_sentences(self.train_sentences)
        print(f"  -> Built Vocabulary Size: {self.tokenizer.vocab_size:,} unique tokens.")

        # Compute UNK rates across splits
        print("\n[EVALUATION] Calculating UNK rates across splits:")
        train_tokens, train_unk, train_rate = self.tokenizer.compute_unk_rate(self.train_sentences)
        val_tokens, val_unk, val_rate = self.tokenizer.compute_unk_rate(self.val_sentences)
        test_tokens, test_unk, test_rate = self.tokenizer.compute_unk_rate(self.test_sentences)

        print(f"  - Train Split:      {train_unk:,} / {train_tokens:,} tokens (<UNK> rate: {train_rate:.2f}%)")
        print(f"  - Validation Split: {val_unk:,} / {val_tokens:,} tokens (<UNK> rate: {val_rate:.2f}%)")
        print(f"  - Test Split:       {test_unk:,} / {test_tokens:,} tokens (<UNK> rate: {test_rate:.2f}%)")

        # Save artifacts to disk
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(self.processed_dir / "vocab.json"))

        for name, data in [("train", train_sents), ("val", val_sents), ("test", test_sents)]:
            with open(self.processed_dir / f"{name}_sentences.json", "w", encoding="utf-8") as f:
                json.dump(data, f)

        stats = {
            "total_sentences": len(all_sentences),
            "train_sentences": len(train_sents),
            "val_sentences": len(val_sents),
            "test_sentences": len(test_sents),
            "vocab_size": self.tokenizer.vocab_size,
            "min_count": self.min_count,
            "train_tokens": train_tokens,
            "train_unk": train_unk,
            "train_unk_rate": train_rate,
            "val_tokens": val_tokens,
            "val_unk": val_unk,
            "val_unk_rate": val_rate,
            "test_tokens": test_tokens,
            "test_unk": test_unk,
            "test_unk_rate": test_rate
        }

        with open(self.processed_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats

    def create_sliding_window_dataset(
        self,
        sentences: List[str],
        k: int,
        max_samples: Optional[int] = None
    ) -> NextWordDataset:
        """
        Creates a sliding-window dataset across whole sentences.
        Each sentence is tokenized with <s> and </s>, then windows of size k are constructed.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not initialized. Run prepare_and_split_corpus() first.")

        all_X = []
        all_y = []

        for sent in sentences:
            tokens = self.tokenizer.sentence_to_tokens(sent)
            if len(tokens) <= k:
                # If sentence length <= k, pad with PAD_TOKEN to allow next-word prediction
                continue
            encoded = self.tokenizer.encode(tokens)
            X_sent, y_sent = self.tokenizer.create_sliding_windows(encoded, k)
            all_X.extend(X_sent)
            all_y.extend(y_sent)

            if max_samples is not None and len(all_y) >= max_samples:
                all_X = all_X[:max_samples]
                all_y = all_y[:max_samples]
                break

        return NextWordDataset(all_X, all_y)

    def get_dataloader(
        self,
        dataset: NextWordDataset,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

if __name__ == "__main__":
    manager = CorpusManager()
    stats = manager.prepare_and_split_corpus()
