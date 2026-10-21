"""
Sliding Window Tokenizer & Vocabulary Module for Assignment 2.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Implements:
1. Word tokenization with sentence boundaries (<s> and </s>).
2. Vocabulary builder built strictly from the training split.
3. Frequency thresholding (< min_count -> <UNK>).
4. Sliding window sequence generation for variable k.
5. <UNK> rate reporting across splits.
"""

import re
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

class Tokenizer:
    def __init__(self, min_count: int = 3):
        self.min_count = min_count
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        self.is_fitted = False

        # Pre-populate special tokens
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    @property
    def pad_id(self) -> int:
        return self.word2idx[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.word2idx[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.word2idx[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """
        Tokenizes text into words and punctuation while preserving contractions.
        Example: "It is a truth, isn't it?" -> ['it', 'is', 'a', 'truth', ',', "isn't", 'it', '?']
        """
        text_clean = text.lower().strip()
        # Match words (including internal apostrophes for contractions) and punctuation
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[.,!?;:\"'-]", text_clean)
        return tokens

    def sentence_to_tokens(self, sentence: str) -> List[str]:
        """Converts raw sentence into token list bounded by <s> and </s>."""
        words = self.tokenize_text(sentence)
        if not words:
            return []
        return [BOS_TOKEN] + words + [EOS_TOKEN]

    def fit_on_sentences(self, sentences: List[str]) -> None:
        """
        Builds vocabulary strictly from the training split sentences.
        Tokens with count < min_count are mapped to <UNK>.
        """
        counter = Counter()
        for sent in sentences:
            tokens = self.sentence_to_tokens(sent)
            for token in tokens:
                if token not in SPECIAL_TOKENS:
                    counter[token] += 1

        self.word_counts = dict(counter)

        # Assign indices to tokens meeting minimum frequency
        next_idx = len(SPECIAL_TOKENS)
        # Sort by frequency descending, then alphabetically for deterministic index assignment
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        for token, count in sorted_tokens:
            if count >= self.min_count:
                self.word2idx[token] = next_idx
                self.idx2word[next_idx] = token
                next_idx += 1

        self.is_fitted = True

    def encode(self, tokens: List[str]) -> List[int]:
        """Encodes token list to IDs, replacing OOV words with <UNK> ID."""
        unk = self.unk_id
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, token_ids: List[int]) -> List[str]:
        """Decodes token IDs back to string tokens."""
        unk = UNK_TOKEN
        return [self.idx2word.get(i, unk) for i in token_ids]

    def compute_unk_rate(self, sentences: List[str]) -> Tuple[int, int, float]:
        """
        Computes UNK token statistics over a list of sentences.
        Returns: (total_tokens, unk_count, unk_percentage)
        """
        total_tokens = 0
        unk_count = 0
        unk_idx = self.unk_id

        for sent in sentences:
            tokens = self.sentence_to_tokens(sent)
            if not tokens:
                continue
            encoded = self.encode(tokens)
            total_tokens += len(encoded)
            unk_count += encoded.count(unk_idx)

        rate = (unk_count / total_tokens * 100.0) if total_tokens > 0 else 0.0
        return total_tokens, unk_count, rate

    def create_sliding_windows(self, token_ids: List[int], k: int) -> Tuple[List[List[int]], List[int]]:
        """
        Converts a continuous sequence of token IDs into sliding windows.
        Input: sequence of k words.
        Target: the next word (scalar).
        """
        if len(token_ids) <= k:
            return [], []

        X = []
        y = []
        for i in range(len(token_ids) - k):
            X.append(token_ids[i : i + k])
            y.append(token_ids[i + k])
        return X, y

    def save(self, filepath: str) -> None:
        """Saves vocabulary configuration and mapping to JSON."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "min_count": self.min_count,
            "vocab_size": self.vocab_size,
            "word2idx": self.word2idx,
            "word_counts": self.word_counts
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Vocabulary ({self.vocab_size} tokens) saved to: {out_path.resolve()}")

    @classmethod
    def load(cls, filepath: str) -> "Tokenizer":
        """Loads vocabulary from saved JSON."""
        in_path = Path(filepath)
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(min_count=data["min_count"])
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(idx): word for word, idx in tok.word2idx.items()}
        tok.word_counts = data.get("word_counts", {})
        tok.is_fitted = True
        return tok
