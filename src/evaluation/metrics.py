"""
Metrics Module for Next-Word Prediction & Perplexity Analysis.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Implements:
1. Batch & Corpus Perplexity: PP = exp(Loss).
2. Top-1 Prediction Accuracy.
3. Sentence-level Perplexity and Log-Perplexity:
   PP(W) = exp(- 1/N * sum_{i=1}^N ln P(w_i | w_{<i}))
   where <s> is prompt context (not predicted) and </s> is predicted.
"""

import math
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F

def compute_loss_and_perplexity(loss_value: float) -> Tuple[float, float]:
    """Given cross entropy loss, returns (loss, perplexity). Clamped to avoid overflow."""
    loss_clamped = min(loss_value, 20.0)
    ppl = math.exp(loss_clamped)
    return loss_value, ppl

def evaluate_model(model: torch.nn.Module, dataloader, criterion, device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluates model on dataloader.
    Returns: (mean_loss, perplexity, top1_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 1.0, 0.0

    mean_loss = total_loss / total_samples
    ppl = math.exp(min(mean_loss, 20.0))
    accuracy = (total_correct / total_samples) * 100.0

    return mean_loss, ppl, accuracy

def compute_sentence_perplexity(
    model: torch.nn.Module,
    token_ids: List[int],
    k: int,
    device: torch.device
) -> Tuple[float, float, int]:
    """
    Computes exact sentence perplexity according to Phase 5 specification:
    Sentence tokens: [<s>, w_1, w_2, ..., w_M, </s>]
    - <s> is prompt context (never predicted).
    - </s> is counted as a predicted token.
    Total predicted tokens N = len(tokens) - 1.
    Formula: PP(W) = exp(- 1/N * sum_{i=1}^N ln P(w_i | w_{<i}))

    Returns: (sentence_perplexity, sentence_log_perplexity, N)
    """
    model.eval()
    if len(token_ids) <= 1:
        return 1.0, 0.0, 0

    # Predicted tokens are from index 1 to end (including </s>)
    # Context before token i is token_ids[:i] (padded to length k if shorter, or sliced to last k)
    n_tokens = len(token_ids) - 1
    total_log_prob = 0.0

    with torch.no_grad():
        for i in range(1, len(token_ids)):
            target_id = token_ids[i]
            context = token_ids[:i]

            if len(context) < k:
                # Left pad with pad_idx (0) to maintain input size k
                input_seq = [0] * (k - len(context)) + context
            else:
                input_seq = context[-k:]

            inp_tensor = torch.tensor([input_seq], dtype=torch.long, device=device)
            logits = model(inp_tensor)  # (1, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)  # (1, vocab_size)

            target_log_prob = log_probs[0, target_id].item()
            total_log_prob += target_log_prob

    # Mean negative log likelihood
    nll = - (total_log_prob / n_tokens)
    sent_ppl = math.exp(min(nll, 20.0))
    sent_log_ppl = math.log(sent_ppl)

    return sent_ppl, sent_log_ppl, n_tokens
