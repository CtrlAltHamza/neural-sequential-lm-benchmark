"""
Main Pipeline Runner for Assignment 2.
Deep Learning for Perception (CS4045)
Student: Hamza Khurram (i210735)

Orchestrates all phases end-to-end:
- Phase 1: Sequential Data Engineering
- Phase 2: Vanishing Gradient Experiment
- Phase 3: Hyperparameter & Architecture Search
- Phase 4: Next-Word Prediction Chatbot Verification
- Phase 5: Perplexity & Confidence Analysis
- Phase 6: Multi-Style NLP Adaptation (BBC News)
"""

import sys
import os
import json
import argparse
from pathlib import Path
import torch

from src.utils.seed import set_seed
from src.data.clean_austen import clean_austen_text
from src.data.clean_bbc import clean_and_deduplicate_bbc
from src.data.dataset import CorpusManager
from src.evaluation.phase2_experiment import run_phase2_experiments
from src.evaluation.phase3_experiment import run_phase3_experiments
from src.models.chatbot import NextWordChatbot
from src.evaluation.phase5_experiment import run_phase5_analysis
from src.evaluation.phase6_experiment import run_phase6_adaptation

def run_pipeline(quick_mode: bool = False, device: str = "cpu"):
    print("=" * 70)
    print(" DEEP LEARNING FOR PERCEPTION (CS4045) - ASSIGNMENT #2")
    print(" Student: Hamza Khurram | Roll No: i210735")
    print(f" Execution Mode: {'QUICK (Sanity Verification)' if quick_mode else 'FULL BENCHMARK'}")
    print(f" Hardware Device: {device} (CUDA Available: {torch.cuda.is_available()})")
    print("=" * 70)

    set_seed(42)

    # -------------------------------------------------------------
    # PHASE 1: SEQUENTIAL DATA ENGINEERING
    # -------------------------------------------------------------
    print("\n>>> [PHASE 1] RUNNING SEQUENTIAL DATA ENGINEERING <<<")
    clean_austen_text()
    clean_and_deduplicate_bbc()

    manager = CorpusManager(seed=42, min_count=3)
    corpus_stats = manager.prepare_and_split_corpus()
    tokenizer = manager.tokenizer

    train_sents = manager.train_sentences
    val_sents = manager.val_sentences
    test_sents = manager.test_sentences

    # Sample budget adjustments for quick mode vs full run
    p2_train_samples = 8000 if quick_mode else 30000
    p2_eval_samples = 1500 if quick_mode else 5000
    p2_epochs = 3 if quick_mode else 5

    p3_train_samples = 10000 if quick_mode else 40000
    p3_val_samples = 2000 if quick_mode else 6000
    p3_test_samples = 2000 if quick_mode else 6000
    p3_epochs = 3 if quick_mode else 8

    p5_eval_sentences = 400 if quick_mode else 1500
    p6_eval_samples = 4000 if quick_mode else 15000

    # -------------------------------------------------------------
    # PHASE 2: THE VANISHING GRADIENT EXPERIMENT
    # -------------------------------------------------------------
    print("\n>>> [PHASE 2] RUNNING THE VANISHING GRADIENT EXPERIMENT <<<")
    p2_results = run_phase2_experiments(
        train_sentences=train_sents,
        val_sentences=val_sents,
        tokenizer=tokenizer,
        k_values=[5, 10, 20, 40, 80],
        embedding_dim=128,
        hidden_dim=256,
        lr=0.001,
        batch_size=128,
        epochs=p2_epochs,
        max_train_samples=p2_train_samples,
        max_eval_samples=p2_eval_samples,
        device=device
    )

    # -------------------------------------------------------------
    # PHASE 3: HYPERPARAMETER & ARCHITECTURE SEARCH
    # -------------------------------------------------------------
    print("\n>>> [PHASE 3] RUNNING HYPERPARAMETER & ARCHITECTURE SEARCH <<<")
    p3_results = run_phase3_experiments(
        train_sentences=train_sents,
        val_sentences=val_sents,
        test_sentences=test_sents,
        tokenizer=tokenizer,
        k=20,
        embedding_dim=128,
        lr=0.001,
        batch_size=128,
        epochs=p3_epochs,
        patience=2,
        max_train_samples=p3_train_samples,
        max_val_samples=p3_val_samples,
        max_test_samples=p3_test_samples,
        device=device
    )

    # -------------------------------------------------------------
    # PHASE 4: NEXT-WORD PREDICTION CHATBOT VERIFICATION
    # -------------------------------------------------------------
    print("\n>>> [PHASE 4] TESTING NEXT-WORD PREDICTION CHATBOT <<<")
    chatbot = NextWordChatbot(
        checkpoint_path="saved_models/best_lstm.pt",
        vocab_path="data/processed/vocab.json",
        device=device
    )
    test_prompts = [
        ("it is a truth", 8, "greedy"),
        ("the king sat upon his", 10, "temperature"),
        ("she walked toward the garden", 10, "temperature")
    ]
    for prompt, n, mode in test_prompts:
        output_text, tokens = chatbot.generate_next_words(prompt, n_words=n, mode=mode, temperature=0.7)
        print(f"  [Chatbot Prompt: '{prompt}'] ({mode.upper()}) -> \"{output_text}\"")

    # -------------------------------------------------------------
    # PHASE 5: PERPLEXITY AND CONFIDENCE ANALYSIS
    # -------------------------------------------------------------
    print("\n>>> [PHASE 5] RUNNING PERPLEXITY & CONFIDENCE ANALYSIS <<<")
    p5_results = run_phase5_analysis(
        test_sentences=test_sents,
        tokenizer=tokenizer,
        best_rnn_path="saved_models/best_rnn.pt",
        best_lstm_path="saved_models/best_lstm.pt",
        rare_threshold=50,
        regression_train_ratio=0.70,
        max_eval_sentences=p5_eval_sentences,
        device=device
    )

    # -------------------------------------------------------------
    # PHASE 6: MULTI-STYLE NLP ADAPTATION (BBC NEWS)
    # -------------------------------------------------------------
    print("\n>>> [PHASE 6] RUNNING MULTI-STYLE NLP ADAPTATION <<<")
    p6_results = run_phase6_adaptation(
        bbc_dir="data/processed/bbc",
        tokenizer_path="data/processed/vocab.json",
        primary_test_results_path="experiments/phase3_hyperparameters/phase3_results.json",
        best_rnn_path="saved_models/best_rnn.pt",
        best_lstm_path="saved_models/best_lstm.pt",
        k=20,
        batch_size=128,
        max_eval_samples=p6_eval_samples,
        device=device
    )

    print("\n" + "=" * 70)
    print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(" Figures saved in:   reports/figures/")
    print(" Checkpoints in:     saved_models/")
    print(" Logs & metrics in:  experiments/")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Assignment 2 Pipeline")
    parser.add_argument("--quick", action="store_true", help="Run in fast sanity-check mode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Hardware device")
    args = parser.parse_args()

    run_pipeline(quick_mode=args.quick, device=args.device)
