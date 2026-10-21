# Experimental Analysis of RNNs and LSTMs for Next Word Prediction

**Course:** CS4045 – Deep Learning for Perception  
**Instructor:** Dr. Ahmad Raza Shahid  
**Author:** Hamza Khurram (`i210735`)  
**Assignment:** #2  

---

## 📌 Executive Summary

This repository presents an experimental and research-oriented comparative analysis between **Recurrent Neural Networks (Simple RNNs)** and **Long Short-Term Memory Networks (LSTMs)** for character- and word-level autoregressive modeling. It addresses the vanishing gradient problem empirically, performs systematic hyperparameter search across depth, capacity, and regularization, builds an interactive generation chatbot, assesses sentence hardness via linear regression, and analyzes the Generalization Gap on out-of-domain modern news corpora.

```
       [Primary Training Corpus: Austen + Shakespeare]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Simple RNN (Tanh)]               [LSTM Network (CEC)]
            │                                 │
     Gradients Vanish                  Gradients Preserved
  Over Long Contexts (k>20)         Across Window Sizes (k=80)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
     [Generalization Evaluation: Modern BBC News Articles]
             Surge in Perplexity (+100% to +150%)
```

---

## 🏗️ Repository Architecture & MLOps Structure

```
├── .gitignore
├── README.md
├── README.txt
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── clean_austen.py
│   │   ├── clean_bbc.py
│   │   ├── dataset.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   └── chatbot.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── early_stopping.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── phase2_experiment.py
│   │   ├── phase3_experiment.py
│   │   ├── phase5_experiment.py
│   │   └── phase6_experiment.py
│   └── utils/
│       └── seed.py
├── notebooks/
│   └── Assignment_02_Hamza_Khurram_i210735.ipynb
├── experiments/
├── saved_models/
└── reports/
    ├── figures/
    └── report.pdf
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execution Options
- **Jupyter Notebook:** Open `notebooks/Assignment_02_Hamza_Khurram_i210735.ipynb` and run all cells.
- **Pipeline Runner:**
  ```bash
  python main.py --quick     # Fast sanity-check mode
  python main.py             # Full benchmark run
  ```
- **Interactive Chatbot:**
  ```bash
  python src/models/chatbot.py
  ```

---

## 🔬 Experimental Highlights

1. **Phase 1: Sequential Data Engineering**
   - Gutenberg headers and transcriber notices removed.
   - Exact BBC duplicates deduplicated (4,450 files $\to$ 2,225 unique articles).
   - Sentence boundaries tagged with `<s>` and `</s>`.
   - Primary corpus split 80% train / 10% val / 10% test by whole sentences with seed 42.
   - Vocabulary built strictly from training split ($< 3$ counts mapped to `<UNK>`).

2. **Phase 2: Vanishing Gradient Analysis**
   - Context length sweep: $k \in \{5, 10, 20, 40, 80\}$.
   - Target words strictly aligned at offset $t \ge 80$ for direct comparability.
   - Observed Simple RNN perplexity saturation vs. LSTM resilience due to Constant Error Carousel.

3. **Phase 3: Hyperparameter & Architecture Search**
   - 9 systematic configurations (1–3 layers, 128–512 units, 0.0–0.5 dropout).
   - Early stopping on validation perplexity.
   - Best models chosen strictly on validation perplexity.

4. **Phase 4: Next-Word Prediction Chatbot**
   - Interactive prompt supporting Greedy Decoding and Temperature Sampling ($T \in [0.1, 1.5]$).

5. **Phase 5: Perplexity & Confidence Analysis**
   - Exact sentence perplexity: $PP(W) = \exp(-\frac{1}{N}\sum \ln P(w_i \mid w_{<i}))$.
   - Tertiles comparison across Short, Medium, and Long sentences.
   - OLS Linear Regression predicting sentence $\ln(PP)$ from length and rare words share.

6. **Phase 6: Multi-Style NLP Adaptation**
   - Zero-shot domain transfer on 2,225 modern journalistic BBC News articles.
   - Measured Generalization Gap ($\Delta PP$) and per-topic degradation.
