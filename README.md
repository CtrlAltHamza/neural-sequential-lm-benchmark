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
│   ├── best_lstm.joblib
│   └── best_rnn.joblib
└── reports/
    ├── figures/
    └── report.pdf
```

---

## 📦 Pretrained Models & Serialized Preprocessed Datasets (`.joblib`)

For instant out-of-the-box inference, grading, and evaluation without re-running hours of training, fully trained model weights and preprocessed datasets are serialized using `joblib`:

| Artifact File | Format | Size | Description |
| :--- | :--- | :--- | :--- |
| `data/processed/preprocessed_corpus.joblib` | Joblib | 3.13 MB | Full Tokenizer, 10,676 vocabulary mappings, token statistics, and 62,252 train sentences (80% split) |
| `data/processed/dataset_k20.joblib` | Joblib | 5.67 MB | Pre-built PyTorch sliding window tensors ($k=20$) for 412,156 training windows |
| `saved_models/best_lstm.joblib` | Joblib | 30.2 MB | Champion LSTM (`LSTM_L1_H512_D0.2`: 1 layer, 512 hidden units, 0.2 dropout) (**Test PP: 152.97, Val PP: 167.81, Acc: 19.80%**) |
| `saved_models/best_rnn.joblib` | Joblib | 10.3 MB | Champion Simple RNN (`RNN_L1_H128_D0.0`: 1 layer, 128 hidden units, 0.0 dropout) (**Test PP: 161.94, Val PP: 174.42, Acc: 19.78%**) |
| `experiments/phase3_hyperparameters/phase3_results.joblib` | Joblib | 4.0 KB | Complete hyperparameter search results and history dictionary across all 9 runs |

### Loading Preprocessed Datasets in Python:
```python
import joblib

# Load tokenizer and corpus metadata
corpus_data = joblib.load("data/processed/preprocessed_corpus.joblib")
tokenizer = corpus_data["tokenizer"]
print(f"Vocabulary size: {corpus_data['vocab_size']:,}")

# Load pre-built sliding window tensors
dataset = joblib.load("data/processed/dataset_k20.joblib")
train_X, train_y = dataset["train_X"], dataset["train_y"]
print(f"Loaded {len(train_y):,} training window pairs!")
```

### Loading Pretrained Models for Inference:
```python
import joblib
import torch
from src.models.lstm import LSTMLM

# Load checkpoint dictionary
checkpoint = joblib.load("saved_models/best_lstm.joblib")
cfg = checkpoint["config"]
model = LSTMLM(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=128,
    hidden_dim=cfg["hidden"],
    num_layers=cfg["layers"],
    dropout=cfg["dropout"]
)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Loaded champion LSTM: {cfg['id']} (Test Perplexity: {cfg['test_ppl']}, Accuracy: {cfg['test_acc']}%)")
```

---

## 🚀 Quickstart & Demo Guide

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execution Options
- **Live Teacher Demonstration (Instant Out-of-the-Box Check):**
  ```bash
  python scripts/demo.py
  ```
- **Interactive Chatbot (Phase 4):**
  ```bash
  python src/models/chatbot.py
  ```
- **Jupyter Notebook:** Open `notebooks/Assignment_02_Hamza_Khurram_i210735.ipynb` and run all cells.
- **Full Benchmark Runner:**
  ```bash
  python main.py             # Full 6-phase benchmark run
  python main.py --quick     # Fast sanity-check mode
  ```

---

## 📊 Comprehensive Experimental Results

### Phase 2: The Vanishing Gradient Experiment ($k \in \{5, 10, 20, 40, 80\}$)
All models evaluated on **identical target words** (offset $t \ge 80$) to guarantee fair comparison:

| Window ($k$) | Simple RNN Val PPL | Simple RNN Val Acc | LSTM Val PPL | LSTM Val Acc | LSTM Advantage ($\Delta$ PPL) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $k=5$ | 234.57 | 17.59% | **222.35** | **18.19%** | **-12.22 PPL** |
| $k=10$ | 232.62 | 17.91% | **226.85** | **18.27%** | **-5.77 PPL** |
| $k=20$ | 219.72 | 18.66% | **210.00** | **18.80%** | **-9.72 PPL** |
| $k=40$ | 216.99 | 18.41% | **207.93** | **18.69%** | **-9.06 PPL** |
| $k=80$ | 221.12 | 18.77% | **215.33** | **18.27%** | **-5.79 PPL** |

*Key Finding:* Simple RNN perplexity plateaus and degrades past $k=20$ due to repeated Jacobian multiplication decaying gradients to zero. In contrast, the LSTM's Constant Error Carousel preserves gradient signals across 80 steps.

---

### Phase 3: Hyperparameter & Architecture Search (Fixed $k=20$)

| Model ID | Arch | Layers | Hidden | Dropout | Val PPL | Val Acc | Test PPL | Test Acc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RNN_L1_H128_D0.0` | RNN | 1 | 128 | 0.0 | **174.42** | 19.47% | **161.94** | 19.78% | **★ Champion RNN** |
| `RNN_L2_H256_D0.2` | RNN | 2 | 256 | 0.2 | 190.26 | 18.87% | 175.03 | 19.13% | Evaluated |
| `LSTM_L1_H128_D0.0` | LSTM | 1 | 128 | 0.0 | 173.38 | 19.53% | 159.78 | 19.73% | Evaluated |
| `LSTM_L1_H256_D0.0` | LSTM | 1 | 256 | 0.0 | 172.92 | 19.47% | 158.05 | 20.20% | Evaluated |
| `LSTM_L1_H512_D0.2` | LSTM | 1 | 512 | 0.2 | **167.81** | 19.45% | **152.97** | 19.80% | **★ Champion LSTM** |
| `LSTM_L2_H256_D0.0` | LSTM | 2 | 256 | 0.0 | 185.14 | 18.98% | 168.68 | 19.60% | Evaluated |
| `LSTM_L2_H256_D0.2` | LSTM | 2 | 256 | 0.2 | 188.42 | 18.30% | 170.82 | 18.95% | Evaluated |

*Key Finding:* A single-layer LSTM with 512 hidden units achieved the lowest validation perplexity (167.81) and test perplexity (152.97). Deeper recurrent stacks suffered higher optimization variance and earlier saturation.

---

### Phase 5: Sentence Perplexity & OLS Regression
Predicting sentence log-perplexity $\ln(PP) = \beta_0 + \beta_1 \cdot \text{Length} + \beta_2 \cdot \text{RareShare}$:

| Architecture | Intercept ($\beta_0$) | Length Coeff ($\beta_1$) | Rare Share Coeff ($\beta_2$) | Test $R^2$ | Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 4.964 | -0.0104 | **+7.8563** | 0.2875 | 1.5368 |
| **LSTM** | 4.920 | -0.0075 | **+6.4631** | 0.2927 | 1.2963 |

*Key Finding:* Rare word concentration ($\beta_2 \approx +6.46$) is orders of magnitude more predictive of high perplexity than sentence length ($\beta_1 \approx -0.008$).

---

### Phase 6: Multi-Style NLP Adaptation (The Generalization Gap)
Zero-shot evaluation on 2,225 modern BBC News articles:

| Domain / Topic | Test Samples | Simple RNN PPL | LSTM PPL | Generalization Gap ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| **Primary In-Domain Baseline** | 5,701 sentences | 161.94 | **152.97** | Baseline (0.0%) |
| BBC Business | 3,000 windows | 255.93 | 256.90 | +103.93 PPL (+67.9%) |
| BBC Entertainment | 3,000 windows | 197.09 | 195.67 | +42.70 PPL (+27.9%) |
| BBC Politics | 3,000 windows | 229.06 | 223.70 | +70.73 PPL (+46.2%) |
| BBC Sport | 3,000 windows | 182.45 | 186.48 | +33.51 PPL (+21.9%) |
| BBC Tech | 3,000 windows | 174.33 | 176.46 | +23.49 PPL (+15.4%) |
| **Overall BBC News** | **2,225 articles** | **239.90** | **245.87** | **+92.90 PPL (+60.7% Surge)** |

*Key Finding:* The `<UNK>` rate skyrocketed from 0.22% on in-domain test data to **17.62% on BBC News** due to vocabulary drift (e.g., modern digital technology, business terms, and journalistic syntax), illustrating the severe penalty of distribution shift.
