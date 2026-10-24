================================================================================
DEEP LEARNING FOR PERCEPTION (CS4045) - ASSIGNMENT #2
Analysis of RNNs and LSTMs for Next Word Prediction
================================================================================

--------------------------------------------------------------------------------
1. PROJECT STRUCTURE
--------------------------------------------------------------------------------
i210735_Hamza_Khurram_A2/
├── notebooks/
│   └── Assignment_02_Hamza_Khurram_i210735.ipynb   <- Master notebook containing all phases
├── src/                                             <- Modular MLOps package
│   ├── data/                                        <- Data cleaning, tokenizers, dataset loaders
│   ├── models/                                      <- Simple RNN, LSTM, and Chatbot implementations
│   ├── training/                                    <- Trainer with early stopping & gradient clipping
│   ├── evaluation/                                  <- Perplexity, regression, and phase scripts
│   └── utils/                                       <- Deterministic seed lockers & loggers
├── configs/
│   └── config.yaml                                  <- Centralized experiment hyperparameter config
├── reports/
│   ├── figures/                                     <- High-resolution plots for all phases
│   └── report.pdf                                   <- Formal academic report (PDF)
├── requirements.txt                                 <- Pinned environment dependencies
├── README.txt                                       <- Execution guide (this file)
└── main.py                                          <- End-to-end command-line runner

--------------------------------------------------------------------------------
2. ENVIRONMENT SETUP & DEPENDENCIES
--------------------------------------------------------------------------------
Ensure Python 3.9+ (Python 3.12 recommended) is installed.

Step 2.1: Create and activate a virtual environment
  On Windows:
    python -m venv .venv
    .\.venv\Scripts\activate
  On Linux / macOS:
    python3 -m venv .venv
    source .venv/bin/activate

Step 2.2: Install dependencies
  pip install -r requirements.txt

The core dependencies include:
  torch >= 2.2.0
  numpy >= 1.26.0
  pandas >= 2.2.0
  scikit-learn >= 1.4.0
  matplotlib >= 3.8.0
  seaborn >= 0.13.0
  nltk >= 3.8.0
  reportlab >= 4.0.0

--------------------------------------------------------------------------------
3. RUNNING THE CODE
--------------------------------------------------------------------------------
Option A: Run via Jupyter Notebook (Recommended for Grading & Visualizations)
  1. Start Jupyter:
       jupyter notebook
  2. Open: notebooks/Assignment_02_Hamza_Khurram_i210735.ipynb
  3. Run all cells sequentially from top to bottom.

Option B: Run via Command-Line Pipeline (main.py)
  1. Fast sanity-check run (3 epochs per model):
       python main.py --quick
  2. Full benchmark run:
       python main.py
  3. Run on GPU (if CUDA available):
       python main.py --device cuda

Option C: Run Live Teacher Demonstration (scripts/demo.py)
  Instant out-of-the-box demonstration of dataset requirements, joblib models, and text generation:
       python scripts/demo.py

Option D: Run in Google Colab / Kaggle
  1. Upload the project folder or clone the repository to Colab.
  2. Open notebooks/Assignment_02_Hamza_Khurram_i210735.ipynb.
  3. Ensure GPU runtime is selected (Runtime -> Change runtime type -> T4 GPU).
  4. Run all cells. GPU acceleration will automatically be detected and utilized.

--------------------------------------------------------------------------------
4. INTERACTIVE CHATBOT (PHASE 4)
--------------------------------------------------------------------------------
To launch the interactive next-word prediction chatbot in your console:
  python src/models/chatbot.py

Chatbot Controls:
  - Enter any seed phrase when prompted (e.g., "it is a truth").
  - Enter number of words N to generate (e.g., 10).
  - Select decoding strategy: [1] Greedy Decoding  or  [2] Temperature Sampling.
  - If temperature sampling is selected, enter temperature (e.g., 0.7).
  - Type 'quit' or 'exit' at any prompt to terminate cleanly.

--------------------------------------------------------------------------------
5. REPRODUCIBILITY GUARANTEE
--------------------------------------------------------------------------------
All experiments, data splits, and model initializations use fixed random seed 42
across Python, NumPy, and PyTorch (CPU & CUDA deterministic modes).
All evaluation target positions are strictly aligned for fair comparison.
================================================================================
