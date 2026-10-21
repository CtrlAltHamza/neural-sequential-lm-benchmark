"""
Phase 4: Next-Word Prediction Interactive Chatbot.
Assignment 2 - DLP (CS4045)
Student: Hamza Khurram (i210735)

Features:
1. Loads the Best LSTM model.
2. Supports Greedy Decoding (argmax) and Temperature Sampling (scaled categorical).
3. Robust input() loop with quit command and crash-proof error handling.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when run directly
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F

from src.models.lstm import LSTMLM
from src.data.tokenizer import Tokenizer

class NextWordChatbot:
    def __init__(
        self,
        checkpoint_path: str = "saved_models/best_lstm.pt",
        vocab_path: str = "data/processed/vocab.json",
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.tokenizer = Tokenizer.load(vocab_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = ckpt["config"]
        self.vocab_size = ckpt["vocab_size"]
        self.embedding_dim = ckpt["embedding_dim"]
        self.k = ckpt.get("k", 20)

        self.model = LSTMLM(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.config["hidden"],
            num_layers=self.config["layers"],
            dropout=self.config["dropout"]
        ).to(self.device)

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def generate_next_words(
        self,
        seed_phrase: str,
        n_words: int,
        mode: str = "temperature",
        temperature: float = 0.7
    ) -> Tuple[str, List[str]]:
        """
        Generates the next n_words from the seed phrase.
        mode: 'greedy' or 'temperature'
        """
        self.model.eval()
        raw_words = self.tokenizer.tokenize_text(seed_phrase)
        if not raw_words:
            current_tokens = ["<s>"]
        else:
            current_tokens = ["<s>"] + list(raw_words)

        generated_words = []

        with torch.no_grad():
            for _ in range(n_words):
                # Prepare input window of length k
                encoded = self.tokenizer.encode(current_tokens)
                if len(encoded) < self.k:
                    inp = [self.tokenizer.pad_id] * (self.k - len(encoded)) + encoded
                else:
                    inp = encoded[-self.k:]

                inp_tensor = torch.tensor([inp], dtype=torch.long, device=self.device)
                logits = self.model(inp_tensor)[0]  # (vocab_size,)

                # Mask special tokens so they are never generated into text
                logits[self.tokenizer.pad_id] = -1e9
                logits[self.tokenizer.unk_id] = -1e9
                logits[self.tokenizer.bos_id] = -1e9

                # If user requested more words, mask premature EOS
                if len(generated_words) < n_words - 1:
                    logits[self.tokenizer.eos_id] = -1e9

                if mode.lower() == "greedy":
                    # Mild repetition penalty on recent tokens to prevent deterministic loops
                    for prev_w in set(generated_words[-4:]):
                        prev_id = self.tokenizer.word2idx.get(prev_w)
                        if prev_id is not None:
                            logits[prev_id] -= 1.8
                    next_token_id = torch.argmax(logits).item()
                else:
                    temp = max(float(temperature), 1e-4)
                    scaled_logits = logits / temp
                    probs = F.softmax(scaled_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()

                next_word = self.tokenizer.idx2word.get(next_token_id, "")
                if next_word == "</s>":
                    if generated_words and generated_words[-1] not in ".!?":
                        generated_words.append(".")
                    # Reset context to start of sentence so generation continues
                    current_tokens = ["<s>"]
                    continue

                generated_words.append(next_word)
                current_tokens.append(next_word)

        # Clean detokenization and typography formatting
        import re
        full_tokens = list(raw_words) + generated_words
        formatted_text = " ".join(full_tokens)
        formatted_text = re.sub(r"\s+([.,!?;:])", r"\1", formatted_text)
        formatted_text = re.sub(r"\s+'\s*", "'", formatted_text)
        formatted_text = re.sub(r"\.{2,}", ".", formatted_text)

        return formatted_text, generated_words

    def interactive_loop(self):
        """Runs the interactive console loop."""
        print("=" * 65)
        print("   PHASE 4: NEXT-WORD PREDICTION INTERACTIVE CHATBOT")
        print(f"   Model: {self.config['id']} (Validation Perplexity: {self.config['val_ppl']})")
        print("   Commands: Type 'quit' or 'exit' at any prompt to terminate.")
        print("=" * 65)

        while True:
            try:
                seed = input("\n[USER] Enter seed phrase: ").strip()
                if seed.lower() in ["quit", "exit"]:
                    print("\n[CHATBOT] Goodbye! Session ended.")
                    break
                if not seed:
                    print("  -> Please enter at least one word.")
                    continue

                # Number of words
                n_str = input("[USER] Enter number of words to generate (e.g. 10): ").strip()
                if n_str.lower() in ["quit", "exit"]:
                    print("\n[CHATBOT] Goodbye! Session ended.")
                    break
                try:
                    n_words = int(n_str)
                    if n_words <= 0:
                        print("  -> Number of words must be positive. Defaulting to 10.")
                        n_words = 10
                except ValueError:
                    print("  -> Invalid number format. Defaulting to 10 words.")
                    n_words = 10

                # Mode selection
                print("  Decoding strategies: [1] Greedy Decoding  |  [2] Temperature Sampling")
                mode_choice = input("[USER] Choose strategy (1 or 2, default=2): ").strip()
                if mode_choice.lower() in ["quit", "exit"]:
                    break

                if mode_choice == "1":
                    mode = "greedy"
                    temp = 1.0
                else:
                    mode = "temperature"
                    temp_input = input("[USER] Enter temperature (0.1 to 1.5, default=0.7): ").strip()
                    try:
                        temp = float(temp_input) if temp_input else 0.7
                    except ValueError:
                        temp = 0.7

                print(f"\n[GENERATING] (Mode: {mode.upper()}" + (f", T={temp}" if mode == "temperature" else "") + f", N={n_words})...")
                completed_text, generated_tokens = self.generate_next_words(seed, n_words, mode=mode, temperature=temp)
                print(f"[CHATBOT] Continuation: \"{completed_text}\"")
                print(f"  -> Generated {len(generated_tokens)} tokens: {generated_tokens}")

            except (KeyboardInterrupt, EOFError):
                print("\n\n[CHATBOT] Interrupted by user. Exiting cleanly.")
                break
            except Exception as e:
                print(f"  [ERROR] An unexpected error occurred: {e}. Recovering...")

if __name__ == "__main__":
    bot = NextWordChatbot()
    bot.interactive_loop()
