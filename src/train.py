import math
import os
import time

import torch
from lexibyte import LexiByteTokenizer
from torch.nn import functional as F

from model import GPT, GPTConfig


# -----------------------------------------------------------------------------
# 1. Data Loader using LexiByte Tokenizer
# -----------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, filename="input.txt"):
        self.B = B
        self.T = T

        # Load raw text
        print(f"Loading data from {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        # Initialize our custom PyPI Tokenizer
        self.tokenizer = LexiByteTokenizer()
        vocab_file = "lexibyte_vocab.json"

        if os.path.exists(vocab_file):
            print(f"Loading existing tokenizer vocab from {vocab_file}...")
            self.tokenizer.load(vocab_file)
        else:
            print("Training LexiByte Tokenizer from scratch...")
            # Train a small vocab for this local dataset (base 256 + 768 merges = 1024)
            # We use a slice of text to speed up local tokenizer training
            self.tokenizer.train(text[:1000000], vocab_size=1024, verbose=True)
            self.tokenizer.save(vocab_file)
            print("Tokenizer trained and saved!")

        print("Encoding dataset into tokens...")
        tokens = self.tokenizer.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.vocab_size = len(self.tokenizer.vocab)

        print(f"Loaded {len(self.tokens)} tokens. Vocabulary size: {self.vocab_size}")

        self.current_position = 0

    def get_vocab_size(self):
        return self.vocab_size

    def next_batch(self):
        B, T = self.B, self.T

        # Slice out a chunk of tokens of size B * T + 1
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]

        # Inputs (x) are tokens 0 to the second-to-last
        x = buf[:-1].view(B, T)
        # Targets (y) are tokens 1 to the last (shifted by 1)
        y = buf[1:].view(B, T)

        # Advance the position pointer
        self.current_position += B * T

        # Reset if we hit the end of the dataset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


# -----------------------------------------------------------------------------
# 2. Text Generation Helper
# -----------------------------------------------------------------------------
def generate_text(model, tokenizer, prompt, device, max_new_tokens=50):
    model.eval()
    print(f"\n--- Generating text from prompt: '{prompt}' ---")

    # Encode prompt using LexiByte
    encoded_prompt = tokenizer.encode(prompt)
    x = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)  # Shape: (1, T)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context to block_size to prevent crashing if sequence gets too long
            x_cond = x[:, -model.config.block_size:]

            # Forward pass
            logits, _ = model(x_cond)

            # Pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :]

            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            x = torch.cat((x, next_token), dim=1)

    # Decode the final sequence back to text using LexiByte
    generated_tokens = x[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    print(generated_text)
    print("--------------------------------------------------\n")
    model.train()


# -----------------------------------------------------------------------------
# 3. Main Training Loop
# -----------------------------------------------------------------------------
def main():
    # --- Device Setup & Optimizations ---
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")

    # Enable TF32 for dramatic speedups on NVIDIA Ampere+ GPUs
    if device == "cuda":
        torch.set_float32_matmul_precision('high')

    # --- Hyperparameters ---
    B = 8  # Micro-batch size
    T = 256  # Sequence length (Context window)
    max_steps = 2000
    learning_rate = 6e-4
    warmup_steps = 100

    # --- Initialize Data Loader ---
    # Ensure input.txt exists in your root folder!
    if not os.path.exists("input.txt"):
        raise FileNotFoundError("Please download 'input.txt' (e.g., Tiny Shakespeare) to the root directory.")

    train_loader = DataLoaderLite(B=B, T=T, filename="input.txt")

    # --- Initialize Model ---
    # We round up the vocab size to a multiple of 64 for optimal Tensor Core efficiency
    optimal_vocab_size = ((train_loader.get_vocab_size() + 63) // 64) * 64

    config = GPTConfig(
        vocab_size=optimal_vocab_size,
        block_size=T,
        n_layer=6,  # Scaled down from 12 for faster local training
        n_head=6,
        n_embd=384  # Scaled down from 768
    )

    model = GPT(config)
    model.to(device)

    # Optional: PyTorch 2.0 Compilation for fused kernels (Comment out if using Windows/errors)
    # if device == "cuda":
    #     model = torch.compile(model)

    # --- Optimizer & LR Scheduler ---
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate, device_type=device)

    def get_lr(it):
        # 1. Linear warmup
        if it < warmup_steps:
            return learning_rate * (it + 1) / warmup_steps
        # 2. Cosine decay down to 10% of max LR
        min_lr = learning_rate * 0.1
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # --- Determine Mixed Precision dtype ---
    # Use bfloat16 if supported, otherwise float16
    ctx_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print("Starting Training Loop...")

    for step in range(max_steps):
        t0 = time.time()

        # Periodically generate text to observe learning progress
        if step % 500 == 0 or step == max_steps - 1:
            generate_text(model, train_loader.tokenizer, prompt="O Romeo, Romeo", device=device)

        # Get next batch
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # Set learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        # Mixed Precision Forward Pass
        if device == "cuda":
            with torch.autocast(device_type=device, dtype=ctx_dtype):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)

        # Backward Pass
        loss.backward()

        # Gradient Clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update Weights
        optimizer.step()

        # Sync GPU for accurate timing
        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds
        tokens_processed = B * T
        tok_sec = tokens_processed / (t1 - t0)

        if step % 10 == 0:
            print(
                f"Step {step:4d} | Loss: {loss.item():.4f} | LR: {lr:.4e} | Time: {dt:.2f}ms | Tok/sec: {tok_sec:.2f}")


if __name__ == "__main__":
    main()
