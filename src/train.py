import math
import time

import torch

from dataset import DataLoaderLite
from generate import generate_text
from model import GPT, GPTConfig


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

    # --- Initialize Data Loader (Imported from dataset.py) ---
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
    ctx_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print("Starting Training Loop...")

    for step in range(max_steps):
        t0 = time.time()

        # Periodically generate text to observe learning progress (Imported from generate.py)
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

    # --- Save the Trained Model ---
    checkpoint_path = "nanotransformer.pt"
    print(f"\nTraining Complete! Saving model to {checkpoint_path}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }
    torch.save(checkpoint, checkpoint_path)
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
