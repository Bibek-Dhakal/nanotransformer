import argparse
import os

import torch
from lexibyte import LexiByteTokenizer
from torch.nn import functional as F

from model import GPT


def generate_text(model, tokenizer, prompt, device, max_new_tokens=50):
    """
    Autoregressively generates text from a prompt using the provided model and tokenizer.
    """
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

            # Pluck the logits at the final step
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

    return generated_text


if __name__ == "__main__":
    # Allows generating text directly from the command line
    parser = argparse.ArgumentParser(description="Generate text from a trained NanoTransformer.")
    parser.add_argument("--prompt", type=str, default="O Romeo, Romeo", help="The starting text prompt.")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument("--checkpoint", type=str, default="nanotransformer.pt", help="Path to model weights.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    tokenizer = LexiByteTokenizer()
    if os.path.exists("lexibyte_vocab.json"):
        tokenizer.load("lexibyte_vocab.json")
    else:
        print("Warning: lexibyte_vocab.json not found. Output will be gibberish. Run train.py first.")
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}  # Fallback dummy vocab

    # Load Model Weights
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found. Please train the model first.")
        exit(1)

    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Reconstruct model from saved config
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Generate!
    generate_text(model, tokenizer, prompt=args.prompt, device=device, max_new_tokens=args.tokens)
