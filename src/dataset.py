import os

import torch
from lexibyte import LexiByteTokenizer


class DataLoaderLite:
    """
    Handles data ingestion, tokenizer initialization/training, 
    and serving (B, T) tensor batches for the Transformer.
    """

    def __init__(self, B, T, filename="input.txt"):
        self.B = B
        self.T = T

        # Load raw text
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Dataset file '{filename}' not found. Please download it first.")

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
            # Train a small vocab for local testing
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
