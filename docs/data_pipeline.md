# 💾 Data Pipeline & LexiByte Integration

Neural networks do not understand raw text; they require sequences of dense integer tokens. NanoTransformer Core
features a fully modular `DataLoaderLite` (`src/dataset.py`) that handles this bridge automatically.

---

## 1. Getting a Dataset

The repository expects a single text file named `input.txt` in the root directory.

### Recommended: Tiny Shakespeare

For local training and rapid prototyping, Andrej Karpathy's "Tiny Shakespeare" dataset (approx. 1 MB of text) is
perfect.

You can download it directly via your terminal:

```bash
# Mac/Linux/Windows PowerShell
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Using Custom Data

To train on your own data (e.g., Wikipedia articles, Python code, or your own chat logs), simply create a text file,
paste your raw text into it, and save it as `input.txt` in the project root.

---

## 2. LexiByte Tokenizer Integration

Instead of relying on a black-box third-party library like HuggingFace `transformers`, this repository relies
on [LexiByte](https://pypi.org/project/lexibyte/)—a custom PyPI package explicitly built for this ecosystem.

### How it works at Runtime:

When you run `train.py`, the `DataLoaderLite` orchestrates the following:

1. **Check for existing vocabulary:** It looks for `lexibyte_vocab.json`.
2. **Dynamic Training:** If the file is missing, it passes your `input.txt` into LexiByte. LexiByte performs a Byte-Pair
   Encoding (BPE) compression using Sennrich frequency optimization, building a custom vocabulary tailored perfectly to
   your dataset (e.g., learning Shakespearean words like `"Romeo"` or `"Juliet"` as single tokens).
3. **Encoding:** Once the rulebook is built (or loaded), it encodes the entire `input.txt` into a 1D PyTorch tensor.

### Tensor Shape & Batching

The DataLoader dynamically slices the 1D token tensor into `(B, T)` batches:

- **`B` (Batch Size):** The number of parallel sequences processed at once (e.g., `8`).
- **`T` (Time / Context Length):** The number of tokens in the context window (e.g., `256`).

It returns `x` (the inputs) and `y` (the targets). The `y` tensor is identical to `x`, but shifted to the left by one
position to enforce Next-Token Prediction.

---
