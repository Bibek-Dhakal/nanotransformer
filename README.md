# ⚡ NanoTransformer Core ("Spark")

**NanoTransformer Core** is a custom-built, hardware-optimized Transformer Language Model developed entirely from
scratch in pure PyTorch. It features a GPT-2 style decoder architecture and is deeply integrated
with [LexiByte](https://pypi.org/project/lexibyte/), a custom Byte-Pair Encoding (BPE) tokenizer published on PyPI.

This repository demonstrates elite, production-grade Deep Learning systems engineering, bypassing high-level wrappers (
like HuggingFace) to explicitly implement low-level hardware optimizations and modular training mechanics.

## 📊 Hardware Benchmarks & Performance

By leveraging mixed-precision (`bfloat16`), TF32, and FlashAttention, this architecture achieves massive throughput on
consumer hardware.

**Training Hardware:** NVIDIA GeForce RTX 3060 (140W Mobile)  
**Dataset:** Tiny Shakespeare  
**Batch Size (B) x Context (T):** 8 x 256

* **Time per step:** ~14.5 ms
* **Throughput:** ~142,000 Tokens/sec
* **Final Loss (2000 steps):** ~3.15

## 🎭 Sample Generation

After just 2000 steps of training (less than 1 minute on an RTX 3060), the model learns English syntax, spacing, and
Shakespearean dialogue structures from scratch:

**Prompt:** `"To be, or not to be"`
**Output:**

```text
To be, or not to bend and all the deplAy.
Thou cause to hence, let it do not have forth,
Most matter, being country this sunsel.

ore your words;ake, thereod keep boy to your budy.
Lo you;INC upon the Vaurctaction,
It is a sentent.

BIO:
You are you?
ANTONZALrus of you; come but you, you Kate, trighte?
```

## 🧠 ML Theory: Understanding the Output (Why not train longer?)

A common question is: *"Why does the output contain gibberish words like 'ANTONZALrus', and why not just train it for 3
hours to fix it?"*

This repository is designed as an **Architectural Proof of Concept**, bounded by the limitations of the dataset:

1. **The Dataset Size:** The "Tiny Shakespeare" dataset is extremely small (~1 Megabyte, or ~300,000 tokens). Modern
   LLMs require terabytes of text to learn semantic world-knowledge.
2. **The Overfitting Trap:** Because the dataset is so small, training for more than 2,000 steps causes the model to
   severely overfit. Instead of generalizing the English language, it will simply memorize the exact text of the input
   file, resulting in an artificially low loss but zero generative capabilities.
3. **The Success Criterion:** The goal of this 1-minute training run is not to build ChatGPT. The goal is to prove that
   the Transformer architecture and custom tokenizer work. In just 60 seconds, the model successfully deduces script
   formatting, punctuation usage, and structural syntax strictly from random noise.

*(For massive, multi-hour training pipelines on gigabyte-scale datasets, see the upcoming `ModularML` orchestration
repository in this portfolio).*

## 🚀 Architectural & Hardware Optimizations

- **FlashAttention Integration:** Utilizes `F.scaled_dot_product_attention` to execute causal self-attention in highly
  optimized CUDA kernels, bypassing massive $O(T^2)$ High-Bandwidth Memory (HBM) writes.
- **Mixed Precision (`bfloat16` & TF32):** Maximizes NVIDIA Ampere Tensor Cores by executing heavy GEMM operations in
  `bfloat16` via `torch.autocast` while preserving FP32 precision for unstable loss reductions.
- **Architectural Weight Tying:** The Token Embedding matrix (`wte`) and the final Output Projection matrix (`lm_head`)
  share the exact same physical memory. This stabilizes the gradients and saves ~30% of the total model parameters
  footprint.
- **LexiByte Tokenizer Ecosystem:** Text ingestion does not rely on third-party black-box libraries. It utilizes a
  custom-built, PyPI-published NLP Tokenizer featuring Sennrich frequency maps and O(1) inference caching.
- **Scaled Residual Initialization:** Custom layer initialization scales down residual projection weights
  by $1/\sqrt{2L}$ to prevent activation variance explosion in deep networks.
- **Fused AdamW & Matrix Separation:** Prevents artificial degradation of network capacity by applying Weight Decay
  *only* to 2D matrices, intentionally ignoring 1D biases and LayerNorm scales.

## 📚 Official Documentation

Detailed explanations of the math, hardware scaling, and codebase usage can be found in the `docs/` folder:

- [Architecture & Optimizations](docs/architecture.md) - Deep dive into GPT-2 mechanics, FlashAttention, and Weight
  Tying.
- [Data Pipeline & LexiByte](docs/data_pipeline.md) - Guide on downloading datasets and how tokenization orchestrates
  with batching.
- [Usage & Generation Guide](docs/usage_and_generation.md) - Run-book for executing training loops, checking CUDA/GPU
  compatibility, and standalone CLI inference generation.

## ⚙️ Modular Project Structure

Following ML Engineering best practices, the codebase is segregated into distinct, reusable modules:

```text
nanotransformer/
├── src/
│   ├── model.py     # The pure PyTorch GPT-2 neural architecture
│   ├── dataset.py   # Data ingestion, LexiByte tokenizer integration, and batching
│   ├── train.py     # Training orchestrator, LR scheduling, and checkpointing
│   └── generate.py  # Standalone CLI inference script for loading .pt weights
├── input.txt        # Training corpus (e.g., Tiny Shakespeare)
└── README.md
```

## 🛠️ Quick Start (Run it Yourself!)

### 1. Install Requirements

By default, the `requirements.min.txt` file is configured to download the **CUDA 13.2** GPU version of PyTorch.
If you are on a Mac, CPU, or older GPU, please open `requirements.min.txt` and adjust/remove the `--extra-index-url`
line before installing. (See [docs/usage_and_generation.md](docs/usage_and_generation.md) for details).

```bash
pip install -r requirements.min.txt
```

### 2. Download a Dataset

Download the famous "Tiny Shakespeare" dataset to train the model locally:

```bash
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 3. Start the Training Loop

Run the training script. The script will dynamically train the LexiByte tokenizer, encode the dataset, and launch the
Transformer training loop.

```bash
python src/train.py
```

*The model will automatically save its weights to `nanotransformer.pt` when finished.*

### 4. Standalone Inference Generation

Once the model is trained, you can interact with it directly from the command line without running a training loop!

```bash
python src/generate.py --prompt "O Romeo, Romeo" --tokens 150
```

## 🧠 Technical Inspiration

Built as part of a high-impact AI/ML Systems Engineering portfolio, heavily influenced by modern large-scale training
paradigms (OpenAI, DeepMind) and Andrej Karpathy's "Zero to Hero" series.

---
