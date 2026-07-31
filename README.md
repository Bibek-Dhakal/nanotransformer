# ⚡ NanoTransformer Core ("Spark")

**NanoTransformer Core** is a custom-built, hardware-optimized Transformer Language Model developed entirely from
scratch in pure PyTorch. It features a GPT-2 style decoder architecture and is deeply integrated
with [LexiByte](https://pypi.org/project/lexibyte/), a custom Byte-Pair Encoding (BPE) tokenizer published on PyPI.

This repository demonstrates elite, production-grade Deep Learning systems engineering, bypassing high-level wrappers (
like HuggingFace) to explicitly implement low-level hardware optimizations and modular training mechanics.

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

Make sure you have PyTorch installed, and grab the custom tokenizer from PyPI:

```bash
pip install torch numpy lexibyte
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
