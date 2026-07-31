# 🛠️ Usage & Generation Guide

This guide covers how to execute the training loop and how to interact with the trained model using the standalone
generation script.

---

## 1. System Requirements

- Python 3.8+
- PyTorch (compiled with CUDA support if using an NVIDIA GPU)
- LexiByte (`pip install lexibyte`)

---

## 2. Training the Model

The training orchestrator is fully self-contained in `src/train.py`.

### Execution

Ensure `input.txt` is in your root directory, then run:

```bash
python src/train.py
```

### What happens during training:

1. **Device Detection:** The script automatically detects `cuda` (NVIDIA), `mps` (Apple Silicon), or defaults to `cpu`.
2. **Tokenizer:** It will train the LexiByte tokenizer (taking a few seconds) if a vocabulary file isn't found.
3. **Training Loop:** You will see a printout every 10 steps showing the current `Loss`, `Learning Rate (LR)`, execution
   time, and `Tokens/sec` throughput.
4. **Validation Generation:** Every 500 steps, training pauses to autoregressively generate text from a starting prompt.
   You will see the text evolve from gibberish to structural English.
5. **Checkpointing:** At `max_steps`, the script saves the architecture configuration and neural weights to
   `nanotransformer.pt`.

*Note: You can tweak `B` (Batch Size), `T` (Context Length), and `max_steps` directly inside `train.py` depending on
your hardware limits.*

---

## 3. Standalone Text Generation

You don't need to run a training loop just to talk to your model! `src/generate.py` is a standalone CLI tool that loads
your saved weights and generates text on demand.

### Basic Generation

```bash
python src/generate.py
```

*(By default, this looks for `nanotransformer.pt` and uses the prompt "O Romeo, Romeo")*

### Advanced CLI Arguments

You can control the prompt and the length of the generated output using command-line arguments:

```bash
python src/generate.py --prompt "To be, or not to be" --tokens 200 --checkpoint "nanotransformer.pt"
```

**Arguments:**

- `--prompt`: The string of text to kickstart the generation (Default: `"O Romeo, Romeo"`).
- `--tokens`: The maximum number of tokens to generate (Default: `100`).
- `--checkpoint`: Path to the `.pt` weights file (Default: `"nanotransformer.pt"`).

---
