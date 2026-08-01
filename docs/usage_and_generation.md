# 🛠️ Usage & Generation Guide

This guide covers how to set up your environment (GPU vs. CPU), how to execute the training loop, and how to interact
with the trained model using the standalone generation script.

---

## 1. Environment Setup (PyTorch & CUDA)

Transformer architectures are highly compute-intensive. While you *can* train this on a CPU, using an NVIDIA GPU will
reduce training time from hours to minutes.

To utilize your GPU, you must install a version of PyTorch that has CUDA bundled inside it. We have set up
`requirements.min.txt` to help with this.

### Step 1: Check your System's CUDA Version (NVIDIA GPUs only)

Open your terminal or command prompt and run:

```bash
nvidia-smi
```

Look at the top right of the output table. You will see something like `CUDA Version: 13.3` (or `12.6`, `11.8`, etc.).

### Step 2: Edit `requirements.min.txt` (If necessary)

**The Golden Rule:** The PyTorch CUDA version you install must be **less than or equal to** your system's CUDA version.

Open `requirements.min.txt`. You will see this at the top:

```text
--extra-index-url https://download.pytorch.org/whl/cu132
```

- **If your `nvidia-smi` shows 13.2 or higher:** Do nothing! Leave it as is.
- **If your `nvidia-smi` shows an older version (e.g., 12.1 or 11.8):** Change `cu132` to `cu121` or `cu118`.
- **If you are on a Mac (Apple Silicon) or CPU-only machine:** Delete the `--extra-index-url` line entirely. PyTorch
  will install the default Mac/CPU version.

### Step 3: Install

Once your file is adjusted, run:

```bash
pip install -r requirements.min.txt
```

*(Verify your GPU is active by checking that `Using device: cuda` prints when you run the training script).*

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
   You will see this printed on the very first line.
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
