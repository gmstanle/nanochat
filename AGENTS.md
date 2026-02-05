# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

nanochat is a full-stack ChatGPT clone implementation designed to train LLMs from scratch on modest budgets ($100-$1000). The codebase is deliberately minimal, hackable, and dependency-lite - it's designed to be a single cohesive baseline rather than a configurable framework. The entire pipeline runs on a single 8XH100 node (or can be scaled down to CPUs/single GPUs).

## Development Commands

### Environment Setup
This project uses `uv` for dependency management. Two virtual environments can be used:
- **uv venv** (preferred for production): `uv venv` creates `.venv/`, activate with `source .venv/bin/activate`
- **Standard venv** (for development): `python3 -m venv venv`, activate with `source venv/bin/activate`

Install dependencies:
```bash
uv sync --extra gpu     # For GPU (CUDA 12.8)
uv sync --extra cpu     # For CPU/MPS
```

### Training Scripts

**Quick start ($100 tier, ~4 hours on 8XH100):**
```bash
bash speedrun.sh
```

**Larger models:**
```bash
bash run1000.sh         # $1000 tier, ~42 hours
bash miniseries.sh      # Miniseries experiments
```

**Individual training stages (typically run via torchrun):**
```bash
# Tokenizer training
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=65536
python -m scripts.tok_eval

# Base model pretraining (distributed)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --target-param-data-ratio=20
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Midtraining (conversation format)
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# Supervised Fine-Tuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Reinforcement Learning (optional)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

**Single GPU:** Omit `torchrun` entirely - the code automatically switches to gradient accumulation:
```bash
python -m scripts.base_train -- --depth=20
```

**CPU/MPS:** See `dev/runcpu.sh` for scaled-down hyperparameters.

### Inference

**Web UI (ChatGPT-like interface):**
```bash
python -m scripts.chat_web
```

**CLI:**
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"  # Single prompt
python -m scripts.chat_cli                            # Interactive mode
```

### Testing
```bash
python -m pytest tests/test_engine.py -v -s
```

### Data Management
```bash
# Download pretraining data shards (each ~250M chars, ~100MB)
python -m nanochat.dataset -n 8      # Download 8 shards
python -m nanochat.dataset -n 370    # For full speedrun
```

## Architecture

### Training Pipeline
1. **Tokenizer** (`nanochat/tokenizer.py`): BPE tokenizer (vocab size 65536 by default)
2. **Base Model** (`scripts/base_train.py`): Pretraining on raw text using distributed dataloader
3. **Midtraining** (`scripts/mid_train.py`): Teach conversation format, special tokens, tool use
4. **SFT** (`scripts/chat_sft.py`): Supervised fine-tuning on task datasets
5. **RL** (`scripts/chat_rl.py`): Optional reinforcement learning (GSM8K only currently)

### Model Architecture (`nanochat/gpt.py`)
- GPT-style Transformer with modern features:
  - Rotary embeddings (no positional embeddings)
  - QK normalization
  - Untied embedding/unembedding weights
  - ReLU² activation in MLP
  - Group-Query Attention (GQA) for efficient inference
  - Flash Attention 3 integration
  - Sliding window attention support (configurable pattern: L=full, S=half context)
  - No bias terms, no learnable params in RMSNorm

Model sizing: `model_dim = depth × aspect_ratio`, heads determined by `head_dim`

### Data Loading (`nanochat/dataloader.py`)
Two implementations:
1. **Original**: 100% token utilization, rows can start mid-document
2. **BOS-aligned (default)**: Every row starts with BOS token, best-fit packing, ~35% cropping overhead

The BOS-aligned loader is the default as it provides cleaner training signal when sufficient data is available.

### Distributed Training
- Uses PyTorch DDP via `torchrun`
- Two optimizers: **Muon** (matrix params) and **AdamW** (embeddings)
- Automatic gradient accumulation when batch size doesn't fit in VRAM
- Checkpoint management with resumption support

### Inference Engine (`nanochat/engine.py`)
Efficient inference with KV caching. Used by both CLI and web UI.

### Evaluation System (`tasks/`)
Tasks inherit from `Task` base class with two eval types:
- **categorical**: Multiple choice (ARC, MMLU)
- **generative**: Free-form generation (GSM8K, HumanEval)

Core benchmarks: ARC, MMLU, GSM8K, HumanEval, ChatCORE, CORE (base model)

### Important Files
- `nanochat/common.py`: Utilities, DDP helpers, device detection
- `nanochat/checkpoint_manager.py`: Save/load model checkpoints
- `nanochat/core_eval.py`: CORE score evaluation (base model quality metric)
- `nanochat/report.py`: Generate markdown report cards (`report.md`)
- `nanochat/execution.py`: Python code execution tool for LLM
- `nanochat/ui.html`: Frontend for web UI

## Key Concepts

### Hyperparameter Management
- Critical params: `--depth`, `--device-batch-size`, `--total-batch-size`
- **Memory management**: Reduce `--device-batch-size` until fit (32→16→8→4→2→1), gradient accumulation compensates
- **Training horizon**: Controlled by `--target-param-data-ratio` (Chinchilla optimal is 20, speedrun uses 20)
- Data requirements: ~20× params in tokens for Chinchilla optimal

### Model Tags & Checkpoints
Models are saved in `$NANOCHAT_BASE_DIR` (default: `~/.cache/nanochat/`) with auto-generated tags based on architecture (e.g., `d20_w1280_h10`). Override with `--model-tag`.

### Resume Training
Use `--resume-from-step=N` to resume from a checkpoint. The dataloader automatically resumes from approximately the same position.

### Customization
- **Identity/Personality**: See Discussion #139 - mix synthetic identity data into midtraining/SFT
- **New Abilities**: See Discussion #164 - example of teaching letter counting

### Wandb Integration
Set `WANDB_RUN=my_run_name` environment variable to enable wandb logging (must run `wandb login` first). Use `dummy` to disable.

## Code Conventions

- **No configuration monsters**: Simple, direct code over abstractions
- **Vanilla PyTorch**: Should run on any PyTorch-compatible device (cuda/cpu/mps/xpu)
- **Distributed-aware**: All training scripts handle both single and multi-GPU execution
- **Stateless where possible**: Clean interfaces, minimal global state
- **Self-documenting**: Parameter names and comments explain intent

### Variable Naming
- CLI args use dashes: `--device-batch-size`
- Python variables use underscores: `device_batch_size`
- This is the idiomatic convention (recent refactor from prior Configurator object)

## Development Notes

- Test changes with `dev/runcpu.sh` for quick iteration without GPUs
- The codebase is designed to be packaged and sent to LLMs for questions - keep it minimal
- When adding features, prefer direct implementation over adding abstraction layers
- PR policy: Disclose substantial LLM contributions that you don't fully understand
- ~330KB total, ~8K lines across 44 files - stay lean
