# Experiment 2: RL vs Extra SFT at Constant FLOPs

## AI NOTE:

AI GENERATED, NOT YET REVIEWED FOR ACCURACY. 
gstanley: Broad overview is correct but the FLOP-matching numbers are unreviewed.

## Motivation

Prior runs show that appending RL (on GSM8K) after SFT improves model performance.
However, RL also consumes additional training FLOPs beyond what the SFT-only pipeline uses.
This leaves open the question: **is the RL training signal (reward-based optimization)
actually contributing, or would spending the same FLOPs on additional supervised training
achieve the same gains?**

## Design

Both arms share an identical base model and initial SFT stage. They diverge only in
how the final training FLOPs are spent.

| Arm | Pipeline | Final stage details |
|-----|----------|---------------------|
| A (RL) | base → SFT → **RL on GSM8K** | Default RL config: GRPO with binary correctness reward, 16 samples/example, ~466 steps over 1 epoch of GSM8K train (7,473 examples) |
| B (SFT+) | base → SFT → **additional SFT on GSM8K** | Extra SFT steps on GSM8K data only, with total training FLOPs matched to Arm A's RL stage |

### FLOP matching

RL training FLOPs come from two sources:
1. **Generation (rollouts):** For each of the 7,473 training examples, 16 candidate
   completions are sampled (up to 256 tokens each). This is inference-cost dominated.
2. **Gradient updates:** The policy gradient step over the generated sequences.

Approximate RL FLOPs per step:
- Forward passes for generation: `num_examples_per_step × num_samples × avg_seq_len × flops_per_token`
- Training forward+backward: `num_examples_per_step × num_samples × avg_seq_len × 3 × flops_per_token`

For Arm B, we run additional SFT steps on GSM8K-only data (the same 7,473 train examples,
repeated as needed) until the cumulative training FLOPs match Arm A. SFT FLOPs per step
are straightforward: `total_batch_size_in_tokens × 3 × flops_per_token`.

The exact step count for Arm B will be computed at runtime by logging Arm A's total RL
FLOPs and dividing by Arm B's per-step SFT FLOPs.

### Controls

- **Same base model:** Identical pretraining (depth, data, steps).
- **Same initial SFT:** Both arms use the same SFT checkpoint as starting point.
- **Same eval:** Both arms evaluated on the same benchmarks with the same settings.
- **GSM8K data only** for the divergent stage in both arms — this isolates the
  training signal (reward vs supervision) rather than confounding with data differences.

## Evaluation

Primary metrics (via `scripts/chat_eval`):
- **GSM8K** pass@1 (the RL target task — most directly affected)
- **MMLU**, **ARC-Challenge**, **ARC-Easy** (general capability — check for regression)
- **HumanEval** (code generation — check for regression)
- **SpellingBee** (character-level reasoning — check for regression)

## Hypotheses

- **If Arm A (RL) > Arm B (SFT+):** The reward signal from RL provides optimization
  pressure that supervised training cannot replicate — RL is worth keeping.
- **If Arm A ≈ Arm B:** The gains previously attributed to RL are largely from
  additional exposure to GSM8K data, not from the reward signal itself.
- **If Arm A < Arm B:** RL is actively wasteful compared to supervised training on
  the same domain, possibly due to high-variance gradients or the generation overhead
  consuming FLOPs that don't translate to learning.

## Practical notes

- The `d12` depth setting in `runs/exp2.sh` is for debugging the compute environment
  only. The real experiment should use a depth comparable to the speedrun (d26) or
  whatever is standard for the compute budget.
- GSM8K is already included 2x in the default SFT data mixture (~16K rows out of
  ~858K total). Arm B's extra SFT adds GSM8K on top of that.
- Current RL implementation is GRPO-style (REINFORCE with mean-subtracted advantages,
  no KL penalty, no clipping). The reward is binary: 1 if the extracted numerical
  answer matches the reference, 0 otherwise.
