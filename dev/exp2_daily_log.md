# Experiment 2: Log

## 2026-04-09

### SkyPilot / AWS note

- SkyPilot is working end to end on AWS for `exp2`; the current full run is on a single `A10G` (`g5.2xlarge`) with wandb disabled.

### Base pretraining estimates

- `1x A10G` on `g5.2xlarge` at about `$1.21/hr`:
  - steady-state training rate from live logs is about `8.45s/step` and `62k tok/s`
  - full `base_train` from scratch is about `1,785` steps, so about `4.2h` and about `$5.1` of instance cost
  - from the current live run at step `75/1785`, remaining `base_train` ETA is about `241m` (`4.0h`) and about `$4.9`
- `8x A10G` on `g5.48xlarge` at about `$16.29/hr`:
  - ideal scaling from reducing gradient accumulation from `16` to `2` would be about `1.06s/step`, `496k tok/s`, `31.5m`, and about `$8.6`
  - more realistic expectation is about `1.2-1.4s/step`, `375k-440k tok/s`, `36-43m`, and about `$9.8-$11.7`
  - practical takeaway: `8x A10G` is likely about `6-7x` faster for `base_train`, but not cheaper than the `1x A10G` run

## 2026-03-16

Topline goal: tune a `d12` model into the regime where held-out `SpellingBee` accuracy is roughly `10-25%` after SFT, so it becomes a cheap, learnable target for RL experiments.

### Plan

- Shift exp2 to a `d12` letter-counting sweep instead of GSM8K.
- Build a word-disjoint split for both `SpellingBee` and `SimpleSpelling` so held-out validation/test words never appear in SFT train data.
- Use a fixed `SpellingBee` validation split of `2,000` examples and a fixed held-out test split for eval.
- Train one shared `d12` base checkpoint, then run SFT sweeps with `SpellingBee` train sizes of `5k`, `20k`, `40k`, and `80k`.
- Save the base and every SFT variant to distinct checkpoint tags and run eval on each of them.
- Temporarily comment out GSM8K SFT/eval usage and skip the RL stage while this letter-counting sweep is in progress.

## 2026-02-06

- Research on past RL results
   - indeed, GSM8K perf was very low.	Base: 0.0250	SFT: 0.0455	RL: 0.0758
   - seems like real improvement, but is 8% success on ~1500 validation samples
      enough to read off small improvements and hill climb? Possible, but not ideal.
   - Real problem is 90-95% of compute is used on wrong answers. So if compute
     efficiency is more important (e.g., for pedagogy, rapid experimentation),
     then we want higher success rate.

- IDEA: INFRA: want to make run script automatically push all eval results to google drive.
  even w/chatgpt theres gonna be a bit of pain with setup so for now just leave evals on labmda.



## 2026-02-05

### Next steps

- dig up
    GSM8k scores (old nanochat commmit or gstanly/exp1). Answer question: **Is it even worth trying to
    RL on GSM8k with a d20-size model?** I recall karpathy posting 4% success rate after
    fine tuning and 8% after RL on a d26 model. This may be too small to give meaningful
   signal for experimentation anyway.
    
- Switch to a simpler task (counting letters?) for small-model RL experimentation?
  - Use SpellingBee?
  - TODO: Check SpellingBee scores @ small model size. If 30-60%, could be a great candidate for RL.


### Notes

- Running RL in test mode (12 layers). GPU mem usage seems high (~65GB) but utilization is low (<100W/400W). This may just be a function of using a small test model, although still annoying and wasting GPU time.
- `scripts/chat_rl.py` training loop is L229.
- At this rate it would take ~170 minutes (~2.8 hours, ~2h 50m) to get even the small model through the default 467 RL steps.
- This ~3 hours estimate was on a full 8xA100 node, which is my max computational budget to try out RL.
- There is no way it will be faster with a full model (and it will probably be slower).
- This is RL on *just* GSM8K, nevermind all of the other datasets used for training. Something is very slow here.
- QUESTION: there are 8.5k RL tasks ("environments"?) via GSM8K. What does 1 step correspond to?
- At least I got ~90 steps of RL to run without error on the d12 model, so I can be confident that larger-model runs will also work (unless OOM).
- Observation: the average reward of the d12 model is usually zero, so there is no learning happening. Unlike supervised training (where there are enough correct tokens in even small models for learning to happen), it seems like I need a model large enough to get some answers right for GSM8K-based RL to work.
- From `README.md`, Karpathy suggests using d12 models to experiment and optimize against validation loss, CORE metric, and compute. Seems this won’t work for GSM8K-based RL.



## 2025-02-04

### Progress

1. **Got the full pipeline working in test mode on an A100 node**

2. **Made some QoL improvements to the run script `runs/exp2.sh`**
   - Should now automatically work on H100 and A100 nodes of any number of GPUs

3. **Tried out RL and noted that, while it ran without error, it ran at very low ~10% GPU utilization**
   - Did some research/talked to AI and seems like that is due to using on-policy RL
   - On-policy has stability benefits but is slow — I think because it runs one sample at a time rather than batch processing
   - Offline is supposedly much faster due to batch processing but has stability issues, presumably because each sample in the batch starts from the same initial policy but produces a different gradient-updated policy? Something like that. Rather than policy updating sequentially. In supervised batch learning, the gradient is accumulated over the examples in the batch, all backpropped with the same initial weights, and then the weights are updated. Why can't RL have the same concept of gradient accumulation?

Currently reading through https://cameronrwolfe.substack.com/p/reinforce

Playing around with Codex app.
