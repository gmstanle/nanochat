# Experiment 2: Daily Log

## 2026-02-05

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
### Next steps

- IDEA: switch to a simpler task (counting letters?) for RL experimentation.

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
