# Experiment 2: Log

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
