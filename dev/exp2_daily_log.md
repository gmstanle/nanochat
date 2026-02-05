# Experiment 2: Daily Log

## 2025-02-04

### Progress

1. **Got the full pipeline working in test mode on an A100 node**

2. **Made some QoL improvements to the run script `runs/exp2.sh`**
   - Should now automatically work on H100 and A100 nodes of any number of GPUs

3. **Tried out RL and noted that, while it ran without error, it ran at very low ~10% GPU utilization**
   - Did some research/talked to AI and seems like that is due to using on-policy RL
   - On-policy has stability benefits but is slow — I think because it runs one sample at a time rather than batch processing
   - Offline is supposedly much faster due to batch processing but has stability issues, presumably because each sample in the batch produces a different policy? Something like that.

Currently reading through https://cameronrwolfe.substack.com/p/reinforce 
