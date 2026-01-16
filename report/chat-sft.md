## Chat SFT
timestamp: 2026-01-16 05:45:39

- run: dummy_mistakes_no
- device_type: 
- dtype: bfloat16
- source: mid
- model_tag: mistakes_no
- model_step: None
- num_epochs: 1
- num_iterations: -1
- device_batch_size: 4
- target_examples_per_step: 32
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- use_mistakes: no
- Training rows: 4966
- Number of iterations: 155
- Training loss: 0.4103
- Validation loss: 0.9448

