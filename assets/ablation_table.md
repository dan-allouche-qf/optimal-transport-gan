# Ablation: critic_sign

Legacy protocol: in-training per-epoch FID at n_eval=2048 — biased
upward vs the @10k tables; see headline_results.md for the re-evaluations.

| run | final FID | final IS | final D^2 |
|-----|-----------|----------|-----------|
| critic_sign=True | 77.17 | 2.32 | +0.0018 |
| critic_sign=False | 185.31 | 2.32 | +0.0000 |
