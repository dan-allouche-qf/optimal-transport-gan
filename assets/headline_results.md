# Headline results (FID/KID @ 10k samples, torchmetrics-Inception + MNIST-LeNet)

| model | budget | FID@10k | KIDx1e3 | FID-LeNet | IS |
|---|---|---|---|---|---|
| OT-GAN, buggy critic sign (final, 18 ep) | ~6 h MPS | 177.9 | 191.0 | 208.8 | 2.34 |
| OT-GAN, corrected (final, 18 ep) | ~6 h MPS | 70.8 | 65.7 | 77.0 | 2.33 |
| OT-GAN, energy distance (8 ep) | ~3.5 h MPS | 81.2 | 75.3 | 69.1 | 2.33 |
| OT-GAN, Sinkhorn divergence (8 ep) | ~3.5 h MPS | 24.4 | 13.8 | 60.1 | 1.98 |
| DCGAN baseline (10 ep) | ~15 min MPS | 72.3 | 74.6 | 36.8 | 2.19 |
| I-CFM, no coupling (30 ep) | ~1.2 h MPS | 10.9 | 8.9 | 12.8 | 1.91 |
| OT-CFM, Sinkhorn coupling (30 ep) | ~1.2 h MPS | 11.4 | 9.6 | 12.6 | 1.90 |
| *train-vs-test floor* | - | 1.45 | - | 1.76 | - |

## Sampler cost (NFE = Euler steps), best checkpoints

| NFE | OT-CFM FID@10k | OT-CFM KIDx1e3 | OT-CFM FID-LeNet | I-CFM FID@10k |
|---|---|---|---|---|
| 10 | 13.5 | 11.3 | 34.6 | 13.3 |
| 50 | 10.8 | 8.7 | 15.2 | - |
| 100 | 11.1 | 9.1 | 13.6 | 10.9 |
