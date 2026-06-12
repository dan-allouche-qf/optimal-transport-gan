# Model card — OT-GAN / OT-CFM MNIST weights (v1.0.0)

Two checkpoints are released with
[v1.0.0](https://github.com/dan-allouche-qf/optimal-transport-gan/releases/tag/v1.0.0)
of [optimal-transport-gan](https://github.com/dan-allouche-qf/optimal-transport-gan):

| file | size | contents | FID@10k | KIDx1e3 | FID-LeNet |
|---|---|---|---|---|---|
| `otgan_mnist_sinkdiv_ema_v1.0.0.pt` | 59.3 MB | EMA generator only (Sinkhorn-divergence OT-GAN, best epoch) | 24.4 | 13.8 | 60.1 |
| `otcfm_mnist_v1.0.0.pt` | 12.7 MB | full OT-CFM checkpoint (model + EMA + optimizer + RNG state) | 11.4 | 9.6 | 12.6 |

Download URL pattern:
`https://github.com/dan-allouche-qf/optimal-transport-gan/releases/download/v1.0.0/<file>`

## Model details

### `otgan_mnist_sinkdiv_ema_v1.0.0.pt`

- **Type:** GAN generator (`otgan.models.OTGANGenerator`), trained adversarially against a
  CReLU critic with an L2-normalized embedding under the **debiased Sinkhorn-divergence**
  objective (`loss: sinkhorn_divergence` in `otgan/energy.py`; Genevay, Peyré & Cuturi,
  AISTATS 2018; Feydy et al., AISTATS 2019), the repo's modernization of OT-GAN
  (Salimans et al., ICLR 2018).
- **Architecture:** 100-d Gaussian latent → Linear → GLU → reshape 512×8×8 →
  2 × (nearest-neighbor upsample ×2 + 5×5 conv + GLU) → conv → Tanh → 1×32×32.
- **What is in the file:** EMA generator weights only (`ema`, decay 0.999), the full training
  `config`, the source epoch and provenance — exported by `scripts/export_generator.py` from
  `loss_compare/sinkdiv/ckpt/ot_gan_best.pt`. The critic and optimizer states are not
  included (the full checkpoint is ~250 MB; sampling needs none of it).
- **Checkpoint selection:** this is the **best** checkpoint of the run by FID, which is
  also the **final** epoch (epoch 7 of epochs 0–7) — best and final coincide here.

### `otcfm_mnist_v1.0.0.pt`

- **Type:** optimal-transport conditional flow matching (OT-CFM; Tong et al., TMLR 2024)
  implemented in `otgan/cfm.py` — a time-conditioned convolutional vector field
  (GroupNorm/SiLU blocks, sinusoidal time embedding) trained by MSE regression on
  straight-line velocities, with noise–data pairs coupled by the repo's own log-domain
  Sinkhorn solver (`otgan/sinkhorn.py`, `cfm_coupling: sinkhorn`). No critic, no minimax.
- **Sampling:** Euler integration of the learned flow, `ode_steps: 100` by default.
- **What is in the file:** the full training checkpoint — model, EMA weights (used for
  sampling), optimizer state, RNG state, and the training `config` (final epoch of 30).

## Training data

MNIST (LeCun et al.), train split (60,000 images), auto-downloaded by torchvision, resized
to 32×32 and normalized to [-1, 1] (`Normalize((0.5,), (0.5,))`). No other data was used.

## Training procedure

Both runs: **seed 11**, single seed, on an Apple M3 (MPS); FID/KID metrics computed on CPU
(MPS lacks float64). Both configs are stored inside the checkpoints and can be printed with
`otgan config` or recovered automatically by `otgan sample/eval --ckpt`.

**OT-GAN (Sinkhorn divergence)** — `configs/mnist.yaml` with overrides, i.e.:

```bash
otgan train -c configs/mnist.yaml --override n_epochs=8 sinkhorn_iters=50 fid_every=2 \
  loss=sinkhorn_divergence eval_dir=loss_compare/sinkdiv/eval \
  ckpt_dir=loss_compare/sinkdiv/ckpt log_dir=loss_compare/sinkdiv/logs
```

Key resolved hyperparameters: batch 64 (two independent real half-batches per step), Adam
lr 3e-4 (β₁ 0.5, β₂ 0.999), 1 critic step per `g2c_ratio=3` steps, ε 1.0, 50 Sinkhorn
iterations, EMA decay 0.999, `critic_sign=true`, 8 epochs. Wall-clock ~3.5 h.

**OT-CFM** — `configs/cfm_mnist.yaml` verbatim:

```bash
otgan train -c configs/cfm_mnist.yaml
```

Key hyperparameters: batch_size 128 (the shared loader yields 2×128 = 256 images per step,
so OT coupling plans are 256×256), Adam lr 2e-4 (β₁ 0.5, β₂ 0.999), `cfm_coupling=sinkhorn`,
`cfm_eps=0.05` (entropic ε on the mean-normalized squared-Euclidean cost), 50 Sinkhorn
iterations, `ode_steps=100`, EMA decay 0.999, 30 epochs. Wall-clock ~1.2 h.

## Evaluation

Protocol: 10,000 generated samples (EMA) vs 10,000 real test images, one fixed harness
(`otgan/metrics.py`): FID in torchmetrics-InceptionV3 2048-d features; KID (unbiased MMD²,
cubic kernel, 1000-sample subsets; Binkowski et al., ICLR 2018); FID-LeNet in the 128-d
penultimate features of an MNIST-trained LeNet (`otgan/lenet.py`); IS reported as a sanity
check only. Train-vs-test floor under this protocol: **FID 1.45 / FID-LeNet 1.76**.

Full context table (canonical source `assets/headline_results.md`):

| model | budget | FID@10k | KIDx1e3 | FID-LeNet | IS |
|---|---|---|---|---|---|
| OT-GAN, buggy critic sign (final, 18 ep) | ~6 h MPS | 177.9 | 191.0 | 208.8 | 2.34 |
| OT-GAN, corrected (final, 18 ep) | ~6 h MPS | 70.8 | 65.7 | 77.0 | 2.33 |
| OT-GAN, energy distance (8 ep) | ~3.5 h MPS | 81.2 | 75.3 | 69.1 | 2.33 |
| **OT-GAN, Sinkhorn divergence (8 ep) — released** | ~3.5 h MPS | **24.4** | **13.8** | **60.1** | 1.98 |
| DCGAN baseline (10 ep) | ~15 min MPS | 72.3 | 74.6 | 36.8 | 2.19 |
| I-CFM, no coupling (30 ep) | ~1.2 h MPS | 10.9 | 8.9 | 12.8 | 1.91 |
| **OT-CFM, Sinkhorn coupling (30 ep) — released** | ~1.2 h MPS | **11.4** | **9.6** | **12.6** | 1.90 |
| *train-vs-test floor* | - | 1.45 | - | 1.76 | - |

The I-CFM row is the no-coupling control at identical capacity and budget: it matches the
released OT-CFM within evaluation noise — a null result for the coupling at this scale.

Reproduce the numbers for the released files:

```bash
otgan eval --ckpt otgan_mnist_sinkdiv_ema_v1.0.0.pt
otgan eval --ckpt otcfm_mnist_v1.0.0.pt
```

## Caveats — read before comparing

- **Final-vs-best.** The 18-epoch OT-GAN rows above report the *final* epoch; the
  energy-distance critic collapses late, so their best checkpoints were earlier (legacy
  protocol, n_eval=2048: best FID 69.9 at epoch 8 vs final 77.2). The released OT-GAN
  weight is the **best-FID** checkpoint of the Sinkhorn-divergence run, which is also its
  **final** epoch (epoch 7 of epochs 0–7) — that run shows no collapse, so best and final
  coincide; the final-epoch caveat applies only to the 18-epoch legacy rows.
- **Protocol change.** Numbers 77.2 / 69.9 / 185.3 anywhere in the repo history are the old
  n_eval=2048 protocol (`assets/ablation_table.md`); everything in this card is the
  n_eval=10,000 protocol. Do not mix the two.
- **Not SOTA, not converged.** Single seed, laptop budgets; Lucic et al. (NeurIPS 2018)
  report WGAN ≈ 6.7 on MNIST at much larger budgets. The Sinkhorn-divergence run was still
  improving when its 8-epoch budget ended.
- **Evaluation noise.** Repeat evaluations of one checkpoint differ by ≈ ±0.3 FID — the
  released OT-CFM scored 11.4 in-training vs 11.1 re-evaluated at the default 100 steps —
  so re-running `otgan eval` on it gives ≈ 11.1. Treat sub-unit gaps as noise.
- **Inception features are weak on MNIST** — FID and FID-LeNet can disagree on rankings
  (they do for the DCGAN row). Compare models on all three metrics, never on IS.

## Intended use

Education, portfolio and research: studying minibatch-OT objectives for generative models
(2018 energy distance → 2019 Sinkhorn divergence → 2024 flow matching) under a controlled
evaluation harness. Generating MNIST digits has no product value in itself.

**Out of scope:** any production use; generation of natural images (MNIST-only training);
the finance track ships no weights (its runs take minutes — retrain from configs); any
claim of state-of-the-art performance.

## Checksums

```
98d813494c0989fef900491416a78f304af2d2ea09e28b1611bf43d4254e5258  otgan_mnist_sinkdiv_ema_v1.0.0.pt
da0939da2f9a17d36adb7a646c4b89e4a222ae8afa955c6fcfaf39ba4a8acd38  otcfm_mnist_v1.0.0.pt
```

Verify either file with `shasum -a 256 <file>`.

## How to load

```bash
# Sample a grid (config is recovered from the checkpoint — works for both files):
otgan sample --ckpt otgan_mnist_sinkdiv_ema_v1.0.0.pt -n 64 -o samples.png
otgan sample --ckpt otcfm_mnist_v1.0.0.pt -n 64 -o samples_cfm.png

# Integrity + sample check for the generator-only export:
python scripts/export_generator.py --verify otgan_mnist_sinkdiv_ema_v1.0.0.pt
```

```python
# Pure-PyTorch load of the generator-only file:
import torch
from otgan.config import Config
from otgan.models import OTGANGenerator
from otgan.trainer import denormalize

ckpt = torch.load("otgan_mnist_sinkdiv_ema_v1.0.0.pt", map_location="cpu", weights_only=False)
cfg = Config(**ckpt["config"])
gen = OTGANGenerator(z_dim=cfg.z_dim, channels=cfg.channels)
gen.load_state_dict(ckpt["ema"])
gen.eval()
imgs = denormalize(gen(torch.randn(64, cfg.z_dim)))  # 64 x 1 x 32 x 32 in [0, 1]
```

## Authors & license

Dan Allouche and Nicolas Dahan. MIT license (see `LICENSE`). Please cite Salimans et al.
(ICLR 2018) for OT-GAN and Tong et al. (TMLR 2024) for OT-CFM; BibTeX in the
[README](README.md#citing).
