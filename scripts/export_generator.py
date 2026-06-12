"""Export a release-ready, generator-only checkpoint from a full training checkpoint.

A full OT-GAN checkpoint carries the critic and both optimizer states (~250 MB);
sampling only needs the EMA generator (~59 MB). The export keeps the config and
provenance so `otgan sample/eval` can rebuild everything from the released file.

Usage:
    python scripts/export_generator.py <full_ckpt.pt> [-o dist/out.pt]
    python scripts/export_generator.py --verify dist/out.pt   # sample a grid
"""

import argparse
import hashlib
from pathlib import Path

import torch


def export(src: str, out: str) -> str:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "ema" in ckpt and "config" in ckpt):
        raise SystemExit(f"{src} is not a full OT-GAN checkpoint (need 'ema' and 'config' keys)")
    slim = {
        "ema": ckpt["ema"],  # EMA generator weights — what sample() uses
        "config": ckpt["config"],
        "epoch": ckpt.get("epoch"),
        "source_checkpoint": str(src),
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  ({size_mb:.1f} MB)")
    print(f"sha256  {digest}")
    return digest


def verify(path: str, n: int = 64) -> None:
    from torchvision.utils import make_grid, save_image

    from otgan.config import Config
    from otgan.models import OTGANGenerator
    from otgan.trainer import denormalize

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    gen = OTGANGenerator(z_dim=cfg.z_dim, channels=cfg.channels)
    gen.load_state_dict(ckpt["ema"])
    gen.eval()
    with torch.no_grad():
        imgs = denormalize(gen(torch.randn(n, cfg.z_dim)))
    out = Path(path).with_suffix(".verify.png")
    save_image(make_grid(imgs, nrow=8), out)
    print(f"verification grid -> {out} (epoch {ckpt.get('epoch')}, config OK)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Full checkpoint to export, or slim file with --verify")
    p.add_argument("-o", "--out", default=None, help="Output path (default dist/<name>)")
    p.add_argument("--verify", action="store_true", help="Load a slim export and sample a grid")
    args = p.parse_args()
    if args.verify:
        verify(args.checkpoint)
    else:
        out = args.out or f"dist/otgan_mnist_generator_ema_{Path(args.checkpoint).stem}.pt"
        export(args.checkpoint, out)


if __name__ == "__main__":
    main()
