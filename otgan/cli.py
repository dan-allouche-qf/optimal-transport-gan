"""Command-line interface: ``otgan train|sample|eval|ablate|config``."""

import argparse
from dataclasses import fields

import yaml

from otgan.config import Config


def _coerce(name: str, value: str, cls=Config):
    """Coerce a string override to the dataclass field's type.

    ``none``/``null`` map to ``None`` only for Optional fields — for plain
    ``str`` fields they stay literal strings (e.g. ``cfm_coupling=none``).
    """
    t = {f.name: str(f.type) for f in fields(cls)}.get(name, "str")
    if value.lower() in ("null", "none") and ("None" in t or "Optional" in t):
        return None
    if "bool" in t:
        return value.lower() in ("1", "true", "yes")
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)
    return value


def _parse_overrides(overrides, cls=Config) -> dict:
    """Parse a list of ``key=value`` strings into a coerced dict."""
    out = {}
    valid = {f.name for f in fields(cls)}
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--override expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        if key not in valid:
            raise SystemExit(f"Unknown config key in --override: {key}")
        out[key] = _coerce(key, value, cls)
    return out


def _load_config(path: str, overrides) -> Config:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    data.update(_parse_overrides(overrides))
    return Config(**data)


def _load_finance_config(path: str, overrides):
    from otgan.finance.config import FinanceConfig  # lazy

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    data.update(_parse_overrides(overrides, FinanceConfig))
    return FinanceConfig(**data)


def _config_from_ckpt(path: str, overrides) -> Config:
    """Recover the Config stored inside a checkpoint (new fields keep defaults)."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "config" in ckpt):
        raise SystemExit(f"{path} stores no config; pass -c/--config explicitly")
    data = dict(ckpt["config"])
    data.update(_parse_overrides(overrides))
    return Config(**data)


def _resolve_config(args) -> Config:
    if args.config:
        return _load_config(args.config, args.override)
    if getattr(args, "ckpt", None):
        return _config_from_ckpt(args.ckpt, args.override)
    raise SystemExit("either -c/--config or --ckpt is required")


def _cmd_train(args):
    from otgan.trainer import build_trainer

    cfg = _load_config(args.config, args.override)
    trainer = build_trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at epoch {trainer.start_epoch}")
    trainer.fit()


def _cmd_sample(args):
    from torchvision.utils import make_grid, save_image

    from otgan.trainer import build_trainer, denormalize

    cfg = _resolve_config(args)
    trainer = build_trainer(cfg)
    if args.ckpt:
        trainer.load_checkpoint(args.ckpt)
    imgs = denormalize(trainer.sample(args.n))
    grid = make_grid(imgs, nrow=args.nrow)
    save_image(grid, args.out)
    print(f"Wrote {args.n} samples to {args.out}")


def _cmd_eval(args):
    import json

    from otgan.metrics import FIDISEvaluator
    from otgan.trainer import build_trainer

    cfg = _resolve_config(args)
    trainer = build_trainer(cfg)
    if args.ckpt:
        trainer.load_checkpoint(args.ckpt)
    evaluator = FIDISEvaluator(cfg, trainer.device)
    if getattr(args, "floor", False):
        metrics = evaluator.fid_floor()
        print(json.dumps({"fid_floor": metrics}, indent=2))
        return
    metrics = evaluator.evaluate(trainer)
    print(json.dumps(metrics, indent=2))


def _cmd_ablate(args):
    from otgan.ablation import run_ablation

    run_ablation(args.config, args.axis, _parse_overrides(args.override))


def _cmd_finance_train(args):
    from otgan.finance.trainer import ReturnsTrainer  # lazy

    cfg = _load_finance_config(args.config, args.override)
    trainer = ReturnsTrainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.step}")
    trainer.fit()


def _cmd_finance_eval(args):
    import json

    from otgan.finance.evaluate import (  # lazy
        sinkhorn_divergence_metric,
        stylized_facts_table,
        to_markdown,
    )
    from otgan.finance.trainer import ReturnsTrainer

    cfg = _load_finance_config(args.config, args.override)
    trainer = ReturnsTrainer(cfg)
    trainer.load_checkpoint(args.ckpt)
    fake = trainer.sample(cfg.n_eval_paths).squeeze(1)
    real = trainer.eval_paths.squeeze(1) * trainer.scale
    print(to_markdown(stylized_facts_table(real, fake)))
    divergence = sinkhorn_divergence_metric(real, fake, cfg.epsilon, cfg.sinkhorn_iters)
    print(json.dumps({"sinkhorn_divergence": float(divergence)}, indent=2))


def _cmd_finance_reduce(args):
    import json

    from otgan.finance.reduce import (  # lazy
        holdout_split,
        kmeans_reduce,
        random_subsample,
        reduction_report,
        sinkhorn_reduce,
    )
    from otgan.finance.trainer import ReturnsTrainer

    cfg = _load_finance_config(args.config, args.override)
    train, _ = ReturnsTrainer._build_target(cfg)
    fit_half, eval_half = holdout_split(train)
    reduced = sinkhorn_reduce(
        fit_half, args.k, epsilon=args.epsilon, sinkhorn_iters=cfg.sinkhorn_iters, seed=cfg.seed
    )
    report = {"sinkhorn": reduction_report(eval_half, reduced)}
    if args.compare:
        report["kmeans"] = reduction_report(eval_half, kmeans_reduce(fit_half, args.k, cfg.seed))
        report["random"] = reduction_report(eval_half, random_subsample(fit_half, args.k, cfg.seed))
    print(json.dumps(report, indent=2))


def _cmd_config(args):
    _load_config(args.config, args.override).print_config()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="otgan", description="OT-GAN training and evaluation.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, config_required: bool = True):
        sp.add_argument("-c", "--config", required=config_required, help="Path to a YAML config")
        sp.add_argument("--override", nargs="*", default=[], help="Override config: key=value ...")

    sp = sub.add_parser("train", help="Train an OT-GAN")
    common(sp)
    sp.add_argument("--resume", help="Checkpoint to resume from")
    sp.set_defaults(func=_cmd_train)

    sp = sub.add_parser("sample", help="Sample images from a checkpoint")
    common(sp, config_required=False)
    sp.add_argument("--ckpt", help="Checkpoint to load (config recovered from it if -c omitted)")
    sp.add_argument("-n", type=int, default=64, help="Number of samples")
    sp.add_argument("--nrow", type=int, default=8)
    sp.add_argument("-o", "--out", default="samples.png")
    sp.set_defaults(func=_cmd_sample)

    sp = sub.add_parser("eval", help="Compute FID/KID/IS for a checkpoint")
    common(sp, config_required=False)
    sp.add_argument("--ckpt", help="Checkpoint to load (config recovered from it if -c omitted)")
    sp.add_argument("--floor", action="store_true", help="Report the train-vs-test FID floor")
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("ablate", help="Run an ablation study")
    common(sp)
    sp.add_argument(
        "--axis", default="critic_sign", choices=["critic_sign", "epsilon", "g2c_ratio"]
    )
    sp.set_defaults(func=_cmd_ablate)

    sp = sub.add_parser("config", help="Print the resolved configuration")
    common(sp)
    sp.set_defaults(func=_cmd_config)

    sp = sub.add_parser("finance-train", help="Train the returns generator (FinanceConfig YAML)")
    common(sp)
    sp.add_argument("--resume", help="Checkpoint to resume from")
    sp.set_defaults(func=_cmd_finance_train)

    sp = sub.add_parser("finance-eval", help="Stylized facts + Sinkhorn divergence for a ckpt")
    common(sp)
    sp.add_argument("--ckpt", required=True, help="returns_gan.pt checkpoint")
    sp.set_defaults(func=_cmd_finance_eval)

    sp = sub.add_parser("finance-reduce", help="OT scenario reduction (zero training)")
    common(sp)
    sp.add_argument("-K", "--k", type=int, default=100, help="Number of scenarios to keep")
    sp.add_argument("--epsilon", type=float, default=0.01, help="Entropic eps (dimensionless)")
    sp.add_argument("--compare", action="store_true", help="Also report k-means and random")
    sp.set_defaults(func=_cmd_finance_reduce)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
