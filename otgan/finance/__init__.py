"""Finance track — the same OT engine, from images to markets.

This subpackage points the image pipeline's core at simulated market
log-returns instead of pixels, reusing it verbatim by direct import:
``otgan.sinkhorn.sinkhorn``, ``otgan.energy.energy_distance`` /
``compute_loss``, ``otgan.ema.EMAGenerator``, ``otgan.data.split_real_pair``
and ``otgan.device.seed_everything``. Zero new dependencies (numpy / torch /
matplotlib only), and CPU by default: return paths are tiny, and CPU
sidesteps MPS float64 quirks.

Re-exports are attached lazily via PEP 562 ``__getattr__``, mirroring
``otgan/__init__.py``: importing ``otgan.finance`` (or a single submodule)
must not pull in matplotlib or tensorboard.
"""

_LAZY = {
    "FinanceConfig": "otgan.finance.config",
    "gbm_paths": "otgan.finance.simulate",
    "heston_paths": "otgan.finance.simulate",
    "gjr_garch_paths": "otgan.finance.simulate",
    "prices_from_returns": "otgan.finance.simulate",
    "sinkhorn_reduce": "otgan.finance.reduce",
    "kmeans_reduce": "otgan.finance.reduce",
    "ReturnsTrainer": "otgan.finance.trainer",
    "stylized_facts_table": "otgan.finance.evaluate",
    "sinkhorn_divergence_metric": "otgan.finance.evaluate",
}

__all__ = [*_LAZY.keys()]


def __getattr__(name):
    import importlib

    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
