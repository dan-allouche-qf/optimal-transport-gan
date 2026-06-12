"""Portable output paths. Honors the ``OT_GAN_ROOT`` env var (e.g. a Drive folder
on Colab); defaults to the current working directory locally.
"""

import os
from pathlib import Path


def get_root() -> Path:
    return Path(os.environ.get("OT_GAN_ROOT", Path.cwd()))


def resolve(sub, create: bool = False) -> Path:
    path = get_root() / sub
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
