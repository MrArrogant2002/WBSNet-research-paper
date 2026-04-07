from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        loaded[key] = value

    if os.environ.get("WAND_API_KEY") and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.environ["WAND_API_KEY"]
        loaded["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    return loaded
