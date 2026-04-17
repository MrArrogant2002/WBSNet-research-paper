from __future__ import annotations

import compileall
import sys
from pathlib import Path


def validate_configs(root: Path) -> list[str]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wbsnet.config import load_config

    errors: list[str] = []
    for path in sorted((root / "configs").glob("*.yaml")):
        try:
            load_config(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def runtime_smoke(root: Path) -> list[str]:
    try:
        import torch
    except Exception:
        return []

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from wbsnet.config import load_config
    from wbsnet.losses import total_loss
    from wbsnet.models import build_model

    errors: list[str] = []
    for path in sorted((root / "configs").glob("*.yaml")):
        try:
            config = load_config(path)
            config["model"]["encoder_pretrained"] = False
            config["model"]["encoder_pretrained_checkpoint"] = None
            height = 64
            width = 64
            model = build_model(config)
            model.eval()
            with torch.no_grad():
                images = torch.randn(1, int(config["model"].get("in_channels", 3)), height, width)
                masks = torch.zeros(1, int(config["model"].get("num_classes", 1)), height, width)
                output = model(images)
                if tuple(output["logits"].shape) != tuple(masks.shape):
                    raise RuntimeError(
                        f"Unexpected logits shape {tuple(output['logits'].shape)} for expected {tuple(masks.shape)}"
                    )
                total_loss(output, masks, float(config["train"].get("boundary_loss_weight", 0.5)))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ok = compileall.compile_dir(root / "wbsnet", quiet=1)
    ok = compileall.compile_file(root / "train.py", quiet=1) and ok
    ok = compileall.compile_file(root / "evaluate.py", quiet=1) and ok
    ok = compileall.compile_file(root / "predict.py", quiet=1) and ok
    ok = compileall.compile_file(root / "aggregate_results.py", quiet=1) and ok
    ok = compileall.compile_dir(root / "scripts", quiet=1) and ok
    config_errors = validate_configs(root)
    runtime_errors = runtime_smoke(root)

    if not ok or config_errors or runtime_errors:
        if not ok:
            print("Python compilation failed for one or more files.")
        for item in config_errors:
            print(item)
        for item in runtime_errors:
            print(item)
        sys.exit(1)

    print("Repository verification passed.")


if __name__ == "__main__":
    main()
