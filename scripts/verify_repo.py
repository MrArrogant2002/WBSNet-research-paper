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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ok = compileall.compile_dir(root / "wbsnet", quiet=1)
    ok = compileall.compile_file(root / "train.py", quiet=1) and ok
    ok = compileall.compile_file(root / "evaluate.py", quiet=1) and ok
    ok = compileall.compile_file(root / "predict.py", quiet=1) and ok
    ok = compileall.compile_file(root / "aggregate_results.py", quiet=1) and ok
    ok = compileall.compile_dir(root / "scripts", quiet=1) and ok
    config_errors = validate_configs(root)

    if not ok or config_errors:
        if not ok:
            print("Python compilation failed for one or more files.")
        for item in config_errors:
            print(item)
        sys.exit(1)

    print("Repository verification passed.")


if __name__ == "__main__":
    main()
