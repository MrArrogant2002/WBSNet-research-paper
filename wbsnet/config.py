from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")
    return data


def _load_recursive(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    base_config = data.pop("base_config", None)
    if not base_config:
        return data
    base_path = Path(base_config)
    if not base_path.is_absolute():
        candidate = (path.parent / base_path).resolve()
        if candidate.exists():
            base_path = candidate
        else:
            repo_candidate = (Path.cwd() / base_path).resolve()
            base_path = repo_candidate
    return _deep_merge(_load_recursive(base_path), data)


def _parse_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_nested_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    unknown_prefix: list[str] = []
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            if key not in cursor:
                unknown_prefix = keys[: keys.index(key) + 1]
            cursor[key] = {}
        cursor = cursor[key]
    if keys[-1] not in cursor and not unknown_prefix:
        unknown_prefix = keys
    if unknown_prefix:
        warnings.warn(
            f"Override key '{dotted_key}' does not match any existing config field "
            f"(created new path: {'.'.join(unknown_prefix)}). Check for typos.",
            stacklevel=3,
        )
    cursor[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    merged = deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override}")
        key, raw_value = override.split("=", 1)
        _set_nested_key(merged, key, _parse_value(raw_value))
    return merged


def _ensure_defaults(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    resolved = deepcopy(config)
    experiment = resolved.setdefault("experiment", {})
    if not experiment.get("run_name"):
        experiment["run_name"] = experiment.get("name", config_path.stem)
    experiment["config_path"] = str(config_path.resolve())
    return resolved


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = _load_recursive(config_path)
    config = apply_overrides(config, overrides)
    return _ensure_defaults(config, config_path)
