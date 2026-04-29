from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ..utils.distributed import DistributedState
from .transforms import build_transform


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path
    sample_id: str


def _discover_samples_under(root: Path, dataset_config: dict[str, Any]) -> list[SampleRecord]:
    image_dir = root / dataset_config.get("image_dir", "images")
    mask_dir = root / dataset_config.get("mask_dir", "masks")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    image_map = _list_files(image_dir, dataset_config.get("image_extensions", [".png", ".jpg", ".jpeg"]))
    mask_map = _list_files(mask_dir, dataset_config.get("mask_extensions", [".png", ".jpg", ".jpeg"]))

    shared_keys = sorted(set(image_map) & set(mask_map))
    if not shared_keys:
        raise RuntimeError(f"No matching image/mask pairs found under {root}")

    return [SampleRecord(image_map[key], mask_map[key], key) for key in shared_keys]


def _list_files(path: Path, extensions: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    allowed_extensions = {ext.lower() for ext in extensions}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in allowed_extensions:
            if item.stem in files:
                raise RuntimeError(
                    f"Duplicate sample stem '{item.stem}' under {path}. "
                    "Image/mask pairing is stem-based, so stems must be unique."
                )
            files[item.stem] = item
    return files


def discover_samples(dataset_config: dict[str, Any]) -> list[SampleRecord]:
    root = Path(dataset_config["root"])
    return _discover_samples_under(root, dataset_config)


def _available_pre_split_dirs(dataset_config: dict[str, Any]) -> list[str]:
    root = Path(dataset_config["root"])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    return sorted(item.name for item in root.iterdir() if item.is_dir())


def _discover_all_pre_split_samples(dataset_config: dict[str, Any]) -> list[SampleRecord]:
    root = Path(dataset_config["root"])
    if (root / dataset_config.get("image_dir", "images")).exists():
        return _discover_samples_under(root, dataset_config)

    records: list[SampleRecord] = []
    for split_name in _available_pre_split_dirs(dataset_config):
        split_root = root / split_name
        try:
            records.extend(_discover_samples_under(split_root, dataset_config))
        except FileNotFoundError:
            continue
    if not records:
        raise RuntimeError(f"No image/mask pairs found in pre-split dataset root: {root}")
    return records


def _resolve_split_file(dataset_config: dict[str, Any], split: str) -> Path:
    split_files = dataset_config.get("split_files", {})
    if split not in split_files:
        raise ValueError(f"Missing split file for '{split}' in dataset.split_files")
    split_file = Path(split_files[split])
    if split_file.is_absolute():
        return split_file
    return Path(dataset_config["root"]) / split_file


def _split_from_file(samples: list[SampleRecord], dataset_config: dict[str, Any], split: str) -> list[SampleRecord]:
    sample_map = {sample.sample_id: sample for sample in samples}
    split_path = _resolve_split_file(dataset_config, split)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    selected_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [sample_id for sample_id in selected_ids if sample_id not in sample_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Split file {split_path} references unknown sample ids: {preview}")
    return [sample_map[sample_id] for sample_id in selected_ids]


def split_samples(samples: list[SampleRecord], dataset_config: dict[str, Any], split: str) -> list[SampleRecord]:
    if split == "all":
        if dataset_config.get("split_strategy") == "pre_split_dirs":
            return _discover_all_pre_split_samples(dataset_config)
        return list(samples)

    strategy = dataset_config.get("split_strategy", "ratio")
    if strategy == "pre_split_dirs":
        root = Path(dataset_config["root"])
        split_root = root / split
        return _discover_samples_under(split_root, dataset_config)
    if strategy == "predefined":
        return _split_from_file(samples, dataset_config, split)
    if strategy != "ratio":
        raise ValueError(f"Unsupported split strategy: {strategy}")

    ratios = dataset_config.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = int(dataset_config.get("split_seed", 3407))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(samples), generator=generator).tolist()
    ordered = [samples[idx] for idx in indices]

    n_total = len(ordered)
    n_train = math.floor(n_total * ratios["train"])
    n_val = math.floor(n_total * ratios["val"])
    n_test = n_total - n_train - n_val

    slices = {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val : n_train + n_val + n_test],
    }
    if split not in slices:
        raise ValueError(f"Unsupported split: {split}")
    return slices[split]


def _require_non_empty(samples: list[SampleRecord], split: str, dataset_config: dict[str, Any]) -> None:
    if not samples:
        raise RuntimeError(
            f"Split '{split}' for dataset '{dataset_config.get('name', 'unknown')}' is empty. "
            f"Check dataset.root={dataset_config.get('root')} and split_strategy={dataset_config.get('split_strategy', 'ratio')}."
        )


class BinarySegmentationDataset(Dataset):
    def __init__(self, samples: list[SampleRecord], dataset_config: dict[str, Any], train: bool) -> None:
        self.samples = samples
        self.transform = build_transform(dataset_config, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            with Image.open(sample.mask_path) as mask:
                image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }


def _loader_kwargs(dataset_config: dict[str, Any]) -> dict[str, Any]:
    num_workers = int(dataset_config.get("num_workers", 4))
    persistent_workers = bool(dataset_config.get("persistent_workers", True))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(dataset_config.get("pin_memory", True)),
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0 and "prefetch_factor" in dataset_config:
        prefetch_factor = int(dataset_config["prefetch_factor"])
        if prefetch_factor > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return loader_kwargs


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_loader(
    dataset: Dataset,
    dataset_config: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    distributed_state: DistributedState,
) -> DataLoader:
    sampler = None
    generator = torch.Generator()
    generator.manual_seed(int(dataset_config.get("split_seed", 3407)))
    if distributed_state.enabled:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=shuffle,
        worker_init_fn=_seed_worker,
        generator=generator,
        **_loader_kwargs(dataset_config),
    )


def build_dataloaders(config: dict[str, Any], distributed_state: DistributedState) -> dict[str, DataLoader]:
    dataset_config = config["dataset"]
    samples = [] if dataset_config.get("split_strategy") == "pre_split_dirs" else discover_samples(dataset_config)
    splits = {
        "train": split_samples(samples, dataset_config, "train"),
        "val": split_samples(samples, dataset_config, "val"),
    }
    _require_non_empty(splits["train"], "train", dataset_config)
    _require_non_empty(splits["val"], "val", dataset_config)
    batch_size = int(config["train"]["batch_size"])
    loaders = {
        "train": _build_loader(
            BinarySegmentationDataset(splits["train"], dataset_config, train=True),
            dataset_config,
            batch_size=batch_size,
            shuffle=True,
            distributed_state=distributed_state,
        ),
        "val": _build_loader(
            BinarySegmentationDataset(splits["val"], dataset_config, train=False),
            dataset_config,
            batch_size=batch_size,
            shuffle=False,
            distributed_state=distributed_state,
        ),
    }
    try:
        splits["test"] = split_samples(samples, dataset_config, "test")
    except (FileNotFoundError, ValueError, RuntimeError):
        return loaders
    _require_non_empty(splits["test"], "test", dataset_config)

    loaders["test"] = _build_loader(
        BinarySegmentationDataset(splits["test"], dataset_config, train=False),
        dataset_config,
        batch_size=batch_size,
        shuffle=False,
        distributed_state=distributed_state,
    )
    return loaders


def build_inference_loader(
    dataset_config: dict[str, Any],
    split: str,
    batch_size: int,
    distributed_state: DistributedState,
) -> DataLoader:
    base_samples = [] if dataset_config.get("split_strategy") == "pre_split_dirs" else discover_samples(dataset_config)
    samples = split_samples(base_samples, dataset_config, split)
    _require_non_empty(samples, split, dataset_config)
    dataset = BinarySegmentationDataset(samples, dataset_config, train=False)
    return _build_loader(
        dataset,
        dataset_config,
        batch_size=batch_size,
        shuffle=False,
        distributed_state=distributed_state,
    )
