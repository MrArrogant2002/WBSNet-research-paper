from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _list_files(path: Path, extensions: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in {ext.lower() for ext in extensions}:
            files[item.stem] = item
    return files


def discover_samples(dataset_config: dict[str, Any]) -> list[SampleRecord]:
    root = Path(dataset_config["root"])
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


def split_samples(samples: list[SampleRecord], dataset_config: dict[str, Any], split: str) -> list[SampleRecord]:
    strategy = dataset_config.get("split_strategy", "ratio")
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
    return {
        "num_workers": num_workers,
        "pin_memory": bool(dataset_config.get("pin_memory", True)),
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }


def _build_loader(
    dataset: Dataset,
    dataset_config: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    distributed_state: DistributedState,
) -> DataLoader:
    sampler = None
    if distributed_state.enabled:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=shuffle,
        **_loader_kwargs(dataset_config),
    )


def build_dataloaders(config: dict[str, Any], distributed_state: DistributedState) -> dict[str, DataLoader]:
    dataset_config = config["dataset"]
    samples = discover_samples(dataset_config)
    splits = {
        "train": split_samples(samples, dataset_config, "train"),
        "val": split_samples(samples, dataset_config, "val"),
        "test": split_samples(samples, dataset_config, "test"),
    }
    batch_size = int(config["train"]["batch_size"])
    return {
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
        "test": _build_loader(
            BinarySegmentationDataset(splits["test"], dataset_config, train=False),
            dataset_config,
            batch_size=batch_size,
            shuffle=False,
            distributed_state=distributed_state,
        ),
    }


def build_inference_loader(
    dataset_config: dict[str, Any],
    split: str,
    batch_size: int,
    distributed_state: DistributedState,
) -> DataLoader:
    samples = split_samples(discover_samples(dataset_config), dataset_config, split)
    dataset = BinarySegmentationDataset(samples, dataset_config, train=False)
    return _build_loader(
        dataset,
        dataset_config,
        batch_size=batch_size,
        shuffle=False,
        distributed_state=distributed_state,
    )
