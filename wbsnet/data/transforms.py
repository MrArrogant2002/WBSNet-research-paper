from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance


@dataclass
class SegmentationTransform:
    image_size: tuple[int, int]
    train: bool
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    augment: dict[str, Any]

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        mask = mask.convert("L")

        if self.train:
            image, mask = self._augment(image, mask)

        image = image.resize(self.image_size[::-1], resample=Image.BILINEAR)
        mask = mask.resize(self.image_size[::-1], resample=Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        mask_np = np.asarray(mask, dtype=np.float32)
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
        mask_tensor = torch.from_numpy((mask_np > 0.5).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor

    def _augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.augment.get("horizontal_flip", False) and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment.get("vertical_flip", False) and random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if self.augment.get("rotate90", False) and random.random() < 0.5:
            rotation = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
            image = image.transpose(rotation)
            mask = mask.transpose(rotation)
        if self.augment.get("color_jitter", False):
            image = self._color_jitter(image)
        return image, mask

    @staticmethod
    def _color_jitter(image: Image.Image) -> Image.Image:
        enhancers = (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color)
        for enhancer in enhancers:
            factor = random.uniform(0.9, 1.1)
            image = enhancer(image).enhance(factor)
        return image


def build_transform(dataset_config: dict[str, Any], train: bool) -> SegmentationTransform:
    return SegmentationTransform(
        image_size=tuple(dataset_config.get("image_size", [352, 352])),
        train=train,
        mean=tuple(dataset_config.get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(dataset_config.get("normalize_std", [0.229, 0.224, 0.225])),
        augment=dataset_config.get("augment", {}),
    )
