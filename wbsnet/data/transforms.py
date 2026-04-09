from __future__ import annotations

import math
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
        if self.augment.get("random_resized_crop", False):
            image, mask = self._random_resized_crop(image, mask)
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

    def _random_resized_crop(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        area = width * height
        scale = tuple(self.augment.get("random_resized_crop_scale", [0.8, 1.0]))
        ratio = tuple(self.augment.get("random_resized_crop_ratio", [0.9, 1.1]))
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

        for _ in range(10):
            target_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                box = (left, top, left + crop_width, top + crop_height)
                return image.crop(box), mask.crop(box)

        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        box = (left, top, left + min_side, top + min_side)
        return image.crop(box), mask.crop(box)


def build_transform(dataset_config: dict[str, Any], train: bool) -> SegmentationTransform:
    return SegmentationTransform(
        image_size=tuple(dataset_config.get("image_size", [352, 352])),
        train=train,
        mean=tuple(dataset_config.get("normalize_mean", [0.485, 0.456, 0.406])),
        std=tuple(dataset_config.get("normalize_std", [0.229, 0.224, 0.225])),
        augment=dataset_config.get("augment", {}),
    )
