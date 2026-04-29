from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .utils.io import ensure_dir


def _denormalize(image: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    mean_np = np.array(mean, dtype=np.float32)
    std_np = np.array(std, dtype=np.float32)
    image_np = (image_np * std_np) + mean_np
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return image_np


def _mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    return (mask_np > 0.5).astype(np.uint8) * 255


def _make_overlay(image_np: np.ndarray, target_np: np.ndarray, pred_np: np.ndarray) -> np.ndarray:
    overlay = image_np.copy()
    overlay[..., 0] = np.where(pred_np > 0, 255, overlay[..., 0])
    overlay[..., 1] = np.where(target_np > 0, 255, overlay[..., 1])
    return overlay


def _tile_with_title(tile: np.ndarray, title: str, title_height: int = 28) -> np.ndarray:
    canvas = np.full((tile.shape[0] + title_height, tile.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_height:, :, :] = tile
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), title, fill=(0, 0, 0))
    return np.asarray(image)


def _hstack_with_padding(tiles: list[np.ndarray]) -> np.ndarray:
    max_height = max(tile.shape[0] for tile in tiles)
    padded_tiles = []
    for tile in tiles:
        if tile.shape[0] == max_height:
            padded_tiles.append(tile)
            continue
        canvas = np.full((max_height, tile.shape[1], 3), 255, dtype=np.uint8)
        canvas[: tile.shape[0], :, :] = tile
        padded_tiles.append(canvas)
    return np.concatenate(padded_tiles, axis=1)


def _find_focus_box(target_np: np.ndarray, pred_np: np.ndarray, min_size: int = 96, padding: int = 24) -> tuple[int, int, int, int]:
    target_mask = target_np > 0
    pred_mask = pred_np > 0
    focus = np.logical_xor(target_mask, pred_mask)
    if not focus.any():
        focus = np.logical_or(target_mask, pred_mask)

    height, width = target_np.shape[:2]
    if not focus.any():
        crop = min(min_size, height, width)
        top = max((height - crop) // 2, 0)
        left = max((width - crop) // 2, 0)
        return left, top, min(left + crop, width), min(top + crop, height)

    ys, xs = np.where(focus)
    left = max(int(xs.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, width)
    top = max(int(ys.min()) - padding, 0)
    bottom = min(int(ys.max()) + padding + 1, height)

    box_width = right - left
    box_height = bottom - top
    target_size = max(min_size, box_width, box_height)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    half = target_size // 2
    left = max(cx - half, 0)
    top = max(cy - half, 0)
    right = min(left + target_size, width)
    bottom = min(top + target_size, height)
    left = max(right - target_size, 0)
    top = max(bottom - target_size, 0)
    return left, top, right, bottom


def _draw_focus_box(tile: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    image = Image.fromarray(tile)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    draw.rectangle([left, top, right - 1, bottom - 1], outline=color, width=3)
    return np.asarray(image)


def create_prediction_visuals(
    image: torch.Tensor,
    target_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    mean: list[float],
    std: list[float],
    zoom_size: int = 256,
) -> dict[str, np.ndarray | tuple[int, int, int, int]]:
    image_np = _denormalize(image, mean, std)
    target_np = _mask_to_uint8(target_mask)
    pred_np = _mask_to_uint8(pred_mask)
    overlay_np = _make_overlay(image_np, target_np, pred_np)

    focus_box = _find_focus_box(target_np, pred_np)
    left, top, right, bottom = focus_box

    image_box = _draw_focus_box(image_np, focus_box)
    target_box = _draw_focus_box(np.repeat(target_np[..., None], 3, axis=2), focus_box)
    pred_box = _draw_focus_box(np.repeat(pred_np[..., None], 3, axis=2), focus_box)
    overlay_box = _draw_focus_box(overlay_np, focus_box)

    zoom_crop = overlay_np[top:bottom, left:right]
    zoom_panel = np.asarray(Image.fromarray(zoom_crop).resize((zoom_size, zoom_size), resample=Image.NEAREST))

    return {
        "image": image_np,
        "target": target_np,
        "prediction": pred_np,
        "overlay": overlay_np,
        "focus_box": focus_box,
        "paper_panel": _hstack_with_padding(
            [
                _tile_with_title(image_box, "Image"),
                _tile_with_title(target_box, "Ground Truth"),
                _tile_with_title(pred_box, "Prediction"),
                _tile_with_title(overlay_box, "Overlay"),
                _tile_with_title(zoom_panel, "Boundary Zoom"),
            ]
        ),
    }


def save_prediction_triplet(
    save_dir: str | Path,
    sample_id: str,
    image: torch.Tensor,
    target_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> dict[str, str]:
    save_path = ensure_dir(save_dir)
    visuals = create_prediction_visuals(image, target_mask, pred_mask, mean, std)
    image_np = visuals["image"]
    target_np = visuals["target"]
    pred_np = visuals["prediction"]
    overlay = visuals["overlay"]
    paper_panel = visuals["paper_panel"]

    image_path = save_path / f"{sample_id}_image.png"
    target_path = save_path / f"{sample_id}_target.png"
    pred_path = save_path / f"{sample_id}_prediction.png"
    overlay_path = save_path / f"{sample_id}_overlay.png"
    paper_panel_path = save_path / f"{sample_id}_paper_panel.png"

    Image.fromarray(image_np).save(image_path)
    Image.fromarray(target_np).save(target_path)
    Image.fromarray(pred_np).save(pred_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(paper_panel).save(paper_panel_path)

    return {
        "image": str(image_path),
        "target": str(target_path),
        "prediction": str(pred_path),
        "overlay": str(overlay_path),
        "paper_panel": str(paper_panel_path),
    }


def save_contact_sheet(
    panel_paths: list[str | Path],
    output_path: str | Path,
    columns: int = 2,
    panel_spacing: int = 20,
    margin: int = 20,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> str:
    if not panel_paths:
        raise ValueError("At least one panel path is required to build a contact sheet.")

    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    try:
        max_width = max(panel.width for panel in panels)
        max_height = max(panel.height for panel in panels)
        columns = max(1, columns)
        rows = (len(panels) + columns - 1) // columns

        canvas_width = (columns * max_width) + ((columns - 1) * panel_spacing) + (2 * margin)
        canvas_height = (rows * max_height) + ((rows - 1) * panel_spacing) + (2 * margin)
        canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)

        for index, panel in enumerate(panels):
            row = index // columns
            column = index % columns
            x = margin + column * (max_width + panel_spacing)
            y = margin + row * (max_height + panel_spacing)
            canvas.paste(panel, (x, y))

        target = Path(output_path)
        ensure_dir(target.parent)
        canvas.save(target)
        return str(target)
    finally:
        for panel in panels:
            panel.close()
