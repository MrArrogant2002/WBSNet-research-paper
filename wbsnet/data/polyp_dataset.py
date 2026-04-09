from __future__ import annotations

from .datasets import BinarySegmentationDataset


class PolypDataset(BinarySegmentationDataset):
    """Dataset alias for Kvasir-SEG and CVC-style polyp benchmarks."""
