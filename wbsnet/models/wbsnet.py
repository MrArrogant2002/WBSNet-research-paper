from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .decoder import DecoderBlock
from .resnet import ResNetEncoder
from .wbs_module import WBSModule


class WBSNet(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        decoder_channels = model_cfg.get("decoder_channels", [256, 128, 64, 32])
        reduction_ratio = int(model_cfg.get("reduction_ratio", 16))

        self.encoder = ResNetEncoder(in_channels=int(model_cfg.get("in_channels", 3)))
        pretrained_mode = model_cfg.get("encoder_pretrained", False)
        if pretrained_mode in {True, "imagenet", "ImageNet", "IMAGENET"}:
            self.encoder.load_imagenet_pretrained()
        if model_cfg.get("encoder_pretrained_checkpoint"):
            self.encoder.load_checkpoint(model_cfg["encoder_pretrained_checkpoint"])

        self.skip_modules = nn.ModuleList(
            [
                WBSModule(
                    64,
                    reduction_ratio=reduction_ratio,
                    use_wavelet=bool(model_cfg.get("use_wavelet", True)),
                    use_lfsa=bool(model_cfg.get("use_lfsa", True)),
                    use_hfba=bool(model_cfg.get("use_hfba", True)),
                    boundary_supervision=bool(model_cfg.get("boundary_supervision", True)),
                    wavelet_type=model_cfg.get("wavelet_type", "haar"),
                ),
                WBSModule(
                    64,
                    reduction_ratio=reduction_ratio,
                    use_wavelet=bool(model_cfg.get("use_wavelet", True)),
                    use_lfsa=bool(model_cfg.get("use_lfsa", True)),
                    use_hfba=bool(model_cfg.get("use_hfba", True)),
                    boundary_supervision=bool(model_cfg.get("boundary_supervision", True)),
                    wavelet_type=model_cfg.get("wavelet_type", "haar"),
                ),
                WBSModule(
                    128,
                    reduction_ratio=reduction_ratio,
                    use_wavelet=bool(model_cfg.get("use_wavelet", True)),
                    use_lfsa=bool(model_cfg.get("use_lfsa", True)),
                    use_hfba=bool(model_cfg.get("use_hfba", True)),
                    boundary_supervision=bool(model_cfg.get("boundary_supervision", True)),
                    wavelet_type=model_cfg.get("wavelet_type", "haar"),
                ),
                WBSModule(
                    256,
                    reduction_ratio=reduction_ratio,
                    use_wavelet=bool(model_cfg.get("use_wavelet", True)),
                    use_lfsa=bool(model_cfg.get("use_lfsa", True)),
                    use_hfba=bool(model_cfg.get("use_hfba", True)),
                    boundary_supervision=bool(model_cfg.get("boundary_supervision", True)),
                    wavelet_type=model_cfg.get("wavelet_type", "haar"),
                ),
            ]
        )

        self.dec4 = DecoderBlock(512, 256, decoder_channels[0])
        self.dec3 = DecoderBlock(decoder_channels[0], 128, decoder_channels[1])
        self.dec2 = DecoderBlock(decoder_channels[1], 64, decoder_channels[2])
        self.dec1 = DecoderBlock(decoder_channels[2], 64, decoder_channels[3])
        self.head = nn.Conv2d(decoder_channels[3], int(model_cfg.get("num_classes", 1)), kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        stem, layer1, layer2, layer3, bottleneck = self.encoder(x)
        skip1, bnd1 = self.skip_modules[0](stem)
        skip2, bnd2 = self.skip_modules[1](layer1)
        skip3, bnd3 = self.skip_modules[2](layer2)
        skip4, bnd4 = self.skip_modules[3](layer3)

        dec4 = self.dec4(bottleneck, skip4)
        dec3 = self.dec3(dec4, skip3)
        dec2 = self.dec2(dec3, skip2)
        dec1 = self.dec1(dec2, skip1)
        logits = self.head(torch.nn.functional.interpolate(dec1, scale_factor=2, mode="bilinear", align_corners=False))

        boundary_logits = [item for item in [bnd1, bnd2, bnd3, bnd4] if item is not None]
        return {"logits": logits, "boundary_logits": boundary_logits}


def build_model(config: dict[str, Any]) -> WBSNet:
    return WBSNet(config)


def variant_name_from_config(config: dict[str, Any]) -> str:
    model_cfg = config["model"]
    if (
        model_cfg.get("use_wavelet", True)
        and model_cfg.get("use_lfsa", True)
        and model_cfg.get("use_hfba", True)
        and model_cfg.get("boundary_supervision", True)
        and str(model_cfg.get("wavelet_type", "haar")).lower() in {"db2", "daubechies2", "daubechies-2"}
    ):
        return "A7_db2_wavelet"
    if not model_cfg.get("use_wavelet", True) and not model_cfg.get("use_lfsa", True) and not model_cfg.get("use_hfba", True):
        return "A1_identity_unet"
    if model_cfg.get("use_wavelet", True) and model_cfg.get("use_lfsa", True) and model_cfg.get("use_hfba", True) and model_cfg.get("boundary_supervision", True):
        return "A2_full_wbsnet"
    if model_cfg.get("use_wavelet", True) and model_cfg.get("use_lfsa", True) and not model_cfg.get("use_hfba", True):
        return "A3_lfsa_only"
    if model_cfg.get("use_wavelet", True) and not model_cfg.get("use_lfsa", True) and model_cfg.get("use_hfba", True):
        return "A4_hfba_only"
    if not model_cfg.get("boundary_supervision", True):
        return "A5_no_boundary_supervision"
    if not model_cfg.get("use_wavelet", True):
        return "A6_no_wavelet"
    return "custom_variant"
