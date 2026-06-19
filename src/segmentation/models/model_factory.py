"""
Route Resilience — Phase 2
src/segmentation/models/model_factory.py

Registry-based Model Factory supporting SMP backbones and custom architectures.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Registry for model architectures. Allows configuration-based switching
    and dynamic registration of new models.
    """
    _registry: Dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a new model builder function.
        """
        def decorator(builder_fn: Callable[..., nn.Module]) -> Callable:
            key = name.lower().strip()
            cls._registry[key] = builder_fn
            return builder_fn
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """
        Build and return a registered model by name.
        """
        key = name.lower().strip()
        if key not in cls._registry:
            raise KeyError(
                f"Model architecture '{name}' is not registered. "
                f"Available architectures: {list(cls._registry.keys())}"
            )
        builder = cls._registry[key]
        return builder(**kwargs)


# ─────────────────────────────────────────────────────────────
#  Pre-registered Models
# ─────────────────────────────────────────────────────────────

@ModelFactory.register("deeplabv3plus")
def build_deeplabv3plus(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    classes: int = 1,
    **kwargs
) -> nn.Module:
    """DeepLabV3+ builder using segmentation_models_pytorch."""
    logger.info(f"Building DeepLabV3+ with {encoder_name} encoder...")
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        **kwargs
    )


@ModelFactory.register("unet")
def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    classes: int = 1,
    **kwargs
) -> nn.Module:
    """UNet builder using segmentation_models_pytorch."""
    logger.info(f"Building UNet with {encoder_name} encoder...")
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        **kwargs
    )


@ModelFactory.register("unetplusplus")
def build_unetplusplus(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    classes: int = 1,
    **kwargs
) -> nn.Module:
    """UNet++ builder using segmentation_models_pytorch."""
    logger.info(f"Building UNet++ with {encoder_name} encoder...")
    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=classes,
        **kwargs
    )


@ModelFactory.register("segformer")
def build_segformer(
    encoder_name: str = "nvidia/mit-b0",
    classes: int = 1,
    **kwargs
) -> nn.Module:
    """
    SegFormer builder using Hugging Face transformers.
    Wraps the Hugging Face model to automatically upsample logits back to input size.
    """
    logger.info(f"Building SegFormer with {encoder_name} encoder...")
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required to use SegFormer. "
            "Install it via: pip install transformers"
        )

    class SegFormerWrapper(nn.Module):
        def __init__(self, model_name: str, num_labels: int):
            super().__init__()
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Hugging Face SegFormer expects input format (batch_size, channels, H, W)
            outputs = self.model(pixel_values=x)
            logits = outputs.logits
            # Interpolate to match the original input resolution (512x512)
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            return upsampled_logits

    return SegFormerWrapper(encoder_name, classes)
