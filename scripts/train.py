#!/usr/bin/env python3
"""
Route Resilience — scripts/train.py  [Phase 2 — Not yet implemented]

Trains a road segmentation model (U-Net / DeepLabV3+ / SegFormer).

Usage (Phase 2):
    python scripts/train.py --config configs/config.yaml --model unet
    python scripts/train.py --model segformer --epochs 100 --resume

Dependencies (Phase 2):
    torch, torchvision, segmentation-models-pytorch, albumentations
"""

import sys
print("Phase 2 training not yet implemented.", file=sys.stderr)
print("Implement src/segmentation/training/ first.", file=sys.stderr)
sys.exit(1)
