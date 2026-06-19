"""
Route Resilience — Phase 2 Verification
scratch/verify_pipeline.py

Rigorously verifies the deep learning training pipeline end-to-end using a mock dataset.
"""

import os
import shutil
import sys
import yaml
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# Add project root to python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.segmentation.datasets.road_dataset import RoadDataset, SyntheticOcclusionTransform
from src.segmentation.models.model_factory import ModelFactory
from src.segmentation.losses.losses import CombinedLoss
from src.segmentation.training.trainer import Trainer
from src.segmentation.inference.inference import predict_image

def setup_dummy_dataset(dummy_dir: str) -> str:
    """Generate mock images, masks, and master_dataset.csv for testing."""
    os.makedirs(os.path.join(dummy_dir, "tiles/train/images"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "tiles/train/masks"), exist_ok=True)

    rows = []
    # Generate 4 training tiles and 2 validation tiles
    splits = ["train", "train", "train", "train", "val", "val"]
    
    for i, split in enumerate(splits):
        tile_id = f"tile_test_{i:03d}"
        img_name = f"{tile_id}.jpg"
        msk_name = f"{tile_id}.png"
        
        img_path = os.path.join(dummy_dir, "tiles/train/images", img_name)
        msk_path = os.path.join(dummy_dir, "tiles/train/masks", msk_name)
        
        # 1. Create dummy RGB image (512x512)
        # Create a simple grey background with a horizontal white line as road
        img = np.random.randint(50, 150, (512, 512, 3), dtype=np.uint8)
        img[240:272, :] = 200  # mock road pixels
        cv2.imwrite(img_path, img)
        
        # 2. Create dummy binary mask (512x512)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[240:272, :] = 255
        cv2.imwrite(msk_path, mask)
        
        rows.append({
            "image_id": tile_id,
            "source_dataset": "deepglobe",
            "source_image_id": f"img_test_{i // 2}",
            "image_dimensions": "512x512",
            "road_density": 0.0625,
            "avg_road_width_px": 32.0,
            "max_road_width_px": 32.0,
            "road_pixel_pct": 6.25,
            "background_pixel_pct": 93.75,
            "occlusion_type": "",
            "occlusion_severity": "",
            "occlusion_coverage_pct": "",
            "occlusion_road_coverage_pct": "",
            "split": split,
            "image_tile_path": f"data/tiles/train/images/{img_name}",
            "mask_tile_path": f"data/tiles/train/masks/{msk_name}",
            "geospatial_metadata_reference": ""
        })
        
    csv_path = os.path.join(dummy_dir, "master_dataset.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Mock dataset created in: {dummy_dir}")
    return csv_path


def main():
    dummy_dir = os.path.join(project_root, "scratch/dummy_data")
    output_dir = os.path.join(project_root, "scratch/output")
    
    # Clean up any old test run directories
    if os.path.exists(dummy_dir):
        shutil.rmtree(dummy_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    try:
        # 1. Create mock data
        csv_path = setup_dummy_dataset(dummy_dir)
        
        # 2. Create temporary validation config
        test_config = {
            "project": {
                "name": "Route Resilience Pipeline Verification",
                "phase": 2,
                "version": "1.0.0-test",
                "random_seed": 42
            },
            "model": {
                "architecture": "deeplabv3plus",
                "encoder": "resnet34",
                "encoder_weights": None,  # Skip downloading weights during test
                "num_classes": 1
            },
            "dataset": {
                "dataset_dir": dummy_dir,
                "image_size": 512,
                "occlusion_prob": 0.5,
                "num_workers": 0,         # Run synchronously in test
                "pin_memory": False,
                "persistent_workers": False
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "grad_clip_norm": 1.0,
                "early_stopping_patience": 2,
                "device": "cpu"           # Run on CPU for validation safety
            },
            "loss": {
                "bce_weight": 0.5,
                "dice_weight": 0.5,
                "jaccard_weight": 0.0
            },
            "scheduler": {
                "type": "cosine",
                "t_max": 1,
                "eta_min": 0.000001
            },
            "paths": {
                "output_dir": output_dir
            }
        }
        
        # 3. Load Splits
        df = pd.read_csv(csv_path)
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        
        # 4. Augmentations
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        val_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # 5. Loaders
        occ_transform = SyntheticOcclusionTransform(p=test_config["dataset"]["occlusion_prob"])
        
        train_dataset = RoadDataset(train_df, dummy_dir, transform=train_transform, occlusion_transform=occ_transform)
        val_dataset = RoadDataset(val_df, dummy_dir, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=test_config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=test_config["training"]["batch_size"], shuffle=False)
        
        # 6. Model building
        model = ModelFactory.create(
            test_config["model"]["architecture"],
            encoder_name=test_config["model"]["encoder"],
            encoder_weights=test_config["model"]["encoder_weights"],
            classes=test_config["model"]["num_classes"]
        )
        
        criterion = CombinedLoss(
            bce_weight=test_config["loss"]["bce_weight"],
            dice_weight=test_config["loss"]["dice_weight"],
            jaccard_weight=test_config["loss"]["jaccard_weight"]
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=test_config["training"]["learning_rate"])
        
        # 7. Trainer setup
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_dataset=val_dataset,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            config=test_config
        )
        
        # 8. Run 1 epoch
        print("Running training loop test...")
        trainer.fit()
        print("✓ Training loop executed successfully.")
        
        # 9. Verify Checkpoint files
        exp_dir = trainer.experiment_dir
        expected_files = [
            "config.yaml",
            "models/last_checkpoint.pth",
            "models/optimizer_state.pth",
            "models/config_used.yaml",
            "logs/training_history.csv",
            "predictions/validation_grid_epoch_001.png"
        ]
        
        for f in expected_files:
            fpath = os.path.join(exp_dir, f)
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Expected file was not created: {fpath}")
        print("✓ All expected training output artifacts exist.")
        
        # 10. Test Inference script
        test_img_path = os.path.join(dummy_dir, "tiles/train/images/tile_test_000.jpg")
        binary_mask = predict_image(model, test_img_path, device=trainer.device, threshold=0.5, transform=val_transform)
        
        if binary_mask.shape != (512, 512) or binary_mask.dtype != np.uint8:
            raise ValueError(f"Inference output shape/dtype mismatch. Shape: {binary_mask.shape}, Dtype: {binary_mask.dtype}")
        print("✓ Inference helper function verified successfully.")
        
        print("\n==============================================")
        print("  VERIFICATION SUCCESSFUL: PIPELINE IS ROBUST ")
        print("==============================================")
        
    except Exception as exc:
        print(f"\n❌ Pipeline verification failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup mock folders
        print("Cleaning up test directories...")
        if os.path.exists(dummy_dir):
            shutil.rmtree(dummy_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    main()
