"""
U-Net Slum Segmentation — Training Script
==========================================
Architecture : U-Net with ResNet34 encoder (pretrained on ImageNet)
Loss         : Dice Loss + Binary Cross-Entropy
Optimizer    : AdamW with Cosine Annealing LR schedule
Input        : 256x256 RGB satellite patches (Sentinel-2 B04/B03/B02)
Output       : Binary mask — 1=slum, 0=non-slum
Expected IoU : 0.75–0.85 on good Sentinel-2 + GRID3 labels

Requirements:
    pip install torch torchvision segmentation-models-pytorch albumentations rasterio

GPU Recommended: ~2-4 hours on NVIDIA GPU, ~10-20h on CPU
For free GPU: use Google Colab → https://colab.research.google.com
              Upload this file and run: !python train_segmentation.py

Usage:
    python train_segmentation.py
    python train_segmentation.py --epochs 100 --batch_size 16 --lr 5e-5
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
import rasterio
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class SlumDataset(Dataset):
    """
    Loads paired satellite image patches + binary masks.

    Expected directory structure:
        data/images/train/patch_00001.tif   ← 3-band GeoTIFF (uint16)
        data/masks/train/patch_00001.tif    ← 1-band GeoTIFF (uint8: 0/1)

    The image and mask filenames MUST match exactly.
    Run preprocess_sentinel2.py to generate these files.
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Only keep patches that have a matching mask
        self.images = sorted([
            p for p in self.image_dir.glob("*.tif")
            if (self.mask_dir / p.name).exists()
        ])

        if len(self.images) == 0:
            raise RuntimeError(
                f"No matching image-mask pairs found!\n"
                f"  Images dir: {image_dir}\n"
                f"  Masks dir:  {mask_dir}\n"
                f"Run preprocess_sentinel2.py first."
            )

        logger.info(f"Dataset: {len(self.images)} image-mask pairs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        # Load satellite image
        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)  # 3 bands, shape (3, H, W)
            image = image / 10000.0                          # Sentinel-2 scale factor (DN → reflectance)
            image = np.clip(image, 0, 1)
            image = np.transpose(image, (1, 2, 0))           # CHW → HWC for albumentations

        # Load binary mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)            # Shape: (H, W), values: 0.0 or 1.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)  # mask shape: (1, H, W)


# ─────────────────────────────────────────────
#  Augmentations
# ─────────────────────────────────────────────

def get_train_transforms():
    """Data augmentation for training — makes model robust to rotation, lighting, etc."""
    return A.Compose([
        A.RandomCrop(256, 256, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        # Simulate different lighting conditions (season, time of day)
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.4),
        # Simulate atmospheric haze
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # Simulate partial clouds
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.CenterCrop(256, 256, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=1.0,
        ),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    return (2 * intersection / (pred_binary.sum() + target.sum() + 1e-6)).item()


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    if device.type == "cpu":
        logger.warning("GPU not available — training will be slow. Consider Google Colab.")

    # ── Data ──
    full_dataset = SlumDataset(args.image_dir, args.mask_dir)
    val_size = max(1, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms by wrapping
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            # Temporarily set transform to None to get raw data
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            # Get raw data by bypassing transforms
            dataset = self.subset.dataset
            original_transform = dataset.transform
            dataset.transform = None
            image, mask = dataset[self.subset.indices[idx]]
            dataset.transform = original_transform
            # Apply our transform
            image = image.numpy().transpose(1, 2, 0) if isinstance(image, torch.Tensor) else image
            mask = mask.numpy().squeeze() if isinstance(mask, torch.Tensor) else mask
            aug = self.transform(image=image, mask=mask)
            return aug["image"], aug["mask"].unsqueeze(0)

    train_data = TransformedSubset(train_dataset, get_train_transforms())
    val_data = TransformedSubset(val_dataset, get_val_transforms())

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Train: {len(train_data)} samples | Val: {len(val_data)} samples")

    # ── Model ──
    # U-Net with ResNet34 backbone pretrained on ImageNet
    # Options for encoder_name: resnet18, resnet34, resnet50, efficientnet-b4
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights="imagenet",   # Use ImageNet pretrained weights (transfer learning)
        in_channels=3,
        classes=1,
        activation="sigmoid",         # Output is probability 0-1
    ).to(device)

    logger.info(f"Model: U-Net + {args.encoder}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    # ── Loss ──
    # Dice Loss handles class imbalance well (slums are minority class in image)
    # BCE provides stable gradients, especially early in training
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()

    def criterion(pred, target):
        return dice_loss(pred, target) + bce_loss(pred, target)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )

    # Warm-up + cosine annealing: prevents early overfitting
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training Loop ──
    best_iou = 0.0
    history = []
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()

            # Gradient clipping prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_ious, val_dices = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                preds = model(images)
                val_ious.append(iou_score(preds, masks))
                val_dices.append(dice_score(preds, masks))

        mean_iou = np.mean(val_ious)
        mean_dice = np.mean(val_dices)
        avg_loss = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history.append({
            "epoch": epoch, "loss": avg_loss,
            "iou": mean_iou, "dice": mean_dice, "lr": current_lr
        })

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | IoU: {mean_iou:.4f} | "
            f"Dice: {mean_dice:.4f} | LR: {current_lr:.2e}"
        )

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "encoder": args.encoder,
            }, args.output)
            logger.info(f"  ✓ Saved best model (IoU={best_iou:.4f})")

    # Save training history
    history_path = Path(args.output).parent / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete!")
    logger.info(f"Best Validation IoU: {best_iou:.4f}")
    logger.info(f"Model saved to:      {args.output}")
    logger.info(f"History saved to:    {history_path}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for slum segmentation")
    parser.add_argument("--image_dir", default="data/images/train", help="Training images directory")
    parser.add_argument("--mask_dir", default="data/masks/train", help="Training masks directory")
    parser.add_argument("--output", default="models/unet_slum.pt", help="Output model path")
    parser.add_argument("--encoder", default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50", "efficientnet-b4"],
                        help="Encoder backbone (default: resnet34)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (default: 0 for Windows compatibility)")

    args = parser.parse_args()
    train(args)
