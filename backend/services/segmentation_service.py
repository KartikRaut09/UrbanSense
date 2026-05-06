"""
Segmentation Service — Advanced Production Implementation
=========================================================
Uses trained EfficientNet-B4 + DeepLabV3+ model (models/segmentation_model.pt).
Input: 6-channel multi-spectral (RGB + NIR + NDVI + NDWI).
Falls back to simulation mode if model not trained yet.
"""

import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import segmentation_models_pytorch as smp
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("PyTorch/SMP not installed. Running in simulation mode.")


class SegmentationService:
    """
    Advanced slum segmentation service.
    Model: EfficientNet-B4 + DeepLabV3+ with 6-channel multi-spectral input.
    Falls back to simulation if model not trained.
    """

    MODEL_PATH = "models/segmentation_model.pt"
    CONFIDENCE_THRESHOLD = 0.5
    # ImageNet mean/std for RGB channels; custom for NIR/NDVI/NDWI
    NORMALIZE_MEAN = [0.485, 0.456, 0.406, 0.3, 0.5, 0.5]
    NORMALIZE_STD  = [0.229, 0.224, 0.225, 0.15, 0.2, 0.2]

    def __init__(self):
        self.model = None
        self.device = None
        self.in_channels = 6
        self.encoder = "efficientnet-b4"

        if ML_AVAILABLE and Path(self.MODEL_PATH).exists():
            self._load_model()
        else:
            msg = "PyTorch not installed" if not ML_AVAILABLE else \
                  f"Model not found at {self.MODEL_PATH}"
            logger.info(f"SegmentationService: {msg}. Running in SIMULATION mode. "
                        f"Train: python ml/train_segmentation.py")

    def _load_model(self):
        """Load trained EfficientNet-B4 + DeepLabV3+ from checkpoint."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.MODEL_PATH, map_location=self.device)
            self.encoder = checkpoint.get("encoder", "efficientnet-b4")
            self.in_channels = checkpoint.get("in_channels", 6)

            self.model = smp.DeepLabV3Plus(
                encoder_name=self.encoder,
                encoder_weights=None,
                in_channels=self.in_channels,
                classes=1,
                activation="sigmoid",
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval().to(self.device)
            best_iou = checkpoint.get("best_iou", "unknown")
            logger.info(f"✓ {self.encoder}+DeepLabV3+ loaded (IoU={best_iou}, "
                        f"channels={self.in_channels}, device={self.device})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Falling back to simulation.")
            self.model = None

    def _build_6channel(self, image_rgb: np.ndarray,
                         nir: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Build 6-channel input: [R, G, B, NIR, NDVI, NDWI].
        If NIR not available, estimate it from RGB luminance.
        """
        r, g, b = image_rgb[:,:,0], image_rgb[:,:,1], image_rgb[:,:,2]
        if nir is None:
            nir = 0.4 * r + 0.4 * g + 0.2 * b  # synthetic NIR
        eps = 1e-8
        ndvi = np.clip((nir - r) / (nir + r + eps), -1, 1)
        ndwi = np.clip((g - nir) / (g + nir + eps), -1, 1)
        return np.stack([
            np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1),
            np.clip(nir, 0, 1),
            (ndvi + 1) / 2,   # normalize -1..1 → 0..1
            (ndwi + 1) / 2,
        ], axis=-1)  # (H, W, 6)

    def detect_slums(self, image: np.ndarray,
                     nir: Optional[np.ndarray] = None) -> Dict:
        """
        Detect slum areas in a satellite image patch.

        Args:
            image : (H, W, 3) float32 array — RGB values 0–1
            nir   : (H, W) float32 array — NIR band 0–1 (optional but improves accuracy)

        Returns:
            segmentation_mask   : (H, W) probability map 0–1
            slum_percentage     : % of image classified as slum
            confidence          : mean model confidence in slum pixels
            ndvi_mean           : mean NDVI in slum zones (vegetation indicator)
            model_used          : "efficientnet_deeplabv3plus" or "simulation"
        """
        if self.model is not None:
            return self._real_inference(image, nir)
        else:
            return self._simulate(image)

    def _real_inference(self, image: np.ndarray,
                         nir: Optional[np.ndarray] = None) -> Dict:
        """Run 6-channel EfficientNet-B4+DeepLabV3+ inference."""
        img_6ch = self._build_6channel(image, nir)   # (H, W, 6)

        # Normalize each channel
        mean = np.array(self.NORMALIZE_MEAN)
        std = np.array(self.NORMALIZE_STD)
        img_norm = (img_6ch - mean) / std

        # (H, W, 6) → (1, 6, H, W)
        tensor = torch.tensor(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            mask = self.model(tensor).squeeze().cpu().numpy()   # (H, W)

        slum_pixels = np.sum(mask > self.CONFIDENCE_THRESHOLD)
        slum_pct = (slum_pixels / mask.size) * 100

        # Compute NDVI in slum zones for additional context
        r = image[:,:,0]
        nir_ch = nir if nir is not None else 0.4*image[:,:,0] + 0.4*image[:,:,1] + 0.2*image[:,:,2]
        ndvi = (nir_ch - r) / (nir_ch + r + 1e-8)
        slum_mask_binary = mask > self.CONFIDENCE_THRESHOLD
        ndvi_in_slums = float(np.mean(ndvi[slum_mask_binary])) if slum_pixels > 0 else 0.0

        return {
            "segmentation_mask": mask.tolist(),
            "slum_percentage": round(float(slum_pct), 2),
            "confidence": float(np.mean(mask[slum_mask_binary])) if slum_pixels > 0 else 0.0,
            "ndvi_mean_in_slums": round(ndvi_in_slums, 4),
            "model_used": f"{self.encoder}_deeplabv3plus",
        }

    def _simulate(self, image: np.ndarray) -> Dict:
        """Return simulated segmentation results for development."""
        h, w = image.shape[:2] if len(image.shape) > 1 else (256, 256)
        # Simulate realistic slum detection patterns
        mask = np.zeros((h, w), dtype=np.float32)
        # Add a few "slum cluster" regions
        np.random.seed(42)
        for _ in range(np.random.randint(2, 6)):
            cy, cx = np.random.randint(32, h-32), np.random.randint(32, w-32)
            r = np.random.randint(20, 60)
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask += np.clip(1 - dist/r, 0, 1) * np.random.uniform(0.5, 0.95)
        mask = np.clip(mask, 0, 1)
        slum_pct = np.mean(mask > 0.5) * 100
        return {
            "segmentation_mask": mask.tolist(),
            "slum_percentage": round(float(slum_pct), 2),
            "confidence": 0.73,
            "model_used": "simulation",
        }

    def extract_features(self, segmentation_mask: np.ndarray) -> Dict:
        """Extract urban features from segmentation mask using connected components."""
        from scipy import ndimage

        binary_mask = segmentation_mask > self.CONFIDENCE_THRESHOLD
        labeled_array, num_features = ndimage.label(binary_mask)

        if num_features == 0:
            return {
                "cluster_count": 0,
                "average_cluster_size_px": 0.0,
                "density": 0.0,
                "fragmentation_index": 0.0,
            }

        cluster_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
        return {
            "cluster_count": num_features,
            "average_cluster_size_px": float(np.mean(cluster_sizes)),
            "density": float(np.mean(binary_mask)),
            "fragmentation_index": float(num_features / (np.sum(binary_mask) + 1)),
        }


class GraphAnalysisService:
    """Service for graph-based urban structure analysis."""

    def build_urban_graph(self, building_centers: np.ndarray,
                          connection_distance: float = 50.0) -> Dict:
        """
        Build a proximity graph from building centroids.

        Args:
            building_centers : (N, 2) array of (x, y) pixel coordinates
            connection_distance : max distance (px) to connect two buildings

        Returns:
            dict with graph metrics (no actual networkx Graph object
            — serializable for API responses)
        """
        try:
            import networkx as nx
        except ImportError:
            return {"error": "networkx not installed. pip install networkx"}

        G = nx.Graph()
        for i, center in enumerate(building_centers):
            G.add_node(i, pos=tuple(center))

        for i in range(len(building_centers)):
            for j in range(i + 1, len(building_centers)):
                dist = float(np.linalg.norm(building_centers[i] - building_centers[j]))
                if dist < connection_distance:
                    G.add_edge(i, j, weight=dist)

        if G.number_of_nodes() == 0:
            return {"num_nodes": 0, "num_edges": 0, "density": 0.0,
                    "clustering_coefficient": 0.0, "connected_components": 0}

        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "clustering_coefficient": nx.average_clustering(G),
            "connected_components": nx.number_connected_components(G),
        }
