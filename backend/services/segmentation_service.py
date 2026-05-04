"""
Segmentation Service — Production Implementation
=================================================
Uses trained U-Net model (models/unet_slum.pt) to detect slum areas
in satellite imagery patches.

Falls back to simulation mode if model weights are not found,
so the API still works during development before training.
"""

import numpy as np
from typing import Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Try importing ML deps — gracefully degrade if not installed
try:
    import torch
    import segmentation_models_pytorch as smp
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("PyTorch/SMP not installed. Segmentation will run in simulation mode.")


class SegmentationService:
    """
    Service for slum detection from satellite imagery.

    If model file exists → runs real U-Net inference.
    If not → returns simulated results (for development/demo).
    """

    MODEL_PATH = "models/unet_slum.pt"
    ENCODER = "resnet34"
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self):
        self.model = None
        self.device = None

        if ML_AVAILABLE and Path(self.MODEL_PATH).exists():
            self._load_model()
        else:
            if not ML_AVAILABLE:
                logger.info("SegmentationService: Running in SIMULATION mode (PyTorch not installed)")
            else:
                logger.info(f"SegmentationService: Model not found at {self.MODEL_PATH}. "
                            f"Running in SIMULATION mode. Train model first: python ml/train_segmentation.py")

    def _load_model(self):
        """Load trained U-Net model from disk."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = smp.Unet(
                encoder_name=self.ENCODER,
                encoder_weights=None,   # No pretrained weights — we load our trained weights
                in_channels=3,
                classes=1,
                activation="sigmoid",
            )
            checkpoint = torch.load(self.MODEL_PATH, map_location=self.device)
            # Handle both raw state_dict and checkpoints with metadata
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval().to(self.device)
            logger.info(f"✓ U-Net model loaded from {self.MODEL_PATH} (device: {self.device})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}. Falling back to simulation.")
            self.model = None

    def detect_slums(self, image: np.ndarray) -> Dict:
        """
        Detect slum areas in a satellite image patch.

        Args:
            image: np.ndarray of shape (H, W, 3), values 0–1 (float32)
                   RGB bands from Sentinel-2 (already normalized)

        Returns:
            dict with:
                segmentation_mask  : 2D array (H, W), values 0.0–1.0
                slum_percentage    : float, % of pixels classified as slum
                confidence         : float, mean confidence in slum areas
                model_used         : str, "unet" or "simulation"
        """
        if self.model is not None:
            return self._real_inference(image)
        else:
            return self._simulate(image)

    def _real_inference(self, image: np.ndarray) -> Dict:
        """Run actual U-Net inference."""
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = (image - mean) / std

        # Convert to tensor: (H, W, C) → (1, C, H, W)
        tensor = torch.tensor(image_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            mask = self.model(tensor).squeeze().cpu().numpy()  # Shape: (H, W)

        slum_pixels = np.sum(mask > self.CONFIDENCE_THRESHOLD)
        total_pixels = mask.size
        slum_pct = (slum_pixels / total_pixels) * 100

        return {
            "segmentation_mask": mask.tolist(),
            "slum_percentage": round(float(slum_pct), 2),
            "confidence": float(np.mean(mask[mask > self.CONFIDENCE_THRESHOLD]))
                          if slum_pixels > 0 else 0.0,
            "model_used": "unet",
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
