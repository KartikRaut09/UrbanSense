# UrbanSense — Complete Development Guide

> Last updated: May 2026 | Every file audited and cleaned.

---

## 1. Project Status Overview

### ✅ Done (Scaffold / UI)
| Component | Status | Notes |
|---|---|---|
| Frontend UI (5 pages) | ✅ Done | Glassmorphism design, charts, animations |
| Backend API skeleton | ✅ Done | 20+ endpoints — all return **mock data** |
| Database models (SQLAlchemy) | ✅ Done | 5 tables defined, never actually queried |
| Service classes | ✅ Done | Written but routes don't call them yet |
| Docker / docker-compose | ✅ Done | Needs real env vars |
| Python package structure | ✅ Fixed | `__init__.py` files added |
| GitHub repo | ✅ Done | https://github.com/KartikRaut09/UrbanSense |

### ❌ NOT Done (Core Logic)
| Component | Status |
|---|---|
| U-Net segmentation model (training) | ❌ Missing |
| XGBoost / LightGBM risk model | ❌ Missing |
| LSTM growth prediction model | ❌ Missing |
| Real database queries in routes | ❌ Routes return hardcoded JSON |
| Image upload + preprocessing pipeline | ❌ Missing |
| PostgreSQL + PostGIS connection | ❌ Not configured |

---

## 2. Recommended Completion Order

```
Phase 1 → Run locally (30 min)
Phase 2 → Wire routes → services → DB (2-3 days)
Phase 3 → Build & train ML models (1-2 weeks)
Phase 4 → Connect ML to API (2-3 days)
Phase 5 → Polish + Deploy (2-3 days)
```

---

## 3. Phase 1 — Run Locally

### Frontend
```bash
cd d:\UrbanSense\frontend
npm install
npm run dev
# → http://localhost:5173
```

### Backend (minimal install — no PyTorch yet)
```bash
cd d:\UrbanSense\backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn pydantic-settings python-multipart numpy scipy
python main.py
# → http://localhost:8000/docs
```

Create a `.env` file in `backend/`:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/urbansense
REDIS_URL=redis://localhost:6379
SECRET_KEY=dev-secret-key-change-in-production
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:5173
```

---

## 4. Phase 2 — Wire Routes to Real Database

### Step 4a — Set Up PostgreSQL with PostGIS

Install PostgreSQL + PostGIS locally, then:
```sql
CREATE DATABASE urbansense;
\c urbansense
CREATE EXTENSION postgis;
```

### Step 4b — Create Tables

In `backend/`, run:
```bash
python -c "from db import engine; from models.database import Base; Base.metadata.create_all(engine)"
```

### Step 4c — Seed Initial Region Data

Create `backend/seed_data.py`:
```python
from db import get_session
from models.database import Region

regions = [
    Region(name="Dharavi", city="Mumbai", country="India",
           latitude=19.0396, longitude=72.8556,
           area_sqkm=2.39, population=1000000, slum_percentage=92.0),
    Region(name="Kibera", city="Nairobi", country="Kenya",
           latitude=-1.3133, longitude=36.7833,
           area_sqkm=2.5, population=250000, slum_percentage=85.0),
    Region(name="Mathare", city="Nairobi", country="Kenya",
           latitude=-1.2592, longitude=36.8539,
           area_sqkm=1.1, population=180000, slum_percentage=78.0),
]

with get_session() as session:
    for region in regions:
        session.add(region)
    session.commit()
    print("Seeded regions!")
```

### Step 4d — Fix `routes/data.py` to Query Real DB

Replace the mock return in `/data/regions`:
```python
# In routes/data.py
from db import get_session
from models.database import Region

@router.get("/data/regions")
async def get_regions():
    with get_session() as session:
        regions = session.query(Region).all()
        return [
            {
                "id": str(r.id), "name": r.name, "city": r.city,
                "country": r.country, "latitude": r.latitude,
                "longitude": r.longitude, "area_sqkm": r.area_sqkm,
                "population": r.population,
                "slum_percentage": r.slum_percentage,
            }
            for r in regions
        ]
```

---

## 5. Phase 3 — Datasets

### 5a. Satellite Imagery (for U-Net training)

| Dataset | What it is | How to get |
|---|---|---|
| **SpaceNet 8** | High-res satellite + building footprints | https://spacenet.ai/datasets/ (free) |
| **UN-Habitat Slum Dataset** | Labeled slum polygons in 8 African cities | https://www.unhabitat.org/open-data |
| **Sentinel-2 (ESA)** | Free 10m multispectral imagery worldwide | https://scihub.copernicus.eu (free, register) |
| **Landsat 8/9 (USGS)** | Free 30m imagery, 40+ years of history | https://earthexplorer.usgs.gov (free) |
| **OpenStreetMap** | Roads, buildings, water bodies | https://overpass-turbo.eu (free) |

**Quickest dataset to start with:**
Download Sentinel-2 tiles for Mumbai or Nairobi (free, ~500MB per tile).

### 5b. Ground Truth Labels

| Source | Description |
|---|---|
| **GRID3** | Slum boundaries for 12+ African cities (GeoJSON) — https://grid3.org |
| **Map Kibera** | Detailed Kibera slum mapping — https://mapkibera.org |
| **OSM Overpass** | `amenity=slum` or `landuse=residential` tags |

Download ground truth GeoJSON:
```bash
# Example: Get Kibera slum boundaries from Overpass API
curl "https://overpass-api.de/api/interpreter?data=[out:json];(way[landuse=residential]({{bbox}}););out geom;" > kibera.json
```

---

## 6. Phase 3 — Build & Train the Models

### 6a. U-Net Segmentation Model

Install full ML stack:
```bash
pip install torch torchvision rasterio geopandas shapely albumentations segmentation-models-pytorch
```

Create `backend/ml/train_segmentation.py`:
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import rasterio
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SlumDataset(Dataset):
    """Dataset for slum segmentation from satellite imagery"""

    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(self.image_dir.glob("*.tif"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name.replace("image", "mask")

        # Load satellite image (RGB bands from Sentinel-2)
        with rasterio.open(img_path) as src:
            # Bands: B04 (Red), B03 (Green), B02 (Blue)
            image = src.read([4, 3, 2]).astype(np.float32)
            image = image / 10000.0  # Sentinel-2 scale factor
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

        # Load binary mask (1=slum, 0=non-slum)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0)


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.CenterCrop(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def train_model(image_dir: str, mask_dir: str, output_path: str = "models/unet_slum.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Model: U-Net with ResNet34 encoder (pretrained on ImageNet)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    ).to(device)

    # Dataset
    train_dataset = SlumDataset(image_dir, mask_dir, get_transforms(train=True))
    val_dataset = SlumDataset(image_dir, mask_dir, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Loss: Dice + BCE
    criterion = smp.losses.DiceLoss(mode="binary") + smp.losses.SoftBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_iou = 0.0
    for epoch in range(50):
        # Training loop
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        iou_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = (model(images) > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum() - intersection
                iou = (intersection / (union + 1e-6)).item()
                iou_scores.append(iou)

        mean_iou = np.mean(iou_scores)
        scheduler.step()
        print(f"Epoch {epoch+1}/50 | Loss: {train_loss/len(train_loader):.4f} | IoU: {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best model (IoU={best_iou:.4f})")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    train_model(
        image_dir="data/images/train",
        mask_dir="data/masks/train",
        output_path="models/unet_slum.pt"
    )
```

**Expected results:** ~0.78–0.85 IoU on Sentinel-2 with good labels.

---

### 6b. Risk Assessment Model (XGBoost)

Install:
```bash
pip install xgboost scikit-learn pandas joblib
```

Create `backend/ml/train_risk_model.py`:
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

def create_synthetic_training_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic training data.
    Replace this with your real dataset (CSV with ground truth risk scores).
    Real data sources: UN-Habitat field surveys, World Bank slum data.
    """
    np.random.seed(42)
    df = pd.DataFrame({
        "building_density": np.random.beta(2, 5, n_samples),
        "roof_flammability": np.random.beta(2, 3, n_samples),
        "road_accessibility": np.random.beta(3, 2, n_samples),
        "elevation_m": np.random.uniform(0, 200, n_samples),
        "dist_to_water_m": np.random.uniform(50, 5000, n_samples),
        "drainage_quality": np.random.beta(2, 4, n_samples),
        "hospital_dist_m": np.random.uniform(200, 10000, n_samples),
        "road_density": np.random.beta(3, 3, n_samples),
        "population_density": np.random.lognormal(8, 1, n_samples),
        "ndvi": np.random.uniform(-0.2, 0.8, n_samples),  # vegetation index
    })

    # Ground truth risk score (0-1) derived from weighted formula
    df["risk_score"] = (
        df["building_density"] * 0.25 +
        df["roof_flammability"] * 0.15 +
        (1 - df["road_accessibility"]) * 0.15 +
        (1 - df["elevation_m"] / 200) * 0.15 +
        (1 - df["dist_to_water_m"] / 5000) * 0.10 +
        (1 - df["drainage_quality"]) * 0.10 +
        (df["hospital_dist_m"] / 10000) * 0.10
    ).clip(0, 1)

    return df


def train_risk_model(output_dir: str = "models/"):
    df = create_synthetic_training_data(10000)

    features = [c for c in df.columns if c != "risk_score"]
    X = df[features]
    y = df["risk_score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=50,
    )

    preds = model.predict(X_test_scaled)
    print(f"\nMAE: {mean_absolute_error(y_test, preds):.4f}")
    print(f"R²:  {r2_score(y_test, preds):.4f}")

    joblib.dump(model, f"{output_dir}/risk_model.joblib")
    joblib.dump(scaler, f"{output_dir}/risk_scaler.joblib")
    joblib.dump(features, f"{output_dir}/risk_features.joblib")
    print("Risk model saved!")


if __name__ == "__main__":
    train_risk_model()
```

---

### 6c. Growth Prediction (LSTM)

Install:
```bash
pip install torch
```

Create `backend/ml/train_growth_model.py`:
```python
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

class LSTMGrowthPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def generate_growth_data(n_series=200, length=20):
    """
    Generate synthetic slum area time series.
    Replace with real data from: UN-Habitat, World Bank Urban Development data.
    """
    X, y = [], []
    for _ in range(n_series):
        base = np.random.uniform(0.5, 10.0)
        rate = np.random.uniform(0.02, 0.12)
        noise = np.random.normal(0, 0.05, length + 1)
        series = [base * ((1 + rate) ** t) + noise[t] for t in range(length + 1)]
        X.append(series[:length])
        y.append(series[length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_growth_model(output_path="models/lstm_growth.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = generate_growth_data(2000, length=10)
    X = torch.tensor(X).unsqueeze(-1)
    y = torch.tensor(y).unsqueeze(-1)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = LSTMGrowthPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train.to(device))
        loss = criterion(pred, y_train.to(device))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val.to(device))
                val_loss = criterion(val_pred, y_val.to(device))
            print(f"Epoch {epoch+1} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss.item():.6f}")

    torch.save(model.state_dict(), output_path)
    print(f"LSTM model saved to {output_path}")


if __name__ == "__main__":
    train_growth_model()
```

---

## 7. Phase 4 — Connect ML Models to the API

After training, update `services/segmentation_service.py` to load the real model:

```python
import torch
import segmentation_models_pytorch as smp
import numpy as np

class SegmentationService:
    def __init__(self, model_path: str = "models/unet_slum.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=1, activation="sigmoid"
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval().to(self.device)

    def detect_slums(self, image: np.ndarray) -> dict:
        """Run real U-Net inference on a satellite image patch."""
        tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            mask = self.model(tensor).squeeze().cpu().numpy()
        slum_pixels = np.sum(mask > 0.5)
        slum_percentage = (slum_pixels / mask.size) * 100
        return {
            "segmentation_mask": mask.tolist(),
            "slum_percentage": round(slum_percentage, 2),
            "confidence": float(np.mean(mask[mask > 0.5])) if slum_pixels > 0 else 0.0,
        }
```

Update `services/risk_service.py` to load real XGBoost:

```python
import joblib
import numpy as np

class RiskAnalysisService:
    def __init__(self, model_dir: str = "models/"):
        self.model = joblib.load(f"{model_dir}/risk_model.joblib")
        self.scaler = joblib.load(f"{model_dir}/risk_scaler.joblib")
        self.features = joblib.load(f"{model_dir}/risk_features.joblib")

    def predict_risk(self, feature_dict: dict) -> dict:
        import pandas as pd
        X = pd.DataFrame([feature_dict])[self.features]
        X_scaled = self.scaler.transform(X)
        score = float(self.model.predict(X_scaled)[0])
        return {
            "overall_risk": round(score, 3),
            "risk_level": self._categorize_risk(score),
        }
```

---

## 8. Phase 5 — Full Requirements

Create `backend/requirements.txt` (final version):
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic-settings==2.2.1
sqlalchemy==2.0.30
psycopg2-binary==2.9.9
alembic==1.13.1
redis==5.0.4
python-multipart==0.0.9
torch==2.3.0
torchvision==0.18.0
segmentation-models-pytorch==0.3.3
xgboost==2.0.3
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
rasterio==1.3.10
geopandas==0.14.4
shapely==2.0.4
scipy==1.13.0
networkx==3.3
albumentations==1.4.7
joblib==1.4.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
httpx==0.27.0
```

---

## 9. Data Directory Structure

```
backend/
  data/
    images/
      train/          ← Sentinel-2 .tif patches (256x256)
      val/
      test/
    masks/
      train/          ← Binary GeoTIFF masks (1=slum, 0=non-slum)
      val/
      test/
    raw/              ← Original downloaded tiles
  models/
    unet_slum.pt      ← Trained U-Net weights
    risk_model.joblib ← Trained XGBoost model
    risk_scaler.joblib
    risk_features.joblib
    lstm_growth.pt    ← Trained LSTM weights
  ml/
    train_segmentation.py
    train_risk_model.py
    train_growth_model.py
    preprocess_sentinel2.py   ← (create next)
```

---

## 10. Data Preprocessing Script

Create `backend/ml/preprocess_sentinel2.py`:
```python
"""
Download and tile Sentinel-2 imagery into 256x256 patches.
Requires: sentinelsat or direct download from Copernicus Hub.
"""
import rasterio
from rasterio.windows import Window
import numpy as np
from pathlib import Path

def tile_image(input_path: str, output_dir: str, tile_size: int = 256):
    """Slice a large GeoTIFF into 256x256 training patches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        width, height = src.width, src.height
        count = 0
        for row in range(0, height - tile_size, tile_size // 2):  # 50% overlap
            for col in range(0, width - tile_size, tile_size // 2):
                window = Window(col, row, tile_size, tile_size)
                data = src.read(window=window)

                # Skip mostly black tiles
                if np.mean(data) < 100:
                    continue

                profile = src.profile.copy()
                profile.update(width=tile_size, height=tile_size,
                               transform=src.window_transform(window))

                out_path = output_dir / f"patch_{count:04d}.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)
                count += 1

    print(f"Created {count} tiles in {output_dir}")


if __name__ == "__main__":
    tile_image(
        input_path="data/raw/sentinel2_mumbai_B432.tif",
        output_dir="data/images/train/"
    )
```

---

## 11. Key Commands Reference

```bash
# Run backend
cd backend && python main.py

# Run frontend
cd frontend && npm run dev

# Train U-Net (GPU recommended, ~2-4 hours on GPU)
cd backend && python ml/train_segmentation.py

# Train XGBoost risk model (~5 minutes on CPU)
cd backend && python ml/train_risk_model.py

# Train LSTM growth model (~2 minutes on CPU)
cd backend && python ml/train_growth_model.py

# Seed database
cd backend && python seed_data.py

# Docker (full stack)
docker-compose up --build
```

---

## 12. What to Do Right Now (Quickstart)

1. `cd d:\UrbanSense\frontend && npm install && npm run dev` — verify UI works
2. `cd d:\UrbanSense\backend`, create venv, install minimal deps, run `python main.py` — verify API works
3. Sign up at https://scihub.copernicus.eu — download 1 Sentinel-2 tile for Mumbai or Nairobi
4. Install PostGIS locally and run the DB migration
5. Run `python ml/train_risk_model.py` first (fastest, no GPU needed)
6. Then tackle the U-Net (needs GPU or Google Colab)
