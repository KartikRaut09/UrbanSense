"""
XGBoost Risk Assessment Model — Training Script
================================================
Predicts a composite risk score (0.0–1.0) for a slum area based on:
  - Building density
  - Roof flammability
  - Road accessibility
  - Elevation
  - Distance to water bodies
  - Drainage quality
  - Hospital distance
  - Road density
  - Population density
  - NDVI (vegetation index)

Output: risk_model.joblib, risk_scaler.joblib, risk_features.joblib

Using Real Data (recommended):
  Replace create_synthetic_training_data() with your CSV loaded via pd.read_csv().
  Suggested real datasets:
    - UN-Habitat Field Survey Data: https://www.unhabitat.org/open-data
    - World Bank Slum Indicators: https://data.worldbank.org (search "slum")
    - OpenStreetMap: extract road density, building density per grid cell
    - SRTM Elevation Data: https://earthexplorer.usgs.gov
    - Global Surface Water (JRC): https://global-surface-water.appspot.com
    - WorldPop population density: https://www.worldpop.org

Expected CSV columns (for real data mode):
  building_density, roof_flammability, road_accessibility, elevation_m,
  dist_to_water_m, drainage_quality, hospital_dist_m, road_density,
  population_density, ndvi, risk_score

Usage:
    python train_risk_model.py
    python train_risk_model.py --data_csv path/to/real_data.csv
    python train_risk_model.py --output_dir models/
"""

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES = [
    "building_density",     # 0–1: fraction of land covered by buildings
    "roof_flammability",    # 0–1: proportion of flammable roofs (tin=low, thatch=high)
    "road_accessibility",   # 0–1: quality/density of road access
    "elevation_m",          # meters above sea level
    "dist_to_water_m",      # distance to nearest river/lake in meters
    "drainage_quality",     # 0–1: quality of drainage infrastructure
    "hospital_dist_m",      # distance to nearest hospital in meters
    "road_density",         # 0–1: density of road network
    "population_density",   # people per sq km
    "ndvi",                 # Normalized Difference Vegetation Index (-1 to 1)
]


def create_synthetic_training_data(n_samples: int = 15000) -> pd.DataFrame:
    """
    Generate realistic synthetic training data for risk model.

    This synthetic data models real-world relationships:
    - Dense buildings + flammable roofs → high fire risk
    - Low elevation + near water + poor drainage → high flood risk
    - Far hospital + poor roads → high accessibility risk

    REPLACE THIS with real survey data for production use.
    Real data sources listed in the module docstring above.
    """
    np.random.seed(42)

    # Generate features with realistic distributions
    df = pd.DataFrame({
        # Building density: most slums are 0.4–0.9
        "building_density": np.random.beta(4, 2, n_samples),
        # Roof flammability: bimodal (tin = 0.2–0.4, thatch = 0.7–0.9)
        "roof_flammability": np.where(
            np.random.rand(n_samples) < 0.6,
            np.random.beta(2, 5, n_samples),   # metal roofs (lower)
            np.random.beta(5, 2, n_samples),   # thatch/plastic (higher)
        ),
        # Road accessibility: slums typically have poor access
        "road_accessibility": np.random.beta(2, 4, n_samples),
        # Elevation: most slums are in low-lying areas
        "elevation_m": np.random.lognormal(3, 1, n_samples).clip(0, 500),
        # Distance to water: river-adjacent areas are common
        "dist_to_water_m": np.random.lognormal(6, 1, n_samples).clip(10, 10000),
        # Drainage: typically poor in slums
        "drainage_quality": np.random.beta(2, 5, n_samples),
        # Hospital distance: usually far
        "hospital_dist_m": np.random.lognormal(8, 0.8, n_samples).clip(100, 20000),
        # Road density: correlated with accessibility
        "road_density": np.random.beta(2, 4, n_samples),
        # Population density: extremely high in slums
        "population_density": np.random.lognormal(10, 0.8, n_samples).clip(5000, 200000),
        # NDVI: low in dense slums
        "ndvi": np.random.normal(0.1, 0.15, n_samples).clip(-0.3, 0.8),
    })

    # Ground truth risk score — weighted combination of risk factors
    # These weights are based on UN-Habitat risk assessment methodology
    fire_risk = (
        df["building_density"] * 0.4 +
        df["roof_flammability"] * 0.4 +
        (1 - df["road_accessibility"]) * 0.2
    )
    flood_risk = (
        (1 - np.clip(df["elevation_m"] / 100, 0, 1)) * 0.4 +
        (1 - np.clip(df["dist_to_water_m"] / 2000, 0, 1)) * 0.35 +
        (1 - df["drainage_quality"]) * 0.25
    )
    access_risk = (
        np.clip(df["hospital_dist_m"] / 5000, 0, 1) * 0.4 +
        (1 - df["road_density"]) * 0.35 +
        (1 - df["road_accessibility"]) * 0.25
    )

    df["risk_score"] = (
        fire_risk * 0.35 +
        flood_risk * 0.30 +
        access_risk * 0.35
    ).clip(0, 1)

    # Add small noise to simulate measurement error
    df["risk_score"] += np.random.normal(0, 0.02, n_samples)
    df["risk_score"] = df["risk_score"].clip(0, 1)

    logger.info(f"Synthetic dataset: {len(df)} samples")
    logger.info(f"Risk score distribution: min={df['risk_score'].min():.3f}, "
                f"mean={df['risk_score'].mean():.3f}, max={df['risk_score'].max():.3f}")
    return df


def load_real_data(csv_path: str) -> pd.DataFrame:
    """Load real survey data from CSV."""
    df = pd.read_csv(csv_path)
    missing = [f for f in FEATURES + ["risk_score"] if f not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing these required columns: {missing}")
    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    return df


def train(args):
    # ── Load Data ──
    if args.data_csv:
        df = load_real_data(args.data_csv)
    else:
        logger.info("No CSV provided — using synthetic data for training.")
        logger.info("For production, collect real data and pass --data_csv path/to/data.csv")
        df = create_synthetic_training_data(15000)

    X = df[FEATURES]
    y = df["risk_score"]

    # ── Train/Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # ── Feature Scaling ──
    # XGBoost doesn't strictly need scaling, but it helps with regularization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Model ──
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,      # L1 regularization
        reg_lambda=1.5,      # L2 regularization
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=50,
    )

    logger.info("Training XGBoost risk model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100,
    )

    # ── Evaluation ──
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    logger.info(f"\n{'='*40}")
    logger.info(f"Risk Model Evaluation:")
    logger.info(f"  MAE  : {mae:.4f}  (avg prediction error)")
    logger.info(f"  RMSE : {rmse:.4f}")
    logger.info(f"  R²   : {r2:.4f}  (1.0 = perfect)")
    logger.info(f"{'='*40}")

    # ── Feature Importance ──
    importance = dict(zip(FEATURES, model.feature_importances_))
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("\nFeature Importance:")
    for feat, imp in importance_sorted:
        bar = "█" * int(imp * 40)
        logger.info(f"  {feat:<25} {imp:.4f} {bar}")

    # ── Save ──
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model_path = f"{args.output_dir}/risk_model.joblib"
    scaler_path = f"{args.output_dir}/risk_scaler.joblib"
    features_path = f"{args.output_dir}/risk_features.joblib"
    metrics_path = f"{args.output_dir}/risk_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(FEATURES, features_path)

    metrics = {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": len(df)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\n✓ Model saved to:    {model_path}")
    logger.info(f"✓ Scaler saved to:   {scaler_path}")
    logger.info(f"✓ Features saved to: {features_path}")
    logger.info(f"✓ Metrics saved to:  {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost risk assessment model")
    parser.add_argument("--data_csv", default=None,
                        help="Path to real survey data CSV (optional, uses synthetic if not provided)")
    parser.add_argument("--output_dir", default="models",
                        help="Output directory for saved model files (default: models/)")
    args = parser.parse_args()
    train(args)
