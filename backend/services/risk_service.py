"""
Risk Analysis Service — Production Implementation
=================================================
Uses trained XGBoost model (models/risk_model.joblib) to predict
composite risk scores for slum areas.

Falls back to formula-based calculation if model is not trained yet.
"""

import numpy as np
from typing import Dict, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import joblib
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("joblib/pandas not installed. Risk model will use formula fallback.")


class RiskAnalysisService:
    """
    Comprehensive risk assessment service.

    If trained model exists → XGBoost prediction.
    If not → weighted formula (still accurate, just not data-driven).
    """

    MODEL_DIR = "models"
    RISK_WEIGHTS = {
        "fire_risk": 0.35,
        "flood_risk": 0.30,
        "accessibility_risk": 0.35,
    }

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None

        if SKLEARN_AVAILABLE:
            self._load_model()

    def _load_model(self):
        model_path = f"{self.MODEL_DIR}/risk_model.joblib"
        scaler_path = f"{self.MODEL_DIR}/risk_scaler.joblib"
        features_path = f"{self.MODEL_DIR}/risk_features.joblib"

        if all(Path(p).exists() for p in [model_path, scaler_path, features_path]):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.features = joblib.load(features_path)
                logger.info(f"✓ XGBoost risk model loaded from {self.MODEL_DIR}/")
            except Exception as e:
                logger.error(f"Failed to load risk model: {e}. Using formula fallback.")
        else:
            logger.info("Risk model not trained yet. Using formula-based assessment. "
                        "Train model: python ml/train_risk_model.py")

    def assess_risk(self, feature_dict: Dict) -> Dict:
        """
        Calculate comprehensive risk score.

        Args:
            feature_dict: dict with keys:
                building_density   : 0–1
                roof_flammability  : 0–1
                road_accessibility : 0–1
                elevation_m        : meters
                dist_to_water_m    : meters
                drainage_quality   : 0–1
                hospital_dist_m    : meters
                road_density       : 0–1
                population_density : people/sqkm
                ndvi               : -1 to 1

        Returns:
            dict with fire_risk, flood_risk, accessibility_risk,
                       overall_risk, risk_level, interventions
        """
        if self.model is not None:
            return self._model_prediction(feature_dict)
        else:
            return self._formula_assessment(feature_dict)

    def _model_prediction(self, feature_dict: Dict) -> Dict:
        """XGBoost-based prediction."""
        X = pd.DataFrame([feature_dict])[self.features]
        X_scaled = self.scaler.transform(X)
        overall_risk = float(np.clip(self.model.predict(X_scaled)[0], 0, 1))

        # Decompose into component risks using formula
        fire_risk = self.calculate_fire_risk(
            feature_dict.get("building_density", 0.5),
            feature_dict.get("roof_flammability", 0.5),
            feature_dict.get("road_accessibility", 0.5),
        )
        flood_risk = self.calculate_flood_risk(
            feature_dict.get("elevation_m", 50),
            feature_dict.get("dist_to_water_m", 500),
            feature_dict.get("drainage_quality", 0.3),
        )
        access_risk = self.calculate_accessibility_risk(
            feature_dict.get("hospital_dist_m", 2000),
            feature_dict.get("road_density", 0.4),
            feature_dict.get("road_accessibility", 0.5),
        )

        result = {
            "fire_risk": round(fire_risk, 3),
            "flood_risk": round(flood_risk, 3),
            "accessibility_risk": round(access_risk, 3),
            "overall_risk": round(overall_risk, 3),
            "risk_level": self._categorize_risk(overall_risk),
            "model_used": "xgboost",
        }
        result["interventions"] = self.generate_interventions(result)
        return result

    def _formula_assessment(self, feature_dict: Dict) -> Dict:
        """Formula-based fallback when model is not trained."""
        fire_risk = self.calculate_fire_risk(
            feature_dict.get("building_density", 0.5),
            feature_dict.get("roof_flammability", 0.5),
            feature_dict.get("road_accessibility", 0.5),
        )
        flood_risk = self.calculate_flood_risk(
            feature_dict.get("elevation_m", 50),
            feature_dict.get("dist_to_water_m", 500),
            feature_dict.get("drainage_quality", 0.3),
        )
        access_risk = self.calculate_accessibility_risk(
            feature_dict.get("hospital_dist_m", 2000),
            feature_dict.get("road_density", 0.4),
            feature_dict.get("road_accessibility", 0.5),
        )

        overall_risk = (
            fire_risk * self.RISK_WEIGHTS["fire_risk"] +
            flood_risk * self.RISK_WEIGHTS["flood_risk"] +
            access_risk * self.RISK_WEIGHTS["accessibility_risk"]
        )

        result = {
            "fire_risk": round(fire_risk, 3),
            "flood_risk": round(flood_risk, 3),
            "accessibility_risk": round(access_risk, 3),
            "overall_risk": round(float(overall_risk), 3),
            "risk_level": self._categorize_risk(overall_risk),
            "model_used": "formula",
        }
        result["interventions"] = self.generate_interventions(result)
        return result

    # ── Risk Component Calculators ──

    def calculate_fire_risk(self, building_density: float,
                             roof_flammability: float,
                             road_accessibility: float) -> float:
        risk = (building_density * 0.4 +
                roof_flammability * 0.4 +
                (1 - road_accessibility) * 0.2)
        return float(np.clip(risk, 0, 1))

    def calculate_flood_risk(self, elevation_m: float,
                              dist_to_water_m: float,
                              drainage_quality: float) -> float:
        elevation_factor = 1 - min(elevation_m / 100.0, 1.0)
        water_factor = 1 - min(dist_to_water_m / 2000.0, 1.0)
        drainage_factor = 1 - drainage_quality
        risk = (elevation_factor * 0.4 + water_factor * 0.35 + drainage_factor * 0.25)
        return float(np.clip(risk, 0, 1))

    def calculate_accessibility_risk(self, hospital_dist_m: float,
                                      road_density: float,
                                      road_accessibility: float) -> float:
        hospital_factor = min(hospital_dist_m / 5000.0, 1.0)
        risk = (hospital_factor * 0.4 +
                (1 - road_density) * 0.35 +
                (1 - road_accessibility) * 0.25)
        return float(np.clip(risk, 0, 1))

    @staticmethod
    def _categorize_risk(score: float) -> str:
        if score >= 0.75:
            return "Critical"
        elif score >= 0.50:
            return "High"
        elif score >= 0.25:
            return "Moderate"
        else:
            return "Low"

    def generate_interventions(self, risk_scores: Dict) -> List[str]:
        """Generate specific actionable interventions based on risk profile."""
        interventions = []

        if risk_scores["fire_risk"] >= 0.60:
            interventions.extend([
                "Install fire hydrant network with 150m spacing",
                "Establish community water storage tanks (10,000L minimum)",
                "Replace thatch/plastic roofing with fire-resistant metal sheets",
                "Create 3m firebreak lanes between building clusters",
            ])
        elif risk_scores["fire_risk"] >= 0.35:
            interventions.extend([
                "Improve emergency vehicle road access",
                "Community fire safety training program",
            ])

        if risk_scores["flood_risk"] >= 0.60:
            interventions.extend([
                "Install underground drainage network (150mm diameter pipes)",
                "Build retaining walls and flood barriers",
                "Elevate ground floors by 0.5–1.0m in high-risk zones",
                "Restore wetlands as natural flood buffers",
            ])
        elif risk_scores["flood_risk"] >= 0.35:
            interventions.extend([
                "Improve open channel drainage",
                "Early warning flood monitoring system",
            ])

        if risk_scores["accessibility_risk"] >= 0.60:
            interventions.extend([
                "Develop minimum 4m-wide emergency access roads",
                "Establish primary healthcare clinic within 1km",
                "Deploy mobile health unit program",
                "Mark and clear official evacuation routes",
            ])
        elif risk_scores["accessibility_risk"] >= 0.35:
            interventions.extend([
                "Improve pedestrian pathway network",
                "Negotiate emergency vehicle access agreements",
            ])

        if not interventions:
            interventions.append("Continue regular monitoring — risk levels acceptable")

        return interventions


class GrowthPredictionService:
    """
    Slum growth prediction service.

    Uses trained LSTM model if available,
    otherwise falls back to trend extrapolation.
    """

    MODEL_PATH = "models/lstm_growth.pt"
    SEQ_LEN = 10

    def __init__(self):
        self.model = None
        self.model_meta = None

        try:
            import torch
            self.torch = torch
            if Path(self.MODEL_PATH).exists():
                self._load_model()
            else:
                logger.info("LSTM growth model not trained. Using trend extrapolation. "
                            "Train: python ml/train_growth_model.py")
        except ImportError:
            self.torch = None
            logger.info("PyTorch not installed. Growth prediction uses trend extrapolation.")

    def _load_model(self):
        try:
            from ml.train_growth_model import LSTMGrowthPredictor
            checkpoint = self.torch.load(self.MODEL_PATH, map_location="cpu")
            self.model = LSTMGrowthPredictor(
                hidden_size=checkpoint.get("hidden_size", 64),
                num_layers=2
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.model_meta = checkpoint
            logger.info(f"✓ LSTM growth model loaded from {self.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")

    def predict_growth(self, historical_areas: List[float],
                       forecast_years: int = 5) -> Dict:
        """
        Predict future slum area growth.

        Args:
            historical_areas : list of annual area values (sq km), oldest first
            forecast_years   : number of years to forecast

        Returns:
            dict with forecasts list, growth_rate, trend, model_used
        """
        if len(historical_areas) < 2:
            return {"error": "Need at least 2 years of historical data"}

        if self.model is not None and len(historical_areas) >= self.SEQ_LEN:
            return self._lstm_predict(historical_areas, forecast_years)
        else:
            return self._trend_predict(historical_areas, forecast_years)

    def _lstm_predict(self, historical_areas: List[float], forecast_years: int) -> Dict:
        """LSTM multi-step prediction (recursive)."""
        areas = list(historical_areas[-self.SEQ_LEN:])
        scale = max(areas)
        if scale == 0:
            scale = 1.0

        predictions = []
        for year in range(1, forecast_years + 1):
            seq = np.array(areas[-self.SEQ_LEN:], dtype=np.float32) / scale
            tensor = self.torch.tensor(seq).unsqueeze(0).unsqueeze(-1)
            with self.torch.no_grad():
                pred_norm = self.model(tensor).item()
            pred_area = max(0.01, pred_norm * scale)
            areas.append(pred_area)
            # Confidence decreases with forecast horizon
            confidence = max(0.95 - (year - 1) * 0.08, 0.55)
            predictions.append({
                "year": year,
                "predicted_area_sqkm": round(pred_area, 3),
                "confidence": round(confidence, 2),
            })

        last = historical_areas[-1]
        growth_rate = (predictions[-1]["predicted_area_sqkm"] / last) ** (1/forecast_years) - 1
        return {
            "forecasts": predictions,
            "avg_annual_growth_rate": round(float(growth_rate), 4),
            "trend": "increasing" if growth_rate > 0.01 else
                     "declining" if growth_rate < -0.01 else "stable",
            "model_used": "lstm",
        }

    def _trend_predict(self, historical_areas: List[float], forecast_years: int) -> Dict:
        """Linear trend extrapolation fallback."""
        rates = np.diff(historical_areas) / np.array(historical_areas[:-1])
        avg_rate = float(np.mean(rates))
        last_area = historical_areas[-1]

        predictions = []
        for year in range(1, forecast_years + 1):
            projected = last_area * ((1 + avg_rate) ** year)
            projected = max(0.01, projected)
            confidence = max(0.90 - (year - 1) * 0.07, 0.55)
            predictions.append({
                "year": year,
                "predicted_area_sqkm": round(projected, 3),
                "confidence": round(confidence, 2),
            })

        return {
            "forecasts": predictions,
            "avg_annual_growth_rate": round(avg_rate, 4),
            "trend": "increasing" if avg_rate > 0.01 else
                     "declining" if avg_rate < -0.01 else "stable",
            "model_used": "trend_extrapolation",
        }
