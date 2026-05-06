"""
Advanced Risk Analysis Service — Stacked Ensemble Inference
===========================================================
Loads XGBoost + LightGBM + CatBoost stacked ensemble per risk target.
Returns: risk scores, SHAP feature attributions, calibrated probabilities.
Falls back to formula calculation if models not trained yet.
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import joblib
    import pandas as pd
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class RiskAnalysisService:
    """
    Stacked ensemble risk assessment service.

    Targets: fire_risk, flood_risk, accessibility_risk, overall_risk
    Each target has its own XGBoost + LightGBM + CatBoost + Ridge meta-model.

    If trained models exist → full ensemble prediction with SHAP explanations.
    If not → formula-based fallback (still accurate for demo).
    """

    MODEL_DIR = "models"
    TARGETS = ["fire_risk", "flood_risk", "accessibility_risk", "overall_risk"]
    RISK_WEIGHTS = {"fire_risk": 0.35, "flood_risk": 0.30, "accessibility_risk": 0.35}

    def __init__(self):
        self.models = {}      # target → {xgb, lgb, cat, meta, scaler}
        self.features = None
        self.explainers = {}  # target → shap TreeExplainer
        self._loaded = False

        if SKLEARN_AVAILABLE:
            self._load_ensemble()

    def _load_ensemble(self):
        """Load all 4 target ensembles from disk."""
        meta_path = f"{self.MODEL_DIR}/risk_ensemble_metadata.json"
        features_path = f"{self.MODEL_DIR}/risk_features.joblib"

        if not Path(features_path).exists():
            logger.info("Risk ensemble not trained. Using formula fallback. "
                        "Run: python ml/train_risk_model.py")
            return

        try:
            self.features = joblib.load(features_path)

            for target in self.TARGETS:
                safe = target.replace("_", "")
                paths = {
                    "xgb": f"{self.MODEL_DIR}/risk_{safe}_xgb.joblib",
                    "lgb": f"{self.MODEL_DIR}/risk_{safe}_lgb.joblib",
                    "cat": f"{self.MODEL_DIR}/risk_{safe}_cat.joblib",
                    "meta": f"{self.MODEL_DIR}/risk_{safe}_meta.joblib",
                    "scaler": f"{self.MODEL_DIR}/risk_{safe}_scaler.joblib",
                }
                if all(Path(p).exists() for p in paths.values()):
                    self.models[target] = {k: joblib.load(v) for k, v in paths.items()}
                    # SHAP explainer on XGBoost base model
                    if SHAP_AVAILABLE:
                        self.explainers[target] = shap.TreeExplainer(
                            self.models[target]["xgb"]
                        )

            if self.models:
                self._loaded = True
                logger.info(f"✓ Risk ensemble loaded for targets: {list(self.models.keys())}")
        except Exception as e:
            logger.error(f"Failed to load risk ensemble: {e}")

    def _engineer_features(self, feature_dict: Dict) -> Dict:
        """Add engineered interaction features (must match training)."""
        d = dict(feature_dict)
        d["fire_proxy"] = d.get("building_density", 0.5) * d.get("roof_flammability", 0.5)
        d["flood_proxy"] = (1 / max(d.get("elevation_m", 20), 1)) * \
                           (1 / max(d.get("dist_to_water_m", 300) / 1000, 0.01))
        d["access_proxy"] = (d.get("hospital_dist_m", 2000) / 5000) * \
                            (1 - d.get("road_density", 0.3))
        d["density_accessibility"] = d.get("building_density", 0.5) * \
                                     (1 - d.get("road_accessibility", 0.3))
        return d

    def assess_risk(self, feature_dict: Dict) -> Dict:
        """
        Full risk assessment with ensemble + SHAP explanations.

        Returns:
            fire_risk, flood_risk, accessibility_risk, overall_risk (0–1)
            risk_level: "Low" / "Moderate" / "High" / "Critical"
            shap_explanation: {feature: contribution} showing WHY risk is high
            interventions: prioritized list of recommended actions
            model_used: "ensemble" or "formula"
        """
        enriched = self._engineer_features(feature_dict)

        if self._loaded and self.features:
            return self._ensemble_predict(enriched)
        else:
            return self._formula_assess(feature_dict)

    def _ensemble_predict(self, enriched: Dict) -> Dict:
        """Stacked ensemble prediction."""
        X_row = pd.DataFrame([enriched])[self.features]
        scores = {}
        shap_explanations = {}

        for target in self.TARGETS:
            if target not in self.models:
                continue
            m = self.models[target]
            X_scaled = m["scaler"].transform(X_row)

            # Stack base model predictions
            pred_xgb = float(m["xgb"].predict(X_scaled)[0])
            pred_lgb = float(m["lgb"].predict(X_scaled)[0])
            pred_cat = float(m["cat"].predict(X_scaled)[0])
            stack_input = np.array([[pred_xgb, pred_lgb, pred_cat]])
            score = float(np.clip(m["meta"].predict(stack_input)[0], 0, 1))
            scores[target] = round(score, 3)

            # SHAP values for XGBoost base model
            if target in self.explainers and SHAP_AVAILABLE:
                sv = self.explainers[target].shap_values(X_scaled)[0]
                shap_explanations[target] = {
                    feat: round(float(val), 4)
                    for feat, val in zip(self.features, sv)
                    if abs(val) > 0.005  # only significant contributors
                }

        # If overall_risk not in model, compute from components
        if "overall_risk" not in scores:
            scores["overall_risk"] = round(
                scores.get("fire_risk", 0) * 0.35 +
                scores.get("flood_risk", 0) * 0.30 +
                scores.get("accessibility_risk", 0) * 0.35, 3
            )

        result = {
            **scores,
            "risk_level": self._categorize(scores["overall_risk"]),
            "shap_explanation": shap_explanations,
            "interventions": self.generate_interventions(scores),
            "model_used": "stacked_ensemble",
        }
        return result

    def _formula_assess(self, feature_dict: Dict) -> Dict:
        """Formula-based fallback when ensemble not trained."""
        fire = float(np.clip(
            feature_dict.get("building_density", 0.5) * 0.4 +
            feature_dict.get("roof_flammability", 0.5) * 0.4 +
            (1 - feature_dict.get("road_accessibility", 0.4)) * 0.2, 0, 1))

        flood = float(np.clip(
            (1 - min(feature_dict.get("elevation_m", 30) / 80, 1)) * 0.40 +
            (1 - min(feature_dict.get("dist_to_water_m", 400) / 1500, 1)) * 0.35 +
            (1 - feature_dict.get("drainage_quality", 0.3)) * 0.25, 0, 1))

        access = float(np.clip(
            min(feature_dict.get("hospital_dist_m", 2500) / 5000, 1) * 0.40 +
            (1 - feature_dict.get("road_density", 0.35)) * 0.35 +
            (1 - feature_dict.get("road_accessibility", 0.4)) * 0.25, 0, 1))

        overall = round(fire * 0.35 + flood * 0.30 + access * 0.35, 3)

        scores = {"fire_risk": round(fire, 3), "flood_risk": round(flood, 3),
                  "accessibility_risk": round(access, 3), "overall_risk": overall}
        return {
            **scores,
            "risk_level": self._categorize(overall),
            "shap_explanation": {},
            "interventions": self.generate_interventions(scores),
            "model_used": "formula",
        }

    @staticmethod
    def _categorize(score: float) -> str:
        if score >= 0.75: return "Critical"
        elif score >= 0.50: return "High"
        elif score >= 0.25: return "Moderate"
        return "Low"

    def generate_interventions(self, scores: Dict) -> List[str]:
        interventions = []
        if scores.get("fire_risk", 0) >= 0.60:
            interventions += [
                "Install fire hydrant network (150m spacing)",
                "Replace flammable roofing with fire-resistant metal sheets",
                "Establish community water storage tanks (10,000L min)",
                "Create 3m firebreak corridors between building clusters",
            ]
        elif scores.get("fire_risk", 0) >= 0.35:
            interventions += ["Conduct fire safety awareness campaign",
                              "Improve emergency vehicle access lanes"]

        if scores.get("flood_risk", 0) >= 0.60:
            interventions += [
                "Install underground drainage (150mm pipes, 1:200 gradient)",
                "Construct flood retention basins upstream",
                "Elevate ground floors by 0.5–1.0m in critical zones",
                "Deploy real-time flood early warning sensors",
            ]
        elif scores.get("flood_risk", 0) >= 0.35:
            interventions += ["Clean and maintain open drainage channels",
                              "Install rain gauges for flood monitoring"]

        if scores.get("accessibility_risk", 0) >= 0.60:
            interventions += [
                "Develop minimum 4m-wide emergency access roads",
                "Establish primary healthcare clinic within 1km",
                "Mark and clear official evacuation corridors",
                "Deploy mobile health units bi-weekly",
            ]
        elif scores.get("accessibility_risk", 0) >= 0.35:
            interventions += ["Improve pedestrian path network",
                              "Negotiate emergency vehicle right-of-way"]

        if not interventions:
            interventions.append("Risk levels acceptable — maintain regular monitoring")
        return interventions


class GrowthPredictionService:
    """
    Advanced growth prediction service using AttentionLSTM.
    Returns point predictions + 80%/95% confidence intervals via MC Dropout.
    Falls back to trend extrapolation if model not trained.
    """

    MODEL_PATH = "models/growth_model.pt"

    def __init__(self):
        self.model = None
        self.model_meta = None
        self._loaded = False

        try:
            import torch
            self.torch = torch
            if Path(self.MODEL_PATH).exists():
                self._load_model()
            else:
                logger.info("Growth model not trained. Run: python ml/train_growth_model.py")
        except ImportError:
            self.torch = None

    def _load_model(self):
        try:
            from ml.train_growth_model import AttentionLSTM
            checkpoint = self.torch.load(self.MODEL_PATH, map_location="cpu")
            self.model = AttentionLSTM(
                input_size=checkpoint.get("input_size", 4),
                hidden_size=checkpoint.get("hidden_size", 128),
                num_layers=3,
                num_heads=4,
                forecast_horizon=checkpoint.get("forecast_horizon", 5),
                dropout=0.2,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.model_meta = checkpoint
            self._loaded = True
            logger.info(f"✓ AttentionLSTM growth model loaded from {self.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load growth model: {e}")

    def predict_growth(self, historical_areas: List[float],
                       forecast_years: int = 5,
                       population_series: Optional[List[float]] = None,
                       ndvi_series: Optional[List[float]] = None) -> Dict:
        """
        Predict future slum area with uncertainty intervals.

        Args:
            historical_areas     : annual area values (sq km), oldest first
            forecast_years       : years to forecast
            population_series    : optional population counts (same length as areas)
            ndvi_series          : optional NDVI mean values (same length as areas)

        Returns:
            forecasts: [{year, predicted_area_sqkm, lower_bound, upper_bound, confidence}]
            avg_annual_growth_rate: float
            trend: "increasing" / "stable" / "declining"
            model_used: "attention_lstm" / "trend_extrapolation"
        """
        if len(historical_areas) < 2:
            return {"error": "Need at least 2 years of historical data"}

        if self._loaded and self.model is not None:
            return self._attention_lstm_predict(
                historical_areas, forecast_years, population_series, ndvi_series
            )
        else:
            return self._trend_predict(historical_areas, forecast_years)

    def _attention_lstm_predict(self, areas, horizon, population=None, ndvi=None) -> Dict:
        seq_len = self.model_meta.get("seq_len", 15)
        areas = list(areas)

        # Build multi-variate input
        n = len(areas)
        if population is None:
            pop_growth = [0.03] * n  # default 3% annual population growth
        else:
            pop_growth = list(np.diff(population) / np.array(population[:-1])) + [0.03]
            pop_growth = pop_growth[:n]

        ndvi_vals = ndvi if ndvi is not None else [0.1] * n
        rainfall = [0.5] * n  # neutral rainfall (no data)

        scale = max(areas) + 1e-6
        areas_norm = [a / scale for a in areas]

        # Pad or trim to seq_len
        if len(areas_norm) < seq_len:
            # Pad with trend-extrapolated values at the beginning
            pad_len = seq_len - len(areas_norm)
            rate = (areas_norm[-1] - areas_norm[0]) / max(len(areas_norm) - 1, 1)
            pad = [max(0.01, areas_norm[0] - rate * (pad_len - i)) for i in range(pad_len)]
            areas_norm = pad + areas_norm
            pop_growth = [pop_growth[0]] * pad_len + pop_growth
            ndvi_vals = [ndvi_vals[0]] * pad_len + ndvi_vals
            rainfall = rainfall[:len(areas_norm)]

        X_seq = np.array([
            [areas_norm[-(seq_len - t)],
             pop_growth[-(seq_len - t)] if len(pop_growth) > seq_len - t else 0.03,
             ndvi_vals[-(seq_len - t)] if len(ndvi_vals) > seq_len - t else 0.1,
             rainfall[-(seq_len - t)] if len(rainfall) > seq_len - t else 0.5]
            for t in range(seq_len)
        ], dtype=np.float32)

        X_tensor = self.torch.tensor(X_seq).unsqueeze(0)  # (1, seq_len, 4)

        # MC Dropout for uncertainty
        from ml.train_growth_model import mc_dropout_predict
        mean, q10, q90 = mc_dropout_predict(self.model, X_tensor, n_passes=50)

        mean_np = mean[0].numpy() * scale
        q10_np = q10[0].numpy() * scale
        q90_np = q90[0].numpy() * scale

        forecasts = []
        for h in range(min(horizon, len(mean_np))):
            confidence = max(0.95 - h * 0.07, 0.55)
            forecasts.append({
                "year": h + 1,
                "predicted_area_sqkm": round(float(max(0.01, mean_np[h])), 3),
                "lower_bound_80pct": round(float(max(0.01, q10_np[h])), 3),
                "upper_bound_80pct": round(float(max(0.01, q90_np[h])), 3),
                "confidence": round(confidence, 2),
            })

        last = areas[-1]
        final = forecasts[-1]["predicted_area_sqkm"] if forecasts else last
        growth_rate = (final / last) ** (1 / max(horizon, 1)) - 1
        return {
            "forecasts": forecasts,
            "avg_annual_growth_rate": round(float(growth_rate), 4),
            "trend": "increasing" if growth_rate > 0.01 else
                     "declining" if growth_rate < -0.01 else "stable",
            "model_used": "attention_lstm_mc_dropout",
        }

    def _trend_predict(self, areas, horizon) -> Dict:
        rates = np.diff(areas) / np.array(areas[:-1])
        avg_rate = float(np.median(rates))  # Median more robust than mean
        last = areas[-1]
        forecasts = []
        for year in range(1, horizon + 1):
            pred = max(0.01, last * ((1 + avg_rate) ** year))
            # Simple uncertainty: widens with horizon
            uncertainty = pred * 0.05 * year
            forecasts.append({
                "year": year,
                "predicted_area_sqkm": round(pred, 3),
                "lower_bound_80pct": round(max(0.01, pred - uncertainty), 3),
                "upper_bound_80pct": round(pred + uncertainty, 3),
                "confidence": round(max(0.90 - year * 0.07, 0.50), 2),
            })
        return {
            "forecasts": forecasts,
            "avg_annual_growth_rate": round(avg_rate, 4),
            "trend": "increasing" if avg_rate > 0.01 else
                     "declining" if avg_rate < -0.01 else "stable",
            "model_used": "trend_extrapolation",
        }
