import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskAnalysisService:
    """Service for comprehensive risk assessment"""
    
    def __init__(self):
        self.risk_weights = {
            "fire_risk": 0.35,
            "flood_risk": 0.30,
            "accessibility_risk": 0.25,
            "infrastructure_risk": 0.10
        }
    
    def calculate_fire_risk(self, building_density: float, roof_flammability: float, 
                           road_accessibility: float) -> float:
        """Calculate fire risk based on multiple factors"""
        # Normalized score 0-1
        fire_risk = (building_density * 0.4) + (roof_flammability * 0.4) + ((1 - road_accessibility) * 0.2)
        return np.clip(fire_risk, 0, 1)
    
    def calculate_flood_risk(self, elevation: float, proximity_to_water: float, 
                            drainage_quality: float) -> float:
        """Calculate flood risk"""
        # Low elevation and proximity to water increase risk
        flood_risk = ((100 - elevation) / 100 * 0.4) + ((100 - proximity_to_water) / 100 * 0.4) + ((1 - drainage_quality) * 0.2)
        return np.clip(flood_risk, 0, 1)
    
    def calculate_accessibility_risk(self, hospital_distance: float, road_density: float,
                                    emergency_access: float) -> float:
        """Calculate accessibility/emergency response risk"""
        # Normalized distances
        normalized_hospital_dist = min(hospital_distance / 5000, 1)  # 5km reference
        accessibility_risk = (normalized_hospital_dist * 0.4) + ((1 - road_density) * 0.35) + ((1 - emergency_access) * 0.25)
        return np.clip(accessibility_risk, 0, 1)
    
    def assess_overall_risk(self, fire_risk: float, flood_risk: float, 
                           accessibility_risk: float) -> Dict:
        """Calculate overall risk score"""
        overall_risk = (fire_risk * 0.35) + (flood_risk * 0.30) + (accessibility_risk * 0.35)
        
        return {
            "fire_risk": np.round(fire_risk, 3),
            "flood_risk": np.round(flood_risk, 3),
            "accessibility_risk": np.round(accessibility_risk, 3),
            "overall_risk": np.round(overall_risk, 3),
            "risk_level": self._categorize_risk(overall_risk)
        }
    
    @staticmethod
    def _categorize_risk(score: float) -> str:
        """Categorize risk level"""
        if score >= 0.75:
            return "Critical"
        elif score >= 0.5:
            return "High"
        elif score >= 0.25:
            return "Moderate"
        else:
            return "Low"
    
    def generate_risk_interventions(self, risk_scores: Dict) -> List[str]:
        """Generate specific interventions based on risk profile"""
        interventions = []
        
        if risk_scores["fire_risk"] >= 0.6:
            interventions.append("Install fire hydrant network")
            interventions.append("Establish water storage infrastructure")
            interventions.append("Implement fire-resistant building standards")
        
        if risk_scores["flood_risk"] >= 0.6:
            interventions.append("Upgrade drainage systems")
            interventions.append("Build elevated infrastructure")
            interventions.append("Implement wetland restoration")
        
        if risk_scores["accessibility_risk"] >= 0.6:
            interventions.append("Develop emergency access roads")
            interventions.append("Establish medical clinics")
            interventions.append("Create evacuation zones")
        
        return interventions

class GrowthPredictionService:
    """Service for slum growth prediction using time-series data"""
    
    def __init__(self):
        self.model_type = "LSTM"
    
    def predict_growth(self, historical_areas: List[float], forecast_years: int = 5) -> Dict:
        """Predict slum area growth"""
        # Simplified prediction using trend analysis
        if len(historical_areas) < 2:
            return {"error": "Insufficient historical data"}
        
        # Calculate growth rate
        growth_rates = np.diff(historical_areas) / historical_areas[:-1]
        avg_growth_rate = np.mean(growth_rates)
        
        # Project forward
        last_area = historical_areas[-1]
        predictions = []
        
        for year in range(1, forecast_years + 1):
            projected_area = last_area * ((1 + avg_growth_rate) ** year)
            predictions.append({
                "year": year,
                "predicted_area": np.round(projected_area, 2),
                "confidence": max(0.99 - (year * 0.05), 0.7)
            })
        
        return {
            "forecasts": predictions,
            "growth_rate": np.round(avg_growth_rate, 3),
            "trend": "increasing" if avg_growth_rate > 0 else "decreasing"
        }
