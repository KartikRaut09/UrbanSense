from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from services.risk_service import RiskAnalysisService, GrowthPredictionService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Instantiate services once at module level
risk_service = RiskAnalysisService()
growth_service = GrowthPredictionService()


class RiskAssessmentRequest(BaseModel):
    latitude: float
    longitude: float
    area_name: str
    # Risk input features — all optional with sensible defaults
    building_density: float = 0.7
    roof_flammability: float = 0.6
    road_accessibility: float = 0.3
    elevation_m: float = 20.0
    dist_to_water_m: float = 300.0
    drainage_quality: float = 0.2
    hospital_dist_m: float = 3000.0
    road_density: float = 0.3
    population_density: float = 50000.0
    ndvi: float = 0.05


class GrowthRequest(BaseModel):
    region_id: str
    historical_areas: List[float]  # Annual area in sq km, oldest first
    forecast_years: int = 5


@router.post("/risk/assess")
async def assess_risks(request: RiskAssessmentRequest):
    """
    Assess fire, flood, and accessibility risks for a location.
    Uses XGBoost model if trained, otherwise formula-based calculation.
    """
    try:
        feature_dict = {
            "building_density": request.building_density,
            "roof_flammability": request.roof_flammability,
            "road_accessibility": request.road_accessibility,
            "elevation_m": request.elevation_m,
            "dist_to_water_m": request.dist_to_water_m,
            "drainage_quality": request.drainage_quality,
            "hospital_dist_m": request.hospital_dist_m,
            "road_density": request.road_density,
            "population_density": request.population_density,
            "ndvi": request.ndvi,
        }

        result = risk_service.assess_risk(feature_dict)

        return {
            "location": request.area_name,
            "coordinates": {"latitude": request.latitude, "longitude": request.longitude},
            "risk_scores": {
                "fire_risk": result["fire_risk"],
                "flood_risk": result["flood_risk"],
                "accessibility_risk": result["accessibility_risk"],
                "overall_risk": result["overall_risk"],
            },
            "risk_level": result["risk_level"],
            "model_used": result.get("model_used", "formula"),
            "recommendations": result.get("interventions", []),
        }
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/risk/growth")
async def predict_growth(request: GrowthRequest):
    """
    Predict slum growth for the next N years.
    Uses LSTM model if trained, otherwise trend extrapolation.
    """
    try:
        result = growth_service.predict_growth(
            historical_areas=request.historical_areas,
            forecast_years=request.forecast_years,
        )
        return {"region_id": request.region_id, **result}
    except Exception as e:
        logger.error(f"Growth prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/heatmap/{region_id}")
async def get_risk_heatmap(region_id: str):
    """Get risk heatmap for a region."""
    return {
        "region_id": region_id,
        "heatmap_url": f"s3://urbansense-data/heatmaps/{region_id}.tif",
        "generated_at": "2024-01-10T12:00:00Z",
        "note": "S3 integration not yet configured.",
    }


@router.get("/risk/historical/{area_id}")
async def get_historical_risks(area_id: str, months: int = Query(12, ge=1, le=60)):
    """Get historical risk trends."""
    return {
        "area_id": area_id,
        "time_period_months": months,
        "risk_trend": "increasing",
        "trend_data": [
            {"month": i, "fire_risk": round(0.5 + i * 0.02, 3),
             "flood_risk": round(0.3 + i * 0.01, 3)}
            for i in range(months)
        ],
    }
