from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter()

class RiskAssessmentRequest(BaseModel):
    latitude: float
    longitude: float
    area_name: str

class RiskScores(BaseModel):
    fire_risk: float
    flood_risk: float
    accessibility_risk: float
    overall_risk: float

class RiskResponse(BaseModel):
    location: str
    risk_scores: RiskScores
    risk_factors: dict
    recommendations: List[str]

@router.post("/risk/assess", response_model=RiskResponse)
async def assess_risks(request: RiskAssessmentRequest):
    """
    Assess multiple risk factors for a location
    """
    try:
        risk_scores = RiskScores(
            fire_risk=0.75,
            flood_risk=0.45,
            accessibility_risk=0.65,
            overall_risk=0.62
        )
        
        recommendations = [
            "Install fire hydrants in high-density areas",
            "Improve drainage systems for flood prevention",
            "Develop emergency access roads",
            "Enhance building materials resistance"
        ]
        
        return RiskResponse(
            location=request.area_name,
            risk_scores=risk_scores,
            risk_factors={
                "building_density": 850,
                "roof_flammability": 0.8,
                "elevation": 25,
                "proximity_to_water": 150
            },
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/heatmap/{region_id}")
async def get_risk_heatmap(region_id: str):
    """
    Get risk heatmap for a region
    """
    return {
        "region_id": region_id,
        "heatmap_url": "s3://urbansense-data/heatmaps/region_" + region_id + ".tif",
        "generated_at": "2024-01-10T12:00:00Z"
    }

@router.get("/risk/historical/{area_id}")
async def get_historical_risks(area_id: str, months: int = Query(12, ge=1, le=60)):
    """
    Get historical risk trends
    """
    return {
        "area_id": area_id,
        "time_period_months": months,
        "risk_trend": "increasing",
        "trend_data": [
            {"month": i, "fire_risk": 0.5 + i*0.02, "flood_risk": 0.3 + i*0.01}
            for i in range(months)
        ]
    }
