from fastapi import APIRouter, Query, File, UploadFile, HTTPException
from typing import Optional
from db import get_session
from models.database import Region, Analysis
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/data/sources")
async def get_data_sources():
    """Get available data sources"""
    return {
        "sources": [
            {"name": "Sentinel-2", "type": "Satellite Imagery",
             "resolution": "10m", "update_frequency": "5 days"},
            {"name": "Google Earth Engine", "type": "Processed Satellite Data",
             "resolution": "Variable", "update_frequency": "Daily"},
            {"name": "OpenStreetMap", "type": "Road Networks & Infrastructure",
             "resolution": "Varies", "update_frequency": "Real-time"},
            {"name": "Elevation Data (DEM)", "type": "Terrain Information",
             "resolution": "30m", "update_frequency": "Static"},
            {"name": "Population Density", "type": "Demographic Data",
             "resolution": "100m", "update_frequency": "Annual"},
        ]
    }


@router.get("/data/regions")
async def list_regions(limit: int = Query(50, ge=1, le=500)):
    """
    List analyzed regions — queries the real database.
    Falls back to demo data if DB is not connected.
    """
    try:
        with get_session() as session:
            regions = session.query(Region).limit(limit).all()
            return {
                "total_regions": session.query(Region).count(),
                "regions": [
                    {
                        "id": r.id,
                        "name": r.name,
                        "city": r.city,
                        "country": r.country,
                        "latitude": r.latitude,
                        "longitude": r.longitude,
                        "area_sqkm": r.area_sqkm,
                        "population": r.population,
                        "slum_percentage": r.slum_percentage,
                        "last_analysis": r.last_analysis.isoformat() if r.last_analysis else None,
                    }
                    for r in regions
                ],
            }
    except Exception as e:
        logger.warning(f"DB unavailable, returning demo data: {e}")
        # Fallback demo data so frontend always works during development
        return {
            "total_regions": 8,
            "regions": [
                {"id": "demo-1", "name": "Dharavi", "city": "Mumbai",
                 "country": "India", "latitude": 19.0396, "longitude": 72.8556,
                 "area_sqkm": 2.39, "population": 1000000, "slum_percentage": 92.0,
                 "last_analysis": None},
                {"id": "demo-2", "name": "Kibera", "city": "Nairobi",
                 "country": "Kenya", "latitude": -1.3133, "longitude": 36.7833,
                 "area_sqkm": 2.5, "population": 250000, "slum_percentage": 85.0,
                 "last_analysis": None},
                {"id": "demo-3", "name": "Rocinha", "city": "Rio de Janeiro",
                 "country": "Brazil", "latitude": -22.9868, "longitude": -43.2497,
                 "area_sqkm": 0.86, "population": 100000, "slum_percentage": 95.0,
                 "last_analysis": None},
                {"id": "demo-4", "name": "Orangi Town", "city": "Karachi",
                 "country": "Pakistan", "latitude": 24.9614, "longitude": 66.9943,
                 "area_sqkm": 45.0, "population": 2400000, "slum_percentage": 65.0,
                 "last_analysis": None},
            ],
        }


@router.get("/data/region/{region_id}")
async def get_region_data(region_id: str):
    """Get detailed data for a specific region."""
    try:
        with get_session() as session:
            region = session.query(Region).filter(Region.id == region_id).first()
            if not region:
                raise HTTPException(status_code=404, detail=f"Region '{region_id}' not found")

            # Get analysis history for this region
            analyses = (
                session.query(Analysis)
                .filter(Analysis.region_id == region_id)
                .order_by(Analysis.created_at.desc())
                .limit(10)
                .all()
            )

            return {
                "region_id": region.id,
                "metadata": {
                    "name": region.name,
                    "city": region.city,
                    "country": region.country,
                    "latitude": region.latitude,
                    "longitude": region.longitude,
                    "area_sqkm": region.area_sqkm,
                    "population": region.population,
                    "slum_percentage": region.slum_percentage,
                },
                "analysis_history": [
                    {
                        "id": a.id,
                        "type": a.analysis_type,
                        "slum_area_km2": a.slum_area_km2,
                        "slum_percentage": a.slum_percentage,
                        "infrastructure_score": a.infrastructure_score,
                        "date": a.created_at.isoformat(),
                    }
                    for a in analyses
                ],
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"DB unavailable for region {region_id}: {e}")
        raise HTTPException(status_code=503, detail="Database not connected. Run: docker compose up postgres")


@router.post("/data/upload")
async def upload_region_data(
    region_id: str = Query(...),
    file: UploadFile = File(...),
):
    """Upload new satellite data for a region."""
    contents = await file.read()
    return {
        "region_id": region_id,
        "filename": file.filename,
        "size_bytes": len(contents),
        "upload_status": "success",
        "processing_status": "queued",
        "message": "File received. Connect Celery worker to process.",
    }


@router.get("/data/export/{region_id}")
async def export_region_data(
    region_id: str,
    format: str = Query("geojson", pattern="^(geojson|shapefile|tif|csv)$"),
):
    """Export region data in various formats."""
    return {
        "region_id": region_id,
        "format": format,
        "download_url": f"https://storage.urbansense.io/exports/{region_id}_{format}.zip",
        "expires_in_hours": 48,
        "size_mb": 125.5,
        "note": "Storage integration not yet configured.",
    }
