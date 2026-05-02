from fastapi import APIRouter, Query, File, UploadFile
from typing import Optional
import json

router = APIRouter()

@router.get("/data/sources")
async def get_data_sources():
    """
    Get available data sources
    """
    return {
        "sources": [
            {
                "name": "Sentinel-2",
                "type": "Satellite Imagery",
                "resolution": "10m",
                "update_frequency": "5 days"
            },
            {
                "name": "Google Earth Engine",
                "type": "Processed Satellite Data",
                "resolution": "Variable",
                "update_frequency": "Daily"
            },
            {
                "name": "OpenStreetMap",
                "type": "Road Networks & Infrastructure",
                "resolution": "Varies",
                "update_frequency": "Real-time"
            },
            {
                "name": "Elevation Data (DEM)",
                "type": "Terrain Information",
                "resolution": "30m",
                "update_frequency": "Static"
            },
            {
                "name": "Population Density",
                "type": "Demographic Data",
                "resolution": "100m",
                "update_frequency": "Annual"
            }
        ]
    }

@router.get("/data/regions")
async def list_regions(limit: int = Query(50, ge=1, le=500)):
    """
    List analyzed regions
    """
    return {
        "total_regions": 156,
        "regions": [
            {
                "id": f"REGION_{i:03d}",
                "name": f"Urban Area {i}",
                "country": "Sample Country",
                "area_km2": 2.5 + i*0.5,
                "last_analysis": "2024-01-10"
            }
            for i in range(1, min(limit+1, 11))
        ]
    }

@router.get("/data/region/{region_id}")
async def get_region_data(region_id: str):
    """
    Get detailed data for a region
    """
    return {
        "region_id": region_id,
        "metadata": {
            "name": "Sample Urban Region",
            "country": "Sample Country",
            "coordinates": {
                "north": 28.7041,
                "south": 28.5333,
                "east": 77.2290,
                "west": 77.0369
            },
            "area_km2": 250.5,
            "population": 2500000
        },
        "data_layers": [
            "satellite_rgb",
            "infrared_nir",
            "elevation_dem",
            "road_network",
            "building_footprints",
            "vegetation_ndvi",
            "population_density"
        ],
        "analysis_history": [
            {"date": "2024-01-10", "type": "segmentation"},
            {"date": "2024-01-09", "type": "risk_assessment"},
            {"date": "2024-01-08", "type": "full_analysis"}
        ]
    }

@router.post("/data/upload")
async def upload_region_data(
    region_id: str = Query(...),
    file: UploadFile = File(...)
):
    """
    Upload new data for a region
    """
    contents = await file.read()
    return {
        "region_id": region_id,
        "filename": file.filename,
        "size_bytes": len(contents),
        "upload_status": "success",
        "processing_status": "queued"
    }

@router.get("/data/export/{region_id}")
async def export_region_data(
    region_id: str,
    format: str = Query("geojson", pattern="^(geojson|shapefile|tif|csv)$")
):
    """
    Export region data in various formats
    """
    return {
        "region_id": region_id,
        "format": format,
        "download_url": f"https://storage.urbansense.io/exports/{region_id}_{format}.zip",
        "expires_in_hours": 48,
        "size_mb": 125.5
    }
