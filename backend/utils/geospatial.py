"""Utility functions for backend"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def validate_geospatial_bounds(north: float, south: float, east: float, west: float) -> bool:
    """Validate geospatial boundaries"""
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        return False
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        return False
    if south >= north or west >= east:
        return False
    return True

def calculate_area_from_bounds(north: float, south: float, east: float, west: float) -> float:
    """Rough calculation of area in km²"""
    lat_diff = abs(north - south)
    lon_diff = abs(east - west)
    
    # Approximate at equator
    km_per_degree_lat = 111
    km_per_degree_lon = 111 * np.cos(np.radians((north + south) / 2))
    
    return lat_diff * km_per_degree_lat * lon_diff * km_per_degree_lon

def compress_numpy_array(array: np.ndarray) -> bytes:
    """Compress numpy array for storage"""
    import zlib
    return zlib.compress(array.tobytes())

def decompress_numpy_array(data: bytes, shape: Tuple, dtype) -> np.ndarray:
    """Decompress numpy array from storage"""
    import zlib
    decompressed = zlib.decompress(data)
    return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
