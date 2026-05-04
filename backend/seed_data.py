"""
Seed the database with real-world slum regions.
Run this ONCE after creating tables:
  python seed_data.py
"""

from db import get_session
from models.database import Region
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REGIONS = [
    {
        "name": "Dharavi",
        "city": "Mumbai",
        "country": "India",
        "latitude": 19.0396,
        "longitude": 72.8556,
        "area_sqkm": 2.39,
        "population": 1000000,
        "slum_percentage": 92.0,
    },
    {
        "name": "Kibera",
        "city": "Nairobi",
        "country": "Kenya",
        "latitude": -1.3133,
        "longitude": 36.7833,
        "area_sqkm": 2.5,
        "population": 250000,
        "slum_percentage": 85.0,
    },
    {
        "name": "Mathare",
        "city": "Nairobi",
        "country": "Kenya",
        "latitude": -1.2592,
        "longitude": 36.8539,
        "area_sqkm": 1.1,
        "population": 180000,
        "slum_percentage": 78.0,
    },
    {
        "name": "Korogocho",
        "city": "Nairobi",
        "country": "Kenya",
        "latitude": -1.2425,
        "longitude": 36.8847,
        "area_sqkm": 0.75,
        "population": 150000,
        "slum_percentage": 88.0,
    },
    {
        "name": "Tondo",
        "city": "Manila",
        "country": "Philippines",
        "latitude": 14.6194,
        "longitude": 120.9720,
        "area_sqkm": 8.67,
        "population": 630000,
        "slum_percentage": 70.0,
    },
    {
        "name": "Ciudad Nezahualcoyotl",
        "city": "Mexico City",
        "country": "Mexico",
        "latitude": 19.4022,
        "longitude": -98.9996,
        "area_sqkm": 63.4,
        "population": 1109363,
        "slum_percentage": 55.0,
    },
    {
        "name": "Orangi Town",
        "city": "Karachi",
        "country": "Pakistan",
        "latitude": 24.9614,
        "longitude": 66.9943,
        "area_sqkm": 45.0,
        "population": 2400000,
        "slum_percentage": 65.0,
    },
    {
        "name": "Rocinha",
        "city": "Rio de Janeiro",
        "country": "Brazil",
        "latitude": -22.9868,
        "longitude": -43.2497,
        "area_sqkm": 0.86,
        "population": 100000,
        "slum_percentage": 95.0,
    },
]


def seed():
    with get_session() as session:
        existing = session.query(Region).count()
        if existing > 0:
            logger.info(f"Database already has {existing} regions. Skipping seed.")
            return

        for data in REGIONS:
            region = Region(**data)
            session.add(region)

        session.commit()
        logger.info(f"✓ Seeded {len(REGIONS)} regions successfully.")


if __name__ == "__main__":
    seed()
