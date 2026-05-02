"""Database models and schemas"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import uuid


class Base(DeclarativeBase):
    pass

class Region(Base):
    __tablename__ = "regions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    country = Column(String(255), nullable=False)
    north_latitude = Column(Float, nullable=False)
    south_latitude = Column(Float, nullable=False)
    east_longitude = Column(Float, nullable=False)
    west_longitude = Column(Float, nullable=False)
    area_km2 = Column(Float)
    population_estimate = Column(Integer)
    last_analysis = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String(36), nullable=False)
    analysis_type = Column(String(50), nullable=False)
    slum_area_km2 = Column(Float)
    slum_percentage = Column(Float)
    building_count = Column(Integer)
    infrastructure_score = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String(36), nullable=False)
    fire_risk = Column(Float)
    flood_risk = Column(Float)
    accessibility_risk = Column(Float)
    overall_risk = Column(Float)
    risk_factors = Column(JSON)
    recommendations = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Insight(Base):
    __tablename__ = "insights"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String(36), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    insights_data = Column(JSON)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class GrowthPrediction(Base):
    __tablename__ = "growth_predictions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    region_id = Column(String(36), nullable=False)
    prediction_year = Column(Integer)
    predicted_area_km2 = Column(Float)
    predicted_population = Column(Integer)
    confidence_interval = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
