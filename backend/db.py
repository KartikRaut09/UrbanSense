"""Database initialization and session management"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database import Base
from config import settings

# Create engine once at module level
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def get_session():
    """Get database session (use as context manager or close manually)"""
    return SessionLocal()


if __name__ == "__main__":
    init_db()
