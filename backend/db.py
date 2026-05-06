"""Database initialization and session management"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from models.database import Base
from config import settings
import logging

logger = logging.getLogger(__name__)

# SQLite needs special connect_args; PostgreSQL/others use connection pooling
_is_sqlite = settings.database_url.startswith("sqlite")

if _is_sqlite:
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
    )
else:
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,   # verify connections before use
        pool_recycle=300,     # recycle connections every 5 minutes
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all database tables. Run once on startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Database tables created successfully")


@contextmanager
def get_session():
    """
    Provide a transactional database session as a context manager.

    Usage:
        with get_session() as session:
            regions = session.query(Region).all()

    Automatically commits on success, rolls back on exception, always closes.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
