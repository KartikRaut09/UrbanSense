from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
from config import settings
from routes import analysis, segmentation, risk, insights, data

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UrbanSense - Intelligent Slum Mapping API",
    description="AI-powered geospatial urban risk intelligence system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Trusted Host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "testserver", "*.vercel.app", "*.aws.com"]
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(segmentation.router, prefix="/api/v1", tags=["Segmentation"])
app.include_router(risk.router, prefix="/api/v1", tags=["Risk Analysis"])
app.include_router(insights.router, prefix="/api/v1", tags=["Insights"])
app.include_router(data.router, prefix="/api/v1", tags=["Data"])

@app.get("/")
async def read_root():
    return {
        "service": "UrbanSense",
        "version": "1.0.0",
        "status": "online",
        "environment": settings.environment
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "debug": settings.debug
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
