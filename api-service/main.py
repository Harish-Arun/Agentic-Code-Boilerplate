"""
NNP-AI API Service - FastAPI entry point.
Config-driven, pluggable architecture.
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from config import get_config
from adapters import get_database_adapter
from models import HealthResponse

from routers import documents, processing


# ============================================
# Lifespan - Startup/Shutdown
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - connect/disconnect resources."""
    config = get_config()
    
    # Initialize database
    db = get_database_adapter(config)
    await db.connect()
    app.state.db = db
    app.state.config = config
    
    print(f"🚀 API Service started [DB: {config.database.type}, LLM: {config.llm.provider}]")
    
    yield
    
    # Cleanup
    await db.disconnect()
    print("👋 API Service shutdown complete")


# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="NNP-AI API Service",
    description="Manual Payments AI Platform - Document Processing API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(processing.router, prefix="/process", tags=["Processing"])


# ============================================
# Health Endpoints
# ============================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="api-service"
    )


@app.get("/")
async def root():
    """Root endpoint with service info."""
    config = get_config()
    return {
        "service": "NNP-AI API Service",
        "version": "1.0.0",
        "database": config.database.type,
        "llm_provider": config.llm.provider,
        "features": config.features.model_dump()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
