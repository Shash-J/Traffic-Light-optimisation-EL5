"""
FastAPI Main Application
Entry point for Smart Traffic RL backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes.simulation import router as simulation_router
from app.routes.metrics import router as metrics_router
from app.routes.advanced import router as advanced_router
from app.websocket import ws_router
import uvicorn


# Create FastAPI app
app = FastAPI(
    title="Smart Traffic RL System",
    description="Real-time traffic signal optimization using Reinforcement Learning and SUMO",
    version="2.0.0"
)

# Configure CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(simulation_router)
app.include_router(metrics_router)
app.include_router(advanced_router)  # Weather, Emergency, Evaluation endpoints
app.include_router(ws_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Smart Traffic RL System",
        "version": "2.0.0",
        "status": "running",
        "features": ["weather_awareness", "emergency_priority", "multi_junction_coordination"],
        "endpoints": {
            "simulation": "/api/simulation",
            "metrics": "/api/metrics",
            "advanced": "/api/advanced",
            "websocket": "/ws"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "smart-traffic-rl"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
