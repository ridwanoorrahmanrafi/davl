"""
FastAPI Main Application Entry Point for Personalized CVD Browser Assistant.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from .api.routes import router
from .config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="API for Multi-Type Color Vision Deficiency Recoloring and Personalization.",
)

# Enable CORS for Chrome Extension requests and local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Mount archive static assets for ground-truth comparison
archive_path = Path("archive")
if archive_path.is_dir():
    app.mount("/archive", StaticFiles(directory="archive"), name="archive")


@app.get("/test")
async def get_test_page():
    return FileResponse("extension/test_page.html")


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "test_page": "/test",
        "supported_conditions": ["Protanopia (0)", "Deuteranopia (1)", "Tritanopia (2)"],
    }
