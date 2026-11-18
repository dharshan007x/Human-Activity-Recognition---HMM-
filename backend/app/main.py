"""FastAPI application entrypoint for Human Activity Recognition Platform."""

import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
project_root = backend_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.app.config import TITLE, VERSION, DESCRIPTION, API_PREFIX
from backend.app.api.v1.project import router as project_router
from backend.app.api.v1.dataset import router as dataset_router
from backend.app.api.v1.model import router as model_router
from backend.app.api.v1.inference import router as inference_router
from backend.app.api.v1.recognize import router as recognize_router
from backend.app.api.v1.evaluation import router as evaluation_router
from backend.app.services.model_service import ModelService

app = FastAPI(
    title=TITLE,
    version=VERSION,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for Vite frontend (http://localhost:5173) and any other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    # Preload model artifacts on startup
    try:
        model_service = ModelService.get_instance()
        _ = model_service.hmm
        print("[Startup] HMM model and scaler loaded successfully.")
    except Exception as e:
        print(f"[Startup Warning] Could not preload model artifacts: {e}")


@app.get("/")
def root():
    return {
        "title": TITLE,
        "version": VERSION,
        "status": "online",
        "api_docs": "/docs",
        "endpoints_prefix": API_PREFIX,
    }


@app.get(f"{API_PREFIX}/health")
def healthcheck():
    model_loaded = False
    try:
        model_service = ModelService.get_instance()
        model_loaded = model_service.hmm is not None
    except Exception:
        model_loaded = False

    return {
        "status": "healthy",
        "version": VERSION,
        "model_loaded": model_loaded,
    }


# Mount API v1 Routers
app.include_router(project_router, prefix=API_PREFIX)
app.include_router(dataset_router, prefix=API_PREFIX)
app.include_router(model_router, prefix=API_PREFIX)
app.include_router(inference_router, prefix=API_PREFIX)
app.include_router(recognize_router, prefix=API_PREFIX)
app.include_router(evaluation_router, prefix=API_PREFIX)


# Structured error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
            "path": str(request.url.path),
        },
    )
