from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas import AnalyticsResponse
from app.services.video_processor import VideoProcessor

app = FastAPI(
    title="Theft Detection API",
    version="1.0.0",
    description="AI-powered video analytics for theft detection using YOLOv8 + behavior classification.",
)

# ---------------------------------------------------------------------------
# CORS – allow the standalone frontend (any origin in dev; restrict in prod)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = VideoProcessor(settings=settings)


@app.get("/health", tags=["System"])
def health() -> dict:
    """Health-check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyticsResponse, tags=["Analytics"])
async def analyze_video(
    file: UploadFile = File(...),
    save_video: bool = settings.save_video_default,
) -> AnalyticsResponse:
    """Upload a video and receive theft-detection analytics."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    extension = Path(file.filename).suffix.lower()
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if extension and extension not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {extension}")

    upload_dir = settings.resolved_outputs_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / f"{uuid4().hex}{extension or '.mp4'}"

    with input_path.open("wb") as output_file:
        output_file.write(await file.read())

    try:
        response = processor.process_video(video_path=input_path, save_video=save_video)
        return response
    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)


@app.get("/processed/{filename}", tags=["Media"])
def get_processed_video(filename: str) -> FileResponse:
    """Stream a processed video file by filename."""
    path = settings.resolved_outputs_dir / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Processed video not found")
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        media_type = "video/mp4"
    elif suffix == ".webm":
        media_type = "video/webm"
    else:
        media_type = "video/x-msvideo"
    return FileResponse(path, media_type=media_type, filename=path.name)
