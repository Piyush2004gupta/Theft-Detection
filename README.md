# Theft Detection – AI Video Analytics

An AI-powered system for detecting theft behaviour in CCTV footage. It uses **YOLOv8** for person and object tracking and a custom **behaviour classification model** to flag suspicious activity.

---

## Project Structure

```
theft-detection/
├── backend/               ← FastAPI backend (Python)
│   ├── app/
│   │   ├── config.py      ← Settings & environment variables
│   │   ├── main.py        ← FastAPI app + API routes
│   │   ├── schemas.py     ← Pydantic request/response models
│   │   ├── services/
│   │   │   ├── analytics_service.py   ← Track aggregation & theft logic
│   │   │   ├── classifier_service.py  ← Behaviour classification (PyTorch)
│   │   │   ├── detector_service.py    ← YOLOv8 detection & tracking
│   │   │   └── video_processor.py     ← End-to-end video pipeline
│   │   └── utils/
│   │       └── time_utils.py
│   ├── models/            ← Model weights (not tracked by git)
│   ├── outputs/           ← Processed video output (not tracked by git)
│   ├── scripts/
│   │   └── run_local.py   ← CLI runner (no server needed)
│   ├── .env.example       ← Environment variable template
│   └── requirements.txt
│
└── frontend/              ← Standalone HTML/CSS/JS frontend
    ├── index.html
    ├── styles.css
    └── app.js
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip / virtualenv

### 1 – Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment variables
copy .env.example .env
```

Edit `.env` to point to your model weights and configure CORS origins.

### 2 – Add Model Weights

Place the following files inside `backend/models/`:

| File | Description |
|------|-------------|
| `theft_modelss.keras` | Behaviour classification model |
| `yolov8n.pt` | YOLOv8 nano weights (auto-downloaded by Ultralytics if missing) |

### 3 – Run the Backend

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/docs**

### 4 – Run the Frontend

Open `frontend/index.html` directly in a browser, **or** serve it with any static server:

```bash
# Using Python
python -m http.server 5500 --directory frontend

# Using Node.js (npx)
npx serve frontend
```

> The frontend expects the backend at `http://localhost:8000` by default.  
> To change this, set `window.THEFT_API_BASE` before `app.js` loads (see `index.html` comments).

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/analyze` | Upload + analyse a video |
| `GET` | `/processed/{filename}` | Stream processed video |

### POST `/analyze`

**Form data:**

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Video file (mp4, avi, mov, mkv, webm) |
| `save_video` | bool | Whether to save annotated video (default: `true`) |

**Response:** `AnalyticsResponse` JSON

```json
{
  "people": [
    { "id": 1, "in_time": "00:00:02", "out_time": "00:00:15", "time_spent_seconds": 13, "activity": "Normal" }
  ],
  "total_people": 1,
  "total_cups_detected": 2,
  "suspicious_ids": [],
  "overall_status": "Normal",
  "processed_video_path": "processed_20260408_113000_input.mp4"
}
```

---

## CLI Usage (no server)

```bash
cd backend
python -m scripts.run_local path/to/video.mp4
python -m scripts.run_local path/to/video.mp4 --no-save-video
```

---

## Environment Variables

See `backend/.env.example` for the full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `THEFT_YOLO_MODEL` | `yolov8n.pt` | YOLO model path/name |
| `THEFT_BEHAVIOR_MODEL` | `models/theft_modelss.keras` | Classifier weights |
| `THEFT_OUTPUTS_DIR` | `outputs` | Where to save videos |
| `THEFT_CORS_ORIGINS` | `*` | Allowed frontend origins |
| `THEFT_YOLO_CONF` | `0.25` | Detection confidence |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Detection | YOLOv8 (Ultralytics) + ByteTrack |
| Classification | PyTorch / Keras |
| Video | OpenCV |
| Frontend | Vanilla HTML / CSS / JavaScript |
