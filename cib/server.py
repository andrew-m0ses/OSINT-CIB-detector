"""
FastAPI server for CIB detection.

Endpoints:
    POST /api/detect         - Upload file and run detection
    POST /api/detect/config  - Upload file with custom config
    GET  /api/result         - Get latest result
    GET  /api/graph          - Get graph data for visualization
    GET  /                   - Web UI
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .detector import DetectionConfig, DetectionResult, detect_file
from .synthetic import generate_dataset, write_csv

logger = logging.getLogger(__name__)

_latest_result: DetectionResult | None = None

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="CIB Detector", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        return html_path.read_text(encoding="utf-8")

    @app.post("/api/detect")
    async def detect_endpoint(
        file: UploadFile = File(...),
        window: float = Form(300),
        threshold: float = Form(0.15),
        min_posts: int = Form(5),
    ):
        global _latest_result
        # Save upload to temp file
        suffix = Path(file.filename or "data.csv").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        config = DetectionConfig(
            copost_window_seconds=window,
            min_posts=min_posts,
            edge_threshold=threshold,
        )

        try:
            result = detect_file(tmp_path, config)
            _latest_result = result
            return JSONResponse(result.to_dict())
        except Exception as e:
            logger.exception("Detection failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/result")
    async def get_result():
        if _latest_result is None:
            return JSONResponse({"error": "No analysis run yet"}, status_code=404)
        return JSONResponse(_latest_result.to_dict())

    @app.get("/api/graph")
    async def get_graph():
        if _latest_result is None:
            return JSONResponse({"error": "No analysis run yet"}, status_code=404)
        return JSONResponse(_latest_result.network.get_graph_data())

    @app.post("/api/generate")
    async def load_demo():
        global _latest_result
        # Load bundled IRA dataset
        demo_path = Path(__file__).parent.parent / "data" / "examples" / "ira_demo.json"
        if not demo_path.exists():
            return JSONResponse({"error": "Demo dataset not found. Expected at data/examples/ira_demo.json"}, status_code=404)

        config = DetectionConfig(
            copost_window_seconds=300,
            min_posts=10,
            edge_threshold=0.12,
        )

        try:
            result = detect_file(str(demo_path), config)
            _latest_result = result

            # Load ground truth if available
            gt_path = demo_path.parent / "ira_ground_truth.json"
            response = result.to_dict()
            if gt_path.exists():
                import json as _json
                with open(gt_path) as f:
                    response["ground_truth"] = _json.load(f)

            return JSONResponse(response)
        except Exception as e:
            logger.exception("Demo detection failed")
            return JSONResponse({"error": str(e)}, status_code=500)

    return app
