"""FastAPI application serving the control-room UI and the live API.

One analysis session at a time — matching the "one operator, one camera"
use case. Binds to loopback by default (see the CLI); nothing here is
authenticated, so keep it off untrusted networks.
"""

from __future__ import annotations

import asyncio
import csv
import io
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from trafficlens import __version__
from trafficlens.classes import COCO_CLASSES, TRAFFIC_CLASSES
from trafficlens.config import AppConfig, CalibrationConfig, DetectorConfig, GateConfig, SpeedConfig
from trafficlens.web.session import AnalysisSession

app = FastAPI(title="TrafficLens", version=__version__)

_STATIC = Path(__file__).parent / "static"
_session: AnalysisSession | None = None

KNOWN_MODELS = [
    "yolo26n.pt", "yolo26s.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolo12n.pt", "yolo12s.pt",
]


class SessionRequest(BaseModel):
    source: str = Field(min_length=1)
    model: str = "yolo11n.pt"
    classes: list[str] = Field(default_factory=lambda: list(TRAFFIC_CLASSES[:4]))
    confidence: float = Field(default=0.35, ge=0.05, le=0.95)
    tracker: str = "bytetrack"
    gates: list[GateConfig] = Field(default_factory=list)
    calibration: CalibrationConfig | None = None
    speed_limit: float | None = Field(default=None, gt=0)
    speed_unit: str = Field(default="kmh", pattern="^(kmh|mph)$")
    loop: bool = True

    def to_app_config(self) -> AppConfig:
        return AppConfig(
            source=self.source,
            detector=DetectorConfig(
                model=self.model, classes=self.classes,
                confidence=self.confidence, tracker=self.tracker,
            ),
            gates=self.gates,
            calibration=self.calibration,
            speed=SpeedConfig(speed_limit=self.speed_limit, unit=self.speed_unit),
        )


class GatesRequest(BaseModel):
    gates: list[GateConfig]


class SpeedRequest(BaseModel):
    speed_limit: float | None = Field(default=None, gt=0)
    unit: str | None = Field(default=None, pattern="^(kmh|mph)$")


class CalibrationRequest(BaseModel):
    calibration: CalibrationConfig | None = None


def _require_session() -> AnalysisSession:
    if _session is None or not _session.is_alive():
        raise HTTPException(status_code=409, detail="no analysis session is running")
    return _session


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/app.js", include_in_schema=False)
def app_js() -> FileResponse:
    return FileResponse(_STATIC / "app.js", media_type="application/javascript")


@app.get("/style.css", include_in_schema=False)
def style_css() -> FileResponse:
    return FileResponse(_STATIC / "style.css", media_type="text/css")


@app.get("/api/meta")
def meta() -> dict:
    return {
        "version": __version__,
        "models": KNOWN_MODELS,
        "classes": COCO_CLASSES,
        "traffic_classes": TRAFFIC_CLASSES,
    }


@app.post("/api/session", status_code=201)
def start_session(request: SessionRequest) -> dict:
    global _session
    if _session is not None and _session.is_alive() and not _session.finished:
        raise HTTPException(status_code=409, detail="a session is already running — stop it first")
    config = request.to_app_config()  # pydantic re-validates the full config
    session = AnalysisSession(config, loop_file=request.loop)
    session.start()
    try:
        session.wait_ready(timeout=120.0)
    except Exception as exc:
        session.stop()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _session = session
    return {"status": "running", "source": session.source_info}


@app.delete("/api/session")
def stop_session() -> dict:
    global _session
    if _session is None:
        raise HTTPException(status_code=409, detail="no analysis session is running")
    _session.stop()
    _session.join(timeout=10.0)
    summary = _session.pipeline.summary() if _session.pipeline else {}
    _session = None
    return {"status": "stopped", "summary": summary}


@app.get("/api/session")
def session_status() -> dict:
    if _session is None:
        return {"running": False}
    config = _session.pipeline.config if _session.pipeline else _session.config
    return {
        "running": _session.is_alive() and not _session.finished,
        "error": _session.error,
        "source": _session.source_info,
        "stats": _session.stats,
        "gates": [g.model_dump() for g in config.gates],
        "calibrated": config.calibration is not None,
        "speed_limit": config.speed.speed_limit,
        "speed_unit": config.speed.unit,
    }


@app.get("/api/events")
def events(after: int = 0) -> dict:
    session = _require_session()
    items = session.events_after(after)
    return {"events": items, "latest": items[-1]["seq"] if items else after}


@app.get("/api/stream", include_in_schema=False)
async def stream() -> StreamingResponse:
    session = _require_session()

    async def frames():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last = None
        while session.is_alive() and not session.finished:
            jpeg = session.latest_jpeg
            if jpeg is not None and jpeg is not last:
                yield boundary + jpeg + b"\r\n"
                last = jpeg
            await asyncio.sleep(0.03)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot.jpg", include_in_schema=False)
def snapshot() -> Response:
    session = _require_session()
    if session.latest_jpeg is None:
        raise HTTPException(status_code=404, detail="no frame yet")
    return Response(content=session.latest_jpeg, media_type="image/jpeg")


@app.put("/api/gates")
def put_gates(request: GatesRequest) -> dict:
    session = _require_session()
    session.update_gates(request.gates)
    return {"gates": [g.name for g in request.gates]}


@app.put("/api/speed")
def put_speed(request: SpeedRequest) -> dict:
    session = _require_session()
    session.update_speed(request.speed_limit, request.unit)
    return {"speed_limit": request.speed_limit, "unit": request.unit}


@app.put("/api/calibration")
def put_calibration(request: CalibrationRequest) -> dict:
    session = _require_session()
    try:
        session.update_calibration(request.calibration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"calibrated": request.calibration is not None}


@app.get("/api/violations")
def violations() -> dict:
    session = _require_session()
    return {
        "violations": [
            {
                "seq": v.seq, "track": v.event.track_id, "class": v.event.class_name,
                "gate": v.event.gate, "speed": round(v.event.speed, 1) if v.event.speed else None,
                "t": round(v.event.timestamp, 2),
            }
            for v in session.violations
        ]
    }


@app.get("/api/violations/{seq}.jpg", include_in_schema=False)
def violation_image(seq: int) -> Response:
    session = _require_session()
    for v in session.violations:
        if v.seq == seq:
            return Response(content=v.jpeg, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"no violation snapshot with seq {seq}")


@app.get("/api/export/events.csv")
def export_events() -> Response:
    session = _require_session()
    if session.pipeline is None:
        raise HTTPException(status_code=409, detail="session not ready")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["frame", "timestamp_s", "track_id", "class", "gate", "direction", "speed", "violation"])
    for e in session.pipeline.events:
        writer.writerow([e.frame_index, f"{e.timestamp:.3f}", e.track_id, e.class_name,
                         e.gate, e.direction, f"{e.speed:.1f}" if e.speed is not None else "",
                         int(e.is_violation)])
    return Response(
        content=buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="trafficlens-events-{int(time.time())}.csv"'},
    )


@app.get("/api/export/summary.json")
def export_summary() -> JSONResponse:
    session = _require_session()
    if session.pipeline is None:
        raise HTTPException(status_code=409, detail="session not ready")
    return JSONResponse(session.pipeline.summary())
