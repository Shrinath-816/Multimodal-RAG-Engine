"""
mmrag/main.py
================

FastAPI web layer — deliberately THIN: every handler delegates to TicketAPI
(mmrag/api.py), where the testable logic lives.

Run (from the project root, so relative paths to data/ and config/ resolve):

    DEMO_MODE=1 uvicorn --factory mmrag.main:create_app --port 8000

Endpoints:
    POST /tickets        multipart form: text?, image?, audio?  (any combination)
    WS   /ws/voice       stream raw PCM16 mono 16 kHz binary frames; each detected
                         utterance becomes a ticket and a JSON result is sent back.
                         Send the text message "end" to flush the final utterance.
    GET  /dashboard      live cost-per-ticket dashboard
    GET  /api/metrics    dashboard data (JSON)
    GET  /api/tickets    recent ticket summaries
    GET  /health         liveness

The app is built by a FACTORY (`create_app`) so importing this module has no
side effects (no settings read, no knowledge-base load) until uvicorn calls it.
Requires fastapi, uvicorn, and python-multipart (see requirements.txt).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from mmrag.bootstrap import build_engine
from mmrag.config import Settings
from mmrag.dashboard import DASHBOARD_HTML
from mmrag.ingest.voice import VoiceStreamSession


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. The engine (and knowledge base) is built once at startup."""
    resolved = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = await build_engine(resolved)
        yield

    app = FastAPI(title="The Multimodal RAG Engine", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        return request.app.state.engine.api.health()

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/api/metrics")
    async def metrics(request: Request):
        return request.app.state.engine.api.metrics()

    @app.get("/api/tickets")
    async def tickets(request: Request, limit: int = 20):
        return request.app.state.engine.api.recent_tickets(max(1, min(limit, 200)))

    @app.post("/tickets")
    async def submit_ticket(
        request: Request,
        text: str | None = Form(default=None),
        image: UploadFile | None = File(default=None),
        audio: UploadFile | None = File(default=None),
    ):
        image_bytes = await image.read() if image else None
        audio_bytes = await audio.read() if audio else None
        result = await request.app.state.engine.api.submit(
            text=text,
            image=image_bytes or None,
            audio=audio_bytes or None,
            audio_filename=(audio.filename if audio and audio.filename else "audio.wav"),
            audio_mime=(audio.content_type if audio and audio.content_type else "audio/wav"),
        )
        return JSONResponse(result, status_code=result["http_status"])

    @app.websocket("/ws/voice")
    async def voice_stream(websocket: WebSocket):
        await websocket.accept()
        engine = websocket.app.state.engine
        session = VoiceStreamSession()
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                utterances: list[bytes] = []
                if message.get("bytes") is not None:
                    utterances = session.feed(message["bytes"])
                elif message.get("text") == "end":
                    final = session.flush()
                    utterances = [final] if final else []
                for utterance in utterances:
                    await websocket.send_json(await engine.api.submit_voice_pcm(utterance))
        except WebSocketDisconnect:
            pass

    return app
