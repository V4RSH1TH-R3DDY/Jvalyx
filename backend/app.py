"""FastAPI application entrypoint for the Jvalyx backend.

Preferred launch is from the repo root (`python app.py` / `python -m backend` /
`uvicorn backend.app:app`), but running this file directly from inside `backend/`
also works — the block below puts the repo root on the path first.
"""

import contextlib
from collections.abc import AsyncIterator

if __name__ == "__main__" and __package__ in (None, ""):  # `python app.py` / `python backend/app.py`
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.api import cbs as cbs_routes
from backend.api import config as config_routes
from backend.api import events as events_routes
from backend.api import facilities as facilities_routes
from backend.api import firms as firms_routes
from backend.api import scenarios as scenario_routes
from backend.api import triage as triage_routes
from backend.api import weather as weather_routes
from backend.api import ws as ws_routes
from backend.config import load_config
from backend.pipeline import active_model_version
from backend.runtime import replay_worker

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    # containerised frontend (docker compose `web` service)
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_version: str
    policy_version: str


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await replay_worker.shutdown()


def create_app() -> FastAPI:
    config = load_config()
    app = FastAPI(title="Jvalyx Backend", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        # 127.0.0.1 as well as localhost: the compose frontend is reachable on both.
        # Private LAN ranges too, so a phone on the venue Wi-Fi can load the
        # dashboard and drive the cell broadcast (docs/cell-broadcast.md).
        allow_origin_regex=(
            r"http://(localhost|127\.0\.0\.1"
            r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3}"
            r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}):\d+"
        ),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["system"])
    async def index() -> dict:
        return {
            "service": "jvalyx-backend",
            "version": app.version,
            "model_version": active_model_version(),
            "policy_version": config.versions.policy_version,
            "docs": "/docs",
            "endpoints": [
                "GET /health",
                "GET /config",
                "GET /scenarios",
                "POST /scenarios/{id}/start",
                "POST /scenarios/{id}/reset",
                "GET /events",
                "GET /events/{id}",
                "POST /events/{id}/simulate",
                "POST /events/{id}/verify",
                "GET /audit",
                "WS /ws/events",
                "GET /api/firms/{path:path}",
                "GET /api/weather",
                "POST /api/triage/classify",
            ],
        }

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="jvalyx-backend",
            model_version=active_model_version(),
            policy_version=config.versions.policy_version,
        )

    app.include_router(scenario_routes.router)
    app.include_router(events_routes.router)
    app.include_router(config_routes.router)
    app.include_router(ws_routes.router)
    app.include_router(firms_routes.router)
    app.include_router(weather_routes.router)
    app.include_router(facilities_routes.router)
    app.include_router(triage_routes.router)
    app.include_router(cbs_routes.router)

    return app


app = create_app()


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Jvalyx backend API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
