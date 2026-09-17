"""Cell Broadcast Centre simulator.

A real CBC takes a CAP alert, converts its area into the set of cell sites whose
coverage intersects that area, and tells those towers to broadcast. It never
addresses a subscriber: there is no recipient list, which is exactly why cell
broadcast does not congest a network and why it needs no opt-in.

This module keeps that property. Devices attach to `/ws/cbs` and are placed in a
cell; a broadcast selects *cells inside the alert footprint* and fans out to
whoever happens to be attached there. The server never targets a device by
address, and holds no identity beyond an ephemeral connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.models.schemas import RouteState
from backend.pipeline.cap import AlertStatus, CapAlert, build_alert, to_xml
from backend.models.intelligence import EventIntelligence
from backend.runtime import replay_worker, route_observers

router = APIRouter(tags=["cbs"])

STATIC_DIR = Path(__file__).parent / "static"
EARTH_RADIUS_KM = 6371.0
MAX_HISTORY = 50

# Unlocated devices land in this notional cell. A phone that refuses GPS still
# has to sit somewhere, and on demo night that somewhere is the venue.
VENUE_CELL = "VENUE"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


VENUE_LAT = _env_float("JVALYX_VENUE_LAT", 12.9716)
VENUE_LON = _env_float("JVALYX_VENUE_LON", 77.5946)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def cell_id_for(latitude: float | None, longitude: float | None) -> str:
    """Coarse grid cell, standing in for a tower's coverage area.

    ~0.01 degrees is roughly a kilometre, which is the right order of magnitude
    for an urban macrocell.
    """
    if latitude is None or longitude is None:
        return VENUE_CELL
    return f"CELL-{latitude:.2f}-{longitude:.2f}"


class Device(BaseModel):
    """One attached handset. Deliberately anonymous."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    device_id: str
    cell_id: str
    latitude: float | None = None
    longitude: float | None = None
    located: bool = False
    attached_at: str
    websocket: Any = Field(exclude=True)


class BroadcastRequest(BaseModel):
    """Either replay an existing event, or author an alert by hand."""

    model_config = ConfigDict(extra="forbid")

    event_id: str | None = None
    route_state: RouteState = RouteState.CRITICAL
    latitude: float | None = None
    longitude: float | None = None
    area_desc: str | None = None
    event: str = "Industrial Fire"
    headline: str | None = None
    radius_km: float | None = None
    status: AlertStatus = AlertStatus.EXERCISE
    ttl_minutes: int = Field(default=30, ge=1, le=1440)


class BroadcastReceipt(BaseModel):
    """What the CBC can honestly report: cells lit, not people reached."""

    model_config = ConfigDict(extra="forbid")

    identifier: str
    severity: str
    cmas_class: str
    message_identifier: int
    status: str
    cells_in_footprint: list[str]
    devices_attached: int
    devices_in_footprint: int
    ntfy_dispatched: bool
    cap_xml_url: str


class CellBroadcastCentre:
    """Holds attached devices and fans CAP alerts out by area."""

    def __init__(self) -> None:
        self._devices: dict[str, Device] = {}
        self._lock = asyncio.Lock()
        self._history: list[CapAlert] = []
        self._armed = False

    # -- device lifecycle -------------------------------------------------
    async def attach(
        self,
        websocket: WebSocket,
        latitude: float | None,
        longitude: float | None,
    ) -> Device:
        device = Device(
            device_id=uuid.uuid4().hex[:8].upper(),
            cell_id=cell_id_for(latitude, longitude),
            latitude=latitude,
            longitude=longitude,
            located=latitude is not None and longitude is not None,
            attached_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            websocket=websocket,
        )
        async with self._lock:
            self._devices[device.device_id] = device
        return device

    async def detach(self, device_id: str) -> None:
        async with self._lock:
            self._devices.pop(device_id, None)

    async def update_location(
        self, device_id: str, latitude: float, longitude: float
    ) -> Device | None:
        """A handset that grants GPS after attaching re-registers to its real cell."""
        async with self._lock:
            device = self._devices.get(device_id)
            if device is None:
                return None
            updated = device.model_copy(
                update={
                    "latitude": latitude,
                    "longitude": longitude,
                    "located": True,
                    "cell_id": cell_id_for(latitude, longitude),
                }
            )
            self._devices[device_id] = updated
            return updated

    # -- targeting --------------------------------------------------------
    def _in_footprint(self, device: Device, alert: CapAlert) -> bool:
        """Is this device's cell inside the alert circle?

        An unlocated device is judged by the venue's coordinates, not waved
        through: if the venue is outside the footprint it genuinely gets nothing,
        which is what makes the geographic targeting real rather than decorative.
        """
        lat = device.latitude if device.located else VENUE_LAT
        lon = device.longitude if device.located else VENUE_LON
        assert lat is not None and lon is not None
        return haversine_km(lat, lon, alert.latitude, alert.longitude) <= alert.radius_km

    async def broadcast(self, alert: CapAlert) -> BroadcastReceipt:
        async with self._lock:
            devices = list(self._devices.values())

        targeted = [device for device in devices if self._in_footprint(device, alert)]
        cells = sorted({device.cell_id for device in targeted})

        message = {
            "type": "cell_broadcast",
            "alert": alert.model_dump(mode="json"),
            "cap_xml_url": f"/cbs/alerts/{alert.identifier}.xml",
        }

        dead: list[str] = []
        for device in targeted:
            try:
                await device.websocket.send_json(message)
            except Exception:  # noqa: BLE001 - one dropped handset must not stop the broadcast
                dead.append(device.device_id)
        for device_id in dead:
            await self.detach(device_id)

        self._history.append(alert)
        del self._history[:-MAX_HISTORY]

        ntfy_ok = await dispatch_ntfy(alert)

        return BroadcastReceipt(
            identifier=alert.identifier,
            severity=alert.severity,
            cmas_class=alert.cmas_class.value,
            message_identifier=alert.message_identifier,
            status=alert.status.value,
            cells_in_footprint=cells,
            devices_attached=len(devices),
            devices_in_footprint=len(targeted) - len(dead),
            ntfy_dispatched=ntfy_ok,
            cap_xml_url=f"/cbs/alerts/{alert.identifier}.xml",
        )

    # -- introspection ----------------------------------------------------
    def status(self) -> dict[str, Any]:
        devices = list(self._devices.values())
        cells: dict[str, int] = {}
        for device in devices:
            cells[device.cell_id] = cells.get(device.cell_id, 0) + 1
        last = self._history[-1] if self._history else None
        return {
            "devices_attached": len(devices),
            "located": sum(1 for device in devices if device.located),
            "cells": cells,
            "armed": self._armed,
            "venue": {"latitude": VENUE_LAT, "longitude": VENUE_LON},
            "ntfy_topic": os.environ.get("JVALYX_NTFY_TOPIC"),
            "last_alert": last.model_dump(mode="json") if last else None,
        }

    def find(self, identifier: str) -> CapAlert | None:
        return next((alert for alert in self._history if alert.identifier == identifier), None)

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self, value: bool) -> bool:
        self._armed = value
        return self._armed


centre = CellBroadcastCentre()

# Routing state as of the last frame, per event -- so an armed broadcast fires on
# the *crossing* into CRITICAL rather than on every frame that stays there.
_last_route: dict[str, RouteState] = {}


async def auto_broadcast(intelligence: EventIntelligence) -> None:
    """Fire a cell broadcast when an event crosses into CRITICAL, while armed."""
    previous = _last_route.get(intelligence.event_id)
    _last_route[intelligence.event_id] = intelligence.route_state
    if not centre.armed:
        return
    if intelligence.route_state is not RouteState.CRITICAL or previous is RouteState.CRITICAL:
        return
    alert = build_alert(
        route_state=RouteState.CRITICAL,
        latitude=intelligence.fused_event.latitude,
        longitude=intelligence.fused_event.longitude,
        area_desc=intelligence.fused_event.facility_id or intelligence.label,
        source_event_id=intelligence.event_id,
    )
    await centre.broadcast(alert)


route_observers.append(auto_broadcast)


async def dispatch_ntfy(alert: CapAlert) -> bool:
    """Push to a phone whose screen is locked, via ntfy.

    This is the one path that survives the tab being closed, because it rides a
    channel the OS keeps open -- the nearest civilian analogue to the baseband
    always listening on SIB12.
    """
    topic = os.environ.get("JVALYX_NTFY_TOPIC")
    if not topic:
        return False
    server = os.environ.get("JVALYX_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    priority = "urgent" if alert.severity in {"Extreme", "Severe"} else "default"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{server}/{topic}",
                content=f"{alert.description}\n\n{alert.instruction}".encode(),
                headers={
                    "Title": alert.headline,
                    "Priority": priority,
                    "Tags": "fire,rotating_light",
                },
            )
        return response.status_code < 400
    except Exception:  # noqa: BLE001 - push is best-effort; the broadcast already went out
        return False


def lan_ip() -> str:
    """This machine's address on the local network, for the join QR."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No packet is actually sent; this just picks the outbound interface.
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        probe.close()


def _alert_from_request(request: BroadcastRequest) -> CapAlert:
    """Author the CAP alert, from a live event when one is named."""
    latitude, longitude = request.latitude, request.longitude
    area_desc, route_state = request.area_desc, request.route_state
    source_event_id = request.event_id

    if request.event_id is not None:
        event = replay_worker.store.get_event(request.event_id)
        if event is None:
            raise HTTPException(status_code=404, detail=f"Unknown event: {request.event_id}")
        latitude = event.fused_event.latitude
        longitude = event.fused_event.longitude
        route_state = event.route_state
        area_desc = area_desc or event.fused_event.facility_id or event.label

    if latitude is None or longitude is None:
        raise HTTPException(
            status_code=422,
            detail="Provide event_id, or both latitude and longitude.",
        )

    return build_alert(
        route_state=route_state,
        latitude=latitude,
        longitude=longitude,
        area_desc=area_desc or "Unnamed area",
        event=request.event,
        headline=request.headline,
        status=request.status,
        ttl_minutes=request.ttl_minutes,
        source_event_id=source_event_id,
        radius_km=request.radius_km,
    )


# -- routes ----------------------------------------------------------------


@router.websocket("/ws/cbs")
async def cbs_stream(websocket: WebSocket) -> None:
    """A handset attaching to the network.

    The device may send `{"latitude": .., "longitude": ..}` at any time; that is
    the only thing it ever tells us, and we keep it only while it is connected.
    """
    await websocket.accept()
    device = await centre.attach(websocket, None, None)
    try:
        await websocket.send_json(
            {
                "type": "attached",
                "device_id": device.device_id,
                "cell_id": device.cell_id,
                "located": device.located,
            }
        )
        while True:
            payload = await websocket.receive_json()
            latitude, longitude = payload.get("latitude"), payload.get("longitude")
            if latitude is None or longitude is None:
                continue
            updated = await centre.update_location(device.device_id, float(latitude), float(longitude))
            if updated is not None:
                await websocket.send_json(
                    {
                        "type": "attached",
                        "device_id": updated.device_id,
                        "cell_id": updated.cell_id,
                        "located": updated.located,
                    }
                )
    except (WebSocketDisconnect, ValueError, KeyError):
        pass
    finally:
        await centre.detach(device.device_id)
        with contextlib.suppress(RuntimeError):
            await websocket.close()


@router.post("/cbs/broadcast", response_model=BroadcastReceipt)
async def broadcast(request: BroadcastRequest) -> BroadcastReceipt:
    return await centre.broadcast(_alert_from_request(request))


@router.get("/cbs/status")
async def cbs_status() -> dict[str, Any]:
    return centre.status()


@router.post("/cbs/arm")
async def cbs_arm(armed: bool = True) -> dict[str, Any]:
    """Arm auto-broadcast, so a CRITICAL routing decision fires the alert itself."""
    return {"armed": centre.arm(armed)}


@router.get("/cbs/join")
async def cbs_join() -> dict[str, Any]:
    """Everything the presenter console needs to get phones attached."""
    port = os.environ.get("JVALYX_PORT", "8000")
    url = f"http://{lan_ip()}:{port}/alert"
    return {"join_url": url, "lan_ip": lan_ip(), "port": port}


@router.get("/cbs/qr")
async def cbs_qr() -> Response:
    """QR for the join URL. Degrades to JSON when `qrcode` is not installed."""
    payload = await cbs_join()
    try:
        import io

        import qrcode
    except ImportError:
        return JSONResponse(
            {**payload, "qr": None, "hint": "pip install 'qrcode[pil]' for a QR image"}
        )
    buffer = io.BytesIO()
    qrcode.make(payload["join_url"]).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@router.get("/cbs/alerts/{identifier}.xml")
async def cap_xml(identifier: str) -> Response:
    alert = centre.find(identifier)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Unknown alert: {identifier}")
    return Response(content=to_xml(alert), media_type="application/xml")


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / name).read_text(encoding="utf-8"))


@router.get("/alert", response_class=HTMLResponse)
async def alert_page() -> HTMLResponse:
    """The handset page. This is what the audience scans into."""
    return _page("alert.html")


@router.get("/cbs/console", response_class=HTMLResponse)
async def console_page() -> HTMLResponse:
    """The presenter's control surface."""
    return _page("console.html")
