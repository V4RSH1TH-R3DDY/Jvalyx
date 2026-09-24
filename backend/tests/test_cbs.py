"""Cell Broadcast Centre: CAP authoring, area targeting, and the WebSocket path."""

from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from backend.api.cbs import CellBroadcastCentre, cell_id_for, haversine_km
from backend.app import app
from backend.models.schemas import RouteState
from backend.pipeline.cap import AlertStatus, build_alert, to_xml

client = TestClient(app)

# MRPL, Mangaluru -- the facility the demo routes on.
MRPL = (12.9716, 74.8631)


def _alert(radius_km: float = 10.0, route_state: RouteState = RouteState.CRITICAL):
    return build_alert(
        route_state=route_state,
        latitude=MRPL[0],
        longitude=MRPL[1],
        area_desc="MRPL Petrochemical Complex",
        radius_km=radius_km,
    )


class _FakeSocket:
    """Stands in for a handset; records what the tower sent it."""

    def __init__(self) -> None:
        self.received: list[dict] = []

    async def send_json(self, message: dict) -> None:
        self.received.append(message)


# -- CAP authoring ---------------------------------------------------------


def test_route_state_maps_to_3gpp_message_identifier():
    assert build_alert(
        route_state=RouteState.CRITICAL, latitude=0.0, longitude=0.0, area_desc="x"
    ).message_identifier == 4371
    assert build_alert(
        route_state=RouteState.UNCERTAIN, latitude=0.0, longitude=0.0, area_desc="x"
    ).message_identifier == 4373
    assert build_alert(
        route_state=RouteState.NORMAL, latitude=0.0, longitude=0.0, area_desc="x"
    ).message_identifier == 4379


def test_alert_defaults_to_exercise_not_actual():
    # A demo must not emit something that claims to be a real government alert.
    assert _alert().status is AlertStatus.EXERCISE


def test_cap_xml_is_wellformed_and_carries_the_area_circle():
    root = ET.fromstring(to_xml(_alert()))
    namespace = {"cap": "urn:oasis:names:tc:emergency:cap:1.2"}
    assert root.tag == "{urn:oasis:names:tc:emergency:cap:1.2}alert"
    assert root.find("cap:status", namespace).text == "Exercise"
    assert root.find("cap:info/cap:severity", namespace).text == "Extreme"
    assert root.find("cap:info/cap:area/cap:circle", namespace).text == "12.9716,74.8631 10"


# -- geography -------------------------------------------------------------


def test_haversine_matches_known_distance():
    # Bengaluru -> Mangaluru is roughly 300 km.
    assert 280 < haversine_km(12.9716, 77.5946, *MRPL) < 320


def test_unlocated_device_falls_into_the_venue_cell():
    assert cell_id_for(None, None) == "VENUE"
    assert cell_id_for(12.9716, 74.8631) == "CELL-12.97-74.86"


# -- targeting -------------------------------------------------------------


@pytest.mark.asyncio
async def test_broadcast_reaches_only_devices_inside_the_footprint():
    centre = CellBroadcastCentre()
    inside, outside = _FakeSocket(), _FakeSocket()
    # 2 km from MRPL, and Bengaluru ~300 km away.
    await centre.attach(inside, 12.9880, 74.8631)
    await centre.attach(outside, 12.9716, 77.5946)

    receipt = await centre.broadcast(_alert(radius_km=10.0))

    assert receipt.devices_attached == 2
    assert receipt.devices_in_footprint == 1
    assert len(inside.received) == 1
    assert outside.received == []


@pytest.mark.asyncio
async def test_radius_widens_the_footprint():
    centre = CellBroadcastCentre()
    far = _FakeSocket()
    await centre.attach(far, 12.9716, 77.5946)

    assert (await centre.broadcast(_alert(radius_km=10.0))).devices_in_footprint == 0
    assert (await centre.broadcast(_alert(radius_km=400.0))).devices_in_footprint == 1


@pytest.mark.asyncio
async def test_broadcast_payload_carries_the_full_cap_alert():
    centre = CellBroadcastCentre()
    handset = _FakeSocket()
    await centre.attach(handset, *MRPL)

    await centre.broadcast(_alert())

    message = handset.received[0]
    assert message["type"] == "cell_broadcast"
    assert message["alert"]["cmas_class"] == "Extreme"
    assert message["alert"]["instruction"].startswith("Evacuate")


@pytest.mark.asyncio
async def test_a_dropped_handset_does_not_break_the_broadcast():
    class _Dead(_FakeSocket):
        async def send_json(self, message: dict) -> None:
            raise ConnectionResetError("handset gone")

    centre = CellBroadcastCentre()
    alive = _FakeSocket()
    await centre.attach(_Dead(), *MRPL)
    await centre.attach(alive, *MRPL)

    receipt = await centre.broadcast(_alert())

    assert len(alive.received) == 1
    assert receipt.devices_in_footprint == 1


@pytest.mark.asyncio
async def test_location_update_moves_the_device_into_range():
    centre = CellBroadcastCentre()
    handset = _FakeSocket()
    device = await centre.attach(handset, 12.9716, 77.5946)  # out of range

    assert (await centre.broadcast(_alert())).devices_in_footprint == 0

    await centre.update_location(device.device_id, *MRPL)
    assert (await centre.broadcast(_alert())).devices_in_footprint == 1


# -- HTTP surface ----------------------------------------------------------


def test_status_reports_an_empty_network():
    body = client.get("/cbs/status").json()
    assert body["devices_attached"] == 0
    assert body["armed"] is False


def test_broadcast_requires_coordinates_or_an_event():
    response = client.post("/cbs/broadcast", json={"route_state": "CRITICAL"})
    assert response.status_code == 422


def test_broadcast_rejects_an_unknown_event():
    response = client.post("/cbs/broadcast", json={"event_id": "nope"})
    assert response.status_code == 404


def test_manual_broadcast_returns_a_receipt_and_serves_its_cap_xml():
    response = client.post(
        "/cbs/broadcast",
        json={
            "route_state": "CRITICAL",
            "latitude": MRPL[0],
            "longitude": MRPL[1],
            "area_desc": "MRPL Petrochemical Complex",
        },
    )
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["message_identifier"] == 4371
    assert receipt["status"] == "Exercise"

    xml = client.get(receipt["cap_xml_url"])
    assert xml.status_code == 200
    assert "urn:oasis:names:tc:emergency:cap:1.2" in xml.text


def test_unknown_cap_identifier_is_404():
    assert client.get("/cbs/alerts/JVALYX-nope.xml").status_code == 404


def test_arm_toggles_auto_broadcast():
    try:
        assert client.post("/cbs/arm", params={"armed": True}).json()["armed"] is True
    finally:
        assert client.post("/cbs/arm", params={"armed": False}).json()["armed"] is False


def test_handset_attaches_over_websocket_and_receives_a_broadcast():
    with client.websocket_connect("/ws/cbs") as socket:
        attached = socket.receive_json()
        assert attached["type"] == "attached"
        assert attached["located"] is False
        assert attached["cell_id"] == "VENUE"

        # Move the handset onto the facility, then broadcast over it.
        socket.send_json({"latitude": MRPL[0], "longitude": MRPL[1]})
        relocated = socket.receive_json()
        assert relocated["located"] is True
        assert relocated["cell_id"] == "CELL-12.97-74.86"

        client.post(
            "/cbs/broadcast",
            json={
                "route_state": "CRITICAL",
                "latitude": MRPL[0],
                "longitude": MRPL[1],
                "area_desc": "MRPL Petrochemical Complex",
            },
        )
        message = socket.receive_json()
        assert message["type"] == "cell_broadcast"
        assert message["alert"]["severity"] == "Extreme"


def test_pages_and_join_url_render():
    assert "Jvalyx Emergency Alert System" in client.get("/alert").text
    assert "Cell Broadcast Console" in client.get("/cbs/console").text
    assert client.get("/cbs/join").json()["join_url"].endswith("/alert")


def test_qr_endpoint_returns_a_png():
    response = client.get("/cbs/qr")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
