"""CAP 1.2 alert construction and CMAS / 3GPP severity mapping.

This mirrors the real government alerting chain. An authority authors an alert
in CAP (Common Alerting Protocol, OASIS standard); an aggregator such as India's
SACHET hands it to each carrier's Cell Broadcast Centre; the CBC turns the CAP
area into a list of cell sites and those towers broadcast it.

We reproduce every layer except the radio one: the payload, the severity classes,
the 3GPP message identifiers and the area-based targeting are the genuine
article. Delivery rides WebSocket/push instead of SIB12, because the broadcast
channel is licensed spectrum.

Default status is EXERCISE, which is CAP's own value for a drill -- an honest
demo, not a spoofed government alert.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import StrEnum
from xml.etree import ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

from backend.models.schemas import RouteState

CAP_NAMESPACE = "urn:oasis:names:tc:emergency:cap:1.2"
SENDER_ID = "jvalyx.fire.intelligence"
SENDER_NAME = "Jvalyx Fire Intelligence"
IST = timezone(timedelta(hours=5, minutes=30))


class CmasClass(StrEnum):
    """CMAS / WEA alert classes. Handsets let users mute the lower ones only."""

    PRESIDENTIAL = "Presidential"
    EXTREME = "Extreme"
    SEVERE = "Severe"
    AMBER = "Amber"


# 3GPP TS 23.041 message identifiers -- the number the baseband actually matches on.
CMAS_MESSAGE_IDENTIFIER: dict[CmasClass, int] = {
    CmasClass.PRESIDENTIAL: 4370,
    CmasClass.EXTREME: 4371,
    CmasClass.SEVERE: 4373,
    CmasClass.AMBER: 4379,
}


class AlertStatus(StrEnum):
    ACTUAL = "Actual"
    EXERCISE = "Exercise"
    TEST = "Test"


class RouteProfile(BaseModel):
    """How one routing state maps onto CAP/CMAS fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cmas_class: CmasClass
    urgency: str
    severity: str
    certainty: str
    response_type: str
    radius_km: float
    instruction: str


ROUTE_PROFILES: dict[RouteState, RouteProfile] = {
    RouteState.CRITICAL: RouteProfile(
        cmas_class=CmasClass.EXTREME,
        urgency="Immediate",
        severity="Extreme",
        certainty="Observed",
        response_type="Evacuate",
        radius_km=10.0,
        instruction=(
            "Evacuate the area immediately. Move upwind and away from the facility "
            "perimeter. Do not approach. Await instructions from emergency services."
        ),
    ),
    RouteState.UNCERTAIN: RouteProfile(
        cmas_class=CmasClass.SEVERE,
        urgency="Expected",
        severity="Severe",
        certainty="Likely",
        response_type="Prepare",
        radius_km=5.0,
        instruction=(
            "A possible thermal anomaly is under verification. Prepare to evacuate "
            "and monitor official channels."
        ),
    ),
    RouteState.NORMAL: RouteProfile(
        cmas_class=CmasClass.AMBER,
        urgency="Future",
        severity="Minor",
        certainty="Possible",
        response_type="Monitor",
        radius_km=2.0,
        instruction="No action required. Advisory only.",
    ),
}


class CapAlert(BaseModel):
    """One CAP 1.2 alert, ready to serialise to XML or to broadcast as JSON."""

    model_config = ConfigDict(extra="forbid")

    identifier: str
    sent: str
    status: AlertStatus
    msg_type: str = "Alert"
    scope: str = "Public"
    category: str = "Fire"
    event: str
    response_type: str
    urgency: str
    severity: str
    certainty: str
    cmas_class: CmasClass
    message_identifier: int
    effective: str
    expires: str
    headline: str
    description: str
    instruction: str
    area_desc: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0)
    source_event_id: str | None = None


def _now_ist() -> datetime:
    return datetime.now(IST)


def build_alert(
    *,
    route_state: RouteState,
    latitude: float,
    longitude: float,
    area_desc: str,
    event: str = "Industrial Fire",
    headline: str | None = None,
    description: str | None = None,
    status: AlertStatus = AlertStatus.EXERCISE,
    ttl_minutes: int = 30,
    source_event_id: str | None = None,
    radius_km: float | None = None,
) -> CapAlert:
    """Turn a routing decision into a CAP alert.

    `radius_km` overrides the per-severity default, which is what lets an
    operator widen or tighten the broadcast footprint the way a CBC operator
    would when choosing cell sites.
    """
    profile = ROUTE_PROFILES[route_state]
    sent = _now_ist()
    expires = sent + timedelta(minutes=ttl_minutes)
    stamp = sent.strftime("%Y%m%dT%H%M%S")

    resolved_headline = headline or f"{profile.severity.upper()} FIRE ALERT - {area_desc}"
    resolved_description = description or (
        f"Jvalyx has routed a detection at {area_desc} to {route_state.value}. "
        f"Confidence: {profile.certainty.lower()}. Broadcast radius {profile.radius_km:g} km."
    )

    return CapAlert(
        identifier=f"JVALYX-{stamp}-{route_state.value}",
        sent=sent.isoformat(timespec="seconds"),
        status=status,
        event=event,
        response_type=profile.response_type,
        urgency=profile.urgency,
        severity=profile.severity,
        certainty=profile.certainty,
        cmas_class=profile.cmas_class,
        message_identifier=CMAS_MESSAGE_IDENTIFIER[profile.cmas_class],
        effective=sent.isoformat(timespec="seconds"),
        expires=expires.isoformat(timespec="seconds"),
        headline=resolved_headline,
        description=resolved_description,
        instruction=profile.instruction,
        area_desc=area_desc,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km if radius_km is not None else profile.radius_km,
        source_event_id=source_event_id,
    )


def to_xml(alert: CapAlert) -> str:
    """Serialise to CAP 1.2 XML -- the exact shape SACHET would publish."""
    root = ET.Element("alert", xmlns=CAP_NAMESPACE)

    def sub(parent: ET.Element, tag: str, text: str) -> ET.Element:
        node = ET.SubElement(parent, tag)
        node.text = text
        return node

    sub(root, "identifier", alert.identifier)
    sub(root, "sender", SENDER_ID)
    sub(root, "sent", alert.sent)
    sub(root, "status", alert.status.value)
    sub(root, "msgType", alert.msg_type)
    sub(root, "scope", alert.scope)

    info = ET.SubElement(root, "info")
    sub(info, "language", "en-IN")
    sub(info, "category", alert.category)
    sub(info, "event", alert.event)
    sub(info, "responseType", alert.response_type)
    sub(info, "urgency", alert.urgency)
    sub(info, "severity", alert.severity)
    sub(info, "certainty", alert.certainty)
    sub(info, "effective", alert.effective)
    sub(info, "expires", alert.expires)
    sub(info, "senderName", SENDER_NAME)
    sub(info, "headline", alert.headline)
    sub(info, "description", alert.description)
    sub(info, "instruction", alert.instruction)

    for name, value in (
        ("CMAM_class", alert.cmas_class.value),
        ("CMAM_message_identifier", str(alert.message_identifier)),
        ("source_event_id", alert.source_event_id or ""),
    ):
        parameter = ET.SubElement(info, "parameter")
        sub(parameter, "valueName", name)
        sub(parameter, "value", value)

    area = ET.SubElement(info, "area")
    sub(area, "areaDesc", alert.area_desc)
    # CAP circles are "lat,lon radius-in-km".
    sub(area, "circle", f"{alert.latitude},{alert.longitude} {alert.radius_km:g}")

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
