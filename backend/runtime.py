"""Process-wide replay runtime: the singleton worker that turns replay frames into
event intelligence, persists it, and broadcasts updates to WebSocket clients.

API routers stay thin by delegating here.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable
from typing import Any

from backend.config import AppConfig, load_config
from backend.models import EventIntelligence, ReplayScenario, RouteState
from backend.models.schemas import DataMode
from backend.pipeline import ReplayEngine, ReplayStatus, ScenarioCatalog, process_frame
from backend.storage import EventStore, event_store


RouteObserver = Callable[[EventIntelligence], Awaitable[None]]

# Notified whenever a frame is published. Registered by backend.api.cbs, so the
# cell-broadcast layer can fire on a CRITICAL decision without runtime having to
# import the API package back.
route_observers: list[RouteObserver] = []


async def notify_route_observers(intelligence: EventIntelligence) -> None:
    for observer in route_observers:
        # An observer is a side channel: a failing alert must never stall replay.
        with contextlib.suppress(Exception):
            await observer(intelligence)


class Broadcaster:
    """Fan-out of JSON messages to connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[Any] = set()
        self._lock = asyncio.Lock()

    async def register(self, websocket: Any) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def unregister(self, websocket: Any) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def publish(self, message: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._clients)
        dead: list[Any] = []
        for client in targets:
            try:
                await client.send_json(message)
            except Exception:  # noqa: BLE001 - a dropped client must not break the worker
                dead.append(client)
        if dead:
            async with self._lock:
                for client in dead:
                    self._clients.discard(client)

    @property
    def client_count(self) -> int:
        return len(self._clients)


class ReplayWorker:
    """Runs one scenario at a time through the deterministic pipeline."""

    def __init__(
        self,
        catalog: ScenarioCatalog | None = None,
        store: EventStore | None = None,
        broadcaster: Broadcaster | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self._catalog = catalog or ScenarioCatalog()
        self._store = store or event_store
        self._broadcaster = broadcaster or Broadcaster()
        self._config = config or load_config()
        # Observation time is collapsed into demo playback time: ~2 scenario-minutes per
        # real second by default, and no single frame gap stalls the demo for more than 6s.
        # Both are overridable (env) for tuning the live demo and for fast tests.
        compression = float(os.getenv("JVALYX_REPLAY_COMPRESSION", "120"))
        max_gap = float(os.getenv("JVALYX_REPLAY_MAX_GAP_SECONDS", "6"))
        self._engine = ReplayEngine(time_compression=compression, max_frame_gap_seconds=max_gap)
        self._scenario: ReplayScenario | None = None
        self._task: asyncio.Task[None] | None = None

    # -- properties ---------------------------------------------------------
    @property
    def broadcaster(self) -> Broadcaster:
        return self._broadcaster

    @property
    def store(self) -> EventStore:
        return self._store

    @property
    def catalog(self) -> ScenarioCatalog:
        return self._catalog

    @property
    def config(self) -> AppConfig:
        return self._config

    # -- lifecycle --------------------------------------------------------
    def load(self, scenario_id: str) -> ReplayScenario:
        scenario = self._catalog.get(scenario_id)
        self._cancel_task()
        self._scenario = scenario
        self._engine.load(scenario)
        self._store.reset()
        self._store.append_audit(
            event_id=f"evt-{scenario_id}", action="SCENARIO_LOAD", notes=scenario.title
        )
        # Seed frame 0 so /events is never empty before playback starts.
        self._persist_frame(0)
        return scenario

    async def start(self) -> None:
        scenario = self._require_scenario()
        if self._task and not self._task.done():
            self._engine.resume()
            return
        # Re-starting a finished (or idle-at-end) replay begins a clean run: rewind the
        # engine and drop any prior operator verification / simulation state.
        if self._engine.status is ReplayStatus.COMPLETED or self._engine.current_frame_index >= len(
            scenario.frames
        ):
            await self.reset()
        self._store.append_audit(event_id=f"evt-{scenario.scenario_id}", action="REPLAY_START")
        self._task = asyncio.create_task(self._run())
        # Let the engine flip to PLAYING before we return a status snapshot.
        await asyncio.sleep(0)

    async def reset(self) -> None:
        scenario = self._require_scenario()
        self._cancel_task()
        self._engine.load(scenario)
        self._store.reset()
        self._store.append_audit(event_id=f"evt-{scenario.scenario_id}", action="REPLAY_RESET")
        self._persist_frame(0)
        await self._broadcaster.publish(self._status_message("reset"))

    def pause(self) -> None:
        self._engine.pause()

    def resume(self) -> None:
        self._engine.resume()

    def set_speed(self, speed: float) -> None:
        self._engine.set_speed(speed)

    async def jump_to(self, checkpoint: str) -> None:
        self._require_scenario()
        # A scrub must win over an in-flight run, otherwise the running loop keeps
        # advancing from its own index and immediately overwrites the jump.
        self._cancel_task()
        self._engine.jump_to(checkpoint)
        await self._publish_frame(self._engine.current_frame_index, "jump")

    async def step(self, delta: int) -> None:
        """Nudge the playhead one frame at a time (operator scrub)."""
        self._require_scenario()
        self._cancel_task()
        index = self._engine.step(delta)
        await self._publish_frame(index, "step")

    async def shutdown(self) -> None:
        self._cancel_task()

    # -- verification / simulation --------------------------------------
    def simulate(self, event_id: str, deviation: float) -> EventIntelligence:
        scenario = self._require_scenario()
        frame_index = self._engine.current_frame_index
        if frame_index >= len(scenario.frames):
            frame_index = len(scenario.frames) - 1
        verification = self._store.verification(event_id)
        intelligence = process_frame(
            scenario,
            frame_index,
            deviation=deviation,
            config=self._config,
            verification_status=verification.status,
            route_override=verification.route_override,
        )
        self._store.upsert_event(intelligence)
        self._store.append_audit(
            event_id=event_id,
            action="SIMULATE_DEVIATION",
            operator="DEMO",
            notes=f"Operational deviation set to {deviation:.2f}",
        )
        return intelligence

    def verify(self, event_id: str, decision: str, operator: str, notes: str) -> EventIntelligence:
        current = self._store.get_event(event_id)
        if current is None:
            raise KeyError(event_id)
        prior = current.route_state
        if decision == "confirm":
            status, override, action = "human_confirmed", RouteState.CRITICAL, "CONFIRM_CRITICAL"
        elif decision == "reject":
            status, override, action = "human_rejected", RouteState.NORMAL, "REJECT_NORMAL"
        else:
            raise ValueError("decision must be 'confirm' or 'reject'")

        self._store.set_verification(event_id, status, override)
        updated = current.model_copy(
            update={
                "route_state": override,
                "verification_status": status,
                "decision": current.decision.model_copy(update={"route_state": override}),
                "fused_event": current.fused_event.model_copy(update={"route_state": override}),
            }
        )
        self._store.upsert_event(updated)
        self._store.append_audit(
            event_id=event_id,
            action=action,
            operator=operator,
            notes=notes
            or (
                "Operator confirmed thermal anomaly escalation."
                if decision == "confirm"
                else "Operator verified routine operational heat signature."
            ),
            prior_route_state=prior,
            new_route_state=override,
        )
        return updated

    # -- status ------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        scenario = self._scenario
        return {
            "scenario_id": scenario.scenario_id if scenario else None,
            "scenario_title": scenario.title if scenario else None,
            "mode": (scenario.mode.value if scenario else DataMode.HISTORICAL_REPLAY.value),
            "replay_status": self._engine.status.value,
            "speed": self._engine.speed,
            "frame_index": self._engine.current_frame_index,
            "frame_count": len(scenario.frames) if scenario else 0,
            "clients": self._broadcaster.client_count,
            "model_version": self._config.versions.model_version,
            "policy_version": self._config.versions.policy_version,
        }

    # -- internals -------------------------------------------------------
    async def _run(self) -> None:
        scenario = self._require_scenario()

        async def publish(frame: Any) -> None:
            index = int(frame.frame_id.split(":")[-1])
            intelligence = self._persist_frame(index)
            await self._broadcaster.publish(
                {
                    "type": "event_update",
                    "timestamp": intelligence.timestamp,
                    "event_id": intelligence.event_id,
                    "frame_index": index,
                    "checkpoint": intelligence.checkpoint,
                    "changed": ["route_state", "risk_score"],
                    "payload": {
                        "route_state": intelligence.route_state.value,
                        "risk_score": intelligence.risk.total,
                        "class_id": intelligence.decision.class_id,
                        "recommended_action": intelligence.decision.recommended_action,
                        "mode": intelligence.mode.value,
                    },
                }
            )
            await notify_route_observers(intelligence)

        with contextlib.suppress(asyncio.CancelledError):
            await self._engine.run(publish)
            await self._broadcaster.publish(self._status_message("completed"))

    async def _publish_frame(self, index: int, reason: str) -> EventIntelligence:
        """Persist a frame outside of playback and push it to connected clients."""
        intelligence = self._persist_frame(index)
        await self._broadcaster.publish(
            {
                "type": "event_update",
                "timestamp": intelligence.timestamp,
                "event_id": intelligence.event_id,
                "frame_index": intelligence.frame_index,
                "checkpoint": intelligence.checkpoint,
                "changed": ["frame_index", "route_state", "risk_score"],
                "payload": {
                    "route_state": intelligence.route_state.value,
                    "risk_score": intelligence.risk.total,
                    "class_id": intelligence.decision.class_id,
                    "recommended_action": intelligence.decision.recommended_action,
                    "mode": intelligence.mode.value,
                },
            }
        )
        await self._broadcaster.publish(self._status_message(reason))
        await notify_route_observers(intelligence)
        return intelligence

    def _persist_frame(self, index: int) -> EventIntelligence:
        scenario = self._require_scenario()
        index = max(0, min(index, len(scenario.frames) - 1))
        event_id = f"evt-{scenario.scenario_id}"
        verification = self._store.verification(event_id)
        intelligence = process_frame(
            scenario,
            index,
            config=self._config,
            verification_status=verification.status,
            route_override=verification.route_override,
        )
        self._store.upsert_event(intelligence)
        return intelligence

    def _status_message(self, reason: str) -> dict[str, Any]:
        return {"type": "replay_status", "reason": reason, "status": self.status()}

    def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None

    def _require_scenario(self) -> ReplayScenario:
        if self._scenario is None:
            raise RuntimeError("No scenario loaded. POST /scenarios/{id}/start first.")
        return self._scenario


replay_worker = ReplayWorker()
