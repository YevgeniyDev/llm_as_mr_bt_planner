"""Deterministic, measured fault injection for the recovery experiment."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .executor import PhysicalExecutor


@dataclass(frozen=True)
class FaultTrigger:
    robot: str
    action: str
    object: str
    stage: str | None = None
    event: str | None = None
    location: str | None = None
    before_robot: str | None = None
    before_action: str | None = None


@dataclass(frozen=True)
class FaultSpec:
    schema_version: str
    fault_id: str
    fault_type: str
    recoverable: bool
    trigger: FaultTrigger
    force_newtons: tuple[float, float, float]
    duration_seconds: float
    minimum_displacement_m: float
    floor_height_threshold_m: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicFaultInjector:
    """Apply an external body force exactly once at a measured handoff boundary."""

    def __init__(self, spec: FaultSpec) -> None:
        self.spec = spec
        self.triggered = False
        self.completed = False
        self.trigger_time: float | None = None
        self.clear_time: float | None = None
        self.origin: np.ndarray | None = None
        self.trigger_evidence: dict[str, Any] = {}

    def update(self, executor: PhysicalExecutor) -> None:
        world = executor.world
        if self.completed:
            return
        if not self.triggered:
            if not self._trigger_matches(executor):
                return
            self.triggered = True
            self.trigger_time = float(world.data.time)
            self.origin = world.object_position(self.spec.trigger.object)
            executor.events.append(
                {
                    "time": round(self.trigger_time, 4),
                    "robot": "fault_injector",
                    "kind": "fault_injected",
                    "message": self.spec.fault_id,
                    "fault_type": self.spec.fault_type,
                    "object": self.spec.trigger.object,
                    "force_newtons": list(self.spec.force_newtons),
                    "seed": self.spec.seed,
                    "trigger_evidence": dict(self.trigger_evidence),
                }
            )
            executor.progress(
                f"[{self.trigger_time:7.2f}s] FAULT INJECTED — {self.spec.fault_id}"
            )

        assert self.trigger_time is not None
        elapsed = float(world.data.time) - self.trigger_time
        if elapsed <= self.spec.duration_seconds:
            world.apply_object_force(
                self.spec.trigger.object,
                np.asarray(self.spec.force_newtons, dtype=float),
            )
            return
        world.clear_object_force(self.spec.trigger.object)
        self.completed = True
        self.clear_time = float(world.data.time)

    def _trigger_matches(self, executor: PhysicalExecutor) -> bool:
        trigger = self.spec.trigger
        if trigger.event is not None:
            matching_event = next(
                (
                    event
                    for event in reversed(executor.events)
                    if event.get("robot") == trigger.robot
                    and event.get("kind") == trigger.event
                    and str(event.get("message", "")).startswith(f"{trigger.action}(")
                ),
                None,
            )
            if matching_event is None:
                return False
            placed_at_location = bool(
                trigger.location
                and executor.observe_literal(
                    f"at({trigger.object},{trigger.location})"
                )
            )
            next_robot_holding = bool(
                trigger.before_robot and executor._object_held(trigger.object)
            )
            next_action_started = bool(
                trigger.before_robot
                and trigger.before_action
                and (cursor := executor.cursors.get(trigger.before_robot)) is not None
                and cursor.action is not None
                and cursor.action.name == trigger.before_action
            )
            if trigger.location and not placed_at_location:
                raise RuntimeError(
                    f"Fault trigger event occurred before {trigger.object} was measured at "
                    f"{trigger.location}."
                )
            if next_robot_holding:
                raise RuntimeError(
                    f"Fault trigger occurred after {trigger.before_robot} had already grasped "
                    f"{trigger.object}."
                )
            self.trigger_evidence = {
                "placement_event": dict(matching_event),
                "placed_at_location": placed_at_location,
                "location": trigger.location,
                "next_robot": trigger.before_robot,
                "next_action": trigger.before_action,
                "next_action_started": next_action_started,
                "next_robot_holding_object": next_robot_holding,
            }
            return True

        cursor = executor.cursors.get(trigger.robot)
        action = cursor.action if cursor is not None else None
        matched = bool(
            action is not None
            and action.name == trigger.action
            and action.object_id == trigger.object
            and action.stage == trigger.stage
        )
        if matched:
            self.trigger_evidence = {
                "action_stage": trigger.stage,
                "next_robot_holding_object": executor._object_held(trigger.object),
            }
        return matched

    def clear(self, executor: PhysicalExecutor) -> None:
        executor.world.clear_object_force(self.spec.trigger.object)
        if self.triggered and not self.completed:
            self.completed = True
            self.clear_time = float(executor.world.data.time)

    def observation(self, executor: PhysicalExecutor) -> dict[str, Any]:
        position = executor.world.object_position(self.spec.trigger.object)
        displacement = (
            float(np.linalg.norm(position - self.origin)) if self.origin is not None else 0.0
        )
        on_floor = bool(position[2] <= self.spec.floor_height_threshold_m)
        at_recovery_location = bool(
            "source_floor" in executor.world.station_sites
            and executor.observe_literal(
                f"at({self.spec.trigger.object},source_floor)"
            )
        )
        object_usable = bool(self.spec.recoverable and on_floor and at_recovery_location)
        classification = (
            "dropped_to_floor"
            if on_floor
            else "displaced_or_missing"
            if displacement >= self.spec.minimum_displacement_m
            else "fault_not_physically_established"
        )
        return {
            "fault_id": self.spec.fault_id,
            "fault_type": self.spec.fault_type,
            "classification": classification,
            "recoverable": self.spec.recoverable,
            "object": self.spec.trigger.object,
            "object_usable": object_usable,
            "recovery_location": "source_floor" if at_recovery_location else None,
            "recovery_strategy": (
                "retrieve_same_object_from_floor" if object_usable else "no_verified_strategy"
            ),
            "trigger_time_seconds": self.trigger_time,
            "force_cleared_time_seconds": self.clear_time,
            "seed": self.spec.seed,
            "trigger_evidence": dict(self.trigger_evidence),
            "position_m": position.round(6).tolist(),
            "displacement_m": round(displacement, 6),
            "minimum_displacement_m": self.spec.minimum_displacement_m,
            "on_floor": on_floor,
            "at_recovery_location": at_recovery_location,
            "floor_height_threshold_m": self.spec.floor_height_threshold_m,
            "nominal_bt_failure": executor.failed_reason,
        }


def load_fault_spec(path: str | Path) -> FaultSpec:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Fault specification root must be a JSON object.")
    trigger = document.get("trigger")
    if not isinstance(trigger, dict):
        raise ValueError("Fault specification requires an object-valued trigger.")
    force = document.get("force_newtons")
    if not isinstance(force, list) or len(force) != 3:
        raise ValueError("Fault force_newtons must contain exactly three numbers.")
    spec = FaultSpec(
        schema_version=str(document.get("schema_version", "")),
        fault_id=str(document.get("fault_id", "")),
        fault_type=str(document.get("fault_type", "")),
        recoverable=bool(document.get("recoverable", False)),
        trigger=FaultTrigger(
            robot=str(trigger.get("robot", "")),
            action=str(trigger.get("action", "")),
            object=str(trigger.get("object", "")),
            stage=str(trigger["stage"]) if trigger.get("stage") else None,
            event=str(trigger["event"]) if trigger.get("event") else None,
            location=str(trigger["location"]) if trigger.get("location") else None,
            before_robot=(
                str(trigger["before_robot"]) if trigger.get("before_robot") else None
            ),
            before_action=(
                str(trigger["before_action"]) if trigger.get("before_action") else None
            ),
        ),
        force_newtons=(float(force[0]), float(force[1]), float(force[2])),
        duration_seconds=float(document.get("duration_seconds", 0.0)),
        minimum_displacement_m=float(document.get("minimum_displacement_m", 0.0)),
        floor_height_threshold_m=float(document.get("floor_height_threshold_m", 0.0)),
        seed=int(document.get("seed", 0)),
    )
    if spec.schema_version != "1.0":
        raise ValueError("Fault schema_version must be '1.0'.")
    if not all(
        (
            spec.fault_id,
            spec.fault_type,
            spec.trigger.robot,
            spec.trigger.action,
            spec.trigger.object,
        )
    ):
        raise ValueError("Fault identifiers and trigger fields cannot be empty.")
    if bool(spec.trigger.stage) == bool(spec.trigger.event):
        raise ValueError("Fault trigger requires exactly one of stage or event.")
    if spec.duration_seconds <= 0 or spec.minimum_displacement_m <= 0:
        raise ValueError("Fault duration and minimum displacement must be positive.")
    return spec
