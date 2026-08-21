"""Action-directed, deterministic camera selection for publication videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraDecision:
    camera: str
    reason: str


@dataclass(frozen=True)
class CameraProgram:
    fallback: str
    action_cameras: dict[str, str]

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.fallback, *self.action_cameras.values())))


CAMERA_PROGRAMS: dict[str, CameraProgram] = {
    "three_robot_courier": CameraProgram(
        fallback="overview",
        action_cameras={
            "pick_source": "courier_source",
            "place_source_cradle": "courier_source",
            "pick_source_cradle": "courier_source",
            "navigate_destination": "courier_route",
            "place_destination_cradle": "courier_destination",
            "stow_arm_destination": "courier_destination",
            "pick_destination_cradle": "courier_destination",
            "install_target": "courier_destination",
        },
    ),
    "three_robot_packaging_delivery": CameraProgram(
        fallback="packaging_recording",
        action_cameras={
            "verify_delivery_readiness": "packaging_assembly",
            "pick_loaded_package_base": "packaging_assembly",
            "place_base_at_packing_station": "packaging_assembly",
            "pick_package_lid": "packaging_assembly",
            "fit_and_seal_package_lid": "packaging_assembly",
            "pick_sealed_parcel": "packaging_assembly",
            "approach_closed_room_door": "packaging_door",
            "push_open_door_and_cross": "packaging_door",
            "cross_already_open_door": "packaging_door",
            "navigate_delivery_room": "packaging_route",
            "place_parcel_at_delivery_station": "packaging_delivery",
            "stow_after_delivery": "packaging_delivery",
        },
    ),
    "three_robot_spare_part_recovery": CameraProgram(
        fallback="overview",
        action_cameras={
            "pick_source_part": "recovery_source",
            "place_source_cradle": "recovery_source",
            "pick_source_cradle": "recovery_source",
            "navigate_destination": "recovery_route",
            "place_destination_cradle": "recovery_destination",
            "stow_arm_destination": "recovery_destination",
            "pick_destination_cradle": "recovery_destination",
            "install_target": "recovery_destination",
        },
    ),
}


class ActionCameraDirector:
    """Follow newly started physical actions without inspecting or changing physics."""

    def __init__(self, program: CameraProgram) -> None:
        self.program = program
        self._event_index = 0
        self._sequence = 0
        self._active_actions: dict[str, tuple[str, int]] = {}
        self._decision = CameraDecision(program.fallback, "initial_overview")

    @property
    def cameras(self) -> tuple[str, ...]:
        return self.program.cameras

    @property
    def decision(self) -> CameraDecision:
        return self._decision

    def update(self, events: list[dict[str, Any]]) -> CameraDecision:
        if len(events) < self._event_index:
            raise RuntimeError("Physical action history was truncated during camera direction.")
        for event in events[self._event_index :]:
            kind = event.get("kind")
            robot = str(event.get("robot", ""))
            action = _action_name(str(event.get("message", "")))
            if kind == "action_started":
                self._sequence += 1
                self._active_actions[robot] = (action, self._sequence)
            elif kind in {"action_success", "action_failure"}:
                active = self._active_actions.get(robot)
                if active is not None and active[0] == action:
                    del self._active_actions[robot]
        self._event_index = len(events)

        candidates = sorted(self._active_actions.values(), key=lambda item: item[1], reverse=True)
        for action, _ in candidates:
            camera = self.program.action_cameras.get(action)
            if camera is not None:
                self._decision = CameraDecision(camera, f"action:{action}")
                break
        return self._decision


def camera_director_for_task(task_id: str) -> ActionCameraDirector:
    try:
        return ActionCameraDirector(CAMERA_PROGRAMS[task_id])
    except KeyError as error:
        raise ValueError(f"No automatic recording camera program exists for task_id {task_id!r}.") from error


def _action_name(label: str) -> str:
    return label.partition("(")[0].strip()
