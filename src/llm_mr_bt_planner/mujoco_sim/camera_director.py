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
    "five_agent_solar_pipe_inspection": CameraProgram(
        fallback="inspection_overview",
        action_cameras={
            "prepare_inspection_kit": "inspection_handoff",
            "place_inspection_kit_handoff": "inspection_handoff",
            "load_inspection_kit": "inspection_handoff",
            "install_thermal_reference": "inspection_service",
            "navigate_b2_solar_view": "inspection_convoy",
            "navigate_b2_pipe_view": "inspection_convoy",
            "return_b2_home": "inspection_convoy",
            "navigate_husky_reference_dock": "inspection_convoy",
            "navigate_husky_anomaly_dock": "inspection_convoy",
            "return_husky_home": "inspection_convoy",
            "deploy_camera_solar": "inspection_solar",
            "scan_solar_panel": "inspection_solar",
            "stow_z1_after_solar": "inspection_solar",
            "deploy_camera_pipe": "inspection_pipe",
            "scan_pipe_rig": "inspection_pipe",
            "stow_z1_after_pipe": "inspection_pipe",
            "confirm_reported_anomaly": "inspection_service",
            "attach_inspection_marker": "inspection_service",
            "isolate_energy_rig": "inspection_service",
            "deploy_camera_verification": "inspection_pipe",
            "verify_isolation_thermal": "inspection_pipe",
            "stow_z1_after_verification": "inspection_pipe",
            "navigate_b2_tool_search": "inspection_search",
            "deploy_camera_tool_search": "inspection_search",
            "localize_fallen_tool": "inspection_search",
            "stow_z1_after_tool_search": "inspection_search",
            "navigate_husky_tool_recovery": "inspection_convoy",
            "recover_localized_tool": "inspection_floor_recovery",
            "navigate_husky_recovery_to_reference": "inspection_convoy",
            "navigate_b2_search_to_solar": "inspection_convoy",
        },
    ),
    "five_agent_pipe_leak_repair": CameraProgram(
        fallback="inspection_overview",
        action_cameras={
            "prepare_repair_tool": "inspection_handoff",
            "place_repair_tool_handoff": "inspection_handoff",
            "load_repair_tool": "inspection_handoff",
            "navigate_b2_pipe_inspection": "inspection_convoy",
            "return_b2_home_after_repair": "inspection_convoy",
            "navigate_husky_leak_repair_dock": "inspection_convoy",
            "return_husky_home_after_repair": "inspection_convoy",
            "deploy_camera_pipe_leak": "inspection_pipe",
            "detect_pipe_leak": "inspection_pipe",
            "stow_z1_after_leak_detection": "inspection_pipe",
            "repair_pipe_leak": "inspection_service",
            "deploy_camera_repair_verification": "inspection_pipe",
            "verify_pipe_repair": "inspection_pipe",
            "stow_z1_after_repair_verification": "inspection_pipe",
        },
    ),
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
    "three_robot_component_installation": CameraProgram(
        fallback="overview",
        action_cameras={
            "pick_source_part": "recovery_source",
            "place_source_cradle": "recovery_source",
            "pick_source_cradle": "recovery_source",
            "recover_fallen_part": "recovery_floor",
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
