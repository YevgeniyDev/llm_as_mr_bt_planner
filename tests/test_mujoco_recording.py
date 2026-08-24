# ruff: noqa: E402 -- optional module-level skip must precede simulator imports.

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from llm_mr_bt_planner.cli import _build_parser
from llm_mr_bt_planner.mujoco_sim import recording
from llm_mr_bt_planner.mujoco_sim.camera_director import camera_director_for_task
from llm_mr_bt_planner.mujoco_sim.recording import RecordingConfig, SimulationVideoRecorder
from llm_mr_bt_planner.mujoco_sim.runner import (
    _default_recording_camera,
    _recording_camera_decision,
    _validate_recording_cli_args,
    _write_recording_manifest,
)


class _Data:
    def __init__(self, time: float = 0.0) -> None:
        self.time = time


class _FakeRenderer:
    def __init__(self, _model, *, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.cameras: list[str] = []
        self.closed = False

    def update_scene(self, _data, *, camera: str) -> None:
        self.cameras.append(camera)

    def render(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class _FakeWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.frames: list[np.ndarray] = []
        self.closed = False
        path.write_bytes(b"fake-mp4")

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def recording_model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002"/>
          <visual><global offwidth="64" offheight="64"/></visual>
          <worldbody>
            <camera name="overview" pos="0 -2 1" xyaxes="1 0 0 0 0.5 1"/>
            <camera name="detail" pos="0 -1 1" xyaxes="1 0 0 0 0.5 1"/>
            <geom type="sphere" size="0.1"/>
          </worldbody>
        </mujoco>
        """
    )


@pytest.fixture
def fake_encoder(monkeypatch):
    state: dict[str, object] = {}

    def open_writer(path: Path, _config: RecordingConfig) -> _FakeWriter:
        writer = _FakeWriter(path)
        state["writer"] = writer
        return writer

    def open_renderer(model, *, height: int, width: int) -> _FakeRenderer:
        renderer = _FakeRenderer(model, height=height, width=width)
        state["renderer"] = renderer
        return renderer

    monkeypatch.setattr(recording, "_open_video_writer", open_writer)
    monkeypatch.setattr(recording.mujoco, "Renderer", open_renderer)
    return state


def test_simulation_time_scheduler_captures_31_frames_for_one_second(
    tmp_path: Path,
    recording_model,
    fake_encoder,
):
    path = tmp_path / "simulation.mp4"
    data = _Data()
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(path=path, fps=30, width=64, height=64),
    )

    with recorder:
        recorder.capture_initial(data)
        for step in range(1, 501):
            data.time = step * 0.002
            recorder.capture_after_step(data)
        recorder.finish(data)

    assert recorder.metadata.frame_count == 31
    assert recorder.metadata.start_sim_time == 0.0
    assert recorder.metadata.end_sim_time == 1.0
    assert recorder.metadata.simulated_duration_seconds == 1.0
    assert recorder.metadata.final_frame_forced is False
    assert path.is_file()
    assert not (tmp_path / "simulation.partial.mp4").exists()
    assert len(fake_encoder["writer"].frames) == 31
    assert fake_encoder["renderer"].cameras == ["overview"] * 31
    assert fake_encoder["writer"].closed is True
    assert fake_encoder["renderer"].closed is True


def test_finish_forces_one_terminal_frame_when_completion_is_between_deadlines(
    tmp_path: Path,
    recording_model,
    fake_encoder,
):
    data = _Data()
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(path=tmp_path / "simulation.mp4", fps=30, width=64, height=64),
    )

    with recorder:
        recorder.capture_initial(data)
        for step in range(1, 476):
            data.time = step * 0.002
            recorder.capture_after_step(data)
        recorder.finish(data)

    assert recorder.metadata.end_sim_time == 0.95
    assert recorder.metadata.frame_count == 30
    assert recorder.metadata.final_frame_forced is True


def test_action_directed_recording_captures_camera_cut_timeline(
    tmp_path: Path,
    recording_model,
    fake_encoder,
):
    data = _Data()
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(
            path=tmp_path / "simulation.mp4",
            fps=30,
            width=64,
            height=64,
            camera="overview",
            camera_mode="action_directed",
            camera_sequence=("overview", "detail"),
        ),
    )

    with recorder:
        recorder.capture_initial(data, camera="overview", reason="initial_overview")
        for step in range(1, 36):
            data.time = step * 0.002
            recorder.capture_after_step(data, camera="detail", reason="action:pick_source")
        recorder.finish(data, camera="detail", reason="action:pick_source")

    assert fake_encoder["renderer"].cameras == ["overview", "detail", "detail", "detail"]
    assert recorder.metadata.camera == "overview"
    assert recorder.metadata.camera_mode == "action_directed"
    assert recorder.metadata.cameras_used == ["overview", "detail"]
    assert recorder.metadata.camera_cut_count == 1
    assert recorder.metadata.camera_cuts == [
        {
            "frame_index": 1,
            "sim_time_seconds": 0.034,
            "from_camera": "overview",
            "to_camera": "detail",
            "reason": "action:pick_source",
        }
    ]


def test_status_overlay_freezes_simulation_and_records_auditable_timing(
    tmp_path: Path,
    recording_model,
    fake_encoder,
):
    data = _Data(time=2.5)
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(
            path=tmp_path / "simulation.mp4",
            fps=10,
            width=64,
            height=64,
            camera="overview",
            camera_mode="action_directed",
            camera_sequence=("overview", "detail"),
        ),
    )

    with recorder:
        recorder.capture_initial(data, camera="overview", reason="initial_overview")
        recorder.append_status_overlay(
            data,
            title="FAILURE DETECTED",
            message="LLM IS ADAPTING THE BEHAVIOR TREE...",
            detail="primary_part fell after placement; the simulator state is preserved.",
            duration_seconds=0.5,
            wall_seconds=12.3456,
            camera="detail",
        )
        recorder.finish(data, camera="detail", reason="resume")

    writer = fake_encoder["writer"]
    assert len(writer.frames) == 6
    assert data.time == 2.5
    assert np.count_nonzero(writer.frames[1]) > 0
    assert recorder.metadata.frame_count == 6
    assert recorder.metadata.simulated_duration_seconds == 0.0
    assert recorder.metadata.encoded_duration_seconds == 0.6
    assert recorder.metadata.status_overlays == [
        {
            "start_frame_index": 1,
            "frame_count": 5,
            "encoded_duration_seconds": 0.5,
            "simulation_time_seconds": 2.5,
            "wall_seconds_represented": 12.3456,
            "title": "FAILURE DETECTED",
            "message": "LLM IS ADAPTING THE BEHAVIOR TREE...",
            "detail": "primary_part fell after placement; the simulator state is preserved.",
            "camera": "detail",
        }
    ]


def test_capture_rejects_camera_outside_configured_sequence(
    tmp_path: Path,
    recording_model,
    fake_encoder,
):
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(path=tmp_path / "simulation.mp4", fps=30, width=64, height=64),
    )

    with pytest.raises(ValueError, match="configured camera sequence"):
        with recorder:
            recorder.capture_initial(_Data(), camera="detail")


def test_exception_closes_resources_and_retains_only_partial_video(
    tmp_path: Path,
    recording_model,
    fake_encoder,
):
    final_path = tmp_path / "simulation.mp4"
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(path=final_path, fps=30, width=64, height=64),
    )

    with pytest.raises(RuntimeError, match="simulation failed"):
        with recorder:
            recorder.capture_initial(_Data())
            raise RuntimeError("simulation failed")

    assert not final_path.exists()
    assert (tmp_path / "simulation.partial.mp4").is_file()
    assert fake_encoder["writer"].closed is True
    assert fake_encoder["renderer"].closed is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fps": 0}, "--video-fps"),
        ({"fps": 501}, "physics rate"),
        ({"width": 63}, "--video-width"),
        ({"height": 65}, "--video-height"),
        ({"width": 66}, "framebuffer width"),
        ({"height": 66}, "framebuffer height"),
        ({"camera": "missing"}, "Unknown MuJoCo recording camera"),
    ],
)
def test_invalid_recording_configuration_is_rejected(
    tmp_path: Path,
    recording_model,
    changes: dict,
    message: str,
):
    values = {"path": tmp_path / "simulation.mp4", "fps": 30, "width": 64, "height": 64}
    values.update(changes)
    recorder = SimulationVideoRecorder(recording_model, RecordingConfig(**values))

    with pytest.raises(ValueError, match=message):
        with recorder:
            pass


@pytest.mark.parametrize("name", ["simulation.mp4", "simulation.partial.mp4"])
def test_recorder_never_overwrites_existing_video_files(
    tmp_path: Path,
    recording_model,
    name: str,
):
    (tmp_path / name).write_bytes(b"existing")
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(path=tmp_path / "simulation.mp4", fps=30, width=64, height=64),
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        with recorder:
            pass


def test_recording_cli_defaults_are_deferred_until_recording_starts():
    args = _build_parser().parse_args(["mujoco", "--record-video", "--headless"])

    _validate_recording_cli_args(args)

    assert args.record_video is True
    assert args.video_fps is None
    assert args.video_width is None
    assert args.video_height is None
    assert args.video_camera is None


def test_publication_camera_is_selected_per_mission():
    assert _default_recording_camera("three_robot_courier") == "overview"
    assert _default_recording_camera("three_robot_packaging_delivery") == "packaging_recording"
    assert _default_recording_camera("three_robot_component_installation") == "overview"


def test_fixed_camera_override_bypasses_action_direction():
    executor = SimpleNamespace(
        events=[
            {
                "kind": "action_started",
                "robot": "franka_a",
                "message": "pick_source(payload)",
            }
        ]
    )
    recorder = SimpleNamespace(config=SimpleNamespace(camera="overview"))

    decision = _recording_camera_decision(executor, recorder, camera_director=None)

    assert decision.camera == "overview"
    assert decision.reason == "fixed_camera_override"


def test_courier_camera_program_follows_latest_active_action_and_holds_between_actions():
    director = camera_director_for_task("three_robot_courier")
    events: list[dict[str, str]] = []

    assert director.update(events).camera == "overview"
    events.append(
        {"kind": "action_started", "robot": "franka_a", "message": "pick_source(payload)"}
    )
    assert director.update(events).camera == "courier_source"
    events.append(
        {"kind": "action_success", "robot": "franka_a", "message": "pick_source(payload)"}
    )
    assert director.update(events).camera == "courier_source"
    events.append(
        {"kind": "action_started", "robot": "unitree_go2_z1", "message": "navigate_destination()"}
    )
    assert director.update(events).camera == "courier_route"
    events.append(
        {
            "kind": "action_started",
            "robot": "franka_b",
            "message": "place_destination_cradle(payload)",
        }
    )
    decision = director.update(events)
    assert decision.camera == "courier_destination"
    assert decision.reason == "action:place_destination_cradle"


def test_packaging_camera_program_covers_assembly_door_route_and_delivery():
    director = camera_director_for_task("three_robot_packaging_delivery")
    events: list[dict[str, str]] = []

    expected = (
        ("pick_loaded_package_base()", "packaging_assembly"),
        ("push_open_door_and_cross()", "packaging_door"),
        ("navigate_delivery_room()", "packaging_route"),
        ("place_parcel_at_delivery_station()", "packaging_delivery"),
    )
    for index, (action, camera) in enumerate(expected):
        robot = f"robot_{index}"
        events.append({"kind": "action_started", "robot": robot, "message": action})
        assert director.update(events).camera == camera
        events.append({"kind": "action_success", "robot": robot, "message": action})
        assert director.update(events).camera == camera


def test_recovery_camera_program_covers_fault_handoff_route_and_installation():
    director = camera_director_for_task("three_robot_component_installation")
    events: list[dict[str, str]] = []
    expected = (
        ("recover_fallen_part(primary_part,source_floor)", "recovery_floor"),
        ("navigate_destination(primary_part)", "recovery_route"),
        ("install_target(primary_part,target_fixture)", "recovery_destination"),
    )
    for index, (action, camera) in enumerate(expected):
        robot = f"robot_{index}"
        events.append({"kind": "action_started", "robot": robot, "message": action})
        assert director.update(events).camera == camera
        events.append({"kind": "action_success", "robot": robot, "message": action})
        assert director.update(events).camera == camera


def test_video_options_require_record_video():
    args = _build_parser().parse_args(["mujoco", "--video-fps", "24"])

    with pytest.raises(ValueError, match="require --record-video"):
        _validate_recording_cli_args(args)


def test_setup_only_rejects_video_recording():
    args = _build_parser().parse_args(["mujoco", "--setup-only", "--record-video"])

    with pytest.raises(ValueError, match="cannot be combined"):
        _validate_recording_cli_args(args)


@pytest.mark.parametrize(
    ("option", "value"),
    [("--video-fps", "0"), ("--video-width", "1279"), ("--video-height", "-2")],
)
def test_basic_video_settings_fail_before_simulator_setup(option: str, value: str):
    args = _build_parser().parse_args(["mujoco", "--record-video", option, value])

    with pytest.raises(ValueError, match=option):
        _validate_recording_cli_args(args)


def test_recording_manifest_hashes_every_portable_artifact(tmp_path: Path):
    contents = {
        "simulation.mp4": b"video",
        "scenario.json": b"scenario",
        "behavior_tree.json": b"tree",
        "physical_execution_report.json": b"report",
    }
    for name, content in contents.items():
        (tmp_path / name).write_bytes(content)
    report = SimpleNamespace(
        success=True,
        simulated_seconds=1.0,
        physics={"timestep_seconds": 0.002},
    )
    manifest_path = tmp_path / "recording_manifest.json"

    _write_recording_manifest(
        manifest_path,
        task_id="test_task",
        report=report,
        recording={"file": "simulation.mp4", "frame_count": 31},
        output_dir=tmp_path,
        command=["lmrbtp", "mujoco", "--record-video"],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == "1.0"
    assert manifest["task_id"] == "test_task"
    assert manifest["run_success"] is True
    assert manifest["command"] == ["lmrbtp", "mujoco", "--record-video"]
    assert set(manifest["files"]) == set(contents)
    assert all(len(digest) == 64 for digest in manifest["files"].values())


@pytest.mark.skipif(
    os.environ.get("LMRBTP_RUN_MUJOCO_VIDEO") != "1",
    reason="Set LMRBTP_RUN_MUJOCO_VIDEO=1 to run an actual off-screen MP4 encoding test.",
)
def test_real_offscreen_mp4_encoding(tmp_path: Path, recording_model):
    path = tmp_path / "simulation.mp4"
    data = mujoco.MjData(recording_model)
    recorder = SimulationVideoRecorder(
        recording_model,
        RecordingConfig(path=path, fps=30, width=64, height=64),
    )

    with recorder:
        mujoco.mj_forward(recording_model, data)
        recorder.capture_initial(data)
        for _ in range(100):
            mujoco.mj_step(recording_model, data)
            recorder.capture_after_step(data)
        recorder.finish(data)

    assert path.stat().st_size > 100
    assert recorder.metadata.frame_count == 7
    assert recorder.metadata.codec == "libx264"
    assert recorder.metadata.pixel_format == "yuv420p"
    assert recorder.metadata.crf == 18
    assert recorder.metadata.preset == "medium"
    assert recorder.metadata.faststart is True
