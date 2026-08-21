# ruff: noqa: E402 -- optional module-level skip must precede simulator imports.

from __future__ import annotations

import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

mujoco = pytest.importorskip("mujoco")

from llm_mr_bt_planner.mujoco_sim.live_viewer import LiveViewerSession


class _FakeViewerHandle:
    def __init__(self) -> None:
        self.running = True
        self.cam = SimpleNamespace(type=None, fixedcamid=-1)
        self.texts: list[tuple[object, object, str, str]] = []
        self.sync_count = 0

    def is_running(self) -> bool:
        return self.running

    def lock(self):
        return nullcontext()

    def set_texts(self, texts) -> None:
        self.texts.append(texts)

    def sync(self) -> None:
        self.sync_count += 1

    def close(self) -> None:
        self.running = False


def _session(monkeypatch: pytest.MonkeyPatch) -> tuple[LiveViewerSession, _FakeViewerHandle]:
    model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
    data = SimpleNamespace(time=12.5)
    session = LiveViewerSession(model, data, realtime_factor=1.0)
    handle = _FakeViewerHandle()
    session._handle = handle
    monkeypatch.setattr(mujoco, "mj_name2id", lambda *_args: 7)
    return session, handle


def test_live_viewer_applies_fixed_camera_and_native_status_overlay(monkeypatch):
    session, handle = _session(monkeypatch)

    session.set_camera("recovery_floor")
    session.set_status("FAILURE DETECTED", "LLM adaptation in progress")
    session.refresh()

    assert handle.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert handle.cam.fixedcamid == 7
    assert handle.texts[-1][2:] == (
        "FAILURE DETECTED",
        "LLM adaptation in progress",
    )
    assert handle.sync_count == 1


def test_live_viewer_keeps_refreshing_without_advancing_physics_during_replanning(
    monkeypatch,
):
    session, handle = _session(monkeypatch)
    initial_simulation_time = session.data.time

    def replan() -> str:
        time.sleep(0.08)
        return "adapted-tree"

    result = session.run_while_frozen(
        replan,
        camera="recovery_floor",
        title="FAILURE DETECTED - BT ADAPTATION IN PROGRESS",
        message="The LLM is generating a recovery tree",
    )

    assert result == "adapted-tree"
    assert session.data.time == initial_simulation_time
    assert handle.sync_count >= 2
    assert "Simulation paused; state preserved" in handle.texts[0][3]
