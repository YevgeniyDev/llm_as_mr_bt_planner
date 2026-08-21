"""Responsive, action-directed live MuJoCo viewing for adaptive demonstrations."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import TracebackType
from typing import Any, Callable, TypeVar

import mujoco

T = TypeVar("T")


class LiveViewerSession:
    """Drive a passive MuJoCo viewer without changing simulation state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        realtime_factor: float,
    ) -> None:
        if realtime_factor <= 0:
            raise ValueError("--realtime-factor must be greater than zero.")
        self.model = model
        self.data = data
        self.realtime_factor = realtime_factor
        self._handle: Any = None
        self._camera: str | None = None
        self._status: tuple[str, str] | None = None
        self._next_step_wall = 0.0
        self._last_viewer_sync = 0.0

    def __enter__(self) -> LiveViewerSession:
        try:
            import mujoco.viewer

            self._handle = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
        except Exception as error:
            raise RuntimeError(
                "Could not open the live MuJoCo viewer. Confirm that a desktop display is "
                "available, or rerun adaptive-demo with --headless."
            ) from error
        now = time.perf_counter()
        self._next_step_wall = now
        self._last_viewer_sync = now
        self.set_camera("overview")
        self.set_status(
            "ADAPTIVE BT DEMO",
            "Preparing the MuJoCo scene\nThe recorded video uses the same action-directed cuts",
        )
        self._handle.sync()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def is_running(self) -> bool:
        return bool(self._handle is not None and self._handle.is_running())

    def set_camera(self, camera: str) -> None:
        """Apply a fixed action camera only when the directed shot changes."""
        self._require_running()
        if camera == self._camera:
            return
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id < 0:
            raise ValueError(f"Live-view camera {camera!r} does not exist in the MuJoCo model.")
        with self._handle.lock():
            self._handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._handle.cam.fixedcamid = camera_id
        self._camera = camera

    def set_status(self, title: str, detail: str) -> None:
        """Show concise live state in MuJoCo's native top-left text overlay."""
        self._require_running()
        status = (title, detail)
        if status == self._status:
            return
        self._handle.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                title,
                detail,
            )
        )
        self._status = status

    def after_step(self, *, camera: str, phase: str, detail: str) -> None:
        """Pace physics in wall time and periodically synchronize the live window."""
        self._require_running()
        camera_changed = camera != self._camera
        self.set_camera(camera)
        self.set_status(phase, f"{detail}\nCamera: {camera}")

        timestep = float(self.model.opt.timestep)
        self._next_step_wall += timestep / self.realtime_factor
        now = time.perf_counter()
        remaining = self._next_step_wall - now
        if remaining > 0:
            time.sleep(remaining)
            now = time.perf_counter()
        elif -remaining > 0.25:
            # Do not accelerate after rendering or another operation temporarily falls behind.
            self._next_step_wall = now

        if camera_changed or now - self._last_viewer_sync >= 1.0 / 60.0:
            self._handle.sync()
            self._last_viewer_sync = time.perf_counter()

    def run_while_frozen(
        self,
        function: Callable[[], T],
        *,
        camera: str,
        title: str,
        message: str,
    ) -> T:
        """Run blocking replanning off-thread while keeping the frozen viewer responsive."""
        self._require_running()
        self.set_camera(camera)
        self.set_status(
            title,
            f"{message}\nSimulation paused; state preserved\nElapsed: 0s",
        )
        self.refresh()
        started = time.perf_counter()
        next_overlay_update = 1.0
        viewer_closed = False
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="bt-replanning") as pool:
            future = pool.submit(function)
            while not future.done():
                if not self.is_running:
                    viewer_closed = True
                else:
                    elapsed = time.perf_counter() - started
                    if elapsed >= next_overlay_update:
                        self.set_status(
                            title,
                            f"{message}\nSimulation paused; state preserved\nElapsed: {elapsed:.0f}s",
                        )
                        next_overlay_update = elapsed + 1.0
                    self._handle.sync()
                try:
                    future.result(timeout=1.0 / 30.0)
                except TimeoutError:
                    pass
            result = future.result()

        if viewer_closed or not self.is_running:
            raise RuntimeError("The MuJoCo viewer was closed while the LLM was adapting the BT.")
        self.reset_pacing()
        return result

    def refresh(self) -> None:
        """Synchronize viewer state without advancing MuJoCo physics."""
        self._require_running()
        self._handle.sync()
        self._last_viewer_sync = time.perf_counter()

    def hold_terminal_state(self, seconds: float = 2.0) -> None:
        """Keep the terminal pose visible briefly before closing the viewer."""
        if not self.is_running:
            return
        deadline = time.perf_counter() + max(seconds, 0.0)
        while self.is_running and time.perf_counter() < deadline:
            self.refresh()
            time.sleep(1.0 / 30.0)

    def reset_pacing(self) -> None:
        now = time.perf_counter()
        self._next_step_wall = now
        self._last_viewer_sync = now

    def _require_running(self) -> None:
        if not self.is_running:
            raise RuntimeError("The MuJoCo viewer was closed before BT execution completed.")
