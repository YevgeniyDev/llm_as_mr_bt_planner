"""Deterministic, simulation-time video recording for MuJoCo execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

DEFAULT_VIDEO_FPS = 30
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_VIDEO_CAMERA = "overview"
VIDEO_CODEC = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_CRF = 18
VIDEO_PRESET = "medium"


@dataclass(frozen=True)
class RecordingConfig:
    """Stable encoding and camera settings for one simulation recording."""

    path: Path
    fps: int = DEFAULT_VIDEO_FPS
    width: int = DEFAULT_VIDEO_WIDTH
    height: int = DEFAULT_VIDEO_HEIGHT
    camera: str = DEFAULT_VIDEO_CAMERA
    camera_mode: str = "fixed"
    camera_sequence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordingMetadata:
    """Measured properties of a finalized recording."""

    file: str
    fps: int
    width: int
    height: int
    camera: str
    camera_mode: str
    cameras_used: list[str]
    camera_cut_count: int
    camera_cuts: list[dict[str, Any]]
    codec: str
    pixel_format: str
    crf: int
    preset: str
    faststart: bool
    frame_count: int
    start_sim_time: float
    end_sim_time: float
    simulated_duration_seconds: float
    encoded_duration_seconds: float
    includes_settling: bool
    final_frame_forced: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimulationVideoRecorder:
    """Observe one ``MjData`` stream and encode frames without advancing physics."""

    def __init__(self, model: mujoco.MjModel, config: RecordingConfig) -> None:
        self.model = model
        self.config = config
        self.timestep = float(model.opt.timestep)
        self.partial_path = config.path.with_name(f"{config.path.stem}.partial{config.path.suffix}")
        self._renderer: Any | None = None
        self._writer: Any | None = None
        self._start_sim_time: float | None = None
        self._end_sim_time: float | None = None
        self._last_observed_sim_time: float | None = None
        self._last_frame_sim_time: float | None = None
        self._next_frame_index = 0
        self._frame_count = 0
        self._final_frame_forced = False
        self._finished = False
        self._last_camera: str | None = None
        self._cameras_used: list[str] = []
        self._camera_cuts: list[dict[str, Any]] = []
        self._allowed_cameras = frozenset((config.camera, *config.camera_sequence))

    def __enter__(self) -> SimulationVideoRecorder:
        self._validate()
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        self._renderer = mujoco.Renderer(
            self.model,
            height=self.config.height,
            width=self.config.width,
        )
        try:
            self._writer = _open_video_writer(self.partial_path, self.config)
        except Exception:
            self._renderer.close()
            self._renderer = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._finished:
            return
        self._close_resources()
        if exc_type is None:
            raise RuntimeError("Video recorder left its context without finish(data).")

    @property
    def metadata(self) -> RecordingMetadata:
        if not self._finished or self._start_sim_time is None or self._end_sim_time is None:
            raise RuntimeError("Recording metadata is available only after the video is finalized.")
        return RecordingMetadata(
            file=self.config.path.name,
            fps=self.config.fps,
            width=self.config.width,
            height=self.config.height,
            camera=self.config.camera,
            camera_mode=self.config.camera_mode,
            cameras_used=list(self._cameras_used),
            camera_cut_count=len(self._camera_cuts),
            camera_cuts=list(self._camera_cuts),
            codec=VIDEO_CODEC,
            pixel_format=VIDEO_PIXEL_FORMAT,
            crf=VIDEO_CRF,
            preset=VIDEO_PRESET,
            faststart=True,
            frame_count=self._frame_count,
            start_sim_time=round(self._start_sim_time, 6),
            end_sim_time=round(self._end_sim_time, 6),
            simulated_duration_seconds=round(self._end_sim_time - self._start_sim_time, 6),
            encoded_duration_seconds=round(self._frame_count / self.config.fps, 6),
            includes_settling=True,
            final_frame_forced=self._final_frame_forced,
        )

    def capture_initial(
        self,
        data: mujoco.MjData,
        *,
        camera: str | None = None,
        reason: str = "initial",
    ) -> None:
        """Capture frame zero before the first physics step."""
        self._require_open()
        if self._start_sim_time is not None:
            raise RuntimeError("The initial video frame was already captured.")
        self._start_sim_time = float(data.time)
        self._last_observed_sim_time = float(data.time)
        self._capture(data, camera=camera, reason=reason)
        self._next_frame_index = 1

    def capture_after_step(
        self,
        data: mujoco.MjData,
        *,
        camera: str | None = None,
        reason: str = "scheduled",
    ) -> None:
        """Capture frames whose simulation-time deadlines were crossed by a step."""
        self._require_open()
        if self._start_sim_time is None:
            raise RuntimeError("capture_initial(data) must be called before capture_after_step(data).")

        sim_time = float(data.time)
        if (
            self._last_observed_sim_time is not None
            and sim_time < self._last_observed_sim_time - self._time_tolerance
        ):
            raise RuntimeError("MuJoCo simulation time moved backwards during recording.")
        self._last_observed_sim_time = sim_time

        target_time = self._start_sim_time + self._next_frame_index / self.config.fps
        while sim_time + self._time_tolerance >= target_time:
            self._capture(data, camera=camera, reason=reason)
            self._next_frame_index += 1
            target_time = self._start_sim_time + self._next_frame_index / self.config.fps

    def finish(
        self,
        data: mujoco.MjData,
        *,
        camera: str | None = None,
        reason: str = "terminal",
    ) -> None:
        """Capture the terminal state if needed, close the encoder, and publish the MP4."""
        self._require_open()
        if self._start_sim_time is None:
            raise RuntimeError("Cannot finalize a recording before its initial frame.")

        self._end_sim_time = float(data.time)
        selected_camera = self.config.camera if camera is None else camera
        if (
            self._last_frame_sim_time is None
            or self._end_sim_time - self._last_frame_sim_time > self._time_tolerance
            or selected_camera != self._last_camera
        ):
            self._capture(data, camera=selected_camera, reason=reason)
            self._final_frame_forced = True

        self._close_resources()
        self.partial_path.replace(self.config.path)
        self._finished = True

    @property
    def _time_tolerance(self) -> float:
        return self.timestep / 2.0 + 1e-12

    def _capture(
        self,
        data: mujoco.MjData,
        *,
        camera: str | None,
        reason: str,
    ) -> None:
        assert self._renderer is not None
        assert self._writer is not None
        selected_camera = self.config.camera if camera is None else camera
        if selected_camera not in self._allowed_cameras:
            allowed = ", ".join(sorted(self._allowed_cameras))
            raise ValueError(
                f"Camera {selected_camera!r} is not in this recording's configured camera "
                f"sequence ({allowed})."
            )
        if selected_camera not in self._cameras_used:
            self._cameras_used.append(selected_camera)
        if self._last_camera is not None and selected_camera != self._last_camera:
            self._camera_cuts.append(
                {
                    "frame_index": self._frame_count,
                    "sim_time_seconds": round(float(data.time), 6),
                    "from_camera": self._last_camera,
                    "to_camera": selected_camera,
                    "reason": reason,
                }
            )
        self._renderer.update_scene(data, camera=selected_camera)
        self._writer.append_data(self._renderer.render())
        self._frame_count += 1
        self._last_frame_sim_time = float(data.time)
        self._last_camera = selected_camera

    def _validate(self) -> None:
        if self.config.path.exists():
            raise FileExistsError(f"Refusing to overwrite existing video: {self.config.path}")
        if self.partial_path.exists():
            raise FileExistsError(f"Refusing to overwrite partial video: {self.partial_path}")
        if self.config.fps <= 0:
            raise ValueError("--video-fps must be greater than zero.")
        if self.config.camera_mode not in {"fixed", "action_directed"}:
            raise ValueError("Recording camera_mode must be 'fixed' or 'action_directed'.")
        simulation_hz = 1.0 / self.timestep
        if self.config.fps > simulation_hz + 1e-9:
            raise ValueError(
                f"--video-fps cannot exceed the {simulation_hz:g} Hz MuJoCo physics rate."
            )
        for option, value in (
            ("--video-width", self.config.width),
            ("--video-height", self.config.height),
        ):
            if value <= 0 or value % 2:
                raise ValueError(f"{option} must be a positive even integer.")
        if self.config.width > int(self.model.vis.global_.offwidth):
            raise ValueError(
                f"--video-width {self.config.width} exceeds the MuJoCo off-screen framebuffer width "
                f"{int(self.model.vis.global_.offwidth)}."
            )
        if self.config.height > int(self.model.vis.global_.offheight):
            raise ValueError(
                f"--video-height {self.config.height} exceeds the MuJoCo off-screen framebuffer height "
                f"{int(self.model.vis.global_.offheight)}."
            )
        for camera in self._allowed_cameras:
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera,
            )
            if camera_id < 0:
                names = [self.model.camera(index).name for index in range(self.model.ncam)]
                available = ", ".join(name for name in names if name) or "none"
                raise ValueError(
                    f"Unknown MuJoCo recording camera {camera!r}; available cameras: {available}."
                )

    def _require_open(self) -> None:
        if self._renderer is None or self._writer is None or self._finished:
            raise RuntimeError("The video recorder is not open.")

    def _close_resources(self) -> None:
        writer, renderer = self._writer, self._renderer
        self._writer = None
        self._renderer = None
        try:
            if writer is not None:
                writer.close()
        finally:
            if renderer is not None:
                renderer.close()


def _open_video_writer(path: Path, config: RecordingConfig):
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError(
            "Video recording dependencies are missing. Install them with "
            'python -m pip install -e ".[mujoco]".'
        ) from error

    return imageio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=config.fps,
        codec=VIDEO_CODEC,
        pixelformat=VIDEO_PIXEL_FORMAT,
        macro_block_size=2,
        output_params=[
            "-crf",
            str(VIDEO_CRF),
            "-preset",
            VIDEO_PRESET,
            "-movflags",
            "+faststart",
        ],
    )
