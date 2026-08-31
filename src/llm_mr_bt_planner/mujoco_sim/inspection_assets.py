"""Pinned source assets for the five-agent inspection scene."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

UNITREE_URL = "https://github.com/unitreerobotics/unitree_mujoco.git"
UNITREE_COMMIT = "4134cb5dc7ff1ba7f484deda48b5274b58694519"
HUSKY_URL = "https://github.com/husky/husky.git"
HUSKY_COMMIT = "41e15d283a8d955938204e79554a875264417bb9"


def default_inspection_asset_root() -> Path:
    override = os.environ.get("LMRBTP_INSPECTION_ASSETS")
    if override:
        return Path(override).expanduser().resolve()
    cache = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).resolve() if cache else Path.home().resolve() / ".cache"
    return base / "llm-mr-bt-planner" / "inspection-assets" / "v1"


def ensure_inspection_assets(root: str | Path | None = None, *, progress=print) -> Path:
    target = Path(root).expanduser().resolve() if root else default_inspection_asset_root()
    if _valid(target):
        progress(f"Inspection robot assets ready: {target}")
        return target
    if target.exists():
        raise RuntimeError(
            f"Inspection asset directory exists but is incomplete or unverified: {target}. "
            "Move it aside or choose another --assets-dir."
        )
    git = shutil.which("git")
    if not git:
        raise RuntimeError("Git is required for the first download of pinned B2 and Husky assets.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.download-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        _checkout(git, UNITREE_URL, UNITREE_COMMIT, staging / "unitree_mujoco", "unitree_robots/b2")
        _checkout(git, HUSKY_URL, HUSKY_COMMIT, staging / "husky", "husky_description")
        (staging / "LMRBTP_ASSET_PROVENANCE.json").write_text(
            json.dumps(
                {
                    "unitree": {"source": UNITREE_URL, "commit": UNITREE_COMMIT, "model": "B2"},
                    "husky": {"source": HUSKY_URL, "commit": HUSKY_COMMIT, "model": "Husky A200"},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if not _valid(target):
        raise RuntimeError(f"Downloaded inspection assets are incomplete at {target}.")
    progress(f"Inspection robot assets ready: {target}")
    return target


def _checkout(git: str, url: str, commit: str, target: Path, sparse_path: str) -> None:
    _run([git, "clone", "--filter=blob:none", "--no-checkout", url, str(target)])
    _run([git, "-C", str(target), "sparse-checkout", "set", sparse_path])
    _run([git, "-C", str(target), "fetch", "--depth", "1", "origin", commit])
    _run([git, "-C", str(target), "checkout", "--detach", commit])
    actual = _capture([git, "-C", str(target), "rev-parse", "HEAD"])
    if actual != commit:
        raise RuntimeError(f"Asset revision mismatch: expected {commit}, received {actual}.")


def _valid(root: Path) -> bool:
    required = (
        root / "unitree_mujoco" / "unitree_robots" / "b2" / "b2.xml",
        root / "husky" / "husky_description" / "meshes" / "base_link.stl",
        root / "husky" / "husky_description" / "meshes" / "wheel.stl",
        root / "LMRBTP_ASSET_PROVENANCE.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        provenance = json.loads(required[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metadata_valid = (
        provenance.get("unitree", {}).get("commit") == UNITREE_COMMIT
        and provenance.get("husky", {}).get("commit") == HUSKY_COMMIT
    )
    git = shutil.which("git")
    if not metadata_valid or not git:
        return False
    try:
        for repository, commit in (
            (root / "unitree_mujoco", UNITREE_COMMIT),
            (root / "husky", HUSKY_COMMIT),
        ):
            if _capture([git, "-C", str(repository), "rev-parse", "HEAD"]) != commit:
                return False
            if _capture([git, "-C", str(repository), "status", "--porcelain"]):
                return False
    except RuntimeError:
        return False
    return True


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())


def _capture(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()
