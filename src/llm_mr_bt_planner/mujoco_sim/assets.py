"""Pinned, auditable retrieval of the three MuJoCo Menagerie robot models."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"
ROBOT_DIRECTORIES = ("franka_emika_panda", "unitree_go2", "unitree_z1")
REQUIRED_FILES = (
    "franka_emika_panda/panda.xml",
    "franka_emika_panda/LICENSE",
    "unitree_go2/go2.xml",
    "unitree_go2/LICENSE",
    "unitree_z1/z1_gripper.xml",
    "unitree_z1/LICENSE",
)


def default_asset_root() -> Path:
    override = os.environ.get("LMRBTP_MUJOCO_ASSETS")
    if override:
        return Path(override).expanduser().resolve()
    cache_base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if cache_base:
        return Path(cache_base).resolve() / "llm-mr-bt-planner" / "mujoco-menagerie" / MENAGERIE_COMMIT
    return Path.home().resolve() / ".cache" / "llm-mr-bt-planner" / "mujoco-menagerie" / MENAGERIE_COMMIT


def ensure_assets(root: str | Path | None = None, *, progress=print) -> Path:
    """Return a verified Menagerie checkout, downloading its three sparse paths if absent."""
    target = Path(root).expanduser().resolve() if root else default_asset_root()
    if _is_valid(target):
        progress(f"MuJoCo assets ready: {target}")
        return target
    if target.exists():
        raise RuntimeError(
            f"The requested asset directory already exists but is not a verified checkout: {target}. "
            "Choose a different --assets-dir or move the incomplete directory aside."
        )

    git = shutil.which("git")
    if not git:
        raise RuntimeError(
            "Robot assets are not installed and Git was not found. Install Git, rerun the command, "
            "or set LMRBTP_MUJOCO_ASSETS to a MuJoCo Menagerie checkout at commit "
            f"{MENAGERIE_COMMIT}."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.download-{uuid.uuid4().hex}"
    progress("Downloading pinned MuJoCo Menagerie models (Panda, Go2, and Z1); this happens once.")
    try:
        _run([git, "clone", "--filter=blob:none", "--no-checkout", MENAGERIE_URL, str(staging)])
        _run([git, "-C", str(staging), "sparse-checkout", "set", *ROBOT_DIRECTORIES])
        _run([git, "-C", str(staging), "fetch", "--depth", "1", "origin", MENAGERIE_COMMIT])
        _run([git, "-C", str(staging), "checkout", "--detach", MENAGERIE_COMMIT])
        actual = _capture([git, "-C", str(staging), "rev-parse", "HEAD"])
        if actual != MENAGERIE_COMMIT:
            raise RuntimeError(f"Asset revision mismatch: expected {MENAGERIE_COMMIT}, received {actual}.")
        metadata = {
            "source": MENAGERIE_URL,
            "commit": MENAGERIE_COMMIT,
            "models": list(ROBOT_DIRECTORIES),
        }
        (staging / "LMRBTP_ASSET_PROVENANCE.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if not _is_valid(target):
        raise RuntimeError(f"Downloaded robot assets are incomplete at {target}.")
    progress(f"MuJoCo assets ready: {target}")
    return target


def _is_valid(root: Path) -> bool:
    if not all((root / relative).is_file() for relative in REQUIRED_FILES):
        return False
    provenance = root / "LMRBTP_ASSET_PROVENANCE.json"
    if not provenance.is_file():
        # An explicitly supplied full Menagerie checkout is also acceptable if its Git identity is exact.
        git = shutil.which("git")
        if not git or not (root / ".git").exists():
            return False
        try:
            return _capture([git, "-C", str(root), "rev-parse", "HEAD"]) == MENAGERIE_COMMIT
        except RuntimeError:
            return False
    try:
        data = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("commit") == MENAGERIE_COMMIT


def _run(command: list[str]) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Asset setup command failed ({' '.join(command[:3])}): {detail}")


def _capture(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()
