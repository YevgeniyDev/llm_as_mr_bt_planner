"""Pinned preparation and verification for the official KIOS source."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from ..config import save_json, save_text

KIOS_REPOSITORY_URL = "https://github.com/ProNeverFake/kios"
KIOS_REPOSITORY_COMMIT = "e9f16f5bd110ab647242077d55d5cb0a71e4fcd9"
SOURCE_ARCHIVE_URL = (
    "https://codeload.github.com/ProNeverFake/kios/zip/" + KIOS_REPOSITORY_COMMIT
)
SOURCE_ARCHIVE_SHA256 = "32468fcbb0be4df496c273968af098134bea87fd588e468ec75cabba124e8bde"
REQUIRED_SOURCE_FILES = (
    "README.md",
    "LICENSE",
    "kios_bt_planning/kios_agent/kios_graph.py",
    "kios_bt_planning/kios_bt/bt_factory.py",
    "experiments/demo/one_step_generation_sync.py",
    "experiments/demo/iterative_generation_sync.py",
    "experiments/demo/human_in_the_loop_syncrun.py",
    "experiments/gearset1/recursive_generation_sync.py",
)


class KiosSourceError(RuntimeError):
    """Raised when the pinned official source cannot be verified."""


@dataclass(frozen=True)
class PreparedKiosSource:
    directory: Path
    source: Path
    archive: Path
    manifest: Path
    file_count: int


Downloader = Callable[[str, Path], None]


def prepare_official_source(
    output: str | Path,
    *,
    force: bool = False,
    downloader: Downloader | None = None,
    expected_sha256: str = SOURCE_ARCHIVE_SHA256,
) -> PreparedKiosSource:
    """Download, hash, and safely extract the pinned authors' repository."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "kios-source.zip"
    source = directory / "source"
    manifest_path = directory / "source_manifest.json"
    attribution_path = directory / "ATTRIBUTION.md"

    fetch = downloader or _download
    if force or not archive.exists():
        fetch(SOURCE_ARCHIVE_URL, archive)
    actual_hash = _sha256_file(archive)
    if actual_hash != expected_sha256:
        raise KiosSourceError(
            "Official KIOS archive hash verification failed: "
            f"expected {expected_sha256}, got {actual_hash}."
        )

    if source.exists():
        # ``source`` is a fixed child of the explicitly selected preparation
        # directory; clearing it prevents stale files from contaminating hashes.
        shutil.rmtree(source)
    extracted = _safe_extract(archive, source)
    missing = [name for name in REQUIRED_SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise KiosSourceError("Pinned KIOS source is missing required file(s): " + ", ".join(missing))
    license_text = (source / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "MIT License" not in license_text:
        raise KiosSourceError("Pinned KIOS source no longer contains the expected MIT license.")

    save_text(
        attribution_path,
        "\n".join(
            [
                "# KIOS source attribution",
                "",
                f"- Authors' repository: {KIOS_REPOSITORY_URL}",
                f"- Pinned commit: `{KIOS_REPOSITORY_COMMIT}`",
                f"- Archive SHA-256: `{actual_hash}`",
                "- Software license: MIT (see `source/LICENSE`)",
                "- Paper: https://arxiv.org/abs/2409.10444",
                "",
                "The extracted repository is retained as separate official-source evidence. The",
                "common-domain runner is a clean-room compatibility implementation and must not be",
                "described as an unmodified execution of the authors' robot stack.",
                "",
            ]
        ),
    )
    files = {
        path.relative_to(source).as_posix(): _sha256_file(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "manifest_version": "1.0",
        "repository": KIOS_REPOSITORY_URL,
        "commit": KIOS_REPOSITORY_COMMIT,
        "archive_url": SOURCE_ARCHIVE_URL,
        "archive_sha256": actual_hash,
        "software_license": "MIT",
        "file_count": len(files),
        "files": files,
    }
    save_json(manifest_path, manifest)
    return PreparedKiosSource(directory, source, archive, manifest_path, len(extracted))


def verify_prepared_source(output: str | Path) -> dict:
    """Verify the archive, commit manifest, and every extracted source file."""
    directory = Path(output)
    manifest_path = directory / "source_manifest.json"
    archive = directory / "kios-source.zip"
    source = directory / "source"
    if not manifest_path.is_file() or not archive.is_file() or not source.is_dir():
        raise KiosSourceError(
            "Prepared KIOS source is incomplete. Run "
            "'lmrbtp compare llm-as-bt-planner prepare' first."
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("commit") != KIOS_REPOSITORY_COMMIT:
        raise KiosSourceError("Prepared KIOS source records a different repository commit.")
    expected_archive = document.get("archive_sha256")
    if expected_archive != SOURCE_ARCHIVE_SHA256 or _sha256_file(archive) != expected_archive:
        raise KiosSourceError("Prepared KIOS source archive failed hash verification.")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise KiosSourceError("Prepared KIOS source manifest has no file hashes.")
    for relative, expected in files.items():
        path = source / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise KiosSourceError(f"Prepared KIOS file failed hash verification: {relative}.")
    return document


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-mr-bt-planner/kios"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_extract(archive: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = PurePosixPath(member.filename)
            parts = relative.parts[1:]
            if not parts or member.is_dir():
                continue
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in parts):
                raise KiosSourceError(f"Unsafe path in KIOS source archive: {member.filename}.")
            target = output.joinpath(*parts)
            if output.resolve() not in target.resolve().parents:
                raise KiosSourceError(f"KIOS archive path escapes output: {member.filename}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
            written.append(target)
    return written


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
