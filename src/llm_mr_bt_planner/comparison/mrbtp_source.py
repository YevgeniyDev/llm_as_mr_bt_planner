"""Pinned preparation and verification of the official MIT-licensed MRBTP source."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from ..config import save_json, save_text

MRBTP_PAPER_URL = "https://arxiv.org/abs/2502.18072v1"
MRBTP_AAAI_URL = "https://doi.org/10.1609/aaai.v39i14.33594"
MRBTP_REPOSITORY_URL = "https://github.com/DIDS-EI/MRBTP"
MRBTP_COMMIT = "3d6bd240aa2903245b2335711a97ee394f174313"
MRBTP_ARCHIVE_URL = f"https://codeload.github.com/DIDS-EI/MRBTP/tar.gz/{MRBTP_COMMIT}"
MRBTP_ARCHIVE_SHA256 = "959d3559d10721b45629074ca944d95df92ba73bc44a9f6a57332a28dcd20030"
REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "README.md",
    "mabtpg/btp/multi_robot.py",
    "mabtpg/btp/multi_robot_basic.py",
    "mabtpg/btp/base/planning_agent.py",
    "mabtpg/btp/base/planning_condition.py",
    "mabtpg/behavior_tree/base_nodes/Action.py",
)


class MRBTPSourceError(RuntimeError):
    """Raised when the pinned official MRBTP source cannot be verified."""


@dataclass(frozen=True)
class PreparedMRBTPSource:
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
    expected_sha256: str = MRBTP_ARCHIVE_SHA256,
) -> PreparedMRBTPSource:
    """Download, hash, license-check, and extract one immutable MRBTP revision."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "mrbtp-source.tar.gz"
    source = directory / "source"
    manifest_path = directory / "source_manifest.json"
    attribution_path = directory / "ATTRIBUTION.md"
    fetch = downloader or _download
    if force or not archive.is_file():
        fetch(MRBTP_ARCHIVE_URL, archive)
    digest = _sha256_file(archive)
    if digest != expected_sha256:
        raise MRBTPSourceError(
            f"MRBTP archive hash verification failed: expected {expected_sha256}, got {digest}."
        )
    if source.exists():
        shutil.rmtree(source)
    files = _safe_extract_tar(archive, source)
    _require_files(source)
    license_text = (source / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "MIT License" not in license_text or "Permission is hereby granted" not in license_text:
        raise MRBTPSourceError("Pinned MRBTP source does not contain the expected MIT license.")
    hashes = {
        path.relative_to(source).as_posix(): _sha256_file(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "manifest_version": "1.0",
        "paper": MRBTP_PAPER_URL,
        "aaai_doi": MRBTP_AAAI_URL,
        "official_repository": MRBTP_REPOSITORY_URL,
        "commit": MRBTP_COMMIT,
        "archive_url": MRBTP_ARCHIVE_URL,
        "archive_sha256": digest,
        "software_license": "MIT",
        "file_count": len(hashes),
        "required_source_files": list(REQUIRED_SOURCE_FILES),
        "files": hashes,
    }
    save_json(manifest_path, manifest)
    save_text(
        attribution_path,
        "\n".join(
            [
                "# MRBTP source attribution",
                "",
                f"- Paper: {MRBTP_PAPER_URL}",
                f"- AAAI DOI: {MRBTP_AAAI_URL}",
                f"- Official repository: {MRBTP_REPOSITORY_URL}",
                f"- Pinned commit: `{MRBTP_COMMIT}`",
                f"- Archive SHA-256: `{digest}`",
                "- License: MIT; copyright (c) 2024 MABTPG",
                "",
                "The local runner is a source-aligned common-domain port of the",
                "paper's FIFO MRBTP path. The complete pinned source remains here",
                "for inspection and is not imported into the proposed planner.",
                "",
            ]
        ),
    )
    return PreparedMRBTPSource(
        directory=directory,
        source=source,
        archive=archive,
        manifest=manifest_path,
        file_count=len(files),
    )


def verify_prepared_source(output: str | Path) -> dict:
    directory = Path(output)
    manifest_path = directory / "source_manifest.json"
    archive = directory / "mrbtp-source.tar.gz"
    source = directory / "source"
    if not manifest_path.is_file() or not archive.is_file() or not source.is_dir():
        raise MRBTPSourceError(
            "Prepared MRBTP source is incomplete. Run 'lmrbtp compare mrbtp prepare' first."
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "commit": MRBTP_COMMIT,
        "archive_sha256": MRBTP_ARCHIVE_SHA256,
        "software_license": "MIT",
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise MRBTPSourceError(f"Prepared MRBTP manifest has unexpected {key}.")
    if _sha256_file(archive) != MRBTP_ARCHIVE_SHA256:
        raise MRBTPSourceError("Prepared MRBTP source archive failed verification.")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise MRBTPSourceError("Prepared MRBTP manifest has no extracted file hashes.")
    for relative, digest in files.items():
        path = source / str(relative)
        if not path.is_file() or _sha256_file(path) != digest:
            raise MRBTPSourceError(f"Prepared MRBTP file failed verification: {relative}.")
    _require_files(source)
    return document


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-mr-bt-planner/mrbtp"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_extract_tar(archive: Path, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        bundle = tarfile.open(archive, mode="r:gz")
    except tarfile.TarError as error:
        raise MRBTPSourceError("Downloaded MRBTP archive is not a valid tar.gz file.") from error
    with bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name)
            parts = relative.parts[1:]
            if not parts:
                continue
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in parts):
                raise MRBTPSourceError(f"Unsafe path in MRBTP archive: {member.name}.")
            target = output.joinpath(*parts)
            if output.resolve() not in target.resolve().parents:
                raise MRBTPSourceError(f"MRBTP archive path escapes output: {member.name}.")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise MRBTPSourceError(f"Could not read MRBTP archived file: {member.name}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            written.append(target)
    return written


def _require_files(source: Path) -> None:
    missing = [relative for relative in REQUIRED_SOURCE_FILES if not (source / relative).is_file()]
    if missing:
        raise MRBTPSourceError(
            "Pinned MRBTP source is missing method-defining file(s): " + ", ".join(missing)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
