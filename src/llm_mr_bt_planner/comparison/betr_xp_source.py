"""Pinned preparation and verification for the official BETR-XP-LLM source."""

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

BETR_XP_REPOSITORY_URL = "https://github.com/jstyrud/BETR-XP-LLM"
BETR_XP_REPOSITORY_COMMIT = "bf83bda4b8921eea7fe0b8756daacb7da9fb6133"
BETR_XP_PAPER_URL = "https://arxiv.org/abs/2409.13356"
BETR_XP_DOI = "10.1109/ICRA55743.2025.11127942"
SOURCE_ARCHIVE_URL = (
    "https://codeload.github.com/jstyrud/BETR-XP-LLM/zip/"
    + BETR_XP_REPOSITORY_COMMIT
)
SOURCE_ARCHIVE_SHA256 = "54bc787eb7ae78901e3d9dee3929dfc1d90bf0412a246428a3e2b4dc7ecb370f"
REQUIRED_SOURCE_FILES = (
    "LICENSE",
    "README.md",
    "llm/llm_utilities.py",
    "llm/LLM-OBTEA/LLM_BTR_prompt.txt",
    "llm/resolve_prompts/intro_and_conditions.txt",
    "llm/resolve_prompts/specification.txt",
    "llm/parameter_prompts/intro_and_conditions.txt",
    "llm/parameter_prompts/specification.txt",
    "behaviors/behavior_tree.py",
    "behaviors/common_behaviors.py",
    "planner/planner.py",
    "planner/test/test_resolve.py",
)


class BetrXPSourceError(RuntimeError):
    """Raised when the pinned official source cannot be verified."""


@dataclass(frozen=True)
class PreparedBetrXPSource:
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
) -> PreparedBetrXPSource:
    """Download, hash, license-check, and safely extract the official repository."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "betr-xp-llm-source.zip"
    source = directory / "source"
    manifest_path = directory / "source_manifest.json"
    attribution_path = directory / "ATTRIBUTION.md"
    fetch = downloader or _download
    if force or not archive.is_file():
        fetch(SOURCE_ARCHIVE_URL, archive)
    actual_hash = _sha256_file(archive)
    if actual_hash != expected_sha256:
        raise BetrXPSourceError(
            "Official BETR-XP-LLM archive hash verification failed: "
            f"expected {expected_sha256}, got {actual_hash}."
        )

    if source.exists():
        shutil.rmtree(source)
    extracted = _safe_extract(archive, source)
    missing = [relative for relative in REQUIRED_SOURCE_FILES if not (source / relative).is_file()]
    if missing:
        raise BetrXPSourceError(
            "Pinned BETR-XP-LLM source is missing required file(s): " + ", ".join(missing)
        )
    license_text = (source / "LICENSE").read_text(encoding="utf-8", errors="replace")
    if "Redistribution and use in source and binary forms" not in license_text:
        raise BetrXPSourceError("Pinned BETR-XP-LLM source no longer has the expected BSD license.")

    files = {
        path.relative_to(source).as_posix(): _sha256_file(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "manifest_version": "1.0",
        "repository": BETR_XP_REPOSITORY_URL,
        "commit": BETR_XP_REPOSITORY_COMMIT,
        "paper": BETR_XP_PAPER_URL,
        "doi": BETR_XP_DOI,
        "archive_url": SOURCE_ARCHIVE_URL,
        "archive_sha256": actual_hash,
        "software_license": "BSD-3-Clause",
        "copyright": "Copyright (c) 2024, ABB",
        "file_count": len(files),
        "files": files,
        "required_method_files": list(REQUIRED_SOURCE_FILES),
    }
    save_json(manifest_path, manifest)
    save_text(
        attribution_path,
        "\n".join(
            [
                "# BETR-XP-LLM source attribution",
                "",
                f"- Official repository: {BETR_XP_REPOSITORY_URL}",
                f"- Pinned commit: `{BETR_XP_REPOSITORY_COMMIT}`",
                f"- Archive SHA-256: `{actual_hash}`",
                f"- Paper: {BETR_XP_PAPER_URL}",
                f"- IEEE DOI: {BETR_XP_DOI}",
                "- Software license: BSD-3-Clause; copyright (c) 2024, ABB",
                "",
                "The archive is retained as separate official-source evidence. The common-domain",
                "runner uses a compatibility representation for this project's capability contracts",
                "and evaluators; it must not be described as an unmodified execution of the authors'",
                "ABB YuMi, Azure OpenAI, vision, collision-planning, or PyTrees application.",
                "",
            ]
        ),
    )
    return PreparedBetrXPSource(directory, source, archive, manifest_path, len(extracted))


def verify_prepared_source(output: str | Path) -> dict:
    """Verify the pinned archive and every extracted file against its manifest."""
    directory = Path(output)
    manifest_path = directory / "source_manifest.json"
    archive = directory / "betr-xp-llm-source.zip"
    source = directory / "source"
    if not manifest_path.is_file() or not archive.is_file() or not source.is_dir():
        raise BetrXPSourceError(
            "Prepared BETR-XP-LLM source is incomplete. Run "
            "'lmrbtp compare betr-xp-llm prepare' first."
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("commit") != BETR_XP_REPOSITORY_COMMIT:
        raise BetrXPSourceError("Prepared BETR-XP-LLM source records a different commit.")
    if document.get("software_license") != "BSD-3-Clause":
        raise BetrXPSourceError("Prepared BETR-XP-LLM source records an unexpected license.")
    if document.get("archive_sha256") != SOURCE_ARCHIVE_SHA256:
        raise BetrXPSourceError("Prepared BETR-XP-LLM manifest has an unexpected archive hash.")
    if _sha256_file(archive) != SOURCE_ARCHIVE_SHA256:
        raise BetrXPSourceError("Prepared BETR-XP-LLM source archive failed hash verification.")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise BetrXPSourceError("Prepared BETR-XP-LLM source manifest has no file hashes.")
    for relative, expected in files.items():
        path = source / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise BetrXPSourceError(f"Prepared BETR-XP-LLM file failed verification: {relative}.")
    for relative in REQUIRED_SOURCE_FILES:
        if relative not in files:
            raise BetrXPSourceError(f"Prepared source manifest omits required file: {relative}.")
    return document


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-mr-bt-planner/betr-xp-llm"})
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
                raise BetrXPSourceError(
                    f"Unsafe path in BETR-XP-LLM source archive: {member.filename}."
                )
            target = output.joinpath(*parts)
            if output.resolve() not in target.resolve().parents:
                raise BetrXPSourceError(f"BETR-XP-LLM archive path escapes output: {member.filename}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))
            written.append(target)
    return written


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
