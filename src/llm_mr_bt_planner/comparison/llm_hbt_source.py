"""Pinned provenance preparation for LLM-HBT's paper and author project page."""

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

LLM_HBT_PAPER_URL = "https://arxiv.org/abs/2510.09963v1"
LLM_HBT_ARXIV_ID = "2510.09963v1"
LLM_HBT_PROJECT_URL = "https://github.com/baoziweiyuebing/LLM-HBT"
LLM_HBT_PROJECT_COMMIT = "17ff0ad9fc8e0f5ef3534086589cfa812b20cf29"
PROJECT_ARCHIVE_URL = (
    "https://codeload.github.com/baoziweiyuebing/LLM-HBT/tar.gz/"
    + LLM_HBT_PROJECT_COMMIT
)
PROJECT_ARCHIVE_SHA256 = "d7e22fc0ce6ea5c30dfdd4f10da7ccf914e9ef1b230f3db9c8140bc4c7f96002"
ARXIV_SOURCE_URL = "https://export.arxiv.org/e-print/2510.09963v1"
ARXIV_SOURCE_SHA256 = "bb40eff629f12f7a9ae58e989abe518a1092f73a9a26288ec4f361f17f29ca28"
REQUIRED_PROJECT_FILES = ("README.md", "index.html")
REQUIRED_PAPER_FILES = ("00README.json", "bare_jrnl_new_sample4.tex")


class LLMHBTSourceError(RuntimeError):
    """Raised when pinned LLM-HBT provenance cannot be verified."""


@dataclass(frozen=True)
class PreparedLLMHBTSource:
    directory: Path
    source: Path
    project_archive: Path
    paper_archive: Path
    manifest: Path
    file_count: int


Downloader = Callable[[str, Path], None]


def prepare_official_source(
    output: str | Path,
    *,
    force: bool = False,
    downloader: Downloader | None = None,
    project_expected_sha256: str = PROJECT_ARCHIVE_SHA256,
    paper_expected_sha256: str = ARXIV_SOURCE_SHA256,
) -> PreparedLLMHBTSource:
    """Download and verify the author page and exact arXiv v1 source.

    The author repository contains a project page only and explicitly marks the
    code repository as "Coming Soon".  It is provenance, not executable source.
    """
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    project_archive = directory / "llm-hbt-project.tar.gz"
    paper_archive = directory / "llm-hbt-arxiv-v1.tar"
    source = directory / "source"
    project_source = source / "project-page"
    paper_source = source / "paper"
    manifest_path = directory / "source_manifest.json"
    attribution_path = directory / "ATTRIBUTION.md"
    fetch = downloader or _download
    if force or not project_archive.is_file():
        fetch(PROJECT_ARCHIVE_URL, project_archive)
    if force or not paper_archive.is_file():
        fetch(ARXIV_SOURCE_URL, paper_archive)
    project_hash = _sha256_file(project_archive)
    paper_hash = _sha256_file(paper_archive)
    if project_hash != project_expected_sha256:
        raise LLMHBTSourceError(
            "LLM-HBT project-page archive hash verification failed: "
            f"expected {project_expected_sha256}, got {project_hash}."
        )
    if paper_hash != paper_expected_sha256:
        raise LLMHBTSourceError(
            "LLM-HBT arXiv source hash verification failed: "
            f"expected {paper_expected_sha256}, got {paper_hash}."
        )
    if source.exists():
        shutil.rmtree(source)
    project_files = _safe_extract_tar(
        project_archive,
        project_source,
        strip_first_component=True,
    )
    paper_files = _safe_extract_tar(
        paper_archive,
        paper_source,
        strip_first_component=False,
    )
    _require_files(project_source, REQUIRED_PROJECT_FILES, "project page")
    _require_files(paper_source, REQUIRED_PAPER_FILES, "paper source")
    index_text = (project_source / "index.html").read_text(
        encoding="utf-8", errors="replace"
    )
    tex_text = (paper_source / "bare_jrnl_new_sample4.tex").read_text(
        encoding="utf-8", errors="replace"
    )
    if "Code Repository" not in index_text or "Coming Soon" not in index_text:
        raise LLMHBTSourceError(
            "Pinned LLM-HBT author page no longer records the unavailable code boundary."
        )
    if "Automatic Design of Behavior Trees for Heterogeneous Multirobots" not in tex_text:
        raise LLMHBTSourceError("Pinned paper source lacks the defining LLM-HBT algorithm.")
    files = {
        path.relative_to(source).as_posix(): _sha256_file(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "manifest_version": "1.0",
        "paper": LLM_HBT_PAPER_URL,
        "arxiv_id": LLM_HBT_ARXIV_ID,
        "arxiv_source_url": ARXIV_SOURCE_URL,
        "arxiv_source_sha256": paper_hash,
        "paper_license": "arXiv non-exclusive distribution license",
        "author_project_page": LLM_HBT_PROJECT_URL,
        "project_commit": LLM_HBT_PROJECT_COMMIT,
        "project_archive_url": PROJECT_ARCHIVE_URL,
        "project_archive_sha256": project_hash,
        "official_executable_code_found": False,
        "software_license": "not declared; repository contains only README.md and index.html",
        "code_availability_statement": "Coming Soon",
        "file_count": len(files),
        "files": files,
        "required_project_files": list(REQUIRED_PROJECT_FILES),
        "required_paper_files": list(REQUIRED_PAPER_FILES),
    }
    save_json(manifest_path, manifest)
    save_text(
        attribution_path,
        "\n".join(
            [
                "# LLM-HBT provenance",
                "",
                f"- Paper: {LLM_HBT_PAPER_URL}",
                f"- arXiv v1 source SHA-256: `{paper_hash}`",
                f"- Author project page: {LLM_HBT_PROJECT_URL}",
                f"- Pinned project commit: `{LLM_HBT_PROJECT_COMMIT}`",
                f"- Project archive SHA-256: `{project_hash}`",
                "- Executable author code: not released; the pinned page says `Coming Soon`",
                "- Software license: not declared because the repository contains no software",
                "",
                "The local comparison runner is clean-room code derived from the published",
                "algorithm and must not be represented as the authors' official implementation.",
                "",
            ]
        ),
    )
    return PreparedLLMHBTSource(
        directory=directory,
        source=source,
        project_archive=project_archive,
        paper_archive=paper_archive,
        manifest=manifest_path,
        file_count=len(project_files) + len(paper_files),
    )


def verify_prepared_source(output: str | Path) -> dict:
    directory = Path(output)
    manifest_path = directory / "source_manifest.json"
    project_archive = directory / "llm-hbt-project.tar.gz"
    paper_archive = directory / "llm-hbt-arxiv-v1.tar"
    source = directory / "source"
    if not manifest_path.is_file() or not project_archive.is_file() or not paper_archive.is_file():
        raise LLMHBTSourceError(
            "Prepared LLM-HBT provenance is incomplete. Run "
            "'lmrbtp compare llm-hbt prepare' first."
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "project_commit": LLM_HBT_PROJECT_COMMIT,
        "project_archive_sha256": PROJECT_ARCHIVE_SHA256,
        "arxiv_source_sha256": ARXIV_SOURCE_SHA256,
        "official_executable_code_found": False,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise LLMHBTSourceError(f"Prepared LLM-HBT manifest has unexpected {key}.")
    if _sha256_file(project_archive) != PROJECT_ARCHIVE_SHA256:
        raise LLMHBTSourceError("Prepared LLM-HBT project archive failed verification.")
    if _sha256_file(paper_archive) != ARXIV_SOURCE_SHA256:
        raise LLMHBTSourceError("Prepared LLM-HBT paper archive failed verification.")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise LLMHBTSourceError("Prepared LLM-HBT manifest has no extracted file hashes.")
    for relative, digest in files.items():
        path = source / str(relative)
        if not path.is_file() or _sha256_file(path) != digest:
            raise LLMHBTSourceError(f"Prepared LLM-HBT file failed verification: {relative}.")
    return document


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-mr-bt-planner/llm-hbt"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_extract_tar(
    archive: Path,
    output: Path,
    *,
    strip_first_component: bool,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        bundle = tarfile.open(archive, mode="r:*")
    except tarfile.TarError as error:
        raise LLMHBTSourceError(f"Invalid LLM-HBT provenance archive: {archive.name}.") from error
    with bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            relative = PurePosixPath(member.name)
            parts = relative.parts[1:] if strip_first_component else relative.parts
            if not parts:
                continue
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in parts):
                raise LLMHBTSourceError(f"Unsafe path in {archive.name}: {member.name}.")
            target = output.joinpath(*parts)
            if output.resolve() not in target.resolve().parents:
                raise LLMHBTSourceError(f"Archive path escapes output: {member.name}.")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise LLMHBTSourceError(f"Could not read archived file: {member.name}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with extracted, target.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            written.append(target)
    return written


def _require_files(root: Path, required: tuple[str, ...], label: str) -> None:
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise LLMHBTSourceError(
            f"Pinned LLM-HBT {label} is missing required file(s): " + ", ".join(missing)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
