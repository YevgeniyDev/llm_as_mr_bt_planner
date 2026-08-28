"""Pinned preparation of the official LLM-BT source and released NER parser."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import save_json, save_text

LLM_BT_REPOSITORY_URL = "https://github.com/henryhaotian/LLM-BT"
LLM_BT_REPOSITORY_COMMIT = "c69c18d0cf4b78f166ed352fc0fa8470823b32f6"
LLM_BT_PAPER_URL = "https://arxiv.org/abs/2404.05134"
PARSER_DIRECTORY = "LLMBT/Parser/keywords_extraction"
PARSER_MODEL_RELATIVE = f"{PARSER_DIRECTORY}/pytorch_model.bin"
PARSER_MODEL_SHA256 = "e77ac3903c0f0fa46f1336e4d8e14de3e17986b61e3c8f3e32663ca3ce264eb8"
PARSER_MODEL_BYTES = 265_510_949

SOURCE_FILES = {
    "README.md": "8e948fe143242167afc301dd60e8303e24677cdf585635ccb1175559621e9572",
    "LLMBT/README.md": "cbec5d7c8ecd13c6e4e72786f4ff635d75b1e98d1dbf2588bb8258bde32b0b71",
    "LLMBT/Parser/parser.py": "1782e131de8766378235e304bd67947f4d2d053a0c850950423c14af05589c36",
    f"{PARSER_DIRECTORY}/config.json": (
        "26a4e17f66bd2db8362ce39e53f574ee10c7bc9b69d08956581fe493beec8850"
    ),
    f"{PARSER_DIRECTORY}/tokenizer.json": (
        "435667fab0c06c165b1283ecb422497c37124f2d6a35b2ac73dc876332fc9518"
    ),
    f"{PARSER_DIRECTORY}/tokenizer_config.json": (
        "bb75bf4f97f4a69e71313e8f3e55b8387075b369ea2f5c75aae416ace006f2c2"
    ),
    f"{PARSER_DIRECTORY}/special_tokens_map.json": (
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"
    ),
    f"{PARSER_DIRECTORY}/vocab.txt": (
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
    ),
    f"{PARSER_DIRECTORY}/trainer_state.json": (
        "27424a01557d4329d48718c39214fbc1fe96b4e70bd5ff6bbd3fe9318614152f"
    ),
    "LLMBT/BTsUpdate/bt_editor/bt_editor/RunBT.cpp": (
        "210a4a43321f787ee6bb454fab111ac0602f6107f2805821eabd0ada14f3d675"
    ),
    "LLMBT/BTsUpdate/bt_editor/bt_editor/RunBT.h": (
        "e7f6f01b1347c0012cf31e8b2e28a35aa888d6df26e62b02b452c1aaffa42b17"
    ),
    "LLMBT/BTsUpdate/initial.xml": (
        "26180d687707ab4484d317988286b9b343f135fd86edf95ece490ffbbefa11af"
    ),
    "LLMBT/BTsUpdate/update.xml": (
        "9a9c5334337c721169a5c3e35eefd280f71cb7cc5df7cb093e2fdda8637f559c"
    ),
    "LLMBT/BTsUpdate/core/include/nodes_in_BTs.h": (
        "1d091af5d5e2c7086a4169180c2fe4b2469583dd9fa5bb481cce60219d7c995e"
    ),
    "LLMBT/BTsUpdate/core/src/nodes_in_BTs.cpp": (
        "c703a4597710e7b906b2986c673303bef80844a9f01636b15cef45f57bde2565"
    ),
    "LLMBT/BTsUpdate/core/LICENSE": (
        "63b5778ff0ba61175b8cc4875be74fae7e6709ec0fef121c587127d74e2308ee"
    ),
}


class LLMBTSourceError(RuntimeError):
    """Raised when a pinned official LLM-BT artifact cannot be verified."""


@dataclass(frozen=True)
class PreparedLLMBTSource:
    directory: Path
    source: Path
    parser: Path
    manifest: Path
    file_count: int
    parser_model_included: bool


Downloader = Callable[[str, Path], None]


def prepare_official_source(
    output: str | Path,
    *,
    force: bool = False,
    include_parser_model: bool = True,
    downloader: Downloader | None = None,
    expected_files: dict[str, str] | None = None,
    expected_model_sha256: str = PARSER_MODEL_SHA256,
    expected_model_bytes: int = PARSER_MODEL_BYTES,
) -> PreparedLLMBTSource:
    """Download only method-defining files from the immutable official commit."""
    directory = Path(output)
    source = directory / "source"
    parser = source / PARSER_DIRECTORY
    manifest_path = directory / "source_manifest.json"
    attribution_path = directory / "ATTRIBUTION.md"
    source.mkdir(parents=True, exist_ok=True)
    fetch = downloader or _download
    file_hashes = expected_files or SOURCE_FILES

    for relative, expected_hash in file_hashes.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if force or not target.is_file():
            fetch(_raw_url(relative), target)
        _verify_file(target, expected_hash, relative)

    model_path = source / PARSER_MODEL_RELATIVE
    if include_parser_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if force or not model_path.is_file():
            fetch(_media_url(PARSER_MODEL_RELATIVE), model_path)
        _verify_file(model_path, expected_model_sha256, PARSER_MODEL_RELATIVE)
        if model_path.stat().st_size != expected_model_bytes:
            raise LLMBTSourceError(
                f"LLM-BT parser checkpoint has {model_path.stat().st_size} bytes; "
                f"expected {expected_model_bytes}."
            )

    _validate_parser_config(parser / "config.json")
    save_text(
        attribution_path,
        "\n".join(
            [
                "# LLM-BT attribution and license status",
                "",
                f"- Official repository: {LLM_BT_REPOSITORY_URL}",
                f"- Pinned commit: `{LLM_BT_REPOSITORY_COMMIT}`",
                f"- Paper: {LLM_BT_PAPER_URL}",
                "- Released parser: DistilBERT token classifier, 8 labels, 265,510,949-byte checkpoint",
                "- Project-wide software/model license: not declared in the pinned repository",
                "- `LLMBT/BTsUpdate/core/LICENSE`: MIT license for the embedded Michele Colledanchise BT core",
                "",
                "The embedded core license must not be interpreted as a license for the complete",
                "LLM-BT repository or its parser checkpoint. Files are downloaded only after an",
                "explicit prepare command into the selected ignored output directory. They are not",
                "bundled or redistributed by this package.",
                "",
            ]
        ),
    )
    manifest = {
        "manifest_version": "1.0",
        "repository": LLM_BT_REPOSITORY_URL,
        "commit": LLM_BT_REPOSITORY_COMMIT,
        "paper": LLM_BT_PAPER_URL,
        "selection": "method-defining LLMBT files only; V-REP scenes and embedded UI sources excluded",
        "project_license": None,
        "license_warning": (
            "No project-wide license was found. The core MIT license applies to the embedded "
            "BT core and is not treated as a license for the full project or parser model."
        ),
        "files": file_hashes,
        "file_count": len(file_hashes),
        "parser": {
            "architecture": "DistilBertForTokenClassification",
            "base_model": "distilbert-base-uncased",
            "labels": 8,
            "reported_training_epochs": 2.0,
            "checkpoint_included": include_parser_model,
            "checkpoint_sha256": expected_model_sha256 if include_parser_model else None,
            "checkpoint_bytes": expected_model_bytes if include_parser_model else None,
            "license": None,
        },
    }
    save_json(manifest_path, manifest)
    return PreparedLLMBTSource(
        directory=directory,
        source=source,
        parser=parser,
        manifest=manifest_path,
        file_count=len(file_hashes),
        parser_model_included=include_parser_model,
    )


def verify_prepared_source(output: str | Path, *, require_parser_model: bool = True) -> dict:
    """Recheck the selected upstream files and optional released checkpoint."""
    directory = Path(output)
    source = directory / "source"
    manifest_path = directory / "source_manifest.json"
    if not source.is_dir() or not manifest_path.is_file():
        raise LLMBTSourceError(
            "Prepared LLM-BT source is incomplete. Run 'lmrbtp compare llm-bt prepare' first."
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("commit") != LLM_BT_REPOSITORY_COMMIT:
        raise LLMBTSourceError("Prepared LLM-BT source records a different commit.")
    for relative, expected_hash in SOURCE_FILES.items():
        _verify_file(source / relative, expected_hash, relative)
    model_path = source / PARSER_MODEL_RELATIVE
    if require_parser_model:
        _verify_file(model_path, PARSER_MODEL_SHA256, PARSER_MODEL_RELATIVE)
        if model_path.stat().st_size != PARSER_MODEL_BYTES:
            raise LLMBTSourceError("Prepared LLM-BT parser checkpoint has an unexpected size.")
    _validate_parser_config(source / PARSER_DIRECTORY / "config.json")
    return document


def parser_directory(output: str | Path) -> Path:
    return Path(output) / "source" / PARSER_DIRECTORY


def _raw_url(relative: str) -> str:
    return f"https://raw.githubusercontent.com/henryhaotian/LLM-BT/{LLM_BT_REPOSITORY_COMMIT}/{relative}"


def _media_url(relative: str) -> str:
    return f"https://media.githubusercontent.com/media/henryhaotian/LLM-BT/{LLM_BT_REPOSITORY_COMMIT}/{relative}"


def _validate_parser_config(path: Path) -> None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LLMBTSourceError(f"LLM-BT parser config is unreadable: {error}.") from error
    expected = {
        "0": "O",
        "1": "B-Action",
        "2": "B-Target",
        "3": "I-Target",
        "4": "B-Destination",
        "5": "I-Destination",
        "6": "B-Location",
        "7": "I-Location",
    }
    if document.get("architectures") != ["DistilBertForTokenClassification"]:
        raise LLMBTSourceError("Released LLM-BT parser architecture changed unexpectedly.")
    if document.get("id2label") != expected:
        raise LLMBTSourceError("Released LLM-BT parser label vocabulary changed unexpectedly.")


def _verify_file(path: Path, expected_hash: str, label: str) -> None:
    if not path.is_file():
        raise LLMBTSourceError(f"Prepared LLM-BT file is missing: {label}.")
    actual = _sha256_file(path)
    if actual != expected_hash:
        raise LLMBTSourceError(
            f"LLM-BT source hash verification failed for {label}: "
            f"expected {expected_hash}, got {actual}."
        )


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "llm-mr-bt-planner/llm-bt"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
