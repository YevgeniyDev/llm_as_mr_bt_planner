"""Released DistilBERT keyword-parser adapter for the LLM-BT comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class LLMBTParserError(ValueError):
    """Raised when NER predictions cannot be parsed by the released state machine."""


@dataclass(frozen=True)
class ParsedMove:
    action: str
    target: str
    destination: str
    location: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "action": self.action,
            "target": self.target,
            "destination": self.destination,
            "location": self.location,
        }


@dataclass(frozen=True)
class ParserResult:
    predictions: list[dict[str, Any]]
    moves: list[ParsedMove]
    metadata: dict[str, Any]


class KeywordParser(Protocol):
    model: str
    real_model_inference: bool

    def parse(self, text: str) -> ParserResult:
        ...


class ReplayKeywordParser:
    """Replay archived token-classification predictions without model claims."""

    real_model_inference = False

    def __init__(
        self,
        predictions: list[dict[str, Any]],
        *,
        model: str = "archived-ner-predictions",
    ) -> None:
        self.model = model
        self._predictions = predictions

    def parse(self, text: str) -> ParserResult:
        moves = parse_predictions(self._predictions)
        return ParserResult(
            predictions=self._predictions,
            moves=moves,
            metadata={
                "mode": "replay",
                "real_model_inference": False,
                "input_characters": len(text),
                "prediction_count": len(self._predictions),
            },
        )


class TransformersKeywordParser:
    """Run the authors' released DistilBERT NER checkpoint without retraining."""

    real_model_inference = True

    def __init__(self, model_directory: str | Path) -> None:
        try:
            import torch
            import transformers
        except ImportError as error:
            raise LLMBTParserError(
                "LLM-BT parser inference requires optional ML packages. Install them with "
                'python -m pip install -e ".[llm-bt]".'
            ) from error
        model_path = Path(model_directory).resolve()
        if not (model_path / "pytorch_model.bin").is_file():
            raise LLMBTParserError(
                f"Released LLM-BT parser checkpoint is missing from '{model_path}'."
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = transformers.pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=device,
            aggregation_strategy="none",
        )
        self.model = f"released-distilbert-ner:{model_path}"
        self._device = "cuda:0" if device == 0 else "cpu"
        self._versions = {
            "torch": getattr(torch, "__version__", "unknown"),
            "transformers": getattr(transformers, "__version__", "unknown"),
        }

    def parse(self, text: str) -> ParserResult:
        raw = self._pipeline(text)
        predictions = [dict(item) for item in raw]
        moves = parse_predictions(predictions)
        return ParserResult(
            predictions=predictions,
            moves=moves,
            metadata={
                "mode": "released_distilbert_ner",
                "real_model_inference": True,
                "input_characters": len(text),
                "prediction_count": len(predictions),
                "device": self._device,
                "library_versions": self._versions,
            },
        )


def parse_predictions(predictions: list[dict[str, Any]]) -> list[ParsedMove]:
    """Apply the released parser.py B/I keyword state machine strictly."""
    records: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None
    field_for_label = {
        "B-Target": "target",
        "I-Target": "target",
        "B-Destination": "destination",
        "I-Destination": "destination",
        "B-Location": "location",
        "I-Location": "location",
    }
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise LLMBTParserError(f"NER prediction {index} must be an object.")
        label = prediction.get("entity", prediction.get("entity_group"))
        word = prediction.get("word")
        if not isinstance(label, str) or not isinstance(word, str) or not word.strip():
            raise LLMBTParserError(
                f"NER prediction {index} requires string 'entity' and non-empty 'word'."
            )
        normalized_word = _token(word)
        if label == "O":
            continue
        if label == "B-Action":
            if normalized_word != "move":
                raise LLMBTParserError(
                    f"Released LLM-BT parser has no action-template mapping for '{word}'."
                )
            current = {
                "action": "move",
                "target": None,
                "destination": None,
                "location": None,
            }
            records.append(current)
            continue
        field = field_for_label.get(label)
        if field is None:
            raise LLMBTParserError(f"NER prediction {index} uses unknown label '{label}'.")
        if current is None:
            raise LLMBTParserError(f"NER label '{label}' appears before a B-Action token.")
        if label.startswith("B-"):
            current[field] = normalized_word
        else:
            previous = current[field]
            if not previous:
                raise LLMBTParserError(f"NER label '{label}' has no preceding B label.")
            current[field] = f"{previous}_{normalized_word}"

    moves: list[ParsedMove] = []
    for index, record in enumerate(records):
        target = record["target"]
        destination = record["destination"]
        if not target or not destination:
            raise LLMBTParserError(
                f"Parsed move {index + 1} is missing a target or destination."
            )
        moves.append(
            ParsedMove(
                action="move",
                target=target,
                destination=destination,
                location=record["location"],
            )
        )
    if not moves:
        raise LLMBTParserError("Released LLM-BT parser extracted no complete move conditions.")
    return moves


def _token(word: str) -> str:
    normalized = word.strip().lower()
    if normalized.startswith("##"):
        normalized = normalized[2:]
    return normalized.replace(" ", "_")
