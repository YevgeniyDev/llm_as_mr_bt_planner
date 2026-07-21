"""Regenerate LaTeX tables from a completed immutable snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=Path("generated/experiment_tables.tex"))
    args = parser.parse_args()
    source = args.snapshot / "paper_tables.tex"
    manifest = json.loads((args.snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not source.exists():
        raise SystemExit("Snapshot is incomplete or has no paper_tables.tex")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(source.read_bytes())
    print(f"Wrote {args.output} from commit {manifest['commit_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
