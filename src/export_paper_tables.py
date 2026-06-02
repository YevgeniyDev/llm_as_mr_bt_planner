from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

from analysis import aggregate_error_types


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    llm_trials_dir = project_root / "outputs" / "llm_trials"
    experiments_dir = project_root / "outputs" / "experiments"
    output_dir = project_root / "outputs" / "paper_tables"

    llm_summary_path = llm_trials_dir / "summary.json"
    llm_csv_path = llm_trials_dir / "summary.csv"
    experiments_summary_path = experiments_dir / "summary.json"

    if not llm_summary_path.exists() or not llm_csv_path.exists():
        print("Run python src/run_llm_trials.py --trials 10 --mode openai first.")
        sys.exit(1)

    if not experiments_summary_path.exists():
        print("Run python src/run_experiments.py first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    llm_summary = _load_json(llm_summary_path)
    experiments_summary = _load_json(experiments_summary_path)
    trial_dirs = _find_trial_dirs(llm_trials_dir)
    error_breakdown = aggregate_error_types(trial_dirs)

    _write_text(output_dir / "llm_trials_table.tex", _make_llm_trials_table(llm_summary))
    _write_text(output_dir / "ablation_table.tex", _make_ablation_table(experiments_summary))
    _write_text(output_dir / "error_breakdown_table.tex", _make_error_breakdown_table(error_breakdown))
    _save_json(output_dir / "error_breakdown.json", error_breakdown)
    shutil.copyfile(llm_csv_path, output_dir / "llm_trials_table.csv")

    _export_optional_task_tables(project_root, output_dir)

    print("Paper tables exported:")
    print(f"- {output_dir / 'llm_trials_table.tex'}")
    print(f"- {output_dir / 'ablation_table.tex'}")
    print(f"- {output_dir / 'error_breakdown_table.tex'}")
    print(f"- {output_dir / 'llm_trials_table.csv'}")
    print(f"- {output_dir / 'error_breakdown.json'}")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _find_trial_dirs(llm_trials_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in llm_trials_dir.iterdir()
        if path.is_dir() and path.name.startswith("trial_")
    )


def _export_optional_task_tables(project_root: Path, output_dir: Path) -> None:
    candidates = [
        project_root / "outputs" / "llm_trials_packaging_delivery",
        project_root / "outputs" / "llm_trials_gear_assembly",
    ]
    for trial_dir in candidates:
        summary_path = trial_dir / "summary.json"
        csv_path = trial_dir / "summary.csv"
        if not summary_path.exists() or not csv_path.exists():
            continue

        summary = _load_json(summary_path)
        task_id = summary.get("task_id") or trial_dir.name.replace("llm_trials_", "")
        error_breakdown = aggregate_error_types(_find_trial_dirs(trial_dir))

        _write_text(output_dir / f"llm_trials_table_{task_id}.tex", _make_llm_trials_table(summary))
        _write_text(output_dir / f"error_breakdown_table_{task_id}.tex", _make_error_breakdown_table(error_breakdown))
        _save_json(output_dir / f"error_breakdown_{task_id}.json", error_breakdown)
        shutil.copyfile(csv_path, output_dir / f"llm_trials_table_{task_id}.csv")


def _make_llm_trials_table(summary: dict) -> str:
    return "\n".join(
        [
            r"\begin{table}[!t]",
            r"\renewcommand{\arraystretch}{1.2}",
            r"\caption{LLM Planning Results Before and After Repair}",
            r"\label{tab:llm_results}",
            r"\centering",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Metric & Raw LLM & After Repair \\",
            r"\midrule",
            f"Valid rate & {_format_float(summary.get('raw_valid_rate', 0.0))} & "
            f"{_format_float(summary.get('repaired_valid_rate', 0.0))} \\\\",
            f"Success rate & {_format_float(summary.get('raw_success_rate', 0.0))} & "
            f"{_format_float(summary.get('repaired_success_rate', 0.0))} \\\\",
            f"Avg. errors & {_format_float(summary.get('average_errors_before_repair', 0.0))} & "
            f"{_format_float(summary.get('average_errors_after_repair', 0.0))} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _make_ablation_table(experiments_summary: list[dict]) -> str:
    rows = []
    for item in experiments_summary:
        rows.append(
            f"{_latex_escape(_humanize_experiment(item.get('experiment', 'unknown')))} & "
            f"{_yes_no(item.get('valid_before_repair'))} & "
            f"{_yes_no(item.get('success_before_repair'))} & "
            f"{_yes_no(item.get('valid_after_repair'))} & "
            f"{_yes_no(item.get('success_after_repair'))} \\\\"
        )

    return "\n".join(
        [
            r"\begin{table}[!t]",
            r"\renewcommand{\arraystretch}{1.2}",
            r"\caption{Ablation and Repair Results}",
            r"\label{tab:ablation_results}",
            r"\centering",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Experiment & Valid & Success & Repaired Valid & Repaired Success \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _make_error_breakdown_table(error_breakdown: dict) -> str:
    rows = []
    for error_type, stats in error_breakdown.items():
        rows.append(
            f"{_latex_escape(error_type)} & {stats.get('count', 0)} & "
            f"{_format_float(stats.get('average_per_trial', 0.0))} \\\\"
        )
    if not rows:
        rows.append(r"No errors & 0 & 0.00 \\")

    return "\n".join(
        [
            r"\begin{table}[!t]",
            r"\renewcommand{\arraystretch}{1.2}",
            r"\caption{Error Types Detected in Raw LLM Plans}",
            r"\label{tab:error_breakdown}",
            r"\centering",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Error type & Count & Avg. per trial \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _format_float(value: object) -> str:
    return f"{float(value):.2f}"


def _yes_no(value: object) -> str:
    return "Yes" if bool(value) else "No"


def _humanize_experiment(name: str) -> str:
    special_names = {
        "full_pipeline": "Full pipeline",
        "no_synchronization": "No synchronization",
        "wrong_assignment": "Wrong assignment",
        "missing_handoff_condition": "Missing handoff condition",
        "bad_execution_order": "Bad execution order",
        "mock_llm_bad": "Mock LLM bad",
    }
    return special_names.get(name, name.replace("_", " ").capitalize())


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


if __name__ == "__main__":
    main()
