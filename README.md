# LLM-as-Multi-Robot-BT-Planner

Capability-aware generation of synchronized Behavior Trees for heterogeneous robot teams.

## Prototype Status

This repository currently contains Prototype v0.7:

- v0.1 symbolic rule-based planning pipeline
- v0.2 ablation experiments, failure cases, and deterministic repair
- v0.3 optional LLM planning backend with mock and OpenAI-compatible modes
- v0.4 repeated LLM trials and automatic result aggregation
- v0.5 paper-ready result export and error analysis
- v0.6 second symbolic task: gear assembly with three-robot synchronization
- v0.7 hardened LLM prompt/repair path for gear assembly

The project is still a symbolic research prototype. It does not control real robots.

## Run The Rule-Based Pipeline

```powershell
python src/main.py
```

Packaging is the default task. You can also choose a task explicitly:

```powershell
python src/main.py --task packaging_delivery
python src/main.py --task gear_assembly
```

This writes:

- `outputs/packaging_plan.json`
- `outputs/validation_result.json`
- `outputs/simulation_result.json`
- `outputs/metrics.json`

For gear assembly, task-specific files are written with the `gear_assembly_` prefix, such as `outputs/gear_assembly_plan.json`.

## Run Ablation Experiments

```powershell
python src/run_experiments.py
```

Task-specific ablations can also be run:

```powershell
python src/run_experiments.py --task packaging_delivery
python src/run_experiments.py --task gear_assembly
```

This writes experiment results under:

- `outputs/experiments/summary.json`
- `outputs/experiments/{experiment_name}/`

Gear assembly ablations are written under `outputs/experiments_gear_assembly/`.

Each experiment folder contains the plan, validation result, simulation result, metrics, and repaired equivalents.

## Run The LLM Experiment

```powershell
python src/run_llm_experiment.py
```

Without an API key, this uses deterministic `mock_bad` mode. The expected behavior is invalid before repair and valid/successful after repair.

With an API key, set:

```powershell
$env:OPENAI_API_KEY = "your_api_key"
$env:OPENAI_MODEL = "gpt-4o-mini"
python src/run_llm_experiment.py
```

Optional environment variables:

- `OPENAI_MODEL`, default `gpt-4o-mini`
- `OPENAI_BASE_URL`, default `https://api.openai.com/v1`
- `OPENAI_API_URL`, explicit chat completions URL override
- `OPENAI_TIMEOUT_SECONDS`, default `60`

LLM experiment outputs are saved under:

- `outputs/llm_experiment/llm_plan.json`
- `outputs/llm_experiment/validation_result.json`
- `outputs/llm_experiment/simulation_result.json`
- `outputs/llm_experiment/metrics.json`
- repaired result files in the same folder

## Run Repeated LLM Trials

```powershell
python src/run_llm_trials.py --trials 10
```

Optional arguments:

- `--trials N`, default `10`
- `--mode auto`, `mock_bad`, `mock_good`, or `openai`
- `--task packaging_delivery` or `gear_assembly`
- `--output-dir outputs/llm_trials`

Examples:

```powershell
python src/run_llm_trials.py --trials 10 --mode openai --task packaging_delivery
python src/run_llm_trials.py --trials 5 --mode openai --task gear_assembly
```

Analyze the latest validation failures:

```powershell
python src/analyze_latest_failures.py --task gear_assembly
```

This writes:

- `outputs/llm_trials/summary.json`
- `outputs/llm_trials/summary.csv`
- `outputs/llm_trials/trial_XX/`

Each trial folder contains the raw LLM plan, validation and simulation results, repaired plan, repaired validation and simulation results, repaired metrics, and `trial_summary.json`. If an API call fails, the trial folder contains `api_error.txt` with sanitized details.

## Export Paper Tables

After running repeated LLM trials and ablations:

```powershell
python src/run_llm_trials.py --trials 10 --mode openai
python src/run_experiments.py
python src/export_paper_tables.py
```

This writes paper-ready tables under:

- `outputs/paper_tables/llm_trials_table.tex`
- `outputs/paper_tables/ablation_table.tex`
- `outputs/paper_tables/error_breakdown_table.tex`
- `outputs/paper_tables/llm_trials_table.csv`
- `outputs/paper_tables/error_breakdown.json`

## Current Limitations

- The domain is the single packaging-delivery task.
- The simulator is symbolic and uses a simple deterministic two-robot schedule.
- The repair loop is deterministic and scenario-specific.
- The LLM planner expects JSON and validates/repairs output, but it is not yet a full robust planner.
- No real robot control, ROS integration, perception, continuous geometry, timing, or collision reasoning is implemented.
