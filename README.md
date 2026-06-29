# LLM-as-MR-BT-Planner

Research framework for generating **synchronized multi-robot Behavior Trees (BTs)** with an LLM, and
validating and symbolically executing them. Given a natural-language instruction and a *declarative*
scenario (initial state, goal state, objects, locations, and per-robot capability libraries), an LLM must
infer the entire plan — task graph, robot assignments, inter-robot synchronization, and per-robot BTs —
with no hidden checklist or fixed ordering. A static validator and a tick-based simulator score the plan and
feed structured errors back to the LLM for self-correction.

**Research stance: the LLM is the planner.** There is no deterministic BT-synthesis or back-chaining
algorithm (unlike MRBTP-style symbolic planners). The only role of the deterministic code is to *verify and
simulate* the LLM's output — never to author or repair plan structure. By default the program runs in **pure
mode**: the LLM receives only the prompt + initial state + the output schema, and the validator reports *what*
is wrong without suggesting task-specific fixes. Optional **assisted mode** (dependency hints / producer
suggestions) exists solely as an ablation baseline — see [Research design](#research-design-pure-vs-assisted).

The pipeline (Algorithm 1):

1. load a multi-robot scenario,
2. ask an LLM to infer a task graph, assignments, synchronization, and per-robot BTs,
3. validate the returned JSON for structure, robot capabilities, predicate support, and synchronization consistency,
4. ask the LLM to correct invalid or deadlocked plans using validator and simulator feedback (no deterministic repair),
5. run a tick-based symbolic BT simulation,
6. write one result file.

Formal, notation-level pseudocode for all four algorithms (self-correction loop, two-stage generation,
tick-based simulation, and the static validator) is in [`docs/algorithms.md`](docs/algorithms.md).

## Highlights

- **Declarative domain model** — capability effects are explicit `add`/`delete` lists (PDDL-style), so
  world-state semantics live in the data, not in hidden engine conventions.
- **Real Behavior Trees** — `Sequence`, `Fallback`, and `Parallel` composites with `Action`/`Condition`
  leaves, executed by a tick-based engine with `SUCCESS`/`FAILURE`/`RUNNING` status and reactive memory.
  Inter-robot waits are modelled as blocking guards; one action per robot per tick gives a readable timeline.
- **LLM is the planner** — pure mode by default (prompt + initial state only); the deterministic code only
  validates and simulates. Three task-agnostic reliability levers — a general back-chaining *method* in the
  prompt, best-of-N sampling, and **two-stage generation** (action plan → behavior trees) — raise success
  without reintroducing task-specific hints.
- **Pluggable LLM providers** — OpenAI (default) and Anthropic, with automatic fallback to Anthropic when
  `OPENAI_API_KEY` is unset.
- **Execution-backend abstraction** — a symbolic backend now, and a ROS/BehaviorTree.CPP scaffold
  (`export_behaviortree_cpp_xml` + a documented `RosExecutionBackend`) for real-robot testing.
- **Visualization** — a self-contained HTML report with a Behavior Trees view and a chronological Action Plan.
- **Reproducible experiments** — a multi-trial runner with per-scenario metrics (success rate, validity rate,
  mean ± std correction rounds) and CSV / Markdown / JSON outputs.
- **Test suite** — deterministic, LLM-free `pytest` covering predicates, domain, BTs, validation, simulation,
  planning modes, execution, and visualization. Planner *quality* is measured by real LLM runs.

## Layout

```
src/llm_mr_bt_planner/
  predicates.py     parse/format/substitute/match/unify over name(arg, ...) facts
  domain.py         Scenario/Robot/Capability/Effects dataclasses, loading, world-state semantics
  bt.py             Behavior Tree node model (Sequence/Fallback/Parallel/Action/Condition)
  plan.py           typed view over the LLM's JSON plan
  validation.py     static plan validator -> structured errors
  simulation.py     tick-based multi-robot BT executor (deadlock/timeout detection)
  prompts.py        prompt + correction-prompt construction, dependency hints, JSON extraction
  planner.py        the generate -> validate -> simulate -> self-correct loop
  llm/              base protocol + OpenAI and Anthropic clients
  execution/        ExecutionBackend protocol + symbolic backend + ROS scaffold
  experiments/      multi-trial runner + metrics/report exporters
  cli.py            `run` and `experiment` subcommands
data/
  scenario.json, scenario2.json          the two declarative scenarios
tests/                                    pytest suite (engine-only, LLM-free)
docs/architecture.md                      design notes (data flow, domain model, real-robot path)
docs/algorithms.md                        paper-ready pseudocode for the four algorithms
```

## Install & run

Python 3.10+; no third-party runtime dependencies (the LLM clients use the standard library).

```powershell
pip install -e .            # optional; or just run with PYTHONPATH=src
copy .env.example .env      # add OPENAI_API_KEY (or ANTHROPIC_API_KEY)
```

Single run. The result file defaults to `outputs/run-<scenario>.json`, so different scenarios don't overwrite each other:

```powershell
python -m llm_mr_bt_planner run --scenario data/scenario.json                  # -> outputs/run-scenario.json
python -m llm_mr_bt_planner run --scenario data/scenario2.json --model gpt-4o  # -> outputs/run-scenario2.json
```

> **Provider:** commands below use OpenAI (the default). Anthropic is also supported — add `--provider anthropic` (e.g. `--provider anthropic --model claude-opus-4-8`); runs also fall back to Anthropic automatically if `OPENAI_API_KEY` is unset.

Pure mode is harder for the model (it must infer the whole producer chain itself). Three task-agnostic levers improve reliability without reintroducing per-task hints:

- the prompt includes a general back-chaining *method*, and corrections show the model its own failed plan;
- **best-of-N** sampling keeps the first plan that validates and simulates;
- **two-stage generation** (`--two-stage`) splits the job: the LLM first emits an ordered per-robot *action
  plan*, which is validated on its own by running it as condition-free sequences (the simulator blocks each
  action until its preconditions hold, so a feasible plan succeeds without explicit conditions). Only then
  does the LLM encode that fixed action plan into behavior trees with explicit synchronization. This isolates
  the step models most often get wrong (choosing/ordering the producer actions) from the BT-encoding step.

```powershell
python -m llm_mr_bt_planner run --scenario data/scenario.json --two-stage --samples 4 --temperature 0.7
```

(`--temperature` sets the OpenAI sampling temperature; raise it so best-of-N produces diverse candidates.)

Export the generated trees to BehaviorTree.CPP XML for a real executor:

```powershell
python -m llm_mr_bt_planner run --scenario data/scenario.json --export-bt outputs/plan.xml
```

Visualize the plan as a self-contained HTML report (Mermaid; opens in a browser, no install) with two tabs:
a **Behavior Trees** view (Actions as stadiums, Conditions as hexagons, composites as rectangles) and an
**Action Plan** view — a chronological table of every robot's BT node as it fires (with tick, robot, node, effects, and synchronization waits):

```powershell
python -m llm_mr_bt_planner run --scenario data/scenario.json --viz outputs/trees.html
```

`llm_mr_bt_planner.viz.bt_to_mermaid(tree)` also returns the raw Mermaid definition for pasting into GitHub Markdown or
https://mermaid.live.

Reproducible experiment across scenarios and trials, with a results table:

```powershell
python -m llm_mr_bt_planner experiment --scenario data/scenario.json --scenario data/scenario2.json `
    --trials 5 --csv outputs/results.csv --markdown outputs/results.md
```

Run the engine tests (deterministic, no API key, no LLM):

```powershell
python -m pytest
```

## Baselines

The same `experiment` command evaluates competing methods via `--method`, all scored by the **same**
validator + simulator on the same scenarios so the comparison is apples-to-apples:

```powershell
python -m llm_mr_bt_planner experiment --method proposed --scenario data/scenario.json --scenario data/scenario2.json --trials 5 --markdown outputs/cmp_proposed.md
python -m llm_mr_bt_planner experiment --method flat     --scenario data/scenario.json --scenario data/scenario2.json --trials 5 --markdown outputs/cmp_flat.md
python -m llm_mr_bt_planner experiment --method hier     --scenario data/scenario.json --scenario data/scenario2.json --trials 5 --markdown outputs/cmp_hier.md
```

- `proposed` — this work (LLM is the planner; full verifier loop, synchronization, levers).
- `flat` — *LLM-MARS-style*: single-shot, one BT per robot, no synchronization machinery, no self-correction.
- `hier` — *LLM-as-BT-Planner-style*: hierarchical decompose → per-robot BTs → recursive self-correction,
  but robots planned independently (no inter-robot synchronization).
- `mrbtp` — *MRBTP* (Cai et al. 2025) run from the authors' code; see `scripts/run_mrbtp.py` and
  `third_party/MRBTP`. Not LLM-driven, so it needs no API key; it ingests `outputs/mrbtp_results.json`.

The two LLM baselines use the same base model as `proposed` (set via `--provider`/`--model`) and are
faithful re-implementations of the published strategies adapted to these scenarios.

### Running the MRBTP baseline (authors' code)

MRBTP is symbolic and lives in its own dependency stack, so it runs out-of-process in a Python 3.10
environment, then its results are ingested:

```powershell
git clone https://github.com/DIDS-EI/MRBTP third_party/MRBTP   # already cloned in this repo layout
conda create -n mrbtp python=3.10 -y
conda run -n mrbtp pip install -e third_party/MRBTP
# MRBTP's generated ANTLR parser needs the matching runtime (hydra pins an older one):
conda run -n mrbtp pip install "antlr4-python3-runtime==4.13.1"
# Port our scenarios -> run MRBTP -> write outputs/mrbtp_results.json (--time-limit secs/scenario):
conda run -n mrbtp python scripts/run_mrbtp.py --scenario data/scenario.json --scenario data/scenario2.json --time-limit 300
# Ingest + re-score under our validator/simulator, into the comparison table:
python -m llm_mr_bt_planner experiment --method mrbtp --scenario data/scenario.json --scenario data/scenario2.json
```

`scripts/run_mrbtp.py` ports each scenario into MRBTP's ground `PlanningAction` form
(`baselines/mrbtp_port.py`, with delete-relaxation reachability pruning), runs `MAOBTP`, and reports
MRBTP's **native** metrics. MRBTP uses standard reactive BT semantics (a Condition returns
SUCCESS/FAILURE), which is incompatible with our blocking-guard simulator (Condition returns RUNNING for
synchronization), so its trees are **not** re-scored by our simulator; instead, since MRBTP is sound and
complete, goal-success is taken from whether it found a plan within the time budget (it sets a timeout
marker otherwise). `mrbtp_bt_to_plan` (`baselines/mrbtp_adapter.py`) converts its `AnyTreeNode` trees to
our Plan JSON for inspection/visualization.

## Research design: pure vs assisted

The core claim under test is *"an LLM, given only a prompt and the initial world state, can produce correct
synchronized multi-robot BTs."* To keep that claim clean, the deterministic, task-specific planning aids are
**off by default** and exposed as flags so you can run a controlled ablation:

| Flag | `pure` (default) | `assisted` |
|---|---|---|
| `--hints none\|full` | no dependency hints in the prompt | precomputed precondition→producer hints injected |
| `--feedback minimal\|rich` | validator says *what* is unsupported | validator also names candidate producer actions |
| `--max-corrections N` | `N>0` = LLM self-correction loop | `0` = single-shot generation |

The general checks (acyclicity, capability match, predicate support, synchronization consistency) are always
on — they are task-agnostic verification, not planning, and define what "working" means. Suggested study:

```powershell
# Pure, single-shot vs pure, with self-correction:
python -m llm_mr_bt_planner experiment --scenario data/scenario.json --trials 10 --max-corrections 0 --csv outputs/pure_oneshot.csv
python -m llm_mr_bt_planner experiment --scenario data/scenario.json --trials 10 --max-corrections 4 --csv outputs/pure_corrected.csv
# Assisted baseline (how much do hints/suggestions help?):
python -m llm_mr_bt_planner experiment --scenario data/scenario.json --trials 10 --hints full --feedback rich --csv outputs/assisted.csv
```

Every experiment JSON records its `mode`, `include_hints`, `suggest_producers`, and `max_corrections` for
reproducibility.

## Result file

A single JSON file (default `outputs/run-<scenario>.json`) with the final plan, provider/model, validity,
goal success, correction count, the final symbolic state, the execution trace, and validation errors if any.

## Scenarios

- **`gear_assembly`** (default): a three-robot symbolic gear-assembly cell. `go2_z1` opens the drawer,
  stages the gear tray and screwdriver, returns the tool, and closes the drawer; `franka1` holds and
  stabilizes the gearbase; `franka2` picks/mounts the gear, picks the screwdriver, and fastens the screw.
- **`sensor_calibration_cell`** (`scenario2.json`): a more dependency-heavy three-robot sensor-calibration
  cell requiring more cross-robot synchronization (calibration, inspection, clamp release, tool return,
  drawer closure).

Both are pure `add`/`delete` declarative domains — the LLM must infer the causal chains from capability
preconditions and effects.

## Real-robot path

The simulator is deliberately small so generated BT structure stays inspectable before the execution backend
is upgraded. The seam for hardware is `llm_mr_bt_planner.execution`: implement `ExecutionBackend` (or fill in
`RosExecutionBackend`) to dispatch the same trees — `export_behaviortree_cpp_xml(plan)` already emits the
BehaviorTree.CPP / py_trees-compatible XML. This framework does not yet control real robots, ROS nodes,
perception, motion planning, continuous geometry, collision checking, or real-time execution.
