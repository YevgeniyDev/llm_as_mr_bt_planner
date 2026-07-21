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
mode**: the LLM receives the instruction, explicit declarative scenario and capability model, schema, and
task-agnostic method, while the validator reports *what*
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
- **LLM is the planner** — pure mode by default (no producer-specific hints); the deterministic code only
  validates and simulates. Three task-agnostic reliability levers — a general back-chaining *method* in the
  prompt, best-of-N sampling, and **two-stage generation** (action plan → behavior trees) — raise success
  without reintroducing task-specific hints.
- **Pluggable LLM providers** — OpenAI (default) and Anthropic, with automatic fallback to Anthropic when
  `OPENAI_API_KEY` is unset.
- **Two-layer failure handling** — *planning-time* self-correction repairs the whole plan with the LLM
  before execution; a separate *execution-time* **recovery ladder** reacts to runtime action failures with
  **Tier 1 retry (same robot) → Tier 2 reassign (another capable robot)**, no LLM in the loop.
- **Markdown-authored skills** — reusable planning guidance authored as `skills/*.md` files (frontmatter +
  body) is selected per scenario and injected into the prompt; additive and off by default.
- **Execution-backend abstraction** — a symbolic backend (default), a **MuJoCo physics backend** that
  replays the same trees on real menagerie robots (`--backend mujoco`, `settle`/`ik` fidelity), and a
  ROS/BehaviorTree.CPP scaffold (`export_behaviortree_cpp_xml` + a documented `RosExecutionBackend`).
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
  planner.py        the generate -> validate -> simulate -> self-correct loop (planning-time)
  recovery.py       execution-time recovery ladder (retry same robot -> reassign to another) + oracles
  skills.py         load/select/render Markdown-authored planning skills for the prompt
  llm/              base protocol + OpenAI and Anthropic clients
  execution/        ExecutionBackend protocol + symbolic backend + MuJoCo physics backend + ROS scaffold
  experiments/      multi-trial runner + metrics/report exporters
  cli.py            `run` and `experiment` subcommands
data/
  scenario.json, scenario2.json          the two declarative scenarios
skills/                                   Markdown planning skills (frontmatter + guidance body)
tests/                                    pytest suite (engine-only, LLM-free)
docs/architecture.md                      design notes (data flow, domain model, backends, real-robot path)
docs/algorithms.md                        paper-ready pseudocode for the four algorithms
docs/roadmap.md                           future-implementations plan (physics oracle, contact grasping, ...)
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

For paper results, use the fixed protocol instead of ad-hoc commands:

```powershell
python scripts/run_experiment_matrix.py --matrix experiments/protocol_v1.json
```

It runs 30 seeded trials for every LLM condition on both scenarios, runs the
native MRBTP condition once per scenario, and creates a non-overwriting snapshot
under `results/snapshots/`. The snapshot contains raw outputs, aggregates, the
matrix, commit SHA, environment metadata, checksums, and the LaTeX tables consumed
by `main.tex`.

Run the engine tests (deterministic, no API key, no LLM):

```powershell
python -m pytest
```

## Failure detection & recovery

Two distinct mechanisms handle failure, at two different phases — keep them separate:

- **Planning-time self-correction (plan repair).** Inside `run_planner`, an invalid or deadlocked plan is
  regenerated by the LLM from validator + simulator feedback (`--max-corrections N`, best-of-N via
  `--samples`, two-stage via `--two-stage`). It runs *before* execution and rewrites the whole plan.
- **Execution-time recovery ladder.** During execution an action can *fail at runtime*; the ladder reacts
  with no LLM: **Tier 1** retries the action on the same robot up to `--max-retries`, then **Tier 2**
  reassigns it to another robot whose capability produces the same predicate (via `candidate_producers`).

Failures come from a pluggable oracle. A deterministic injector makes the whole ladder reproducible with no
LLM and no physics:

```powershell
python -m llm_mr_bt_planner run --scenario data/scenario.json `
    --recovery on --inject-failures "pick_tool:1" --max-retries 2 --reassign on
```

The result JSON gains a `recovery` block (per-event `log`, `episodes`, whether the goal was reached after
recovery). `--fail-prob P --fail-seed S` is a stochastic alternative for robustness experiments. A MuJoCo
physics oracle (fail when an action's target predicate does not hold in the scene) is the next rung — see
[`docs/roadmap.md`](docs/roadmap.md).

Physical object incidents need a stricter path than retrying an action. The
contracts in `execution/anomalies.py` pause the team, invalidate stale holding
facts, and select reacquisition, quarantine-and-replacement, reassignment, or safe
abort/operator escalation for dropped, missing, or damaged items. See
[`docs/failure_mitigation.md`](docs/failure_mitigation.md). Perception and motion
recovery are explicitly future backend integrations.

## Markdown skills (prompt engineering)

Reusable planning guidance lives in `skills/*.md`, each a small frontmatter block + a free-text body:

```markdown
---
name: robot-scoped-predicates
description: Predicates bound to a robot must be produced by that robot
tags: manipulation, sync
applies_to: pick_gear, mount_gear
---
A predicate whose first argument is a robot ... (guidance the model reads)
```

`--skills on` loads the directory (`--skills-dir` to override), selects the skills relevant to the scenario
(`applies_to: "*"`, a capability-name match, or a tag match), and injects them into the prompt between the
scenario context and the output schema:

```powershell
python -m llm_mr_bt_planner run --scenario data/scenario.json --skills on
```

Skills are additive and **off by default** so the pure-mode baseline is unchanged; the core planning
rules/method stay in `prompts.py`. `experiment --skills on` selects per scenario and records `skills` in the
run config (baseline methods ignore it).

## Physics execution (MuJoCo)

Replay the final plan on real [`mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
robots in physics — the *same* behavior trees, one synchronized round per global tick — with
`--backend mujoco`. Predicate/goal success still comes from the symbolic tick; physics adds a faithful
embodiment and a per-action physics trace.

```powershell
pip install -e ".[mujoco]"                                    # mujoco + mink + qpsolvers[daqp]
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
python -m llm_mr_bt_planner run --scenario data/scenario.json --backend mujoco --physics ik --render
```

- `--physics settle` (Stage 1) scripts each motion as a teleport + gravity settle; `--physics ik` (Stage 2)
  drives the arms with `mink` differential inverse kinematics and carries grasped parts (needs the extra's
  `mink` + `qpsolvers[daqp]`; `settle` does not). Grasping is a kinematic snap, not weld-constrained.
- `--render` opens the interactive viewer; `--mjcf` supplies a custom scene; `--menagerie` points at a
  menagerie checkout (default `third_party/mujoco_menagerie`).
- `scripts/view_mujoco.py` replays a previously saved result JSON in the viewer with no LLM/API calls.

## Baselines

The same `experiment` command evaluates competing methods via `--method`. The
proposed, flat, and hierarchical LLM methods use the same validator and simulator.
MRBTP is explicitly labelled `native_mrbtp` because its Condition semantics differ:

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
# Ingest the labelled native outcome into the comparison table:
python -m llm_mr_bt_planner experiment --method mrbtp --scenario data/scenario.json --scenario data/scenario2.json
```

`scripts/run_mrbtp.py` ports each scenario into MRBTP's ground `PlanningAction` form
(`baselines/mrbtp_port.py`, with delete-relaxation reachability pruning), runs `MAOBTP`, and reports
MRBTP's **native** metrics. MRBTP uses standard reactive BT semantics (a Condition returns
SUCCESS/FAILURE), which is incompatible with our blocking-guard simulator (Condition returns RUNNING for
synchronization), so its trees are **not** re-scored by our simulator. Native success
requires three observable conditions: no timeout, a back-chained condition grounded
in the initial state, and extractable trees for all robots. Frontier exhaustion is a
failure; the old `success = not timed_out` shortcut has been removed.
`mrbtp_bt_to_plan` converts `AnyTreeNode` trees only for inspection/visualization.

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

The symbolic simulator is deliberately small so generated BT structure stays inspectable, and the **MuJoCo
backend** (above) already runs the same trees in physics on real menagerie robot models — a bridge between
the symbolic plan and hardware. The seam for actual hardware is `llm_mr_bt_planner.execution`: implement
`ExecutionBackend` (or fill in `RosExecutionBackend`) to dispatch the same trees — `export_behaviortree_cpp_xml(plan)`
already emits the BehaviorTree.CPP / py_trees-compatible XML. What is still not done: control of real robots
and ROS nodes, perception-driven world state, motion planning and collision checking beyond the MuJoCo
scene, contact/friction-based grasping (MuJoCo grasps kinematically today), and reading execution-time
success/failure back from physics into the recovery ladder. These are tracked in
[`docs/roadmap.md`](docs/roadmap.md).
