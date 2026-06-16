# Architecture & design notes

This document explains the design decisions behind `mrbtp`, with an eye toward scientific
reproducibility and an eventual real-robot backend.

## Data flow

```
scenario.json ──load_scenario──▶ Scenario ─┐
                                            ├─ prompts.build_prompt ──▶ LLMClient.complete ──▶ raw JSON
                                            │                                                     │
                                            │                              prompts.extract_json ──┘
                                            │                                          │
                                            │                                  plan.parse_plan ──▶ Plan
                                            ▼                                                       │
            validation.validate_plan(Plan, Scenario) ──▶ ValidationReport ◀──────────────────────┤
                                            │ valid?                                                │
                                            ▼                                                       │
            simulation.simulate(Plan, Scenario) ──▶ SimulationReport                               │
                                            │ success?                                              │
                                            ▼ no → prompts.build_correction_prompt ────────────────┘ (loop)
                                  planner.PlannerResult
```

## The declarative domain model

The original prototype encoded world-state semantics in *the engine* via naming conventions:

- `not_X(args)` meant "delete X";
- a `_open` / `_closed` suffix pair was mutually exclusive;
- any `_at` predicate was single-valued on its first argument;
- `robot_near(r, o)` was derived in code from any `*_at` fact.

For a research artifact these are hidden assumptions that silently break on a scenario that does not
follow the conventions. v0.2 moves all of it into the data. A capability declares its effects as:

```json
"effects": { "add": ["tray_at(tray, location)"], "delete": ["tray_at(tray)", "holding(go2_z1, tray)"] }
```

Delete literals are **patterns** (`predicates.matches_pattern`): a pattern may supply fewer arguments
than the fact (prefix match) or use `_` as a per-argument wildcard. This single rule expresses both exact
deletes and "remove whatever value this functional fluent currently holds" (the old `_at` rule) without any
suffix convention. Mutual exclusion (`drawer_open` vs `drawer_closed`) is expressed directly as an
`add`/`delete` pair on the relevant capabilities. The unused `robot_near` derivation was removed outright.

Legacy list-style effects are still accepted by `domain.normalize_effects`, which converts them and emits a
`DeprecationWarning` so nothing is silent. `tests/test_domain.py` proves the conversion is faithful.

## Behavior Tree semantics

`bt.BTNode` is a real tree. `simulation` ticks each robot's tree once per global tick with standard reactive
semantics:

- **Sequence** stops at the first non-`SUCCESS` child and resumes there next tick (memory);
- **Fallback** stops at the first non-`FAILURE` child;
- **Parallel** ticks all children and succeeds at `success_threshold` successes.

Leaves model multi-robot synchronization as **blocking guards**: a `Condition` whose predicate is false, or
an `Action` whose preconditions are unmet, returns `RUNNING` (the robot waits) rather than `FAILURE`. This
reproduces the prototype's "wait until satisfiable" behavior while generalizing to composites.

To keep the execution trace a readable step-by-step timeline, each robot executes **at most one action per
global tick** (`actions_per_tick`, default 1); Conditions still resolve freely within a tick. So one global
tick is one synchronized round of robot actions — exactly what the visualization's Action Plan tab shows.

**Deadlock** is detected structurally: only an executed action changes state, so a full global tick that
leaves the predicate set unchanged while some tree is still running guarantees every future tick would be
identical — that is a deadlock, not a transient wait. **Timeout** is the tick bound `K` being exhausted.

## Self-correction loop and the "LLM is the planner" stance

`planner.run_planner` runs Algorithm 1. The research stance is that the **LLM does all the planning**; the
deterministic code only *verifies and simulates*. Concretely:

- There is **no symbolic BT synthesis / back-chaining** (the thing MRBTP-style planners do). Plan structure
  comes only from `client.complete(...)`.
- **No deterministic repair.** When a plan fails, `build_correction_prompt` hands the LLM the typed validator
  errors, a compact simulation summary, and its *own previous plan*, and the *LLM* regenerates. The loop
  terminates on success or after the correction budget `C` (set `C=0` for single-shot).

The validator's value is turning "is this plan correct?" into concrete, typed errors
(`unsupported_precondition`, `missing_sync_condition`, `condition_before_producer`, …). Those checks are
**task-agnostic** — they generalize to any scenario — so they stay on always.

### Reliability levers (task-agnostic)

Pure mode asks the model to infer the whole producer chain unaided, which is hard. Three levers improve
success without adding per-task hints:

1. **General planning method** — the prompt includes a domain-independent back-chaining procedure (work back
   from goals; every precondition must be initial, produced earlier by the same robot, or produced by another
   robot and waited on; robot-scoped predicates like `holding(R, x)` are produced only by R's own action).
2. **Best-of-N** (`samples` > 1) — sample several plans per generation and keep the first that validates and
   simulates (needs a non-zero client temperature to diversify; OpenAI only).
3. **Two-stage generation** (`two_stage=True`) — see below.

### Two-stage generation

Most pure-mode failures are in *choosing and ordering the producer actions*, not in BT syntax. Two-stage
isolates that step:

1. **Stage 1 — action plan.** The LLM emits only an ordered list of actions per robot. It is validated on its
   own by synthesizing condition-free `Sequence` trees and running the normal validator + simulator: because
   actions block until their preconditions hold, a feasible action plan simulates to success *without* any
   explicit conditions, and a missing/mis-ordered producer surfaces as `unsupported_precondition`/deadlock.
   Failures feed a stage-1 correction.
2. **Stage 2 — encode.** Given the fixed, validated action plan, the LLM wraps each robot's actions into a
   `Sequence` and inserts `Condition` nodes + `synchronization` entries for cross-robot waits, then the full
   plan is validated and simulated (with its own correction budget).

Both stages stay pure (no task-specific hints) and LLM-driven. The action-plan synthesis
(`planner._synthesize_plan`) is pure bookkeeping — it does not plan; it just lets the existing validator and
simulator judge an action list.

### Pure vs assisted (ablation)

Two deterministic computations *do* perform planning-style causal reasoning, so they are **off by default**
and gated behind flags, to avoid confounding the "LLM alone" claim:

- `prompts.build_dependency_hints` precomputes, for each unmet precondition, which capability could produce it,
  and injects that into the prompt (`include_hints`). This is partial back-chaining handed to the model.
- The validator can append candidate-producer suggestions to its error messages
  (`suggest_producers` → `_candidate_text` / `_same_name_text`).

In **pure mode** (default) neither is active: prompt + initial state + schema in, "what is wrong" out. In
**assisted mode** both can be enabled to measure how much the scaffolding helps. The same fixed hint/suggestion
text would not generalize across different tasks anyway — which is exactly why it is an opt-in baseline, not the
default path.

### Testing stance

The automated test suite is **engine-only**: it exercises the deterministic parser, validator, simulator, and
visualizer with no LLM. Positive cases use a tiny inline two-robot domain that is known to be valid and to
simulate to success (`tests/conftest.py`). Planner *quality* — whether the LLM produces working plans — is not
asserted in unit tests; it is measured by real LLM runs via `mrbtp experiment`. The default provider is OpenAI,
with automatic fallback to Anthropic when `OPENAI_API_KEY` is absent.

## Toward real robots

Everything downstream of plan generation already speaks to `execution.ExecutionBackend`. To run on hardware:

1. Implement an action/skill server per capability (the leaf `name` selects it; `parameters` are the goal).
2. Back the symbolic predicates with a blackboard updated by perception/condition monitors.
3. Load the generated tree with `execution.export_behaviortree_cpp_xml(plan)` into py_trees_ros /
   BehaviorTree.CPP, or fill in `RosExecutionBackend.execute`.

The symbolic backend remains the fast, deterministic inner loop for validating plan *structure* before
committing to physical execution.

## Reproducibility

LLM planning is stochastic, so `experiments.run_experiment` sweeps `scenarios × trials` and reports
per-scenario success rate, validity rate, and mean ± sample-std correction rounds, exported as a JSON report
plus optional CSV / Markdown tables for papers. Each report records its mode (pure/assisted), `samples`,
`two_stage`, and `max_corrections`. Runs are reproducible given a fixed model response (OpenAI temperature
defaults to 0; raise it only for best-of-N diversity; Opus 4.8 has no temperature knob).
