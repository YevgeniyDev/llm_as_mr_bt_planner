# Changelog

## Unreleased

### Added — fixed evaluation protocol and auditable paper tables
- Added `experiments/protocol_v1.json`: two scenarios, seven LLM conditions with
  30 best-effort seeded trials each, plus one native MRBTP run per scenario.
- Added immutable result snapshots containing raw outputs, aggregates, model and
  sampling configuration, commit SHA, environment metadata, LaTeX tables, and
  SHA-256 checksums. `main.tex` consumes the generated tables directly.
- Expanded reporting with Wilson intervals, synchronization/capability/causal/
  structural error rates, deadlock/timeout rates, BT complexity, synchronization
  edges, executed actions, symbolic makespan, correction rounds, and wall time.

### Fixed — MRBTP native success semantics
- Replaced `success = not timed_out` with the auditable `mrbtp_native_v1`
  criterion: no timeout, an expanded condition grounded in the initial state, and
  extractable per-robot trees. Frontier exhaustion is now a failure.
- Labelled MRBTP as a native metric instead of claiming it is re-simulated under
  incompatible blocking-guard Condition semantics.

### Added — physical object-incident mitigation contracts
- Added conservative decisions for dropped, missing, damaged, and tool-failure
  incidents: safe stop, perception refresh, reacquisition, quarantine/replacement,
  reassignment, or operator escalation. Physical perception and recovery remain
  backend integration work.

### Added — continuous integration
- Added Python 3.10/3.12 symbolic test, Ruff, and mypy checks, plus a separate
  manually triggered MuJoCo workflow.

### Added — execution-time recovery ladder (failure detection → retry → reassign)
- New `recovery.py`: a `RecoveryController` that reacts to *runtime* action failures, distinct from the
  existing planning-time LLM self-correction. **Tier 1** retries a failed action on the same robot
  (`--max-retries`), then **Tier 2** reassigns it to another robot whose capability produces the same
  predicate (via `domain.candidate_producers`), rewriting the plan and re-running. No LLM involved.
- `simulation.simulate` gains an optional `action_oracle(event) → Status` seam (checked before effects
  commit) and reports `SimulationReport.failures`; with no oracle, behavior is byte-for-byte unchanged.
- Pluggable oracles: a deterministic `InjectedFailureOracle` (persistent, monotonic attempt tally) and a
  seeded `StochasticFailureOracle`, so the whole ladder is reproducible with no LLM and no physics.
- `SymbolicExecutionBackend` accepts `recovery=`; the ladder timeline lands in
  `ExecutionResult.details["recovery"]`. New `run` flags: `--recovery`, `--max-retries`, `--reassign`,
  `--inject-failures`, `--fail-prob`/`--fail-seed`. New `tests/test_recovery.py`. (Driving the ladder from a
  real MuJoCo physics oracle is on the roadmap; the tick-engine `action_oracle` seam is already in place.)

### Added — Markdown-authored planning skills
- New `skills.py` + `skills/` directory: reusable planning guidance authored as `*.md` files (frontmatter +
  body), parsed with the standard library only (no PyYAML), selected per scenario, and injected into the LLM
  prompt between the scenario context and the output schema. Off by default so the pure-mode baseline is
  unchanged. New `--skills`/`--skills-dir` flags on `run` and `experiment`; the experiment config records
  `skills`. New `tests/test_skills.py`.

### Changed — documentation & comment reconciliation
- Documented the MuJoCo backend, the recovery ladder, and skills in `README.md` and `docs/architecture.md`;
  refreshed the `pyproject` description and the `execution/__init__` docstring (were "symbolic + ROS scaffold
  only"). Added `docs/roadmap.md` (future-implementations plan).
- Fixed misleading in-code comments in `mujoco_backend.py` (an `ik` step mislabeled "Stage 3"; a claim that
  grasped parts are held by weld constraints — they are re-snapped kinematically, the welds are inactive) and
  marked the superseded `mujoco_ik.py` as legacy (the `ik` fidelity runs through the `mink` `ArmController`).

### Changed — mink-based manipulation ("ik" fidelity)
- Rebuilt inverse kinematics on **[mink](https://github.com/kevinzakka/mink)**, a
  differential-IK library that solves each step as a quadratic program. This
  replaces the hand-rolled Jacobian solver and provides, from a well-tested
  implementation:
  - a **frame task** (position + orientation) that keeps each gripper pointed
    straight down,
  - a **posture task** that regularises the arm to a natural configuration,
  - **configuration limits** (joints stay in range), and
  - **collision avoidance between the arms** -- so the manipulators no longer
    drive through one another when reaching a shared workspace.
- mink produces a smooth, collision-free trajectory that is applied kinematically,
  so the whole team stays deterministic and stable (nothing flings, tips, or falls
  through the table). Verified over the full gear plan: minimum arm-to-arm distance
  stays >= 0 (no penetration), dog tilt stays 0 degrees, and every part ends resting
  on the table.
- The Panda fingers and the Z1 jaw close on the part on grasp and open on release;
  a held part is carried rigidly with the gripper (no teleporting).
- Arms are mounted on pedestals above the table so a top-down grasp is comfortably
  in reach. mink is added to the ``mujoco`` optional dependency group.

### Improved — MuJoCo scene realism
- Rebuilt the MuJoCo workcell to look like a believable assembly cell: checker floor + skybox,
  a legged work table, PBR-style materials (brushed steel, brass, painted plastics, rubber),
  and a three-light rig with shadows.
- Shaped parts instead of primitive blocks: a 12-tooth brass gear with hub/spokes/bore, a
  precision steel shaft, a red-handled screwdriver with steel shank + tip, a walled parts tray,
  and a machined gearbase with bolt heads. Parts are laid out on a spaced front arc so nothing
  overlaps and everything rests flat on the table.
- Robots posed in a natural "ready" stance and spread on a wider ring so the arms no longer
  tangle; the Z1 now rides on the Go2's back. Actuators hold the posed configuration while
  physics settles, so arms don't sag during replay.
- The parts drawer is now a cabinet carcass + sliding drawer (contrasting face + handle).
- Silenced the benign menagerie attach-conflict warnings and enlarged the offscreen framebuffer.

### Added — MuJoCo tests
- Expanded `tests/test_execution.py` with physics assertions: realistic-asset checks, parts
  resting flat on the table, drawer open/close, pick-lift/place-down, IK reach + kinematic grasp,
  and full gear-plan execution in both `settle` and `ik` fidelities.


## 0.2.0

Restructured the single-file prototype into the tested `llm_mr_bt_planner` package.

### Added
- `llm_mr_bt_planner` package with a clean module split (predicates, domain, bt, plan, validation, simulation,
  prompts, planner, llm, execution, experiments, cli).
- Declarative domain model: explicit `add`/`delete` capability effects with partial/prefix delete
  patterns and wildcards; no naming-convention magic.
- Real Behavior Tree model and a tick-based executor (`Sequence`/`Fallback`/`Parallel`, `SUCCESS`/
  `FAILURE`/`RUNNING`, reactive memory, blocking-guard synchronization).
- Pluggable LLM providers: OpenAI (default) and Anthropic, with automatic fallback to Anthropic when
  `OPENAI_API_KEY` is unset and `ANTHROPIC_API_KEY` is present.
- Execution-backend abstraction: `SymbolicExecutionBackend` and a `RosExecutionBackend` scaffold with
  `export_behaviortree_cpp_xml`.
- "Pure" vs "assisted" planning modes. Pure (default) gives the LLM only prompt + initial state + schema and
  a task-agnostic validator; assisted enables dependency hints (`--hints full`) and candidate-producer
  feedback (`--feedback rich`) as an ablation baseline. `--max-corrections 0` selects single-shot generation.
- Reproducible multi-trial experiment runner with metrics and CSV/Markdown/JSON exporters; each report records
  its mode (pure/assisted) and correction budget.
- Behavior Tree visualization (`llm_mr_bt_planner.viz`): a self-contained HTML report (`run --viz <path.html>`) with a
  **Behavior Trees** tab (per-robot Mermaid `flowchart` diagrams; `bt_to_mermaid` returns the raw definition)
  and an **Action Plan** tab — a chronological table of every robot's BT node in execution order, with tick,
  effects, and synchronization waits, built from the simulation trace.
- Engine-only `pytest` suite (deterministic, LLM-free); `pyproject.toml` with packaging and ruff/mypy config.
- `lmrbtp` console script and `python -m llm_mr_bt_planner` with `run` / `experiment` subcommands.

### Changed
- Pure-mode reliability improvements (no task-specific hints): the prompt now includes a general,
  domain-independent back-chaining **planning method** (with explicit robot-scoped-predicate guidance, the
  most common failure); the correction prompt includes the previous failed plan plus per-error-type fix
  guidance so the model patches rather than blindly regenerates; and **best-of-N** sampling (`--samples`,
  with OpenAI `--temperature`) keeps the first plan that validates and simulates.
- **Two-stage generation** (`--two-stage`): the LLM first emits an ordered per-robot action plan, validated
  on its own by simulating it as condition-free sequences (precondition-blocking checks feasibility); the
  validated action plan is then encoded into behavior trees with explicit synchronization. Decoupling
  producer-selection from BT-encoding raises pure-mode success without any task-specific hints.
- Simulator now executes at most one action per robot per global tick (`actions_per_tick`, default 1), so a
  tick is one synchronized round and the Action Plan timeline reads chronologically instead of collapsing
  every action into ticks 1-2. Conditions still resolve freely; deadlock/timeout detection is unchanged.
- Both bundled scenarios rewritten into the explicit `add`/`delete` effect form.
- Default LLM provider is OpenAI (with Anthropic fallback); `.env.example` documents both.
- `run --output` defaults to `outputs/run-<scenario>.json` so different scenarios don't overwrite each other.

### Removed
- Hidden world-state conventions from the engine: `_open`/`_closed` mutual exclusion, `_at`
  single-valued fluents, and the unused hard-coded `robot_near` derived predicate. World-state
  semantics are now expressed declaratively in the scenario data.
- Mock-LLM scaffolding: offline mock clients (`ScriptedLLMClient`, `ReferencePlanClient`), the
  `--reference-plan` CLI option, the `data/reference_plans/` fixtures, and the tests that faked the LLM.
  Planner quality is now measured with real LLM runs; the remaining tests cover the deterministic engine only.
