# Changelog

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
