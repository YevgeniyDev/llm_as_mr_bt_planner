# Changelog

All notable changes to the project are recorded here.

## 0.4.0 — 2026-08-11

### Changed

- Restored the core research claim: the selected LLM now generates the complete schema-v2 multi-robot Behavior Trees, including composites, actions, conditions, waits, resource operations, timeouts, and identifiers.
- Removed the deterministic action-sequence-to-BT compiler and all silent insertion or reordering of Behavior Tree nodes.
- The correction loop now returns the model's entire failed BT plus typed validation and simulation diagnostics and requires a complete replacement tree.
- Accepted trees must round-trip exactly through the parser; unknown fields or any representation that would require normalization are rejected.
- Capability contracts remain authoritative for action semantics, while synchronization and resource-control structure remain the LLM's responsibility.
- Artifact provenance now declares `direct_llm_behavior_tree`, an empty `semantic_rewrites` list, and schema-v2 identity.
- Every blocking UI error now raises a persistent explanatory popup with a redacted cause; correction exhaustion explains why the final BT was not published.
- Blank optional UI text fields are now normalized safely, so an empty model override selects the provider default; missing scenario, provider, project-name, and API-key inputs report field-specific causes instead of internal `NoneType` errors.
- Removed Gradio's redundant per-output `Error` overlay from pipeline runs; blocking failures now leave the detailed inline status readable while still showing the persistent explanatory notification and live-log entry.
- Replaced the free-text model override with a provider-dependent dropdown of current, generally available planning models; provider changes update the choices and clear the visible API-key field, while OpenAI and Anthropic defaults now use `gpt-5.6-sol` and `claude-opus-5`.
- Provider-default dropdown choices now submit their displayed model IDs explicitly, preventing legacy `.env` model overrides from making the executed model disagree with the UI.
- GPT-5-family Chat Completions requests no longer send the legacy `temperature=0` override, which GPT-5.6 rejects; OpenAI HTTP failures now extract a readable provider message instead of displaying raw JSON.
- Documented independent `validate` and `simulate` commands and the boundary between symbolic correctness and physical-robot readiness.

### Removed

- `compiler.py`, action-sequence prompts, compiler-owned nodes, and compiler provenance.

## 0.3.0 — 2026-08-11

### Added

- Local Gradio UI for scenario upload/editing, validation, provider selection, project presets, secure key handling, pipeline execution, cancellation, and artifact downloads.
- Real-time UI pipeline log with elapsed time, progress, provider request/response stages, parsing, trusted compilation, validation results, correction rounds, simulation results, publication, and failures.
- Persisted `pipeline.log` in every completed audit bundle. Logs are redacted and never contain API keys, raw prompts, or hidden reasoning.
- Strict Draft 2020-12 scenario schema, downloadable template, and one runnable three-robot courier scenario for two Franka Panda arms and one Unitree Go2 with Z1 arm.
- Trusted compiler that converts LLM action sequences into canonical synchronized BTs with grounded contracts, resource guards, finite waits/timeouts, stable IDs, and per-node provenance.
- Static checks for structure, robot/capability contracts, typed parameters, predicate support, causal ordering, explicit synchronization, resource ownership, cycles, and liveness.
- Deterministic multi-tick contract simulator with real `Condition`, `WaitFor`, resource, timeout, cancellation, failure-propagation, and state-invariant semantics.
- OpenAI and Anthropic clients with explicit provider selection and no silent fallback.
- Bounded correction loop that feeds typed validation and simulation diagnostics back to the selected provider.
- Canonical JSON artifact, derived XML, HTML visualization, validation report, simulation trace, scenario copy, result summary, checksummed manifest, and stable SHA-256 identity.
- OS credential-store integration and local non-secret project presets.
- Shared `PlannerService` used by both CLI and UI.
- `generate`, `validate`, `simulate`, `render`, `template`, `doctor`, and `ui` CLI commands.

### Changed

- Simplified the UI into a four-step scenario, mission, run, and results flow. Raw JSON, provider utilities, run tuning, and project controls are now collapsed by default, while validation and simulation share compact result tabs.
- Made the application use a stable, wider responsive container so opening advanced sections changes height without changing page width.
- Scenario uploads now load and validate immediately without a separate load button.
- Reduced LLM output to ordered capability actions. BT guards, synchronization, resources, contracts, IDs, and provenance are now compiler-owned and deterministic.
- Failed or cancelled runs never publish `behavior_tree.json` or `behavior_tree.xml`.
- XML generation is now an explicitly documented serializer rather than a ROS execution backend.
- Documentation now focuses on the actual standalone user workflow and clearly states the symbolic validation boundary.
- Repository scope is now the standalone planner only.

### Removed

- The two old scenarios and all active references to them.
- Placeholder experiment tables, obsolete experiment runners, baselines, MRBTP adapter code, and stale result snapshots.
- Oracle-based physical failure injection, automatic reassignment, and the unsupported recovery ladder.
- ROS execution scaffolding and the visual-only MuJoCo backend.
- MuJoCo dependencies, workflow, runtime logs, screenshots, cloned third-party trees, and generated outputs.
- Markdown prompt-skill experiments and stale pure/assisted/two-stage planner modes.
- Old papers, presentation generators, generated figures, and documentation tied to superseded scenarios or unsupported claims.

## 0.2.0

- Converted the original prototype into a Python package with typed predicates, declarative capabilities, Behavior Tree parsing, validation, symbolic simulation, provider clients, CLI commands, and tests.
- This historical release included research and execution experiments that were removed in 0.3.0 when the repository was narrowed to the truthful standalone product.
