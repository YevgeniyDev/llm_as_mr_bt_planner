# Changelog

All notable changes to the project are recorded here.

## 0.6.0 - 2026-08-13

### Added

- Added deterministic MuJoCo video recording with simulation-time frame scheduling, action-directed cuts between fixed mission cameras, a single-camera override, 1080p H.264 output, streaming encoding, safe partial-file handling, and CLI capture controls.
- Added portable physical-run evidence bundles containing the MP4, exact scenario and BT inputs, the measured execution report, software/model provenance, and SHA-256 checksums.
- Replaced the old inspection/adaptive mission with `three_robot_packaging_delivery`, a complete generation scenario in which two Panda arms collaboratively assemble a sealed parcel and Go2/Z1 delivers it into another room through an initially closed door.
- Added a complete schema-v2 reference BT with three cross-robot waits, four exclusive resources, concurrent three-robot execution, and a physically selected closed-door/already-open `Fallback`.
- Added the packing scenario to the UI as the default bundled choice while retaining the courier as a one-click alternative. The LLM generation service receives the selected scenario normally; no reference BT is injected into generation.
- Added an independent package base and lid, shared assembly bench, opposed Panda work positions, a separate delivery room, collidable wall and hinged door, and a room-side delivery pedestal to MuJoCo.
- Added physical controller mappings for all twelve packing and delivery capabilities, including separate base/lid retrieval, measured sealing, sealed-parcel grasping, door approach, contact-driven door opening, far-side crossing, room navigation, placement, and stow.
- Added packaging evidence to physical reports: initial door state, measured final hinge angle, seal-constraint state, parcel and lid positions, final delivery state, and resource-release status.
- Added strict symbolic, direct-generation, scene-composition, canonical-predicate, and opt-in packaging MuJoCo end-to-end regression tests. `lmrbtp doctor` now validates and simulates both bundled reference missions.

### Changed

- Replaced the physical executor's flattened leaf cursor with hierarchical `Sequence` and `Fallback` tick semantics. Only the selected branch executes; the adapter rejects unsupported physical composites instead of flattening or rewriting them.
- Physical Action failure now propagates through the BT and can be recovered by a `Fallback`. Recovered failures remain in the live log and `physical_execution_report.json` rather than being hidden.
- Physical blackboard predicates are now canonicalized before storage and lookup, so equivalent LLM predicate spellings behave identically in MuJoCo execution.
- Reworked the second physical scene around truthful collaborative packing: both manipulators move independent dynamic parts, the lid seal is enabled only at the reached assembly pose, Go2 can pick only a measured sealed parcel, and locomotion fails if either the Z1 grasp or seal is lost.
- Corrected the door frame and panel geometry so the hinge is not pinned by overlapping collision bodies. The closed-door action now succeeds only after contact opens the measured hinge beyond `0.70 rad` and the dynamic Go2 base reaches the far side.
- Aligned the post-door room route with the lateral deflection created naturally by Go2-panel contact, preserving the contact-driven opening instead of resetting or teleporting the base.
- `lmrbtp mujoco --scenario PATH` now discovers an adjacent `PATH.bt.json` when `--bt` is omitted, while still requiring an explicit BT for generated/custom files without a matching adjacent reference.
- Stabilized Z1 release verification by allowing the dynamic component to finish settling while the gripper retreats instead of rejecting a transient bounce immediately.
- Updated the README to document both generation scenarios, both MuJoCo commands, the symbolic/physical branch boundary, and the exact claims supported by the packaging report.

## 0.5.0 - 2026-08-12

### Added

- Added an optional MuJoCo subsystem, isolated from LLM generation, for executing the first three-robot courier scenario with two Panda arms and a Go2 carrying a mounted Z1 gripper model.
- Added `lmrbtp mujoco`, an interactive viewer command, plus `--headless`, `--setup-only`, explicit scenario/BT paths, execution limits, and JSON report output.
- Added pinned, sparse MuJoCo Menagerie asset setup with revision verification, per-model license retention, cache provenance, and actionable setup failures.
- Added a single composed dynamic scene with independent robot namespaces, a free payload, natural four-legged worktables, separate source/destination pads, target fixture, and physical floor contacts.
- Added actuator-driven differential-IK controllers for both Pandas and Z1, a 12-motor Go2 contact gait, measured docking/stow/stationary predicates, and base fall/finite-state checks.
- Added physical BT execution for the first scenario's sequences, waits, exclusive resources, and all eight direct-handoff capability actions. The executor waits on measured preconditions and never applies symbolic effects.
- Added contact/proximity-qualified grasp constraints captured at the current relative pose, continuous Z1-held transport, fixture installation, physical timeouts, live CLI action logs, and `physical_execution_report.json`.
- Added scene-composition tests and an opt-in headless end-to-end test requiring all six physical goals, eight successful actions, Z1-held transport, upright contact-driven locomotion, and zero direct Go2 base-state writes.
- Added third-party controller/model provenance and an explicit account of simulation approximations and sim-to-real limitations.

### Changed

- Rebuilt both courier stations as separated laboratory workcells with physical metal Panda mounting plates and tabletop-height robot bases. In the default view, Franka A occupies the upper outer midpoint and Franka B the lower outer midpoint; both transfer from a left external zone to a right Go2 handoff or process zone while crossing their base-joint centerlines.
- Increased the nominal Go2 dock-to-dock route from `1.20 m` to `3.00 m`, aligned both green handoff pads with the right-side travel route, widened the clear aisle, and updated the default overview/free camera for the longer scene.
- Added continuous top-down pose IK for the bench-mounted Pandas and retained strict measured contact checks for every grasp and release.
- Hid the nonphysical station target sites in the finished MuJoCo view while retaining their exact coordinates for controllers and measured predicates.
- Moved the nominal Go2 docking reference to `0.54 m` from the table and preserved fully dynamic lateral motion during execution.
- Corrected the Menagerie Z1 gripper direction and now require measured open/closed joint positions plus opposing finger-pad contacts before grasp acceptance.
- Added Z1 pose-aware IK, rate-limited jaw closure, compliant finger-pad contacts, a stable manipulation stance, and staged destination release while keeping the Go2 base fully dynamic.
- Franka actions now retreat and return home before releasing their exchange-zone resource, preventing overlap with the incoming Z1 arm.
- Updated the first scenario from tray transport to direct Z1-held transport and separated the green destination pad from the red installation fixture.
- Updated user documentation to distinguish contract simulation from physical MuJoCo execution and provide one-command viewer and headless workflows.

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
