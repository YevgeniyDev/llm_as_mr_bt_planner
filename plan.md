# Three-Robot Multi-Robot Behavior-Tree Planner: Updated Implementation Plan

The confirmed hardware configuration is:

- Franka A: one 7-DoF Franka Emika Panda at the source station
- Mobile manipulator: one Unitree Go2 carrying a Unitree Z1 arm
- Franka B: one 7-DoF Franka Emika Panda at the destination station

This is a stronger research demonstration than the two-robot scenario because it combines heterogeneous task allocation, fixed manipulation, mobile manipulation, locomotion, explicit synchronization, shared-zone safety, perception, and physical execution.

The robot models are now fixed, but the Go2 edition, firmware, SDK access, Z1 mounting hardware, Z1 gripper, Panda system images, Panda grippers, and installed controller versions still have to be recorded. Unitree officially supports Go2 communication through SDK2 and ROS 2/CycloneDDS, while Z1 uses its own controller and SDK for joint-space, Cartesian-space, and low-level control. [Unitree ROS 2](https://github.com/unitreerobotics/unitree_ros2), [Z1 documentation](https://dev-z1.unitree.com/)

The mechanical integration needs special attention. Unitree publishes Go2 payload values that vary by edition, while the Z1 documentation lists the arm itself at 4.1 kg before its mount, gripper, carrier, cameras, compute, cabling, power hardware, and transported object are included. The plan therefore treats combined mass, center of mass, power, thermal behavior, and walking stability as measured hardware acceptance items rather than assuming that a Go2–Z1 combination is automatically safe. [Go2 specifications](https://www.unitree.com/mobile/go2/), [Z1 technical parameters](https://dev-z1.unitree.com/brief/parameter.html)

The simulation architecture now has a concrete Panda-controller source: [`kevinzakka/mjctrl`](https://github.com/kevinzakka/mjctrl). It is an Apache-2.0 repository of minimal, single-file MuJoCo controller examples. Its Panda example provides differential inverse kinematics with damped least squares, nullspace bias toward a home posture, joint-velocity limiting, and a Panda MJCF derived from MuJoCo Menagerie. It is a useful controller seed, not a complete multi-robot simulator: the example is mouse-target-driven, loads the no-hand Panda scene, has no ROS 2 interface, and contains no Go2, Z1, task-level skill, perception, or behavior-tree implementation.

ROS 2 remains the intended future integration architecture for MuJoCo and hardware, but it is explicitly outside the immediate deliverable. The first working program must install and run without ROS, `rclpy`, a ROS workspace, MuJoCo, robot SDKs, or robot hardware. ROS 2 work begins only after the standalone BT generator is complete and accepted. ROS 1 will not be maintained as a parallel execution path.

# Immediate delivery target: standalone validated BT generator

The first finished milestone is a normal Python command-line program that:

1. Loads the single three-robot courier scenario and typed domain.
2. Accepts or loads a natural-language instruction.
3. Calls a real configured LLM provider.
4. Parses the response into the canonical BT schema.
5. Runs strict structural, domain, capability, causal, synchronization, resource, and liveness validation.
6. Returns typed validation errors to the LLM for a bounded number of correction attempts.
7. Requires the LLM to replace the complete BT when correction is needed; deterministic code never inserts or reorders nodes.
8. Revalidates the exact LLM-generated tree without semantic rewriting.
9. Runs that exact LLM-generated tree through the deterministic contract simulator by default.
10. Writes the final canonical BT file atomically only if validation and the required simulation checks succeed.
11. Prints the absolute BT path and plan hash as the final console output.

Neither ROS 2 nor MuJoCo is required for this milestone. `mjctrl`, the physics simulator, and hardware adapters remain later validation and execution layers for the same BT format.

## Primary command

The intended command is:

```powershell
lmrbtp generate `
  --scenario scenarios/three_robot_courier/scenario.yaml `
  --instruction "Move the part from the source bin to the target bin using all three robots." `
  --provider openai `
  --output outputs/three_robot_courier/behavior_tree.json
```

The existing `run` command may remain temporarily as a deprecated alias, but documentation and tests should use `generate`.

The command must not silently switch providers. If the requested provider, API key, model, or network access is unavailable, it exits with a clear error before creating a final BT file.

## Canonical BT artifact

The primary deliverable is:

```text
outputs/three_robot_courier/behavior_tree.json
```

It contains one executable mission root and the three robot subtrees:

```text
ParallelAll
├── Panda A (`franka_a`)
├── Go2–Z1 (`unitree_go2_z1`)
└── Panda B (`franka_b`)
```

The file must include:

- Schema version
- Mission ID
- Original instruction
- Scenario/domain version
- Canonical root node
- All three robot subtrees
- Stable unique node and task IDs
- Robot assignments
- Explicit `WaitFor` synchronization nodes
- Action arguments and required resources
- Timeout and recovery-policy references
- Per-node provenance: `llm`
- An explicit empty semantic-rewrite record
- SHA-256 hash of the canonical executable content

The BT file must not contain API keys, hidden prompts, chain-of-thought, or mutable runtime state.

The canonical JSON is the source of truth. An optional derived BehaviorTree.CPP XML file may be produced with `--xml-output`, but XML is not required for the standalone milestone and must never become a conflicting second plan representation.

## Run artifact directory

Each generation run also creates an auditable sidecar directory:

```text
outputs/three_robot_courier/run-<timestamp>/
├── raw_model_response.txt
├── generation_record.json
├── validation_report.json
├── simulation_report.json
├── correction_history.json
└── behavior_tree.html            # only when requested
```

The final `behavior_tree.json` is written through a temporary file and atomic rename after all required checks pass. Rejected or incomplete model responses may be retained in the run directory for debugging, but they must not appear at the final BT path.

## Standalone CLI surface

Implement these commands:

```text
lmrbtp doctor
lmrbtp generate
lmrbtp validate
lmrbtp simulate
lmrbtp render
```

- `doctor` checks Python version, scenario files, provider configuration, output-directory access, and schema availability.
- `generate` performs the complete LLM full-BT generation → correction → validate → simulate → BT-file pipeline.
- `validate` validates an existing BT file without calling an LLM.
- `simulate` executes an existing BT file in the deterministic contract simulator without calling an LLM.
- `render` creates a human-readable HTML visualization without changing the BT.

The last successful console lines from `generate` should be machine-readable:

```text
BT_FILE=<absolute path>
BT_SHA256=<hash>
```

Suggested exit codes:

- `0`: validated BT generated successfully
- `2`: configuration, provider, or input error
- `3`: LLM output remained invalid after the correction limit
- `4`: deterministic simulation failed or did not reach the goal
- `5`: output or serialization failure

## Local graphical user interface

The first standalone release must also provide a local browser-based interface built with Gradio `Blocks`. It is a frontend over the same application service used by the CLI; it must not duplicate planning, validation, simulation, serialization, or provider logic.

Installation and launch:

```powershell
pip install -e ".[ui]"
lmrbtp ui
```

The UI launches on `127.0.0.1`, opens the default browser, and uses `share=False`. It must not create a public Gradio share link. Optional flags may select the local port or suppress automatic browser opening.

### UI layout

Use a clear four-step layout rather than exposing research-only options on the main screen.

#### Step 1 — Scenario

- Upload one `.json` scenario file.
- Download the separate scenario template.
- Download or view the JSON Schema/documentation.
- Show the uploaded file name, scenario ID, schema version, and robot count.
- Parse and validate the file immediately after upload.
- Display typed input errors before any LLM request is allowed.
- Show a read-only formatted JSON preview after successful validation.

#### Step 2 — Instruction and provider

- Enter the natural-language mission instruction in a multiline textbox.
- Select `OpenAI` or `Anthropic (Claude)` explicitly.
- Select a supported model or enter an allowed model override.
- Enter the provider API key in a password textbox.
- Test provider configuration without running the planner.
- Choose the bounded correction limit using a simple numeric control.
- Choose output format: canonical JSON always, with optional BehaviorTree.CPP XML export.

The interface must not silently fall back from one provider to another.

#### Step 3 — Run

- Present one prominent **Run complete pipeline** button.
- Disable or reject duplicate submissions while a run is active.
- Provide a **Cancel** button.
- Stream stage updates:
  - Validating uploaded scenario
  - Building the prompt
  - Requesting the initial plan
  - Parsing the candidate
  - Validation result
  - Correction round number
  - Compiling mandatory guards
  - Revalidating
  - Running deterministic simulation
  - Saving artifacts
- Show elapsed time and the current stage without exposing hidden prompts or chain-of-thought.

Cancellation must prevent publication of a final BT. If an HTTP request cannot be interrupted immediately, its eventual response must be discarded and the run marked cancelled.

#### Step 4 — Result

On success, display:

- A green success summary
- Provider and model used
- Correction-round count
- Validation status
- Contract-simulation goal status
- Canonical BT hash
- Interactive or formatted BT preview
- Robot-by-robot subtree summary
- Explicit synchronization summary
- Validation report
- Simulation trace summary

Provide download components for:

- `behavior_tree.json` — always present and canonical
- `behavior_tree.xml` — only when XML export was requested and export succeeded
- `behavior_tree.html` — optional visualization
- `run_artifacts.zip` — prompt metadata, raw response, validation, simulation, and correction reports without secrets

On failure, show the failure stage and typed errors. A diagnostic bundle may be downloadable, but no final BT download is shown and no final BT path is published.

## Scenario upload schema and template

Add these repository files:

```text
schemas/
└── scenario.schema.json

templates/
└── three_robot_scenario.template.json

examples/
└── three_robot_courier.json
```

The template is a standalone valid JSON document containing obvious placeholder values. Because JSON does not support comments, field explanations belong in the JSON Schema descriptions and UI help text rather than pseudo-comment fields that could enter the planner domain.

The uploaded scenario must define at least:

```text
schema_version
scenario_id
objects and their types
robots
robot capabilities
robot workspaces/reachable locations
predicate signatures
action schemas
initial_state
goal_state
shared resources and zones
observation declarations
timeouts
permitted recovery policies
```

For the showcase template, the robot IDs are:

```text
franka_a
unitree_go2_z1
franka_b
```

The uploader must reject:

- Invalid JSON
- Files over the configured small input limit
- Unsupported schema versions
- Missing required fields
- Unknown fields when the schema is strict
- Duplicate object, robot, action, predicate, or resource IDs
- Invalid identifiers
- Unknown object types
- Invalid predicate/action arity
- Capabilities assigned to unknown robots
- Initial or goal predicates that do not type-check
- Embedded executable code, commands, or path traversal attempts

Uploaded strings are treated as data only. The program must never execute code, load arbitrary local paths, fetch arbitrary URLs, or import Python modules named in a scenario file.

## Saving and loading user settings

The UI should support two independent persistence features.

### Saved API key

Default behavior is session-only:

- The key is held only in process/session memory.
- The password textbox masks it.
- The key is never written to run artifacts, logs, JSON files, HTML, tracebacks, or console output.
- The key is never returned to the browser after submission.

If the user enables **Save this key on this device**, store it using the operating system credential store through Python `keyring`:

- Windows Credential Manager on Windows
- Keychain on macOS
- Secret Service or another supported secure backend on Linux

Store separate entries for OpenAI and Anthropic. The UI may show “saved key available,” but it must never display the saved value. Provide **Forget saved key** and **Replace saved key** actions.

If no secure keyring backend is available, disable persistent saving with an explanation. Do not fall back to plaintext `.env`, browser local storage, cookies, project JSON, or a home-directory text file.

### Saved project preset

Allow the user to save and reopen a local project preset containing:

- Project display name
- Validated scenario JSON or a managed copy of it
- Mission instruction
- Selected provider name
- Selected model
- Correction limit
- Requested output formats

The project preset must not contain the API key. It references the provider’s secure keyring entry when one exists.

Use sanitized generated IDs and a controlled application-data directory. Do not use an unsanitized project name as a filesystem path. Provide rename, duplicate, and delete operations with confirmation.

## Shared application service

Refactor the pipeline behind a UI/CLI-neutral service boundary:

```text
PlannerService.generate(request, progress_callback, cancellation_token)
PlannerService.validate(bt_path, scenario_path)
PlannerService.simulate(bt_path, scenario_path)
PlannerService.render(bt_path)
```

The `GenerateRequest` contains validated scenario data, instruction, provider configuration, correction limit, output choices, and destination directory. Provider secrets are passed separately in memory and are excluded from request serialization and dataclass representations.

Both frontends use the same service:

```text
CLI ───────┐
           ├── PlannerService ── canonical pipeline ── ArtifactManager
Gradio UI ─┘
```

The `ArtifactManager` owns controlled output paths, atomic writes, hashing, ZIP creation, cleanup, and the rule that final BT files exist only for successful runs.

## UI concurrency and isolation

- Default to one active planning run per local process.
- Assign a unique run ID and private temporary directory to every submission.
- Never reuse mutable world state, correction history, or output paths across runs.
- Never store API keys in global variables shared across sessions.
- Use Gradio queueing for progress delivery with a concurrency limit of one unless multi-session isolation has been proven.
- Clean uploaded temporary files and cancelled-run temporary artifacts according to a documented retention policy.
- Ensure one user action cannot overwrite another run’s final BT.

The first UI is a local trusted-user application. Do not deploy it publicly without authentication, TLS, external secret management, per-user authorization, upload isolation, and a separate security review.

## UI implementation structure

Add:

```text
src/llm_mr_bt_planner/
├── application/
│   ├── planner_service.py
│   ├── requests.py
│   ├── progress.py
│   └── artifacts.py
├── ui/
│   ├── app.py
│   ├── handlers.py
│   ├── key_store.py
│   ├── project_store.py
│   └── view_models.py
└── cli.py
```

Add an optional dependency group:

```toml
[project.optional-dependencies]
ui = ["gradio", "keyring"]
```

Lock the tested Gradio and keyring versions in the project’s reproducible environment/lock file. The base CLI installation remains available without UI dependencies.

## UI acceptance criteria

- `pip install -e ".[ui]"` installs the UI in a fresh supported environment.
- `lmrbtp ui` opens a functional local interface without ROS or MuJoCo.
- The template downloads successfully and passes the same schema validator as uploaded files.
- A valid uploaded JSON scenario is previewed and enables Run.
- Invalid JSON or domain data prevents Run and shows typed errors.
- OpenAI and Anthropic are explicit provider choices with no silent fallback.
- API keys are masked and absent from logs, exceptions, artifacts, ZIPs, HTML, and saved projects.
- Session-only keys disappear when the session/process ends.
- Saved keys use a verified OS keyring backend and can be forgotten.
- A saved project reloads its scenario, instruction, provider, model, and output preferences without storing its key.
- Run progress advances through the real pipeline stages.
- Cancellation never publishes a final BT.
- Successful runs expose a downloadable canonical JSON BT.
- Requested XML export is downloadable and derived from the same canonical BT.
- Failed validation or simulation exposes diagnostics but no final BT.
- CLI and UI generation use the same service and produce equivalent canonical artifacts for the same accepted plan.
- Concurrent/double-click runs cannot mix secrets, state, or output files.
- Automated handler tests cover upload, validation, provider errors, progress, success, failure, cancellation, download paths, saved projects, keyring behavior, and secret redaction.

## No fake offline generation

The production `generate` command always invokes the selected real LLM provider. Automated tests may use a clearly named deterministic test client, and the repository may contain a committed reference BT for `validate`, `simulate`, and regression tests. Neither the test client nor the reference BT may be reported as an LLM generation result.

## Immediate implementation order

1. Replace the active scenarios with the single three-robot courier scenario.
2. Finalize the canonical JSON BT schema.
3. Correct BT leaf, synchronization, resource, and liveness semantics.
4. Replace the validator with the strict validator described below.
5. Make the deterministic contract simulator execute the same canonical BT object.
6. Rebuild the LLM prompt and bounded correction loop around typed validation errors.
7. Extract the shared `PlannerService` and `ArtifactManager` used by every frontend.
8. Implement `doctor`, `generate`, `validate`, `simulate`, and `render`.
9. Make `behavior_tree.json` the automatic primary output.
10. Add the strict upload schema, downloadable template, and validated example.
11. Implement the Gradio UI, progress stream, cancellation, previews, and downloads.
12. Implement secure session keys, optional OS-keyring persistence, and saved project presets.
13. Add end-to-end CLI and UI tests from scenario input to validated BT artifact.
14. Update the README with one CLI command and one UI launch command.

## Standalone milestone acceptance criteria

- Installation succeeds in a fresh Python 3.10+ virtual environment with `pip install -e .`.
- `lmrbtp doctor` reports actionable configuration status.
- The base install imports no ROS or MuJoCo packages.
- The optional UI install and `lmrbtp ui` launch work without ROS or MuJoCo.
- A real-provider `lmrbtp generate` run creates a nonempty canonical `behavior_tree.json`.
- The generated file parses back into the internal BT model without loss.
- `lmrbtp validate` accepts the generated file independently.
- `lmrbtp simulate` executes that exact file and reaches the declared goal.
- The BT contains all three robot subtrees and explicit cross-robot waits.
- The final path is not created when validation or required simulation fails.
- Re-running validation and simulation does not call the LLM.
- Output JSON is deterministic for the same canonical plan and has a stable hash.
- The command exits nonzero with a clear message when its provider key is absent.
- The UI performs the same provider check before beginning a run.
- A nontechnical user can download the template, upload a scenario, enter an instruction and key, press one Run button, and download the validated BT without using the terminal after UI launch.
- Unit tests, Ruff, and the supported static checks pass.

# Updated project objective

The revised project should demonstrate:

> An LLM constructs complete coordinated behavior trees for two stationary 7-DoF Franka Emika Panda manipulators and one Unitree Go2 mobile manipulator equipped with a Z1 arm. Deterministic validators and simulation verify the exact LLM-generated trees without inserting planning logic. Physical actions are accepted as successful only when supported by controller and sensor evidence.

The LLM is responsible for high-level planning. It must not be responsible for:

- Collision avoidance
- Low-level motion generation
- Emergency stopping
- Deciding whether a physical action actually succeeded
- Creating arbitrary robot skills
- Overriding workspace or safety restrictions
- Treating expected effects as observations

# The single showcase task

## Three-robot courier relay

The robots collaboratively move one lightweight, visually identifiable part from a source container to a destination container.

Physical flow:

```text
Source bin
    ↓
Franka Emika Panda A
    ↓
Source exchange cradle
    ↓
Go2-mounted Z1 arm
    ↓
Secured onboard carrier
    ↓
Go2 locomotion
    ↓
Z1 arm
    ↓
Destination exchange cradle
    ↓
Franka Emika Panda B
    ↓
Target bin
```

There are no direct robot-to-robot handovers. Every transfer happens through a rigid, calibrated fixture.

That design is intentional: simultaneous handovers between a Panda and the Go2-mounted Z1 arm would introduce unnecessary collision, compliance, localization, timing, and safety risks.

## Required physical fixtures

The demonstration needs:

- A source bin reachable only by Franka A
- A source exchange cradle reachable by Franka A and Z1, at different times
- A rigid source docking marker or docking fixture for the Go2
- A Z1 mounting plate whose strength and fasteners have been checked for the actual Go2 edition
- A secured payload carrier mounted inside the verified Z1 reach envelope and as low and central as practical on the Go2
- A documented power arrangement for Z1, its gripper, sensors, and companion computer
- A destination docking marker or fixture
- A destination exchange cradle reachable by Z1 and Franka B, at different times
- A final target bin reachable by Franka B
- Fiducial markers or another reliable localization mechanism
- A lightweight, non-fragile object with an easy grasp geometry

The onboard carrier is important. The Go2 should not walk while Z1 holds an object in free space. The object should be placed into a mechanically stable pocket, cup, clamp, or latching container, after which the Z1 arm is stowed. The carrier location must be demonstrated to be reachable without self-collision and must not create an unacceptable center-of-mass shift.

## Nominal mission

1. All robots report ready.
2. Go2 is docked at the source station.
3. Z1 is stowed.
4. Franka B moves to its safe preparation pose.
5. Franka A verifies the part is in the source bin.
6. Franka A acquires exclusive access to the source exchange zone.
7. Franka A picks the part.
8. Gripper and vision evidence verify the grasp.
9. Franka A places the part in the source cradle.
10. Vision verifies the part is in the cradle.
11. Franka A retreats from the exchange zone.
12. Franka A releases the zone.
13. Z1 explicitly waits for:
    - `part_at(part, source_cradle)`
    - `outside_zone(franka_a, source_exchange_zone)`
    - `docked(unitree_go2_z1, source_dock)`
14. Z1 acquires the source exchange zone.
15. Z1 moves from stowed pose and picks the part.
16. Z1 grasp evidence is verified.
17. Z1 places the part into the onboard carrier.
18. The system verifies that the part is in the carrier.
19. The carrier is verified secure.
20. Z1 returns to its stowed configuration.
21. Z1 releases the source exchange zone.
22. Go2 undocks and navigates through a predefined flat, obstacle-free safe corridor.
23. Go2 arrives at and docks with the destination station.
24. The docking pose is verified within calibrated tolerances.
25. Z1 acquires the destination exchange zone.
26. Z1 retrieves the part from the carrier.
27. Z1 places it in the destination cradle.
28. Vision verifies the part is in the destination cradle.
29. Z1 stows.
30. Go2 releases the destination zone and backs away to a safe parking pose.
31. Franka B waits for:
    - `part_at(part, destination_cradle)`
    - `outside_zone(unitree_go2_z1, destination_exchange_zone)`
    - `arm_stowed(z1)`
32. Franka B acquires the destination exchange zone.
33. Franka B picks the part.
34. Its grasp is verified.
35. Franka B places the part in the target bin.
36. Vision verifies the final goal.
37. Franka B retreats and releases the zone.
38. The coordinator verifies the complete terminal state.

## Parallel behavior

The mission should not be a purely sequential script.

Safe parallelism includes:

- Franka B preparing while Franka A works.
- Franka A returning home while Z1 loads the carrier.
- Franka A performing cleanup while Go2 travels.
- Franka B preparing its final grasp while Go2 approaches, but remaining outside the exchange zone.
- Logging, observation processing, and plan-state monitoring running concurrently.

# Robot capability model

The three robots should be represented as three planning agents, even though the Go2–Z1 agent internally controls two subsystems.

## Franka Emika Panda A

Software identifier: `franka_a`

Hardware: one fixed-base 7-DoF Franka Emika Panda with its installed Franka Hand or other confirmed gripper.

Permitted capabilities:

- `move_to_named_pose`
- `pick_from_source`
- `place_in_source_cradle`
- `open_gripper`
- `close_gripper`
- `retreat_from_source_zone`
- `stop`
- `recover_controller`

Workspace:

- Source bin
- Source exchange cradle
- Source-side safe poses

## Unitree Go2–Z1 mobile manipulator

The Go2 and Z1 form one planning agent with two internal resources:

- `mobile_base`
- `z1_arm`

Base capabilities:

- `stand`
- `stop`
- `navigate_to`
- `dock`
- `undock`
- `back_away`
- `hold_position`

Z1 capabilities:

- `unstow_arm`
- `pick_from_source_cradle`
- `place_in_carrier`
- `pick_from_carrier`
- `place_in_destination_cradle`
- `stow_arm`
- `open_gripper`
- `close_gripper`

Critical internal constraints:

- The Go2 cannot navigate while Z1 is not stowed.
- The Go2 cannot navigate while the carrier is unsecured.
- Z1 cannot manipulate unless the base is stationary.
- Exchange manipulation requires a verified docking pose.
- Base recovery motions cannot start while Z1 occupies an exchange zone.
- Z1 and the base may have separate execution adapters, but the planner treats them as one robot with mutually constrained resources.
- Go2 navigation is restricted to the qualified speed, acceleration, terrain, payload, and posture envelope established during hardware testing.
- The combined Z1, mount, gripper, carrier, sensors, compute, cabling, and object mass must remain within the approved Go2 payload and center-of-mass envelope.
- Z1 power loss, controller loss, or failure to prove the stow pose prevents all Go2 locomotion.

## Franka Emika Panda B

Software identifier: `franka_b`

Hardware: one fixed-base 7-DoF Franka Emika Panda with its installed Franka Hand or other confirmed gripper.

Permitted capabilities:

- `move_to_named_pose`
- `pick_from_destination_cradle`
- `place_in_target_bin`
- `open_gripper`
- `close_gripper`
- `retreat_from_destination_zone`
- `stop`
- `recover_controller`

Workspace:

- Destination exchange cradle
- Target bin
- Destination-side safe poses

Both fixed manipulators are confirmed as the older Franka Emika Panda/FER generation rather than FR3. Their skill servers should use a Panda-compatible `libfranka` and preferably `franka_ros2`, after matching each arm’s system image to a supported library version. Franka’s official documentation covers Panda/FER through FCI and provides robot state, gripper control, collision information, and ROS/ROS 2 integration. The hardware stack is Linux-based rather than Windows-based. [Franka Control Interface](https://support.franka.de/docs/index.html), [Franka ROS 2 documentation](https://support.franka.de/docs/franka_ros2.html)

# Safety invariants

These rules are mandatory and enforced outside the LLM.

## Exchange zones

- Only one manipulator may own an exchange zone.
- A robot may enter an exchange zone only after acquiring its resource lock.
- A lock is released only after observed retreat, not merely after the retreat command returns.
- Panda A and Z1 must never simultaneously occupy the source exchange zone.
- Z1 and Panda B must never simultaneously occupy the destination exchange zone.
- Go2 locomotion is forbidden while either exchange-zone lock is held by Z1.

## Mobile manipulation

- Z1 must be inside the calibrated stow tolerance before Go2 can walk.
- Go2 velocity must be zero before Z1 can unstow.
- A destination or source docking predicate must be freshly observed before Z1 manipulation.
- Loss of docking confidence immediately cancels arm motion.
- The carrier must report the payload secure before locomotion.
- A missing or dropped payload causes an immediate stop and operator-required recovery.
- Go2 locomotion is limited to the flat, qualified route used in the demonstration; stairs, running, jumping, aggressive turning, and high-dynamic motions are outside the project scope.
- A companion watchdog must be able to issue a Go2 stop independently of the LLM and behavior-tree tick loop.

## Fixed manipulators

- Each Panda must be in its safe pose before Go2 enters the nearby corridor or station, according to the lab’s risk assessment.
- Controller collision or reflex states produce a mission fault.
- A Panda recovery command cannot automatically resume the mission without refreshing the observed world state.
- Motion limits, collision thresholds, and trajectory safety remain the responsibility of the Panda controller and motion-planning layer.

## Independent safety layer

The behavior-tree coordinator is not a safety-rated controller.

Real execution requires:

- Accessible physical emergency stops
- A laboratory-approved risk assessment
- Speed and workspace limits
- A designated operator
- Clear separation from people
- Reliable command cancellation
- Robot heartbeats
- Network-loss behavior
- Independent low-level controller protections

Software resource locks are coordination tools, not substitutes for safety-rated hardware or procedures.

# Updated software architecture

## Immediate standalone architecture

```text
Natural-language instruction + uploaded courier scenario
                  ↓
        CLI or local Gradio interface
                  ↓
             PlannerService
                  ↓
          Real LLM provider client
                  ↓
       Parse canonical BT candidate
                  ↓
       Strict deterministic validator
          ↖ typed errors │ valid
             correction  ↓
        Exact-roundtrip verification
                  ↓
       Revalidate unchanged BT object
                  ↓
   Deterministic contract simulation
                  ↓
 ArtifactManager atomic behavior_tree.json
```

The standalone core owns the schema, parser, exact-roundtrip check, validator, contract simulator, LLM correction loop, serialization, hashing, shared application service, CLI, and local UI. It must not import ROS 2, MuJoCo, `libfranka`, SDK2, or `z1_sdk`.

## Deferred ROS 2 execution architecture

The following architecture is retained for later MuJoCo and hardware integration. It is not part of the first working-program milestone.

```text
User instruction
       ↓
LLM task-level planner
       ↓
Canonical three-robot BT plan
       ↓
Deterministic validation
  ├── schema
  ├── capabilities
  ├── causal support
  ├── cross-robot synchronization
  ├── workspace/reachability
  ├── resource exclusion
  ├── docking/stow constraints
  └── liveness
       ↓
Independent runtime safety layer
  ├── observation verification
  ├── timeout/cancellation enforcement
  ├── emergency-stop supervision
  └── controller limits outside LLM authority
       ↓
ROS 2 multi-robot coordinator
  ├── Panda A executor
  ├── Go2–Z1 agent
  │    ├── Go2 base executor
  │    └── Z1 executor
  └── Panda B executor
       ↓
Versioned evidence-aware world state
       ↓
ROS 2 execution backend selected at launch
  ├── MuJoCo skill servers and physics observers
  └── Hardware skill servers and physical observers
```

## Deployment architecture

This section is deferred until the standalone milestone passes its acceptance criteria.

The current Windows workspace can remain suitable for offline development, tests, prompt evaluation, and reporting.

Simulation and hardware execution should run on Linux. The primary integration baseline is Ubuntu 22.04 with ROS 2 Humble and CycloneDDS because Unitree’s official Go2 ROS 2 support recommends that combination. Each Panda system image must still be matched to a compatible `libfranka`/`franka_ros2` release before this baseline is frozen. If a Panda cannot use the selected ROS 2 stack directly, its control process should remain on a separately supported Linux host and expose only the common ROS 2 high-level skill protocol through a bridge; the project must not upgrade robot system images merely to simplify software integration without laboratory approval.

Recommended nodes:

```text
mission_coordinator
world_state_server
safety_supervisor
perception_server
panda_a_skill_server
panda_b_skill_server
go2_base_skill_server
z1_skill_server
execution_recorder
mujoco_simulator
mujoco_observation_server
```

When ROS 2 work begins, packages should be organized explicitly:

```text
ros2_ws/src/
├── mr_bt_interfaces
├── mr_bt_coordinator
├── mr_bt_world_state
├── mr_bt_safety
├── mr_bt_perception
├── mr_bt_panda_adapter
├── mr_bt_go2_adapter
├── mr_bt_z1_adapter
├── mr_bt_mujoco
├── mr_bt_bringup
└── mr_bt_recording
```

Simulation and hardware must expose the same task-level action names and result schema. Backend selection happens through launch configuration and node composition, never through special behavior-tree nodes that exist only in simulation.

Recommended physical process separation:

- Mission workstation: planner, validator, coordinator, world state, perception integration, and recorder
- Panda A control host: Panda A FCI, motion planning, gripper, and skill server
- Panda B control host: Panda B FCI, motion planning, gripper, and skill server
- Go2 companion/onboard host: SDK2/ROS 2 base interface, navigation/docking skill, and watchdog
- Z1 control host or Go2 companion process: `z1_controller`, `z1_sdk`, gripper interface, and Z1 skill server
- Simulation workstation: MuJoCo physics server, `mjctrl`-derived Panda controllers, Go2/Z1 simulation controllers, simulated observation providers, and `/clock`

The Python planner must not participate in either Panda’s low-level real-time control loop or in Go2/Z1 motor-level control. It should dispatch high-level skills to dedicated control processes.

# Detailed implementation plan

## Deferred Hardware Phase H0 — Hardware discovery and compatibility proof

This phase does not block the standalone BT generator. Begin it only when hardware preparation can proceed without delaying the working CLI and canonical BT output.

Before structural repository changes, record:

- Go2 edition and serial number
- Go2 firmware, SDK2, motion service, and ROS 2 interface versions
- Whether the Go2 edition exposes all required developer interfaces
- Go2 payload limit applicable to that exact edition and configuration
- Go2 battery, external power, and companion-computer arrangement
- Z1 serial number, controller version, SDK version, and gripper type
- Z1 mounting transform, fastening method, cable routing, and self-collision envelope
- Whether `z1_controller` and `z1_sdk` already work while mounted on this Go2
- Measured mass of Z1, mount, gripper, carrier, sensors, compute, cabling, and test object
- Measured combined center of mass and approved Go2 walking envelope
- Panda A and Panda B serial numbers and system images
- Panda A and Panda B compatible `libfranka`, `franka_ros2`, MoveIt, and gripper versions
- FCI availability and FCI mode on both Pandas
- Installed gripper and tool-center-point calibration for each Panda
- Control PCs and Linux distributions
- ROS 2 distribution, RMW implementation, DDS domain allocation, and exact package versions
- Network interfaces and IP configuration
- Cameras, marker systems, and calibration
- Existing Panda MoveIt configurations and collision scenes
- Go2 navigation, localization, obstacle-stop, and docking capabilities
- Emergency-stop arrangement
- Workcell geometry
- Pinned `mjctrl` commit, Apache-2.0 notices, and the exact Panda model files reused or adapted
- Pinned MuJoCo version; `mjctrl` requires MuJoCo 3.1.0 or later
- Pinned official `unitree_mujoco` commit and Go2 MJCF source
- Source, license, conversion procedure, and numerical validation for the Z1 MuJoCo model
- The selected Go2 low-level locomotion controller or policy used in MuJoCo

Produce:

```text
config/hardware/
├── robots.yaml
├── skills.yaml
├── sensors.yaml
├── workspaces.yaml
├── network.yaml
├── frames.yaml
├── mass_properties.yaml
├── software_versions.lock
├── ros2_domains.yaml
├── simulation_sources.yaml
└── safety.yaml
```

`simulation_sources.yaml` must record repository URL, commit SHA, license, local modifications, model origin, controller origin, and validation status for `mjctrl`, the Panda model, the Go2 model, the Z1 model, and any Go2 locomotion policy. Third-party code must retain its license and attribution; copied or adapted controller code must be distinguishable from code written in this project.

Phase 0 physical subtests must be completed before object transport:

1. Run each Panda independently through its named poses and source/destination pick-place skill.
2. Run Z1 independently while Go2 is powered, stationary, and physically restrained according to lab procedure.
3. Verify Z1 stow and unstow without Go2 self-collision.
4. Verify that Z1 reaches both exchange cradles and the onboard carrier from the measured docking poses.
5. Verify the Go2 can stand, stop, and walk the flat route with the complete mounted hardware but no transported object.
6. Repeat with the lightweight test object secured in the carrier.
7. Measure docking repeatability and define tolerances from data rather than guessing them.
8. Demonstrate that loss of the Z1 heartbeat, an unproved stow state, or an unsecured carrier prevents base motion.

Exit criterion: every scenario action maps to an existing callable skill with status, timeout, and cancellation semantics, and the complete Go2–Z1 assembly has passed the lab-approved mass, power, thermal, stability, reachability, docking, and stop tests.

## Phase 1 — Replace the project claims

The project documentation and paper should define three layers:

1. Complete LLM-generated multi-robot Behavior Trees.
2. Deterministic verification of the unchanged generated trees.
3. Robot-specific expert skills and independent runtime safety enforcement.

The system must report that every accepted planning node came from the LLM and that no semantic rewrite was applied.

Remove claims that:

- The LLM alone creates a complete safety-valid controller.
- Symbolic effects prove physical completion.
- A generated ROS file is equivalent to robot integration.
- State stagnation alone proves deadlock.
- Contract-defined anomalies constitute failure detection.

Exit criterion: every public claim has a test, recorded artifact, or clearly marked limitation.

## Phase 2 — Remove old scenarios and create the courier scenario

Create one active scenario package:

```text
scenarios/three_robot_courier/
├── instruction.txt
├── domain.yaml
├── scenario.yaml
├── hardware_requirements.yaml
├── safety_invariants.yaml
├── reference_plan.json
├── simulation_profile.yaml
└── acceptance_tests.yaml
```

Remove old scenarios from active CLI choices, prompts, experiment matrices, screenshots, and documentation.

The domain should use typed entities:

- `robot`
- `fixed_manipulator`
- `mobile_manipulator`
- `mobile_base`
- `arm`
- `part`
- `location`
- `dock`
- `zone`
- `carrier`

The active scenario must instantiate exactly these robot identities:

```text
franka_a: Franka Emika Panda, 7 DoF, fixed source station
unitree_go2_z1: Unitree Go2 base plus Unitree Z1 arm, mobile courier
franka_b: Franka Emika Panda, 7 DoF, fixed destination station
```

## Phase 3 — Build an evidence-aware world model

Each predicate must store:

```text
predicate
arguments
value: TRUE | FALSE | UNKNOWN
source
timestamp
confidence
expiry
run_id
supporting evidence
```

Core predicates include:

- `robot_ready(robot)`
- `part_at(part, location)`
- `holding(robot, part)`
- `gripper_empty(robot)`
- `docked(unitree_go2_z1, dock)`
- `base_stationary(unitree_go2_z1)`
- `arm_stowed(z1)`
- `carrier_empty(carrier)`
- `carrier_secured(carrier)`
- `outside_zone(robot, zone)`
- `owns_resource(robot, resource)`
- `path_clear(route)`
- `goal_verified(part)`

Hardware mode must never turn an expected effect directly into `TRUE`.

For example:

```text
place_in_source_cradle succeeded
```

does not itself establish:

```text
part_at(part, source_cradle)
```

That predicate becomes true only after a designated observer confirms it.

## Phase 4 — Replace the plan schema

Use one canonical representation.

Each action should contain:

- Unique task ID
- Assigned robot
- Capability name
- Typed arguments
- Required resources
- Preconditions
- Expected postconditions
- Timeout
- Recovery policy reference

Synchronization should exist only as executable `WaitFor` nodes. Separate synchronization metadata should be generated from those nodes.

Supported BT nodes:

- `ReactiveSequence`
- `Fallback`
- `ParallelAll`
- `Condition`
- `WaitFor`
- `Action`
- `AcquireResource`
- `ReleaseResource`

Each robot receives one tree:

```text
ParallelAll
├── Panda A (`franka_a`) tree
├── Go2–Z1 (`unitree_go2_z1`) tree
└── Panda B (`franka_b`) tree
```

A derived task graph can be generated for visualization, but it is not a second source of truth.

## Phase 5 — Implement correct BT behavior

Semantics:

- `Condition`: immediate `SUCCESS` or `FAILURE`
- `WaitFor`: `RUNNING` until a predicate is observed or timeout expires
- `Action`: asynchronous dispatch with `RUNNING`, terminal success, failure, cancellation, or timeout
- `AcquireResource`: waits for deterministic coordinator ownership
- `ReleaseResource`: releases ownership and records an event
- `ReactiveSequence`: rechecks safety and environmental conditions
- `Fallback`: skips work when the goal/postcondition is already verified
- `ParallelAll`: completes only when all robot missions terminate successfully

Every physical action should use a target-conditioned unit subtree:

```text
Fallback
├── Verified postcondition
└── ReactiveSequence
    ├── Runtime safety conditions
    ├── Required local conditions
    ├── Explicit remote waits
    ├── Resource acquisition
    ├── Action
    ├── Post-action observation refresh
    └── Verified postcondition
```

## Phase 6 — Implement strict validation

The validator must cover:

### Structural checks

- Schema version
- Valid tree shape
- Nonempty composites
- Unique node and task IDs
- Exactly three participating robot trees
- Valid robot names
- Valid parallel thresholds

### Domain checks

- Known predicates and actions
- Correct argument counts
- Correct object types
- Known constants
- Legal parameter ranges

### Capability checks

- Panda A can execute only source-station Panda skills
- Panda B can execute only destination-station Panda skills
- Z1 manipulation uses Z1 capabilities
- Go2 movement uses SDK2-backed base capabilities
- Assigned workspace is physically reachable
- No plan assigns a 7-DoF Panda trajectory to the 6-DoF Z1 or a Z1 trajectory to a Panda
- Every named pose and trajectory belongs to the correct calibrated robot model, tool, base transform, and controller version

### Mobile-manipulator checks

- Navigation is dominated by `arm_stowed(z1)`
- Navigation is dominated by `carrier_secured(carrier)` when loaded
- Z1 manipulation is dominated by `base_stationary(unitree_go2_z1)`
- Z1 exchange actions are dominated by the corresponding dock condition
- Arm motion and base motion cannot be parallel
- Go2 navigation actions are rejected unless the configured hardware profile has passed payload/stability qualification
- Z1 actions are rejected when their target pose lies outside the calibrated Go2-body-to-Z1-base reach model

### Causal checks

- Every precondition is initially true, locally produced, externally observed, or explicitly awaited
- Every goal has a valid producer
- No consumer executes before its producer
- Delete effects are considered
- The part cannot be in two locations simultaneously
- Two robots cannot both hold the same part

### Synchronization checks

- Panda A → Z1 synchronization is explicit
- Z1 → Go2-base synchronization is explicit
- Go2 docking → Z1 synchronization is explicit
- Z1 → Panda B synchronization is explicit
- A wait after its consumer is rejected
- Same-robot synchronization is rejected unless it represents a genuine internal subsystem barrier
- Every wait names an actual producer or external observer

### Resource checks

- Source and destination exchange zones are mutually exclusive
- All lock paths have releases
- Cancellation releases locks
- No plan allows locomotion while an exchange zone is occupied
- No parallel branch creates conflicting resource ownership

### Liveness checks

- Static wait-for cycles
- Missing producers
- Unreachable waits
- Failed producer with active consumers
- Waits without timeout
- Resource cycles
- Stale external observations

## Phase 7 — Build the layered simulator and integrate `mjctrl`

For the standalone milestone, implement only Level A. Level B and all `mjctrl` physics integration are the next milestone after canonical BT generation works end to end. The project should keep two deliberately different simulation levels.

### Level A: deterministic contract simulator

The contract simulator remains the fast, deterministic reference for planner, validator, BT, synchronization, recovery, and liveness tests. It contains:

- Panda A state
- Panda B state
- Go2 base state
- Z1 arm state
- Carrier state
- Object state
- Docking state
- Resource ownership
- Observation delays
- Action durations
- Network/controller events

This backend applies declared transition effects because it is explicitly a symbolic model. It is not used as evidence of physical feasibility.

### Level B: ROS 2-connected MuJoCo physics simulator

Build a single MuJoCo world for the courier task containing:

- Two independently namespaced 7-DoF Panda models
- Both Panda hands and fingertip collision geometries
- One Go2 model
- One Z1 model rigidly mounted to the Go2 with the measured mounting transform
- The source bin and source exchange cradle
- The secured onboard carrier
- The destination exchange cradle and target bin
- The manipulated object
- Collision geometry for the exchange zones and relevant workcell structures
- Cameras or perfect-state physics observers, explicitly labeled by simulation mode

The simulator must run headless for automated tests and optionally with the MuJoCo viewer for demonstrations. It publishes `/clock`, and every simulation ROS 2 node uses `use_sim_time:=true`.

### `mjctrl` integration boundary

Use `mjctrl` as a pinned and attributed source for the Panda MJCF and differential-IK/nullspace controller mathematics. Do not run `diffik_nullspace.py` unchanged as a production simulator node.

Refactor the reusable controller into a testable class such as:

```text
MjctrlPandaController
├── set_target_pose()
├── set_posture_target()
├── step()
├── at_target()
├── joint_limit_margin()
├── singularity_metric()
└── cancel()
```

The adapted controller should retain the useful `mjctrl` behavior:

- MuJoCo site Jacobian
- Damped least-squares inverse
- Seven-joint nullspace posture bias
- Maximum joint-velocity limiting
- Joint-range clipping
- Gravity compensation where appropriate

It must add what the demonstration requires:

- Targets supplied by a ROS 2 skill server rather than a mouse-controlled mocap body
- Independent namespaced joint, actuator, site, and keyframe identifiers for Panda A and Panda B
- Translation and orientation tolerances
- Timeout and cancellation
- Joint-limit avoidance and explicit near-singularity failure
- Collision/contact abort rules
- Panda hand opening, closing, and grasp control
- Structured completion and failure results
- Deterministic stepping for tests
- No direct assertion of symbolic postconditions

The upstream example loads `panda_nohand.xml`; therefore the project must deliberately use and validate the hand-enabled model for grasping rather than assuming the example already performs pick-and-place.

### Combined Panda scene construction

The two Panda instances cannot share unqualified names. The scene builder must prefix or namespace:

- Bodies
- Joints
- Actuators
- Sites
- Tendons
- Sensors
- Keyframes
- Materials and meshes where MuJoCo requires unique names

The resulting scene must verify that Panda A and Panda B have separate state indices, separate commands, correct base transforms, correct tool frames, and no accidental cross-control.

### Z1 simulation controller

`mjctrl` contains no Z1 model or controller. Add a separate six-axis Z1 MuJoCo controller using an official Unitree model where available, or a documented URDF-to-MJCF conversion if necessary. Validate joint order, joint limits, inertias, flange transform, gripper transform, and forward kinematics against the real Z1/SDK before treating it as the simulation model.

The Z1 simulation skill server should implement named poses and Cartesian target execution with bounded differential IK, joint/velocity limits, self-collision checks, timeout, cancellation, and gripper control. Because Z1 has no redundant seventh joint, it must not reuse the Panda nullspace term blindly.

### Go2 simulation controller and explicit limitation

Use Unitree’s official [`unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco) Go2 model and SDK2/DDS bridge where compatible. That project is intended primarily for low-level controller verification and does not itself supply the task’s high-level autonomous navigation controller. A real Go2 locomotion controller or policy compatible with the selected Go2 model must therefore be selected, pinned, tested, and recorded separately.

The full MuJoCo courier run may be claimed only when the Go2 actually walks under that controller. Moving the Go2 body by directly changing its pose, using a mocap body, or replaying an animation may be useful for visualization but does not count as simulated locomotion and must not be presented as such.

If no suitable Go2 controller is available, the honest reduced result is:

- Full BT and coordination testing in the contract simulator
- Dual-Panda and stationary Go2–Z1 manipulation testing in MuJoCo
- No claim of a full physics-based mobile courier run
- Go2 locomotion validated only through the real robot or a later controller integration

### Physics-derived observations

The MuJoCo observation server must derive predicates from physics state rather than applying action effects:

- `part_at`: object pose lies inside the calibrated fixture volume and is stably supported
- `holding`: appropriate gripper contacts exist and object pose remains stable relative to the tool
- `gripper_empty`: no grasp contact and measured opening is consistent
- `docked`: Go2 base pose/yaw lies inside tolerance and velocity is near zero
- `arm_stowed`: Z1 joints lie inside the stow tolerance
- `carrier_secured`: object lies in the carrier and the simulated latch/sensor condition is true
- `outside_zone`: complete robot collision geometry lies outside the configured zone
- `goal_verified`: object is stably located in the target bin

Controller completion and predicate truth are separate events. Reaching a target pose does not automatically prove a grasp, placement, docking result, or goal.

### MuJoCo failure cases

The physics simulator should produce and detect:

- Unreachable Panda or Z1 target
- IK singularity or insufficient joint-limit margin
- Panda or Z1 controller timeout
- Collision abort
- Failed grasp due to missing contact
- Object slip or drop
- Placement outside the cradle volume
- Carrier not secured
- Dock pose outside tolerance
- Go2 controller timeout or fall, if a real locomotion controller is integrated
- ROS 2 action cancellation
- Stale simulated observation
- Resource contention

Deliberately injected disturbances remain labeled fault injection, but the resulting failure detection must come from controller or MuJoCo state, not from an oracle that simply declares the action failed.

### Phase 7 exit criteria

- The contract simulator remains deterministic and passes all BT semantic tests.
- Both Panda instances execute independent `mjctrl`-derived controllers in one MuJoCo scene.
- Panda hands perform contact-based grasp and release.
- Z1 completes carrier load/unload motions under its own validated controller.
- ROS 2 skill actions and observation messages are identical between simulation and hardware backends.
- Simulation success requires physics-derived postconditions.
- The exact upstream commits and licenses are recorded.
- A full physics courier claim is enabled only after a genuine Go2 locomotion controller passes its acceptance tests.

## Deferred Phase R1 — Implement the ROS 2 interfaces and robot adapters

Do not implement this phase until the standalone milestone is accepted. Nothing in the base CLI package may depend on the interfaces introduced here.

### Common ROS 2 interface

Create `mr_bt_interfaces` with a backend-independent task-level API. At minimum it should define:

```text
ExecuteSkill.action
WorldObservation.msg
ExecutionEvent.msg
RobotHeartbeat.msg
SafetyState.msg
RefreshObservations.srv
```

`ExecuteSkill.action` should carry the immutable run ID, validated plan hash, task ID, robot ID, skill name, typed/validated arguments, and deadline. Feedback reports phase and progress. The result reports terminal status, failure code, controller evidence IDs, observation evidence IDs, and timestamps. ROS 2 action cancellation is mandatory.

Both backends implement the same endpoints:

```text
/franka_a/execute_skill
/franka_b/execute_skill
/unitree_go2_z1/base/execute_skill
/unitree_go2_z1/z1/execute_skill
```

The behavior tree must not know whether those action servers are backed by MuJoCo or hardware.

Use ROS 2 lifecycle nodes for skill servers, observation providers, coordinator, and recorder. Hardware commands are accepted only when the relevant server is active and the safety supervisor reports an executable state.

### ROS 2 time, QoS, and recording

- MuJoCo publishes `/clock`; simulation nodes use ROS time.
- Hardware nodes use synchronized host clocks and include source timestamps.
- Action goals, results, safety state, and execution events use reliable delivery.
- High-rate joint and sensor streams may use sensor-data QoS where appropriate.
- Heartbeat topics use deadline/liveliness policies so loss is detectable.
- Static configuration and validated plan identity use transient-local delivery where useful.
- `rosbag2` records action goals/results, observations, safety state, TF, robot state, `/clock` in simulation, and execution events.

### ROS 2 domain and backend isolation

Simulation and hardware must never share an accidental DDS domain. Define separate domain IDs and network-interface profiles for:

- Offline/local MuJoCo
- Hardware-in-loop
- Physical Go2/Panda execution

The Go2 uses CycloneDDS-compatible Unitree messages, so `ROS_DOMAIN_ID`, `RMW_IMPLEMENTATION`, and `CYCLONEDDS_URI` must be configured explicitly in launch/environment files. A simulation process must not be able to command the physical Go2 through a stale or incorrect DDS configuration.

Provide explicit bringup entry points:

```text
ros2 launch mr_bt_bringup courier_sim.launch.py
ros2 launch mr_bt_bringup courier_hil.launch.py
ros2 launch mr_bt_bringup courier_hardware.launch.py
```

Hardware launch requires the validated hardware profile and plan hash. Simulation launch refuses real robot network interfaces. Hardware launch refuses `use_sim_time`.

### Franka Emika Panda adapters

Create separate namespaced skill servers:

```text
/franka_a/execute_skill
/franka_b/execute_skill
```

Each server should:

- Receive high-level skill goals
- Use MoveIt or existing validated trajectories
- Report feedback
- Support cancellation
- Return controller error codes
- Expose robot and gripper state
- Trigger observation refresh
- Never claim the symbolic goal automatically

Each Panda should have a separate `arm_id`, ROS namespace, controller manager, robot IP, motion-planning group, gripper namespace, and skill-server lifecycle. A reflex or error on one Panda must fail the tasks that depend on that Panda; whether it should stop the other Panda and Go2 is a safety-policy decision, not an accidental consequence of sharing a controller process.

The MuJoCo Panda action servers use the `mjctrl`-derived controller. The hardware Panda action servers use the approved MoveIt/libfranka path. They share task-level skill names and result codes but do not pretend to share the same low-level controller.

### Unitree Go2 base adapter

The Go2 base adapter should:

- Use the confirmed Go2 SDK2/ROS 2 interface over CycloneDDS.
- Use high-level Go2 motion services for standing, controlled velocity, stopping, and posture management rather than publishing low-level joint commands.
- Wrap base movement in `stand`, `stop`, `navigate_to`, `dock`, `undock`, `back_away`, and `hold_position` skills.
- Implement `navigate_to` and `dock` as closed-loop skills using the project’s measured localization source; sending an open-loop velocity for a fixed time is not sufficient.
- Do not permit the LLM or coordinator to publish raw motor commands.
- Expose pose, velocity, mode, heartbeat, docking result, and stop capability.
- Reject locomotion unless the Z1 stow predicate, payload-secured predicate, hardware-qualified flag, and route-clear predicate are fresh and true.
- Clamp speed and acceleration to values approved during physical qualification, regardless of what appears in an LLM plan.

The MuJoCo Go2 action server must dispatch to the selected low-level simulation controller. The hardware Go2 action server dispatches through SDK2/ROS 2. A kinematic pose setter must never be registered under the same `navigate_to` skill name.

### Z1 adapter

The Z1 uses its dedicated `z1_controller` and `z1_sdk`; it is not treated as if it were part of the Go2 SDK2 joint interface. Implement a ROS 2 action wrapper around the real Z1 SDK and keep the vendor control process isolated from the behavior-tree coordinator.

It should provide:

- Named-pose motion
- Cartesian or prevalidated trajectory execution
- Gripper control
- Joint state
- Stow verification
- Cancellation
- Controller failure reporting

The Z1 adapter must publish and validate the complete transform chain:

```text
world
└── go2_body
    └── z1_base
        └── z1_flange
            └── z1_gripper_tcp
```

The measured `go2_body -> z1_base` mounting transform must come from calibration/configuration, not from a guessed URDF offset.

The MuJoCo Z1 server and hardware Z1 SDK server must use the same task-level skill names, pose-frame conventions, tolerances, cancellation results, and failure taxonomy.

### Network and time

The hardware deployment must include:

- Unique namespaces
- Nonconflicting DDS configuration
- Stable wired networking where possible
- Clock synchronization
- Heartbeat monitoring
- Run IDs on every command and observation
- Protection against delayed results from earlier runs
- Separate simulation, hardware-in-loop, and physical DDS domain profiles
- `ros2 doctor`, topic/action discovery, TF-tree, and lifecycle-state preflight checks

## Phase 9 — Implement perception and verification

Required observation providers:

| Predicate | Evidence |
|---|---|
| `part_at` | Camera detection in a calibrated fixture region |
| `holding` | Gripper state plus visual disappearance/attachment evidence |
| `docked` | Fiducial-relative pose and near-zero base velocity |
| `arm_stowed` | Z1 joints within calibrated tolerance |
| `carrier_secured` | Physical switch, latch sensor, or reliable camera observation |
| `outside_zone` | Robot pose/FK outside the calibrated zone |
| `robot_ready` | Controller state and fresh heartbeat |
| `goal_verified` | Part detected in the target bin |

Additional Go2–Z1 observations must include:

- Go2 motion mode and commanded/measured velocity
- Go2 body pose and orientation
- Dock-relative position and yaw error
- Go2 and Z1 heartbeats
- Z1 controller state and measured joints
- Z1 stow-tolerance result
- Carrier sensor state
- Hardware-qualified configuration hash

Additional Panda observations must include:

- Seven measured joint positions and velocities
- Panda robot mode and error/reflex state
- End-effector pose for the configured tool
- Gripper width and grasp result
- Fresh safe-pose/outside-zone result

If no sensor can establish an essential predicate, the system must return `UNKNOWN` and stop. It must not substitute an action’s expected effect.

## Phase 10 — Implement bounded recovery

Supported recovery should be deliberately narrow.

### Franka or Z1 grasp failure

1. Stop the current action.
2. Open gripper if safe.
3. Retreat to a known pose.
4. Refresh object perception.
5. Recompute the grasp using the approved skill.
6. Retry at most the configured number of times.
7. Stop for operator intervention if verification still fails.

### Docking failure

1. Stop.
2. Verify Z1 is stowed.
3. Back away using a predetermined recovery motion.
4. Refresh localization.
5. Retry docking once.
6. Abort if the measured docking pose remains outside tolerance.

### Carrier not secured

- Go2 must not move.
- Z1 may retry placement only after refreshing the carrier and object state.
- If no carrier sensor exists, this automatic recovery is unavailable.

### Navigation fault

- Go2 stops.
- The other two robots remain in safe poses.
- The mission does not resume until pose, route, carrier, and robot state are revalidated.

### Lost part

- Stop all relevant motion.
- Mark the part location `UNKNOWN`.
- Require operator recovery unless perception safely identifies a reachable location and an approved recovery skill exists.

### Reassignment

Automatic cross-robot reassignment should normally be disabled in this scenario because the robots have different bodies and fixed workspaces.

For example, Panda B cannot truthfully replace the Go2 courier, and Z1 may not reach Panda A’s source bin.

Reassignment may be enabled only for a particular task if reachability and skill equivalence are physically certified. Otherwise, the correct result is “no valid recovery,” followed by a safe stop.

## Phase 11 — Rebuild the LLM planner

The prompt contains:

- The user instruction
- Three robot capability profiles
- Typed domain
- Workspace restrictions
- Initial observed state
- Goal state
- BT schema
- Required explicit waits
- Safety-independent resource rules

Pipeline:

1. Interpret the instruction.
2. Generate a task-level three-robot BT.
3. Parse into the canonical schema.
4. Run deterministic validation.
5. Return structured errors to the model.
6. Allow a bounded number of correction attempts.
7. Reject any candidate that omits mandatory guards, waits, resource operations, or timeouts.
8. Revalidate the unchanged LLM-generated plan.
9. Freeze the plan and calculate its hash.
10. Require that exact hash for hardware execution.

No deterministic stage may add planning nodes. Observation refresh, resource acquisition/release, heartbeat checks, arm-stow guards, base-stationary guards, and finite waits must appear in the LLM-generated BT when required. Independent robot controllers may enforce emergency stops and hard safety limits outside the BT, but they must not be reported as LLM planning results.

## Phase 12 — Testing ladder

Testing should progress through increasingly physical stages.

The standalone release requires Levels 1 and 2 plus the standalone CLI acceptance tests. Levels 3 onward are later MuJoCo, ROS 2, and physical-integration milestones.

### Level 1: Core tests

- Schema
- BT semantics
- Capability checking
- Causal validation
- Explicit synchronization
- Resource locking
- Deadlock detection
- Failure propagation

### Level 2: Contract simulation

- Full nominal courier run
- All fault-injection cases
- Deterministic replay
- Cancellation
- No false effects
- No false goal success

### Level 3: `mjctrl`-derived Panda controller tests

- Both Panda instances have independent state and command indices.
- Cartesian targets converge within declared translation/orientation tolerances.
- Nullspace posture bias does not corrupt the primary end-effector task.
- Velocity limiting and joint-range enforcement work.
- Near-singular and unreachable targets fail with structured codes.
- Cancellation stops controller progress.
- The hand-enabled Panda model performs grasp and release through contacts.
- Action completion without the required physics predicate does not return task success.

### Level 4: MuJoCo component integration

- Panda A source pick/place
- Panda B destination pick/place
- Stationary Go2-mounted Z1 carrier load/unload
- Physics-derived `holding`, `part_at`, `arm_stowed`, `carrier_secured`, and `outside_zone`
- Headless deterministic stepping and visual replay
- ROS 2 `/clock` and `use_sim_time`
- Rosbag replay produces the same mission event ordering

### Level 5: Full MuJoCo courier integration

This level is enabled only after a genuine Go2 low-level locomotion controller is integrated.

- Go2 stands and walks under the selected controller.
- Z1 remains stowed during locomotion.
- The transported object remains secured under physics.
- Docking closes the loop on measured simulated pose.
- The complete Panda A → Go2–Z1 → Panda B mission succeeds from physics-derived observations.
- Controller, contact, drop, docking, and cancellation failures propagate through ROS 2 into the BT.

If these conditions are not met, the test report must call Level 5 unavailable rather than substituting kinematic base motion.

### Level 6: ROS 2 adapter tests without physical motion

- Robot connectivity
- Heartbeats
- State acquisition
- Command rejection
- Cancellation
- Namespace isolation
- Evidence timestamps
- Lifecycle activation/deactivation
- Simulation/hardware endpoint equivalence
- DDS domain isolation
- Hardware launch refusal when `use_sim_time` is enabled
- Simulation launch refusal when configured on a physical robot interface

### Level 7: Individual physical robot tests

- Panda A source pick/place using all seven joints through its real Panda skill server
- Panda B destination pick/place using all seven joints through its real Panda skill server
- Go2 navigation and docking with Z1 stowed
- Z1 manipulation while Go2 is stationary
- Carrier loading and unloading
- Go2–Z1 mass, center-of-mass, power, thermal, stop, and low-speed stability checks

### Level 8: Paired physical integration

- Panda A → source cradle → Z1
- Z1 → destination cradle → Panda B
- Go2 docking with both Pandas parked
- Exchange-zone lock verification

### Level 9: Empty full-system rehearsal

Run the entire three-robot mission without an object, using motion-safe placeholder steps and real robot states. This is a physical rehearsal, not counted as task success.

### Level 10: Full physical execution

Run with:

- Lightweight foam or plastic part
- Reduced motion speed
- Safety operator
- Clear workcell
- All logging active
- Immediate emergency-stop access

### Level 11: Controlled physical failures

Only safe, approved cases:

- Part deliberately absent before execution
- Vision temporarily unavailable before motion begins
- Dock pose intentionally outside tolerance
- Gripper intentionally prevented from confirming grasp
- Skill server cancellation before entering the shared zone

Do not induce unsafe walking, dropping, collision, or live network-loss experiments unless the lab’s procedure specifically permits them.

# Deferred physical acceptance criteria

Before calling the program finished:

- Every planned action maps to a real skill.
- Every essential postcondition has a real observer.
- Missing observations cannot produce success.
- Both exchange zones remain mutually exclusive in every run.
- Go2 never walks with Z1 unstowed.
- Go2 never walks with an unsecured payload.
- Z1 never manipulates without verified docking and zero base velocity.
- Controller failure prevents downstream dependent actions.
- Cancellation stops the active skill and releases coordination resources.
- The saved plan hash matches the executed plan.
- At least 10 consecutive nominal full-system runs complete without manual plan intervention.
- All approved failure tests produce the expected safe response.
- Reports clearly distinguish simulation, hardware-in-loop, and physical execution.
- The executed hardware profile identifies the exact Go2 edition, SDK2/firmware versions, Z1 controller/SDK versions, both Panda system images, and both `libfranka`/ROS interface versions.
- The measured complete mobile-manipulator mass and center of mass remain inside the laboratory-approved Go2 configuration envelope.
- The `mjctrl` repository commit, Apache-2.0 license, Panda model origin, and all local modifications are recorded.
- Both simulated Pandas use the adapted `mjctrl` controller through ROS 2 action servers rather than the original mouse-target demo loop.
- MuJoCo grasp, placement, docking, and goal success are derived from physics observations, not declared action effects.
- Simulation and hardware expose the same ROS 2 task-level skill and observation interfaces.
- Simulation runs use `/clock`; physical runs reject simulated time.
- Simulation and physical DDS domains are isolated and verified before every hardware session.
- Every accepted run has a complete rosbag and execution-event trace.
- A full MuJoCo mobile-courier result is reported only if Go2 locomotion is produced by a documented controller, not direct base-pose manipulation.

# Repository cleanup

The redesign should remove or replace:

- Current scenarios
- Placeholder result tables
- Heuristic deadlock logic
- Implicit synchronization through action blocking
- Oracle-style physical failures
- Unverified effect application in hardware mode
- Unsupported automatic reassignment
- ROS/BT exporters advertised as execution backends
- Existing MuJoCo integration that is only visual embodiment; replace it with the ROS 2-connected layered simulator
- Stale prompts and paper claims tied to the old scenarios

Components worth preserving after refactoring:

- Typed predicate/domain foundations
- LLM client abstraction
- Experiment snapshot ideas
- Visualization/reporting framework
- Existing test infrastructure
- Correction-loop structure, once based on the new validator

New third-party integration structure:

```text
third_party/
└── mjctrl/
    ├── LICENSE
    ├── UPSTREAM_COMMIT
    ├── NOTICE.md
    └── required Panda model/controller sources
```

Only the required upstream files should be vendored or referenced. Local adaptations belong in the project’s own simulation package so upstream provenance and project-specific behavior remain clear.

# Final assessment

The confirmed Go2–Z1 plus dual-Panda setup is an excellent fit for the research goal. It makes coordination genuinely heterogeneous:

- Franka Emika Panda A performs source preparation.
- Unitree Go2 provides mobility.
- Z1 provides mobile manipulation.
- Franka Emika Panda B completes destination processing.

It is also significantly harder than adding a third identical manipulator. Docking, localization, arm-stow interlocks, carrier verification, network integration, and mobile-base safety become first-class problems.

The staged-cradle design keeps those problems manageable and scientifically honest. The robot models are no longer unresolved: they are one Unitree Go2 with a Z1 arm and two 7-DoF Franka Emika Panda arms. The remaining Phase 0 unknowns are the Go2 edition and firmware, the physical and electrical Z1 integration, the grippers and sensors, the two Panda system images, and the exact installed middleware versions. Those details determine the final version lock and whether both Pandas can join the ROS 2 Humble deployment directly or need isolated Panda-compatible skill-server hosts.

`mjctrl` materially improves the plan because it gives the two simulated Pandas a concrete, inspectable MuJoCo model and differential-IK/nullspace controller starting point. It does not eliminate the main simulation gap: a complete Go2–Z1 physics model and genuine Go2 locomotion controller still have to be integrated. ROS 2 now provides the common boundary across the deterministic simulator, MuJoCo, hardware-in-loop, and physical robots, allowing the same validated behavior tree and task-level skill contracts to be tested at every stage without presenting visualization or symbolic effects as real execution.

The immediate priority is now a standalone Python application with two frontends: a scriptable CLI and an easy local Gradio interface. A user can download a scenario template, upload a completed JSON scenario, enter an instruction, select OpenAI or Anthropic, supply a session-only or securely saved API key, press one Run button, follow the real pipeline progress, and download the validated canonical `behavior_tree.json` plus optional XML and reports. Both frontends invoke the same `PlannerService`; the UI is not a second or simplified implementation. ROS 2, `mjctrl`, MuJoCo, and hardware execution remain in the plan, but none of them may delay or become a dependency of that first working result.
