# Multi-Robot Behavior Tree Planner

This application uses OpenAI or Anthropic (Claude) to generate complete synchronized Behavior Trees for a heterogeneous three-robot team. It validates and symbolically simulates the model's exact trees, saves successful results as JSON and XML, and can execute three bundled missions in a separate MuJoCo process.

The demonstration team is:

- `franka_a` — Franka Emika Panda at the source station
- `unitree_go2_z1` — Unitree Go2 with a Z1 arm, transporting the part
- `franka_b` — Franka Emika Panda at the destination station

## Install

Python 3.10 or newer is required.

```powershell
python -m pip install -e ".[ui]"
lmrbtp doctor
```

`doctor` checks the required packages and verifies the bundled three-robot example without calling an LLM.

## Start the user interface

```powershell
lmrbtp ui
```

The application opens locally at `http://127.0.0.1:7860`. It never creates a public Gradio share link.

If you do not want it to open a browser automatically:

```powershell
lmrbtp ui --no-browser
```

## Generate a Behavior Tree

1. Start the UI.
2. Select **Collaborative packing and room delivery** or **Three-robot courier** from the bundled-scenario dropdown, or upload another JSON scenario. Uploaded files are loaded and validated automatically.
3. Edit the mission instruction if needed, then press **Validate scenario** after any manual change.
4. Select `openai` or `anthropic` and enter that provider's API key.
5. Press **Run complete pipeline**.
6. Follow the **Live log**, **Validation**, and **Simulation** tabs, then download the completed files from **Generated artifacts**.

The main screen contains only the normal workflow. Open **Advanced: edit scenario JSON** to edit the raw document, **Advanced provider options** to choose a current model or manage a saved key, **Run settings** to change correction and simulation limits, or **Saved projects** to store and reload local presets. The model dropdown follows the selected provider automatically; changing providers also clears the visible API-key field so a key cannot accidentally be sent to the wrong service. Leave **Provider default** selected for the recommended default (`gpt-5.6-sol` for OpenAI or `claude-opus-5` for Anthropic). The UI sends that displayed model explicitly, so an older `OPENAI_MODEL` or `ANTHROPIC_MODEL` value in an existing `.env` file cannot silently select a different model.

The API key is session-only unless **Save key in OS credential store** is selected. Saved keys use the operating-system credential manager. Leave the key field blank to use a saved key. **Check saved key** reports only whether one exists; the saved value is never returned to the browser. Keys are never stored in project files, logs, generated JSON, XML, HTML, or reports.

GPT-5-family requests omit the legacy `OPENAI_TEMPERATURE` override because those models use their provider-supported default sampling value at the configured reasoning setting. The override remains available for older or compatible models that accept custom temperature values.

The **Cancel** button marks the run as cancelled. An HTTP request already being processed by a provider may finish, but its response is discarded before validation or publication. A cancelled run never publishes a final BT.

## Input files

The repository provides:

- [Collaborative packing and room-delivery scenario](examples/three_robot_packaging_delivery.json) and its [committed reference BT](examples/three_robot_packaging_delivery.bt.json)
- [Courier scenario](examples/three_robot_courier.json) and its [committed reference BT](examples/three_robot_courier.bt.json)
- [Spare-part recovery scenario](examples/three_robot_spare_part_recovery.json), [nominal BT](examples/three_robot_spare_part_recovery.bt.json), [fault contract](examples/three_robot_spare_part_recovery.fault.json), and [offline recovery test oracle](examples/three_robot_spare_part_recovery.expected_recovery.bt.json)
- [Blank template](templates/three_robot_scenario.template.json)
- [JSON Schema](schemas/scenario.schema.json)

The UI opens with the collaborative packing and room-delivery scenario. The CLI `generate` command retains the courier as its default; pass `--scenario` to select the packing mission. Reference BT files are used by `doctor`, regression tests, and adjacent-file MuJoCo examples. They are never substituted for an LLM response during generation.

A scenario describes:

- the mission instruction;
- typed entities and shared resources;
- exactly three robots;
- each robot's capabilities, parameters, preconditions, effects, duration, and timeout;
- the initial symbolic state;
- the required goal state.

Unknown fields, invalid identifiers, duplicate IDs, incorrect predicate arguments, unknown resources, and invalid capability types are rejected before an API request is sent.

The JSON is treated only as data. It cannot execute code, import modules, fetch URLs, or select arbitrary local files.

## What the pipeline does

```text
Scenario + instruction
        |
        v
Selected LLM provider
        |
        v
Complete multi-robot Behavior Trees
        |
        v
Static validation
        |
        v
Contract simulation
        |
        v
JSON/XML/report artifacts
```

The LLM must generate the complete tree for every robot, including:

- composite control flow such as `Sequence`, `Fallback`, or `Parallel`;
- capability actions and grounded arguments;
- conditions and explicit cross-robot `WaitFor` synchronization;
- `AcquireResource` and `ReleaseResource` operations;
- finite wait and resource timeouts;
- stable node and task IDs.

The deterministic layer does not generate or silently fix these nodes. It parses the response, rejects unsupported or noncanonical fields, checks that the accepted JSON round-trips exactly, validates it against the scenario, and executes that exact structure in the contract simulator. Capability preconditions, effects, resources, durations, and action timeouts remain authoritative scenario contracts rather than model-invented overrides.

If a candidate fails, typed validation or simulation errors are returned to the same selected provider for a bounded correction round. There is no offline fake planner and no automatic provider fallback.

Every blocking UI failure opens an error dialog that states the redacted cause. Rejected BTs also show why final publication was stopped while retaining their downloadable diagnostics.

## Output files

Every run receives its own directory under `outputs/runs/`.

A successful run contains:

- `behavior_tree.json` — canonical BT and SHA-256 identity
- `behavior_tree.xml` — XML serialization derived from the canonical tree
- `pipeline.log` — timestamped pipeline stages and results
- `validation_report.json` — static validation result
- `simulation_trace.json` — tick-by-tick symbolic execution
- `report.html` — Behavior Tree visualization
- `scenario.json` — validated scenario used for the run
- `result.json` — planner summary
- `manifest.json` — file checksums and publication status

When validation or simulation fails, diagnostic files are written, but `behavior_tree.json` and `behavior_tree.xml` are not published.

## Verify a downloaded Behavior Tree

Run both checks against the exact scenario used to generate the tree:

```powershell
lmrbtp validate --scenario examples/three_robot_courier.json --bt "C:\path\to\behavior_tree.json"
lmrbtp simulate --scenario examples/three_robot_courier.json --bt "C:\path\to\behavior_tree.json" --max-ticks 100
```

`validate` checks the canonical BT structure, robot capabilities, parameters, causal ordering, synchronization, resource ownership, timeouts, and declared contracts. `simulate` executes the same tree in the deterministic symbolic simulator and checks that all declared goals are reached without deadlock, timeout, resource leak, or failed precondition.

These checks establish correctness only against the uploaded symbolic scenario. They do not establish collision-free trajectories, perception accuracy, dynamics, controller compatibility, ROS 2 integration, or physical-robot safety. The XML is a BehaviorTree.CPP-style serialization of the canonical JSON; it is not independently planned or hardware-ready by itself.

For the packing and room-delivery mission, use the matching scenario:

```powershell
lmrbtp validate --scenario examples/three_robot_packaging_delivery.json --bt "C:\path\to\behavior_tree.json"
lmrbtp simulate --scenario examples/three_robot_packaging_delivery.json --bt "C:\path\to\behavior_tree.json" --max-ticks 180
```

## Command-line use

Set the key for the provider you select:

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp generate --provider openai --scenario examples/three_robot_courier.json
```

or:

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
lmrbtp generate --provider anthropic --scenario examples/three_robot_courier.json
```

Generate the packing and room-delivery mission by changing only the scenario path:

```powershell
lmrbtp generate --provider openai --scenario examples/three_robot_packaging_delivery.json --max-ticks 180
```

Other commands do not call an LLM:

```powershell
lmrbtp template --output my-scenario.json
lmrbtp validate --scenario examples/three_robot_courier.json --bt outputs/runs/ROUTE/behavior_tree.json
lmrbtp simulate --scenario examples/three_robot_courier.json --bt outputs/runs/ROUTE/behavior_tree.json
lmrbtp render --bt outputs/runs/ROUTE/behavior_tree.json --html outputs/tree.html --xml outputs/tree.xml
lmrbtp doctor
```

On successful generation, the final console lines provide `BT_FILE=<absolute path>` and `BT_SHA256=<hash>`.

## Run the bundled scenarios in MuJoCo

The physical simulator is a separate optional subsystem. It reads a scenario and an already-generated BT; it does not import the provider clients, call OpenAI or Anthropic, generate a different tree, or apply symbolic capability effects.

Install its dependencies and download the three pinned robot models once:

```powershell
python -m pip install -e ".[mujoco]"
lmrbtp mujoco --setup-only
```

Open the interactive MuJoCo window and execute the bundled courier BT:

```powershell
lmrbtp mujoco
```

For a fast run without a window:

```powershell
lmrbtp mujoco --headless
```

Both forms default to [the courier scenario](examples/three_robot_courier.json) and [its committed BT](examples/three_robot_courier.bt.json). A generated BT for that same scenario can be supplied explicitly:

```powershell
lmrbtp mujoco --bt "C:\path\to\behavior_tree.json"
```

Run the collaborative packing and room-delivery mission with its adjacent reference BT:

```powershell
lmrbtp mujoco --scenario examples/three_robot_packaging_delivery.json
```

Run an LLM-generated packing BT by supplying both matching files:

```powershell
lmrbtp mujoco `
  --scenario examples/three_robot_packaging_delivery.json `
  --bt "C:\path\to\behavior_tree.json"
```

### Record publication-quality simulation videos

Record the complete courier simulation, including the initial one-second gravity/contact settling phase:

```powershell
lmrbtp mujoco `
  --headless `
  --record-video `
  --scenario examples/three_robot_courier.json `
  --bt examples/three_robot_courier.bt.json `
  --output outputs/paper-complementary
```

Record the collaborative packaging and closed-door delivery simulation:

```powershell
lmrbtp mujoco `
  --headless `
  --record-video `
  --scenario examples/three_robot_packaging_delivery.json `
  --bt examples/three_robot_packaging_delivery.bt.json `
  --output outputs/paper-complementary
```

The publication defaults are deterministic action-directed camera cuts, 1920x1080 resolution, 30 frames per simulated second, H.264 encoding at CRF 18 with the `medium` preset, and `yuv420p` output. The courier cuts from its overview to source-workcell, route, and destination-workcell views as the corresponding physical actions begin. The packaging mission similarly follows assembly, door crossing, room route, and final delivery. All angles are fixed MuJoCo cameras, so identical executions produce identical cuts without introducing tracking-camera jitter. Pass `--video-camera NAME` to disable automatic cuts and lock the whole recording to one named camera; `--video-fps`, `--video-width`, and `--video-height` remain available for capture overrides. These options require `--record-video`. `--realtime-factor` changes viewer pacing only and never changes recorded playback speed.

Each timestamped recording directory contains `simulation.mp4`, exact copies of `scenario.json` and `behavior_tree.json`, `physical_execution_report.json`, and `recording_manifest.json`. The manifest records the camera mode, cameras used, every cut's zero-based frame index, simulation timestamp and triggering action, encoder settings, frame count, simulated duration, MuJoCo and source revisions, and SHA-256 checksums. Frames are streamed to the encoder rather than retained in memory. An interrupted encoder leaves `simulation.partial.mp4` and does not publish a final manifest, while a normal BT failure still produces a playable video and failure report for diagnosis.

### Run and record the unexpected-failure recovery experiment

For the complete paper demonstration, set an OpenAI key and run one command:

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp adaptive-demo --output outputs/paper-complementary/adaptive-demo
```

This command opens a live MuJoCo window automatically after nominal BT generation finishes.
The window follows the same action-directed source, failure, route, and destination camera cuts
as the recording, while still allowing normal mouse camera controls between cuts. During the
OpenAI recovery call, physics remains frozen, the window stays responsive, and a native MuJoCo
overlay shows `FAILURE DETECTED - BT ADAPTATION IN PROGRESS` with the elapsed wait time. Use
`--realtime-factor 2` for 2x live playback. Use `--headless` only for a displayless machine or a
batch run; headless mode can still produce `adaptive_demo.mp4`.

`adaptive-demo` performs the whole workflow without accepting a prebuilt BT or an offline
planner:

1. It sends only the nominal scenario to `gpt-5.6-sol`, prints percentage progress and
   five-second heartbeats, and accepts the generated BT only after static validation and
   contract simulation. A semantic gate requires the nominal tree to use `primary_part` and
   rejects any tree that mentions `spare_part`, including a preplanned fallback.
2. Only after that BT is accepted does the process read the fault specification. It then
   launches MuJoCo automatically with both the orange primary part and blue spare part visible,
   while nominal actions use only the primary part.
3. Panda A places and releases the primary part in `source_cradle`. The configured force is
   applied after that measured successful placement and before Go2/Z1 grasps it, causing the
   part to fall to the floor and the nominal pick to fail.
4. The robots safely stop and the MuJoCo clock/state remain frozen. The measured failure
   snapshot and failed nominal BT are sent to the LLM, with live correction/validation progress.
   The final video inserts a dark `FAILURE DETECTED / LLM IS ADAPTING THE BEHAVIOR TREE`
   status layer whose metadata records the real API wait time.
5. Once the replacement BT passes validation, contract simulation, and spare-part checks, the
   exact same `MjModel` and `MjData` resume without a reset. The robots retrieve the initially
   visible spare, complete the handoff and installation, and the recorder switches among
   overview, source/failure, route, and destination angles according to the active action.

The timestamped output directory contains `adaptive_demo.mp4`, clear text and JSON event logs,
the exact nominal prompts and generated BT, non-secret OpenAI response/model/token provenance,
fault-blindness evidence, the measured failure snapshot, every recovery attempt, the accepted
adapted BT and diff, validation/simulation reports, continuity hashes, physical events and final
goals, recording/camera-cut metadata, and SHA-256 hashes. Use `--no-video` for a faster real-LLM
integration check; it still performs both LLM calls and the physical run. Resolution, frame rate,
reasoning effort, heartbeat interval, and both correction limits are explicit CLI options.

For the older two-trial experimental layout (fault-only control, adaptive run, and side-by-side
comparison), run:

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp recovery-experiment --output outputs/paper-complementary/recovery
```

`recovery-experiment` starts from the committed nominal BT and uses the OpenAI Responses API
with `gpt-5.6-sol`, high reasoning effort, and a strict JSON schema by default. It performs two
independently initialized trials with the same deterministic scene and fault trigger:

1. The fault-only control executes the valid nominal BT through Panda A's successful placement in `source_cradle`. After the part is measured in the handoff zone and released, but before Go2/Z1 establishes a grasp, a declared external force pushes `primary_part` from the cradle. The part falls to the floor, Go2's measured pick fails, and all robots enter a safe stopped/home state.
2. The adaptive trial repeats the same fault. From its measured safe-stop snapshot, the LLM returns a complete continuation BT for all three robots. The candidate must pass static validation, use `spare_part` rather than the unavailable primary part, and reach all goals in contract simulation before physical execution resumes.

The adaptive trial does not rebuild or reset MuJoCo after the fault. The same `MjModel`, `MjData`, controller instances, object poses, simulation clock, and reset counter continue through replanning and final spare-part installation. The report records state hashes immediately before and after the API call and verifies that the reset counter remains unchanged through completion. The LLM call consumes wall time but deliberately does not advance simulation time. Its adaptive video also includes the failure-handling status layer.

Every run creates a timestamped evidence directory containing:

- `failure_only.mp4` — nominal control through detected failure and safe stop;
- `adaptive_recovery.mp4` — the same failure followed by the validated continuation, with source/floor/route/destination camera cuts;
- `comparison_side_by_side.mp4` — aligned control and adaptive clips, labeled on-screen; the completed control side freezes while recovery continues;
- exact scenario, nominal BT, and fault inputs;
- the adapted BT, unified nominal/adapted diff, all LLM attempt provenance, static validation, and contract-simulation trace;
- measured failure, object positions, action events, final goals, continuity evidence, SHA-256 file hashes, and software/model provenance.

For an offline integration dry run, use the clearly labeled committed oracle:

```powershell
lmrbtp recovery-experiment --planner oracle --no-video
```

Oracle runs are marked `real_llm_evidence: false`, use an `ORACLE DRY-RUN RECOVERY` comparison label when video is enabled, and never support the paper's LLM-adaptation claim. Use the default `--planner openai` command for research evidence. `--model`, `--reasoning-effort`, `--max-corrections`, resolution, and frame-rate controls are explicit CLI options.

On a displayless Linux host, select an off-screen OpenGL backend before importing MuJoCo, for example `MUJOCO_GL=egl` with a supported GPU or `MUJOCO_GL=osmesa` for software rendering.

The command statically validates the BT first, composes one MuJoCo world, settles all three robots, and then ticks the exact hierarchical trees concurrently. It prints condition/action/resource progress and writes `physical_execution_report.json` under `outputs/mujoco/`. An unrecovered failure returns a nonzero exit code and identifies the robot, node, failed measured predicate, controller stage, or timeout; recovered Action failures remain visible in the successful report.

The adapter deliberately supports only `three_robot_courier`, `three_robot_packaging_delivery`, and `three_robot_spare_part_recovery`; it rejects any other task rather than silently mapping it to a known scene. It executes `Sequence` and `Fallback` control flow hierarchically. A generated tree that uses an unsupported physical composite or action can still pass symbolic checks, but MuJoCo rejects it with the exact unsupported node instead of flattening or rewriting it.

All three scenes contain two independently prefixed 7-DoF Panda models and one Go2 with the Z1 gripper model mounted on its trunk. The courier and recovery scenes use separated workbenches; recovery adds independent primary and spare free bodies plus object-specific grasp and fixture constraints. The packing scene instead uses one shared assembly bench, opposed Panda mounting positions with collision-safe home poses, a separate room boundary and delivery pedestal, and a travel aisle through the doorway. Controller target sites are hidden and have no collision geometry.

The Go2 free-joint pose is initialized once and never written during execution, so measured displacement comes from leg torques, inertia, and floor contact. Z1 retains the payload in its closed grasp throughout each route and releases it only at the declared destination. Dynamic objects move through actuator-driven differential IK, grasp constraints, and a 12-motor alternating contact gait. Arm-link gravity compensation is enabled, as in the `mjctrl` example; the Go2, package parts, door, and other dynamic bodies remain under full gravity.

The grasp model is explicit: each tabletop Panda follows pose-aware IK with a constrained vertical gripper orientation so its fingers approach from above. Z1 first reaches its measured open joint position, follows pose-aware IK to a side-grasp pose, and closes at a rate-limited command. The grasp is accepted only when the measured joint reaches its contact angle and MuJoCo reports payload contact on both dedicated opposing finger-pad geoms. A soft weld is then activated at the current relative pose to prevent numerical slippage without snapping or teleporting the payload. Navigation fails if that measured grasp constraint is lost. At the destination, Z1 opens to its measured joint limit before releasing the constraint and retreating. Final fixture installation uses the same visible constraint mechanism. This is a controlled-simulation approximation, not a claim that fingertip friction, perception, or grasp uncertainty has been validated.

After either Panda completes its collaborative step, it retreats and returns to its home joint configuration before its BT action succeeds and the shared zone is released. This prevents the next robot from entering while a Franka remains extended over the work area.

### What the packing and room-delivery mission demonstrates

The second scenario uses four independently enforced resources (`packing_zone`, `lid_supply_zone`, `doorway_zone`, and `delivery_zone`) and three concurrent robot trees. Franka A moves a loaded package base to the shared packing station while Franka B independently retrieves a separate dynamic lid. Franka B waits for the measured base placement, acquires the shared station, fits the lid with actuator-driven motion, activates the visible seal constraint at the reached pose, returns home, and only then releases the station. Go2 waits for both the base and the measured seal before it can enter.

The room boundary is physical geometry and the door is an initially closed, collidable hinged body. The closed-door BT branch succeeds only after Go2 pushes the panel through contact, the measured hinge angle exceeds `0.70 rad`, and the dynamically controlled base reaches the far side while retaining both the Z1 grasp and the package seal. An alternate BT branch handles a door that is already physically open. Go2 then follows the room-side route, places the sealed parcel on the delivery station, opens the gripper, and stows Z1.

`physical_execution_report.json` records the initial door state, final hinge angle, seal-constraint state, parcel and lid positions, delivery evidence, measured goals, action events, and resource release. The deterministic contract simulator still checks declared nominal effects only. MuJoCo does not copy those effects into the world: every physical Action succeeds or fails from controller motion, contacts, constraints, measured poses, and timeouts in this fixed scene.

Robot MJCF files are fetched from MuJoCo Menagerie at the pinned commit recorded in [third-party notices](THIRD_PARTY_NOTICES.md). The arm controller adapts the damped differential-IK structure demonstrated by `mjctrl`. The native Go2 controller adapts the alternating contact-gait and joint-torque structure from `go2-convex-mpc`; it is not the upstream Pinocchio/CasADi convex-MPC solver. The upstream solver assumes Python 3.10 and a standalone bare-Go2 state layout, while this program controls the combined Go2/Z1 model directly through MuJoCo state.

## Saved projects

The UI can save the validated scenario, mission instruction, provider name, model, and run limits as a local project. Project files are written under `projects/` and never contain an API key.

## What is and is not verified

The planner genuinely verifies the declared symbolic contract:

- capability names and typed arguments;
- grounded preconditions and effects;
- causal ordering and goal support;
- explicit synchronization and finite waits;
- exclusive resource ownership and release;
- wait/resource cycles, timeouts, leaks, and cancellation cleanup;
- symbolic state invariants and final goals.

The separate MuJoCo adapter tests the three bundled BTs against fixed dynamic scenes: hierarchical branch selection, actuator limits, gravity, contacts, reachable Cartesian motion, collaborative assembly, payload transfer, base stability, docking, contact-driven Go2 displacement, hinged-door opening, room delivery, deterministic part loss, and same-state spare-part recovery. Its report is evidence for that exact model, controller configuration, fault contract, and initial state only. It does **not** establish general collision avoidance, perception accuracy, diagnosis from real sensors, ROS 2 integration, sim-to-real validity, or physical-robot safety. XML remains an export format rather than a hardware controller.

## Development checks

```powershell
python -m pip install -e ".[ui,mujoco,dev]"
python -m ruff check src tests
python -m mypy src/llm_mr_bt_planner
python -m pytest -q
$env:LMRBTP_RUN_MUJOCO_E2E = "1"
python -m pytest tests/test_mujoco_sim.py -q
$env:LMRBTP_RUN_MUJOCO_RECOVERY_E2E = "1"
python -m pytest tests/test_mujoco_sim.py -q -k spare_part_recovery
python -m llm_mr_bt_planner doctor
```

Automated generation tests use the clearly named `test-reference-client`; they do not call or impersonate a production provider.
The committed reference BT is a test fixture, not evidence of a live LLM generation run. Production `generate` and UI runs always require the selected provider and an API key.
