# Multi-Robot Behavior Tree Planner

This application uses OpenAI or Anthropic (Claude) to generate complete synchronized Behavior Trees for heterogeneous robot teams. It validates and symbolically simulates the model's exact trees, saves successful results as JSON and XML, and can execute the bundled three-robot and five-agent missions in a separate MuJoCo process.

The demonstration team is:

- `franka_a` — Franka Emika Panda at the source station
- `unitree_go2_z1` — Unitree Go2 with a Z1 arm, transporting the part
- `franka_b` — Franka Emika Panda at the destination station

The five-agent inspection missions additionally model B2 locomotion, its mounted Z1 thermal
arm, a Husky base, the Franka mounted on Husky, and a static Franka.

## Overview

Use these links to jump directly to the part of the README you need:

- **Get started:** [install](#install), [start the UI](#start-the-user-interface), or
  [generate a Behavior Tree](#generate-a-behavior-tree).
- **Understand the data flow:** [input files](#input-files),
  [pipeline behavior](#what-the-pipeline-does), and [output files](#output-files).
- **Use the CLI:** [command-line use](#command-line-use) and
  [validate or simulate an exported BT](#verify-a-downloaded-behavior-tree).
- **Run MuJoCo:** [simulator setup](#run-the-bundled-scenarios-in-mujoco),
  [five-agent inspection](#generate-and-run-the-five-agent-inspection-mission),
  [online dropped-tool recovery](#run-five-agent-inspection-with-online-dropped-tool-recovery),
  [pipe-leak repair](#generate-and-run-the-pipe-only-leak-repair-mission), and
  [video recording](#record-publication-quality-simulation-videos).
- **Run paper experiments:** [unexpected-failure recovery](#run-and-record-the-unexpected-failure-recovery-experiment)
  and [the five comparison baselines](#paper-comparison-baselines):
  [LLM-as-BT-Planner](#llm-as-bt-planner-comparison),
  [LLM-BT](#llm-bt-comparison), [BETR-XP-LLM](#betr-xp-llm-comparison),
  [LLM-HBT](#llm-hbt-comparison), and [MRBTP](#mrbtp-comparison).
- **Reference:** [saved projects](#saved-projects),
  [verification boundaries](#what-is-and-is-not-verified), and
  [development checks](#development-checks).

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
- [Single-component installation scenario](examples/three_robot_component_installation.json), [nominal BT](examples/three_robot_component_installation.bt.json), [fault contract](examples/three_robot_component_installation.fault.json), and [offline fallen-part recovery oracle](examples/three_robot_component_installation.expected_recovery.bt.json)
- [Five-agent solar/pipe inspection scenario](examples/five_agent_solar_pipe_inspection.json) and its [committed regression BT](examples/five_agent_solar_pipe_inspection.bt.json)
- [Blank template](templates/three_robot_scenario.template.json)
- [JSON Schema](schemas/scenario.schema.json)

The UI opens with the collaborative packing and room-delivery scenario. The CLI `generate` command retains the courier as its default; pass `--scenario` to select the packing mission. Reference BT files are used by `doctor`, regression tests, and adjacent-file MuJoCo examples. They are never substituted for an LLM response during generation.

A scenario describes:

- the mission instruction;
- typed entities and shared resources;
- the participating robots/controllers (three in the original missions and five in the inspection missions);
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

### Generate and run the five-agent inspection mission

The inspection mission coordinates B2 locomotion, its mounted Z1 thermal-camera arm, a Husky
base, the Franka mounted on Husky, and a static Franka. The static Franka prepares the inspection
kit; Husky/Franka installs a thermal reference; B2/Z1 scans a solar panel and a three-joint pipe
rig; the measured hot joint is confirmed and marked; the isolation switch is actuated; and Z1
records a post-isolation verification measurement.

Generate the five synchronized BTs with OpenAI and automatically launch the live MuJoCo viewer.
The video is recorded by default with action-directed handoff, route, solar, pipe, and service
angles:

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp inspection-demo --output outputs/paper-complementary/five-agent-inspection
```

Use `--headless` on a displayless machine, `--no-video` for a faster physical integration check,
or the standard video resolution/fps options. The LLM output must pass strict parsing, static
validation, and deterministic contract simulation before MuJoCo is built. The exact published
`behavior_tree.json` is then executed without action rewriting.

For offline development, the committed adjacent BT can exercise the same physical adapter without
an API call:

```powershell
lmrbtp mujoco `
  --scenario examples/five_agent_solar_pipe_inspection.json `
  --headless `
  --record-video `
  --output outputs/five-agent-inspection
```

The committed BT is a regression fixture, not evidence of a fresh model call. For paper results,
use `inspection-demo` and retain its generation bundle. The execution report archives the hidden
localized site, sensor-to-target ranges, 28.5 C baseline, pre-isolation thermal peak, measured
switch state, post-isolation peak, all BT events, final goals, resource release, model commits,
and recording checksums. MuJoCo does not simulate heat transfer; the thermal values are a seeded,
deterministic sensor abstraction gated by the physical Z1 pose, B2 dock, range, and isolation
switch state.

### Run five-agent inspection with online dropped-tool recovery

This experiment keeps the nominal inspection input unchanged and seals the tool-drop fault until
after GPT-5.6 Sol has produced and validated all five nominal BTs. The only inspection kit is then
knocked from the handoff tray as a dynamic MuJoCo free body. Its audit coordinates are not sent to
the recovery model. The adapted BT must move B2 to a search viewpoint, use Z1 to localize the tool,
send Husky to the resulting recovery dock, and have Husky's Panda recover that same tool before the
solar/pipe mission continues in the same model/data state.

```powershell
lmrbtp inspection-adaptive-demo `
  --model gpt-5.6-sol `
  --output outputs/paper-complementary/inspection-adaptive
```

The default run opens the live viewer and records an action-directed video. During the LLM call,
physics stays frozen and both the viewer and final video show a failure-handling status layer. Use
`--headless --no-video` for a fast integration check. The output bundle includes both generated BTs,
the sealed fault, failure snapshot, prompts and provider responses, validation/simulation reports,
same-state evidence, physical report, diff, log, video, and checksummed manifest.

### Generate and run the pipe-only leak-repair mission

The pipe-only scenario still uses all five controllers: B2 positions the stowed Z1, Z1 detects and
localizes a seeded hot leak, the static Panda supplies the single repair tool, Husky transports its
mounted Panda to the leak, and that Panda closes a measured MuJoCo repair collar. Z1 then rejects or
accepts the repair from a second thermal observation.

```powershell
lmrbtp pipe-repair-demo `
  --model gpt-5.6-sol `
  --output outputs/paper-complementary/pipe-repair
```

For an offline adapter check, run `lmrbtp mujoco --scenario
examples/five_agent_pipe_leak_repair.json --headless`; its adjacent BT is a regression fixture and
must not be reported as fresh LLM evidence.

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
   rejects invented parts or any preplanned `recover_fallen_part` action.
2. Only after that BT is accepted does the process read the fault specification. It then
   launches MuJoCo automatically with one orange `primary_part`. Neither a replacement object
   nor a visible floor-recovery target exists in the nominal planning input.
3. Panda A places and releases the primary part in `source_cradle`. The configured force is
   applied after that measured successful placement and before Go2/Z1 grasps it, causing the
   part to fall to the floor and the nominal pick to fail.
4. The robots safely stop and the MuJoCo clock/state remain frozen. The measured failure
   snapshot reports that the same intact object is usable at `source_floor`; that snapshot and
   the failed nominal BT are sent to the LLM, with live correction/validation progress. Only at
   this boundary is Go2/Z1's `recover_fallen_part(primary_part,source_floor)` capability added to
   the runtime planning contract, so the nominal LLM cannot know the failure in advance.
   The final video inserts a dark `FAILURE DETECTED / LLM IS ADAPTING THE BEHAVIOR TREE`
   status layer whose metadata records the real API wait time.
5. Once the adapted BT passes validation, contract simulation, and same-object recovery checks,
   the exact same `MjModel` and `MjData` resume without a reset. Go2/Z1 retrieves the fallen
   `primary_part`, completes the handoff and installation, and the recorder switches among
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
2. The adaptive trial repeats the same fault. From its measured safe-stop snapshot, the LLM returns a complete continuation BT for all three robots. The candidate must pass static validation, retrieve the usable `primary_part` from its measured `source_floor` location, avoid inventing replacement objects, and reach all goals in contract simulation before physical execution resumes.

The adaptive trial does not rebuild or reset MuJoCo after the fault. The same `MjModel`, `MjData`, controller instances, fallen-object pose, simulation clock, and reset counter continue through replanning, floor retrieval, and final installation of that same object. The report records state hashes immediately before and after the API call and verifies that the reset counter remains unchanged through completion. The LLM call consumes wall time but deliberately does not advance simulation time. Its adaptive video also includes the failure-handling status layer.

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

The adapter supports only its explicitly registered scenarios: `three_robot_courier`,
`three_robot_packaging_delivery`, `three_robot_component_installation`,
`five_agent_solar_pipe_inspection`, and `five_agent_pipe_leak_repair`. It rejects any other
task rather than silently mapping it to a known scene. It executes `Sequence` and `Fallback`
control flow hierarchically. A generated tree that uses an unsupported physical composite or
action can still pass symbolic checks, but MuJoCo rejects it with the exact unsupported node
instead of flattening or rewriting it.

All three scenes contain two independently prefixed 7-DoF Panda models and one Go2 with the Z1 gripper model mounted on its trunk. The courier and component-installation scenes use separated workbenches; component installation adds one free `primary_part` plus object-specific grasp and fixture constraints. The packing scene instead uses one shared assembly bench, opposed Panda mounting positions with collision-safe home poses, a separate room boundary and delivery pedestal, and a travel aisle through the doorway. Controller target sites are hidden and have no collision geometry. The recovery-only `source_floor` site is invisible and has no collision geometry; it represents the diagnosed landing region rather than another scene object.

The Go2 free-joint pose is initialized once and never written during execution, so measured displacement comes from leg torques, inertia, and floor contact. Z1 retains the payload in its closed grasp throughout each route and releases it only at the declared destination. Dynamic objects move through actuator-driven differential IK, grasp constraints, and a 12-motor alternating contact gait. Arm-link gravity compensation is enabled, as in the `mjctrl` example; the Go2, package parts, door, and other dynamic bodies remain under full gravity.

The grasp model is explicit: each tabletop Panda follows pose-aware IK with a constrained vertical gripper orientation so its fingers approach from above. Z1 first reaches its measured open joint position, follows pose-aware IK to a side-grasp pose, and closes at a rate-limited command. Normal Z1 handoffs require the measured joint to reach its contact angle and MuJoCo to report payload contact on both dedicated opposing finger-pad geoms. Ground-supported fallen-part retrieval instead requires a closed gripper, actual Z1/object contact, and tool/object proximity below 6 cm because the floor can prevent an opposing-pad pinch. A soft weld is then activated at the current relative pose to prevent numerical slippage without snapping or teleporting the payload. Navigation fails if that measured grasp constraint is lost. At the destination, Z1 opens to its measured joint limit before releasing the constraint and retreating. Final fixture installation uses the same visible constraint mechanism. This is a controlled-simulation approximation, not a claim that fingertip friction, perception, or grasp uncertainty has been validated.

After either Panda completes its collaborative step, it retreats and returns to its home joint configuration before its BT action succeeds and the shared zone is released. This prevents the next robot from entering while a Franka remains extended over the work area.

### What the packing and room-delivery mission demonstrates

The second scenario uses four independently enforced resources (`packing_zone`, `lid_supply_zone`, `doorway_zone`, and `delivery_zone`) and three concurrent robot trees. Franka A moves a loaded package base to the shared packing station while Franka B independently retrieves a separate dynamic lid. Franka B waits for the measured base placement, acquires the shared station, fits the lid with actuator-driven motion, activates the visible seal constraint at the reached pose, returns home, and only then releases the station. Go2 waits for both the base and the measured seal before it can enter.

The room boundary is physical geometry and the door is an initially closed, collidable hinged body. The closed-door BT branch succeeds only after Go2 pushes the panel through contact, the measured hinge angle exceeds `0.70 rad`, and the dynamically controlled base reaches the far side while retaining both the Z1 grasp and the package seal. An alternate BT branch handles a door that is already physically open. Go2 then follows the room-side route, places the sealed parcel on the delivery station, opens the gripper, and stows Z1.

`physical_execution_report.json` records the initial door state, final hinge angle, seal-constraint state, parcel and lid positions, delivery evidence, measured goals, action events, and resource release. The deterministic contract simulator still checks declared nominal effects only. MuJoCo does not copy those effects into the world: every physical Action succeeds or fails from controller motion, contacts, constraints, measured poses, and timeouts in this fixed scene.

Robot MJCF files are fetched from MuJoCo Menagerie at the pinned commit recorded in [third-party notices](THIRD_PARTY_NOTICES.md). The arm controller adapts the damped differential-IK structure demonstrated by `mjctrl`. The native Go2 controller adapts the alternating contact-gait and joint-torque structure from `go2-convex-mpc`; it is not the upstream Pinocchio/CasADi convex-MPC solver. The upstream solver assumes Python 3.10 and a standalone bare-Go2 state layout, while this program controls the combined Go2/Z1 model directly through MuJoCo state.

## Paper comparison baselines

Primary quantitative comparison is deliberately limited to five external methods: LLM-as-BT-Planner, LLM-BT, BETR-XP-LLM, LLM-HBT, and MRBTP. This keeps the table centered on LLM-generated BTs and runtime adaptation, with MRBTP retained as the one non-LLM planning reference. Broader approaches remain discussed in the paper's related-work review, but they are not implemented or reported as primary experimental baselines.

Baseline implementations live behind `lmrbtp compare` and remain separate from the proposed planner. All five selected methods are implemented below with method-specific provenance, native artifacts, common-domain observations, and regression tests.

### Quick start: five-agent comparison

All five runners can use the same [solar/pipe inspection scenario](examples/five_agent_solar_pipe_inspection.json),
capability contracts, validation gates, and symbolic executor. Their planning mechanisms remain distinct:

| Method | How it produces the Behavior Trees |
| --- | --- |
| **LLM-as-BT-Planner** | An LLM decomposes the mission into sequential subgoals and directly emits KIOS-style BTs for them. |
| **LLM-BT** | One LLM call produces descriptive steps, the released DistilBERT parser extracts goal phrases, and a deterministic Action Template Library expands the BT. |
| **BETR-XP-LLM** | One LLM call formalizes the goal; symbolic reactive backchaining constructs fallback branches for unsatisfied conditions. |
| **LLM-HBT** | Failed conditions enter a queue; an LLM coordinator assigns a robot and another LLM decision selects each producing action recursively. |
| **MRBTP** | A non-LLM FIFO regression planner expands per-robot action spaces and records in-tree and cross-tree policy branches. |

Install the project plus LLM-BT's released-parser dependencies, then prepare the pinned upstream
sources once. Preparation downloads and verifies source/model artifacts; it does not run an
experiment or call an LLM.

```powershell
python -m pip install -e ".[llm-bt,dev]"

lmrbtp compare llm-as-bt-planner prepare --output outputs/comparison/llm-as-bt-planner/source
lmrbtp compare llm-bt prepare            --output outputs/comparison/llm-bt/source
lmrbtp compare betr-xp-llm prepare        --output outputs/comparison/betr-xp-llm/source
lmrbtp compare llm-hbt prepare            --output outputs/comparison/llm-hbt/source
lmrbtp compare mrbtp prepare              --output outputs/comparison/mrbtp/source
```

Run one nominal trial of every method in the common five-agent setting:

```powershell
$env:OPENAI_API_KEY = "your-key"
$scenario = "examples/five_agent_solar_pipe_inspection.json"
$model = "gpt-5.6-sol"
$root = "outputs/comparison/five-robot"

lmrbtp compare llm-as-bt-planner run --scenario $scenario --scheme one-step --provider openai --model $model --output "$root/llm-as-bt-planner" --seed 42 --max-ticks 500
lmrbtp compare llm-bt run            --scenario $scenario --model $model --source outputs/comparison/llm-bt/source --output "$root/llm-bt" --seed 42 --max-ticks 500
lmrbtp compare betr-xp-llm run        --scenario $scenario --model $model --source outputs/comparison/betr-xp-llm/source --output "$root/betr-xp-llm" --seed 42 --max-ticks 500
lmrbtp compare llm-hbt run            --scenario $scenario --provider openai --model $model --source outputs/comparison/llm-hbt/source --output "$root/llm-hbt" --seed 42 --max-ticks 500
lmrbtp compare mrbtp run              --scenario $scenario --source outputs/comparison/mrbtp/source --output "$root/mrbtp" --max-expansions 10000 --max-ticks 500
```

MRBTP needs no API key; the other four commands perform live model inference. Each command creates
a timestamped directory containing native artifacts, `canonical_plan.json`, validation and
simulation reports, `metrics.json`, and `manifest.json`. A successful run additionally publishes
`accepted_plan.json`; a rejected run retains diagnostics but does not publish that file. The
currently selected five-agent accepted plans can be collected under
`outputs/comparison/five-robot/final-json/`, with `index.json` recording their hashes and metrics.
Use at least 30 independent trials per stochastic configuration for paper results; a single
accepted run is an integration result, not a success-rate estimate.

The following subsections document provenance, paper-specific settings, adapters, replay inputs,
and recovery behavior for each method.

### LLM-as-BT-Planner comparison

The first implemented baseline is **LLM-as-BT-Planner** (Ao et al., ICRA 2025). The authors' MIT-licensed KIOS repository is pinned at commit `e9f16f5bd110ab647242077d55d5cb0a71e4fcd9`. The common-domain compatibility runner preserves the published two-level planning flow: an assembly planner first creates sequential subgoals, then one of the four reported in-context schemes produces native KIOS `summary`/`name`/`children` JSON trees. `one-step` generates each complete tree once; `iterative` regenerates after native dummy-simulation failure for at most five attempts by default; `human` creates an action plan and applies archived human revisions; and `recursive` performs the reported `MakePlan`, `MakeTree`, and `PredictState` expansion procedure.

Download, hash, license-check, and extract the exact official source for inspection:

```powershell
lmrbtp compare llm-as-bt-planner prepare `
  --output outputs/comparison/llm-as-bt-planner/source
```

Run the paper's main GPT-4 configuration with the unattended one-step scheme:

```powershell
lmrbtp compare llm-as-bt-planner run `
  --scenario examples/three_robot_courier.json `
  --scheme one-step `
  --provider openai `
  --model gpt-4 `
  --output outputs/comparison/llm-as-bt-planner/runs
```

Select `--scheme iterative`, `--scheme human`, or `--scheme recursive` to evaluate the other paper variants. Human runs accept `--human-feedback feedback.json`, where the file maps each subgoal ID to an ordered array of feedback strings, for example `{"source_handoff": ["Keep the source-zone guard explicit."]}`. An absent or empty array accepts the first generated tree. The iterative attempt limit and recursive safety bounds are explicit through `--max-iterations`, `--max-recursive-depth`, and `--max-recursive-expansions`.

The native observer rejects unknown fields and node types, malformed unit subtrees, hallucinated robot/action assignments, undeclared constants, and disagreement between the reported action sequence and tree leaves. It maps KIOS selectors and memoryless sequences to the closest common control nodes, represents an existing cross-robot KIOS precondition as a bounded wait, exposes resource operations already required by the selected low-level capability, and assigns trace IDs. It does not add, remove, reorder, retry, or substitute task actions. Native trees, dummy-simulation feedback, every prompt/response, subgoal and recursive trace, human feedback, the canonical observation, PS/SV/GS evidence, fidelity limits, and checksums are archived. `accepted_plan.json` is emitted only after common static validation and symbolic execution both pass.

Archived calls can be replayed without an API request:

```powershell
lmrbtp compare llm-as-bt-planner run `
  --scenario examples/three_robot_courier.json `
  --scheme one-step `
  --responses path/to/ordered_kios_responses.json `
  --output outputs/comparison/llm-as-bt-planner/replays
```

Replay input has the form `{"responses": [{"stage": "decompose", "response": {...}}, {"stage": "one_step", "subgoal": "source_handoff", "response": {...}}]}`. Entries are consumed in order and optional `subgoal`, `attempt`, and `depth` fields are checked against the actual call context. Replay artifacts are marked as non-model evidence and report zero new model calls.

### LLM-BT comparison

The second implemented baseline is **LLM-BT** (Zhou et al., ICRA 2024). Its nominal path retains the published architecture: one ChatGPT reasoning call converts the instruction and XML semantic map into descriptive move steps, the authors' released DistilBERT token classifier extracts action/target/destination fields, and the deterministic BT Update stage recursively expands failed goal conditions through a manually supplied Action Template Library (ATL). Its recovery path also follows the published boundary: a newly false condition triggers ATL expansion, but ChatGPT and DistilBERT are **not** called again after the disturbance.

Install the released-parser dependencies and explicitly prepare the pinned upstream files:

```powershell
python -m pip install -e ".[llm-bt,dev]"
lmrbtp compare llm-bt prepare `
  --output outputs/comparison/llm-bt/source
```

Preparation downloads 16 method-defining files and the authors' 265,510,949-byte DistilBERT checkpoint from immutable commit `c69c18d0cf4b78f166ed352fc0fa8470823b32f6`, then verifies every SHA-256 hash, the model size, architecture, and eight-label vocabulary. The upstream repository declares no project-wide software license and no parser-model license. Its embedded `BTsUpdate/core/LICENSE` covers only the Michele Colledanchise BT core; the command records this boundary and does not bundle or redistribute upstream files. Use `--without-parser-model` for a provenance-only preparation.

For a real nominal run, set an OpenAI key and choose the model explicitly:

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp compare llm-bt run `
  --scenario examples/three_robot_component_installation.json `
  --model gpt-3.5-turbo `
  --source outputs/comparison/llm-bt/source `
  --output outputs/comparison/llm-bt/runs
```

The paper and repository do not report the ChatGPT model version, original prompt, or decoding settings. `gpt-3.5-turbo`, temperature 0, and seed 42 are therefore recorded reproduction choices, not author-reported settings. The common adapter serializes the measured symbolic state as XML and gives grounded ATL postconditions stable `object_N`/`position_N` move aliases because the released parser implements only its original `move` grammar. The deterministic layer then maps parsed postconditions back to the unchanged capability contract, emits the paper's `Fallback(condition, Sequence(preconditions, action))` expansions, partitions producer work by declared robot ownership, and represents cross-robot dependencies and exclusive resources explicitly. The original tick-wise failed-node updates are materialized to an ATL fixpoint before common static validation. This adaptation and the non-applied central-tree Insert priority move are listed in each manifest.

Archived reasoning and NER results can test the complete downstream path without an API or ML installation:

```powershell
lmrbtp compare llm-bt run `
  --scenario examples/three_robot_component_installation.json `
  --responses path/to/llm_bt_replay.json `
  --output outputs/comparison/llm-bt/replays
```

Replay JSON has this shape:

```json
{
  "reasoning_response": "1. Move object 7 to position 17.",
  "ner_predictions": [
    {"entity": "B-Action", "word": "Move"},
    {"entity": "B-Target", "word": "object"},
    {"entity": "I-Target", "word": "7"},
    {"entity": "B-Destination", "word": "position"},
    {"entity": "I-Destination", "word": "17"}
  ]
}
```

The aliases must come from that run's archived `native/alias_catalog.json`; the abbreviated example is structural, not a complete mission response. Replay manifests explicitly record zero new model and parser calls.

To evaluate the published deterministic recovery mechanism, first complete a nominal component-installation run. Then pass its run directory and the shared post-drop snapshot:

```powershell
lmrbtp compare llm-bt recover `
  --scenario examples/three_robot_component_installation.json `
  --nominal-run outputs/comparison/llm-bt/runs/<completed-run-directory> `
  --failure-snapshot path/to/failure_snapshot.json `
  --output outputs/comparison/llm-bt/recovery
```

The snapshot JSON requires `measured_initial_state` (the exact post-failure fact array) and `failure_observation` (the detected drop record). The runner reuses `native/parsed_goals.json`, exposes the post-failure `recover_fallen_part(primary_part, source_floor)` ATL entry, and archives RPS/RV/RGS evidence. It does not reveal the fault to the nominal prompt, substitute a spare object, or claim an LLM call during recovery.

### BETR-XP-LLM comparison

The third implemented baseline is **BETR-XP-LLM** (Styrud et al., ICRA 2025). The nominal path retains the method's two-layer design: one GPT-4-1106-Preview call maps the natural-language task and described scene into a strict first-order-logic goal formula, then a deterministic PDDL-style planner expands false conditions into reactive `Fallback(condition, Sequence(preconditions, action))` branches. No validator diagnostic or future failure is included in this call. At execution failure, a separate LLM call resolves missing knowledge and permanently updates the planner policy.

Prepare and verify the authors' complete BSD-3-Clause repository at immutable commit `bf83bda4b8921eea7fe0b8756daacb7da9fb6133`:

```powershell
lmrbtp compare betr-xp-llm prepare `
  --output outputs/comparison/betr-xp-llm/source
```

The command verifies the 217,879-byte archive SHA-256 (`54bc787eb7ae78901e3d9dee3929dfc1d90bf0412a246428a3e2b4dc7ecb370f`), all 120 extracted files, required prompt/planner/test files, and the upstream BSD license. The upstream source is preserved separately for inspection. The common runner is an official-source-informed compatibility implementation because the authors' executable stack is specific to ABB YuMi, Azure OpenAI, PyTrees, RWS, camera/segmentation, and collision-planning interfaces.

Run nominal planning with the paper's reported sampling settings (temperature 0.1, top-p 0.1, one user message):

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp compare betr-xp-llm run `
  --scenario examples/three_robot_component_installation.json `
  --model gpt-4-1106-preview `
  --source outputs/comparison/betr-xp-llm/source `
  --output outputs/comparison/betr-xp-llm/runs
```

Every prompt, raw formula, typed condition/object alias, DNF alternative, initial goal forest, expansion trace, native list-form policy, canonical observation, PS/SV/GS result, and checksum is archived. The common-domain representation partitions the original single-YuMi policy by declared robot ownership and renders cross-robot dependencies and resources explicitly; it does not feed common validation results back to the model.

For the dropped-component trial, pass a successful nominal run and the same measured post-failure snapshot used by the other recovery methods:

```powershell
lmrbtp compare betr-xp-llm recover `
  --scenario examples/three_robot_component_installation.json `
  --nominal-run outputs/comparison/betr-xp-llm/runs/<completed-run-directory> `
  --failure-snapshot path/to/failure_snapshot.json `
  --model gpt-4-1106-preview `
  --source outputs/comparison/betr-xp-llm/source `
  --output outputs/comparison/betr-xp-llm/recovery
```

This trial uses the paper's missing-parameter resolution branch. Only after the failure, the LLM receives the observed intact object's location and must change the native generic pickup binding from `Pick(primary_part, source_cradle)` to `Pick(primary_part, source_floor)`. That resolved parameter constrains the grounded action set used for replanning and must bind to `recover_fallen_part(primary_part, source_floor)`; an incorrect or invented location remains a failed RPS/RV/RGS trial. The update retrieves the same `primary_part`, never substitutes a spare, and archives before/after native policies plus the exact parameter update. This dropped-object test is a common capability/interface adaptation, not one of the paper's ten original missing-precondition scenarios.

Offline replay uses `{"goal_response": "Goal: ..."}` with `run` or `{"recovery_response": "Reasoning: ...\nParameter value: source_floor"}` with `recover`. Replays execute the same parsers and planners but are explicitly recorded as archived evidence with zero real model calls.

### LLM-HBT comparison

The fourth implemented baseline is **LLM-HBT** (Wang et al., 2025). It retains the paper's four-module control flow: an LLM initializes ordered condition nodes, failed conditions enter a queue, the coordinator Alex assigns each failed node to a capable robot, and a second LLM decision selects one action from that robot's library. The selected action is inserted locally or delegated to another robot with requester monitoring; its unmet preconditions recursively enter the same queue. The common observer serializes the resulting native selector/sequence forest without changing the LLM-selected robot, action, dependency order, or recovery object.

The author-owned project repository currently contains only a project page saying that the code repository is “Coming Soon.” The paper also does not report the LLM identifier, prompts, response grammar, temperature, or seed. This implementation is therefore a clearly labeled clean-room reproduction, not an execution of official author code. Pin and inspect the exact author page and arXiv v1 source first:

```powershell
lmrbtp compare llm-hbt prepare `
  --output outputs/comparison/llm-hbt/source
```

Preparation verifies the author project page at commit `17ff0ad9fc8e0f5ef3534086589cfa812b20cf29`, project archive SHA-256 `d7e22fc0ce6ea5c30dfdd4f10da7ccf914e9ef1b230f3db9c8140bc4c7f96002`, and arXiv v1 source SHA-256 `bb40eff629f12f7a9ae58e989abe518a1092f73a9a26288ec4f361f17f29ca28`. It records that no executable source or software license was published; the downloaded material is provenance only.

Run a live nominal trial with an explicitly selected reproduction model:

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp compare llm-hbt run `
  --scenario examples/three_robot_component_installation.json `
  --provider openai `
  --model gpt-4o-2024-08-06 `
  --source outputs/comparison/llm-hbt/source `
  --output outputs/comparison/llm-hbt/runs
```

The default uses strict JSON, temperature 0, and seed 42 as disclosed reproduction choices. Every system/user prompt, raw decision, queue event, Alex assignment, robot-selected action, local/delegated update, native forest, canonical observation, metric, and checksum is archived. The deterministic layer checks that a selected action belongs to the assigned robot and establishes the failed condition; a wrong selection fails the run instead of being repaired or sent validator feedback.

For the shared dropped-component trial, reuse a completed nominal run and reveal the measured snapshot only after failure:

```powershell
lmrbtp compare llm-hbt recover `
  --scenario examples/three_robot_component_installation.json `
  --nominal-run outputs/comparison/llm-hbt/runs/<completed-run-directory> `
  --failure-snapshot path/to/failure_snapshot.json `
  --provider openai `
  --model gpt-4o-2024-08-06 `
  --source outputs/comparison/llm-hbt/source `
  --output outputs/comparison/llm-hbt/recovery
```

The recovery runner reuses the nominal LLM-initialized conditions, invokes Alex and robot action selection again from the measured post-drop state, and accepts only a continuation containing `recover_fallen_part(primary_part, source_floor)`. The same `primary_part` must then be transported and installed; no spare object is introduced. The nominal forest, failure snapshot, post-failure calls, online insertions, and RPS/RV/RGS evidence remain separate and auditable.

Offline replay accepts `{"responses": [...]}`. Entries are consumed in exact call order using stages `initialize`, `assign`, and `select_action`; optional `condition`, `requester`, `robot`, and `track` fields are checked against the live call context. A response is either a strict object such as `{"conditions": [...]}`, `{"robot": "franka_a", "mode": "local", "task": "..."}`, or `{"action": "pick_source_part(primary_part,primary_bin)"}`. Replays exercise the full downstream implementation but are marked as archived evidence with zero new model calls.

### MRBTP comparison

The fifth implemented baseline is **MRBTP** (Cai et al., AAAI 2025). This is the focused non-LLM planning reference. The runner retains the official FIFO MRBTP/MABTP path with composite actions disabled: a shared queue starts from the complete team goal, every robot expands each queued condition through its own grounded action space, and the planner records both in-tree and cross-tree additions. Action relevance, regressed premises, subset pruning, common conflicts, and the full backup forest are checked independently before the common observer is accepted.

Download and verify the complete MIT-licensed official repository at immutable commit `3d6bd240aa2903245b2335711a97ee394f174313`:

```powershell
lmrbtp compare mrbtp prepare `
  --output outputs/comparison/mrbtp/source
```

The preparation command verifies archive SHA-256 `959d3559d10721b45629074ca944d95df92ba73bc44a9f6a57332a28dcd20030`, every extracted file, the required planner/action-node sources, and the MIT license. The upstream tree remains in the ignored output directory for inspection.

Run the nominal comparison in the shared component-installation setting:

```powershell
lmrbtp compare mrbtp run `
  --scenario examples/three_robot_component_installation.json `
  --source outputs/comparison/mrbtp/source `
  --output outputs/comparison/mrbtp/runs
```

No API key or model is used. The paper's optional LLM-generated composite-action plugin is deliberately disabled so MRBTP remains the non-LLM reference. `--max-expansions` bounds symbolic construction and `--max-ticks` bounds common execution.

Each run archives the grounded per-robot action spaces, FIFO expansion trace, planning graph,
native backup forest, reconstructed solution witness, intention-sharing protocol, canonical
observation, validation results, simulation trace, metrics, checksums, and exact source
manifest. The full backup policy remains in `native/native_forest.json`; its solved witness is
projected to per-robot common BT JSON for deterministic execution. Common nodes use explicit
`source: planner`, which the validator accepts only through the comparison runner's opt-in
reactive-policy profile. Resource leaves and bounded cross-robot waits expose the shared
executor contract without adding, deleting, substituting, or reordering task actions.

This is an official-source-aligned common-domain port, not an unmodified run of the upstream
MiniGrid or VirtualHome environment. The witness projection uses deterministic round-robin
ticks and does not emulate MRBTP's speculative belief-success semantics. Communication-loss,
homogeneous-action failure, and the optional LLM subtree experiments remain outside this
nominal comparison track.

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

The separate MuJoCo adapter tests the three bundled BTs against fixed dynamic scenes: hierarchical branch selection, actuator limits, gravity, contacts, reachable Cartesian motion, collaborative assembly, payload transfer, base stability, docking, contact-driven Go2 displacement, hinged-door opening, room delivery, deterministic part loss, and same-state fallen-object retrieval. Its report is evidence for that exact model, controller configuration, fault contract, and initial state only. It does **not** establish general collision avoidance, perception accuracy, diagnosis from real sensors, ROS 2 integration, sim-to-real validity, or physical-robot safety. XML remains an export format rather than a hardware controller.

## Development checks

```powershell
python -m pip install -e ".[ui,mujoco,dev]"
python -m ruff check src tests
python -m mypy src/llm_mr_bt_planner
python -m pytest -q
$env:LMRBTP_RUN_MUJOCO_E2E = "1"
python -m pytest tests/test_mujoco_sim.py -q
$env:LMRBTP_RUN_MUJOCO_RECOVERY_E2E = "1"
python -m pytest tests/test_mujoco_sim.py -q -k fallen_part_recovery
python -m llm_mr_bt_planner doctor
```

Automated generation tests use the clearly named `test-reference-client`; they do not call or impersonate a production provider.
The committed reference BT is a test fixture, not evidence of a live LLM generation run. Production `generate` and UI runs always require the selected provider and an API key.
