# Multi-Robot Behavior Tree Planner

This application uses OpenAI or Anthropic (Claude) to generate complete synchronized Behavior Trees for one three-robot courier mission. It then validates and symbolically simulates the model's exact trees and saves successful results as JSON and XML.

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
2. Use the bundled runnable example, or upload a JSON scenario. Uploaded files are loaded and validated automatically.
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

- [Runnable example](examples/three_robot_courier.json)
- [Committed reference BT](examples/three_robot_courier.bt.json) used only by "doctor" command for separate validation, simulation, and regression checks of generated BTs
- [Blank template](templates/three_robot_scenario.template.json)
- [JSON Schema](schemas/scenario.schema.json)

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

Other commands do not call an LLM:

```powershell
lmrbtp template --output my-scenario.json
lmrbtp validate --scenario examples/three_robot_courier.json --bt outputs/runs/ROUTE/behavior_tree.json
lmrbtp simulate --scenario examples/three_robot_courier.json --bt outputs/runs/ROUTE/behavior_tree.json
lmrbtp render --bt outputs/runs/ROUTE/behavior_tree.json --html outputs/tree.html --xml outputs/tree.xml
lmrbtp doctor
```

On successful generation, the final console lines provide `BT_FILE=<absolute path>` and `BT_SHA256=<hash>`.

## Run the first scenario in MuJoCo

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

Both forms default to [the first scenario](examples/three_robot_courier.json) and [its committed BT](examples/three_robot_courier.bt.json). A generated BT for that same scenario can be supplied explicitly:

```powershell
lmrbtp mujoco --bt "C:\path\to\behavior_tree.json"
```

The command statically validates the BT first, composes one MuJoCo world, settles all three robots, and then executes the BT leaves concurrently. It prints action/resource progress and writes `physical_execution_report.json` under `outputs/mujoco/`. A failure returns a nonzero exit code and identifies the robot, node, failed measured predicate, controller stage, or timeout.

This first adapter deliberately supports only `three_robot_courier`; it rejects a different task instead of silently mapping it to the courier. The world contains two independently prefixed 7-DoF Panda models, two separated laboratory workbenches, and one Go2 with the Z1 gripper model mounted on its trunk. In the default overview, Franka A is mounted at the midpoint of the upper outer edge of the source bench and Franka B at the midpoint of the lower outer edge of the destination bench. Both material flows run left to right: Franka A moves the payload from the yellow source pad to the green Go2 handoff, while Franka B moves it from the green Go2 handoff to the red installation fixture. Each arm crosses its base-joint centerline during this side-to-side transfer. Controller target sites are retained internally but hidden in the finished view; they have no collision geometry.

The two Go2 dock centers are `3.00 m` apart, compared with `1.20 m` in the earlier compact layout. Both green handoff pads align directly with the right-side travel route, and the benches leave a clear aisle between them. Z1 retains the payload in its closed grasp for the entire route and releases it only on the green destination pad. The Go2 uses the `0.54 m` table-offset reference, and its free-joint pose is initialized once and never written during execution, so measured displacement comes from leg torques, inertia, and floor contact. The dynamic payload is moved through actuator-driven differential IK, grasp constraints, and a 12-motor alternating contact gait. Arm-link gravity compensation is enabled, as in the `mjctrl` example; the Go2 and payload remain under full gravity.

The grasp model is explicit: each tabletop Panda follows pose-aware IK with a constrained vertical gripper orientation so its fingers approach from above. Z1 first reaches its measured open joint position, follows pose-aware IK to a side-grasp pose, and closes at a rate-limited command. The grasp is accepted only when the measured joint reaches its contact angle and MuJoCo reports payload contact on both dedicated opposing finger-pad geoms. A soft weld is then activated at the current relative pose to prevent numerical slippage without snapping or teleporting the payload. Navigation fails if that measured grasp constraint is lost. At the destination, Z1 opens to its measured joint limit before releasing the constraint and retreating. Final fixture installation uses the same visible constraint mechanism. This is a controlled-simulation approximation, not a claim that fingertip friction, perception, or grasp uncertainty has been validated.

After either Panda places the payload, it retreats and returns to its home joint configuration before its BT action succeeds and the shared exchange-zone resource is released. This prevents the next robot from entering while a Franka remains extended over the handoff area.

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

The separate MuJoCo adapter additionally tests the first courier BT against one fixed dynamic scene: actuator limits, gravity, contacts, reachable Cartesian motion, payload transfer, base stability, docking, and contact-driven Go2 displacement. Its report is evidence for that exact model, controller configuration, and initial state only. It does **not** establish general collision avoidance, perception accuracy, ROS 2 integration, sim-to-real validity, or physical-robot safety. XML remains an export format rather than a hardware controller.

## Development checks

```powershell
python -m pip install -e ".[ui,mujoco,dev]"
python -m ruff check src tests
python -m mypy src/llm_mr_bt_planner
python -m pytest -q
$env:LMRBTP_RUN_MUJOCO_E2E = "1"
python -m pytest tests/test_mujoco_sim.py -q
python -m llm_mr_bt_planner doctor
```

Automated generation tests use the clearly named `test-reference-client`; they do not call or impersonate a production provider.
The committed reference BT is a test fixture, not evidence of a live LLM generation run. Production `generate` and UI runs always require the selected provider and an API key.
