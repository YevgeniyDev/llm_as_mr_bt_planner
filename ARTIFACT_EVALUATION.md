# Artifact Evaluation Guide

This repository is the reproducible software artifact for LLM-as-Multi-Robot-BT-Planner. It
contains the planner, strict schemas, five scenario/reference-BT pairs, comparison adapters, tests,
and optional MuJoCo execution code. Generated runs, downloaded upstream repositories, credentials,
and the working manuscript are intentionally excluded from version control.

## Scope and platform mapping

| Evidence scope | Mobile platform | What is implemented |
| --- | --- | --- |
| Three-robot courier, packing/delivery, and component recovery | Unitree Go2 with Z1 | Full dynamic Go2 free base, 12-motor contact gait, Z1 manipulation, and two Panda arms in MuJoCo |
| Five-agent inspection and pipe repair | Unitree B2 with Z1 | Official B2 geometry, torque-controlled stance, task-space base translation, Husky, and two Panda arms in MuJoCo |
| Planned final real-world evaluation | Unitree B2 with Z1 | Not included in this release; requires guarded hardware skills, perception, motion planning, ROS 2 dispatch, and safety supervision |

The Go2 scenarios are preserved intentionally. Their identifiers and results must not be renamed or
reported as B2 hardware evidence. The planned B2 hardware evaluation is a separate experimental
layer over the same task-level BT contract.

## A1: Install and verify the symbolic artifact

Use Python 3.10 or newer from a source checkout or extracted source distribution. The command-line
defaults intentionally resolve the bundled scenarios, schemas, and templates from that source root.

```powershell
python -m pip install -e ".[ui,dev]"
lmrbtp doctor
python -m ruff check src tests
python -m mypy src/llm_mr_bt_planner
python -m pytest -q
```

`doctor` performs no provider call. It validates and contract-simulates all five committed nominal
scenario/BT pairs. The test suite skips opt-in physics integration tests unless their environment
flags are enabled.

Expected release result:

- Ruff: no errors;
- mypy: no errors;
- pytest: 155 passed, four opt-in tests skipped when the physics flags are unset;
- doctor: dependency checks pass and all five reference plans validate and reach their symbolic goals.

## A2: Inspect one exact Behavior Tree

```powershell
lmrbtp validate `
  --scenario examples/three_robot_courier.json `
  --bt examples/three_robot_courier.bt.json

lmrbtp simulate `
  --scenario examples/three_robot_courier.json `
  --bt examples/three_robot_courier.bt.json `
  --max-ticks 100
```

This establishes correctness only against the declared symbolic contract. It does not establish
trajectory feasibility, perception, hardware compatibility, or safety.

## A3: Run the optional MuJoCo adapter

```powershell
python -m pip install -e ".[mujoco]"
lmrbtp mujoco --setup-only
lmrbtp mujoco --headless
```

The setup command downloads pinned, license-preserving robot assets. The execution command uses the
committed Go2 courier scenario and reference BT by default. See the README for packaging, recovery,
five-agent B2, recording, and live-viewer commands. A successful report applies only to the exact
scene and controller revision recorded in that report.

## A4: Generate a fresh LLM-authored plan

Fresh generation is intentionally separate from deterministic artifact verification and requires a
provider key:

```powershell
$env:OPENAI_API_KEY = "your-key"
lmrbtp generate `
  --provider openai `
  --scenario examples/three_robot_courier.json
```

Each run is written under `outputs/` with exact prompts, model provenance, validation results,
simulation traces, checksums, and publication status. `outputs/` is ignored because provider runs,
downloaded checkpoints, videos, and comparison sources can be large and may contain unreleased
experimental evidence. Archive the exact accepted bundles separately before deriving paper tables.

## A5: Reproduce comparison methods

The five isolated comparison runners are LLM-as-BT-Planner, LLM-BT, BETR-XP-LLM, LLM-HBT, and
MRBTP. Preparation commands download and verify pinned upstream material; run commands translate
the common declarative scenario without silently repairing task-action choices. Start from the
`Paper comparison baselines` section of the README and retain every generated manifest.

Single accepted runs are integration evidence, not statistical estimates. Do not fill aggregate
paper tables until every included configuration has the planned number of independent trials,
identical scenario hashes, complete manifests, documented exclusions, and uncertainty intervals.

## Submission hygiene

- Never commit `.env`, credentials, generated `outputs/`, downloaded third-party repositories, or
  local manuscript files.
- Keep reference BTs labeled as regression fixtures; they are not fresh LLM evidence.
- Report Go2 simulation, B2 simulation, and future B2 hardware evaluation separately.
- Preserve the exact scenario, BT, model ID, prompt/response record, source revision, and checksum
  manifest for every reported run.
- Run `git diff --check` and all A1 checks from a clean clone before creating an archival release.

ICRA 2027 uses double-anonymous review and does not accept a separate software supplement beyond
the permitted accompanying video. Reviewers are not required to follow external links. Accordingly,
this repository should be treated as an archival/reproducibility artifact, not as evidence that is
absent from the eight-page submission. Do not expose a deanonymizing public repository URL in the
blind manuscript. Verify the current conference rules before submission:
https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/
