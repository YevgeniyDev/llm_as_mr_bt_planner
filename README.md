# LLM-as-MR-BT-Planner

Draft prototype for generating synchronized multi-robot Behavior Trees (BTs) with an LLM.

The repository is intentionally small. It keeps one AI-driven path only:

1. load a multi-robot scenario,
2. ask an OpenAI LLM to produce a task graph, assignments, handoffs, and per-robot BTs,
3. validate the returned JSON against robot capabilities and handoff conditions,
4. ask the LLM to correct invalid or deadlocked plans using validator and simulator feedback,
5. run a small symbolic simulation,
6. write one result file.

There are no rule-based baselines, ablations, deterministic repair loops, or paper-table exporters.

## Files

- `src/main.py` - LLM call, validator, symbolic BT simulator, and CLI.
- `data/scenario.json` - the current three-robot gear assembly scenario.
- `data/simple_handoff.json` - a small two-robot handoff scenario used as a smoke test.
- `.env.example` - API configuration template.

## Run

Requires Python 3.10+ and no extra Python packages.

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then run:

```powershell
python src/main.py
```

Optional:

```powershell
python src/main.py --scenario data/scenario.json --output outputs/run.json --model gpt-4o-mini --max-corrections 4
python src/main.py --scenario data/simple_handoff.json --output outputs/simple_handoff.json --model gpt-4o-mini
python src/main.py --scenario data/scenario.json --output outputs/gear_trials.json --model gpt-4o-mini --trials 3
```

The output is a single JSON file:

```text
outputs/run.json
```

It contains the final LLM plan, validation errors if any, correction-attempt counts, final symbolic state, execution trace, and a minimal success summary. With `--trials N`, the result file contains one summary plus each trial result.

## Main Algorithm

```text
Algorithm 1: LLM-guided multi-robot BT generation
Input: scenario S = (I, G, R, A, H), LLM model M, tick bound K, correction bound C
       I: initial predicates
       G: goal predicates
       R: robots
       A: robot capability library
       H: required inter-robot handoffs
Output: result Omega = (Pi, V, X)
        Pi: generated plan
        V: validation report
        X: symbolic simulation report

1: function LLM-MRBTP(S, M, K)
2:     P <- BuildPrompt(S)                         // scenario, capabilities, schema
3:     y <- QueryLLM(M, P)                          // JSON-only response
4:     Pi <- ParsePlan(y)
5:     for r <- 0 to C do
6:         V <- ValidatePlan(Pi, S)
7:         if V.valid = true then
8:             X <- TickBTs(Pi.behavior_trees, I, A, K)
9:         else
10:            X <- SkippedSimulation(V)
11:        end if
12:        if V.valid = true and X.success = true then
13:            return Omega(Pi, V, X)
14:        end if
15:        if r = C then return Omega(Pi, V, X)
16:        P_r <- BuildCorrectionPrompt(S, Pi, V.errors, X)
17:        y <- QueryLLM(M, P_r)
18:        Pi <- ParsePlan(y)                       // complete corrected plan
19:    end for
20: end function

21: function ValidatePlan(Pi, S)
22:     check required fields: task_graph, assignments, synchronization, behavior_trees
23:     check task graph dependencies are acyclic
24:     check required_tasks or required_actions coverage
25:     check each assigned robot can execute its task action
26:     check each assigned action appears in that robot's BT
27:     check each handoff h in H appears as a synchronization condition
28:     check producer BT creates h.condition and consumer BT waits for it
29:     return validation report
30: end function

31: function TickBTs(B, I, A, K)
32:     state <- I
33:     cursor_i <- 0 for each robot tree B_i
34:     for tick <- 1 to K do
35:         if every cursor is finished then return SuccessIfGoalsHold(state)
36:         progress <- false
37:         for each robot i do
38:             n <- next node in B_i
39:             if n is Condition and n.predicate in state then
40:                 cursor_i <- cursor_i + 1; progress <- true
41:             else if n is Action and preconditions(n, state) hold then
42:                 state <- ApplyEffects(n, state)
43:                 cursor_i <- cursor_i + 1; progress <- true
44:             end if
45:         end for
46:         if progress = false then return Deadlock(state)
47:     end for
48:     return Timeout(state)
49: end function
```

## Current Scenario

The included scenario is a symbolic gear assembly task:

- `go2_z1` opens the drawer, moves the gear tray and screwdriver, then returns the tool.
- `franka1` holds and stabilizes the gearbase.
- `franka2` picks the gear, mounts it, picks the screwdriver, and fastens the screw.

Scenarios can list `required_tasks` with stable IDs, robot assignments, actions, parameters, and dependencies. The LLM must turn those constraints into a task graph, assignments, explicit synchronization, and per-robot BTs. If a scenario omits `required_tasks`, the planner falls back to `required_actions`.

## Limitations

This is a simulation-first draft. It does not control real robots, ROS nodes, perception, motion planners, continuous geometry, collision checking, or real-time execution. The symbolic simulator is deliberately small so the generated BT structure stays easy to inspect before upgrading the execution backend.
