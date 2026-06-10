# LLM-as-MR-BT-Planner

Draft prototype for generating synchronized multi-robot Behavior Trees (BTs) with an LLM.

The repository is intentionally small. It keeps one AI-driven path only:

1. load a multi-robot scenario,
2. ask an OpenAI LLM to infer a task graph, assignments, synchronization, and per-robot BTs,
3. validate the returned JSON for structure, robot capabilities, and internal synchronization consistency,
4. ask the LLM to correct invalid or deadlocked plans using validator and simulator feedback,
5. run a small symbolic simulation,
6. write one result file.

There are no rule-based baselines, ablations, deterministic repair loops, or paper-table exporters.

## Files

- `src/main.py` - LLM call, validator, symbolic BT simulator, and CLI.
- `data/scenario.json` - the current three-robot gear assembly scenario.
- `.env.example` - API configuration template.

## Run

Requires Python 3.10+ and no extra Python packages.

Copy `.env.example` to `.env`, set `OPENAI_API_KEY`, then run:

```powershell
python src/main.py
```

Optional:

```powershell
python src/main.py --scenario data/scenario.json --output outputs/run.json --model gpt-4o --max-corrections 4
```

The gear assembly scenario needs a capable model such as `gpt-4o` because the LLM must infer a longer causal chain without a hidden task checklist.

The output is a single JSON file:

```text
outputs/run.json
```

It contains the final LLM plan, validation errors if any, correction count, final symbolic state, execution trace, and a minimal success summary.

## Main Algorithm

```text
Algorithm 1: LLM-guided multi-robot BT generation
Input: scenario S = (u, I, G, O, L, R, A), LLM model M, tick bound K, correction bound C
       u: natural-language task instruction
       I: initial predicates
       G: goal predicates
       O: objects
       L: locations
       R: robots
       A: robot capability library
Output: result Omega = (Pi, V, X)
        Pi: generated plan
        V: validation report
        X: symbolic simulation report

1: function LLM-MRBTP(S, M, K)
2:     P <- BuildPrompt(S)                         // scenario, capabilities, dependency hints, schema
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
16:        P_r <- BuildCorrectionPrompt(S, V.errors, X)
17:        y <- QueryLLM(M, P_r)
18:        Pi <- ParsePlan(y)                       // complete corrected plan
19:    end for
20: end function

21: function ValidatePlan(Pi, S)
22:     check required fields: task_graph, assignments, synchronization, behavior_trees
23:     check task graph dependencies are acyclic
24:     check each assigned robot can execute its task action
25:     check each assigned action appears in that robot's BT
26:     check each BT action has a matching assigned task
27:     check action preconditions and non-initial goals have generated producers
28:     check generated Conditions are initially true or produced by generated actions
29:     check each generated synchronization condition is produced by its producer BT
30:     check each generated synchronization condition appears in its consumer BT
31:     return validation report
32: end function

33: function TickBTs(B, I, A, K)
34:     state <- I
35:     cursor_i <- 0 for each robot tree B_i
36:     for tick <- 1 to K do
37:         if every cursor is finished then return SuccessIfGoalsHold(state)
38:         progress <- false
39:         for each robot i do
40:             n <- next node in B_i
41:             if n is Condition and n.predicate in state then
42:                 cursor_i <- cursor_i + 1; progress <- true
43:             else if n is Action and preconditions(n, state) hold then
44:                 state <- ApplyEffects(n, state)
45:                 cursor_i <- cursor_i + 1; progress <- true
46:             end if
47:         end for
48:         if progress = false then return Deadlock(state)
49:     end for
50:     return Timeout(state)
51: end function
```

## Current Scenario

The included scenario is a symbolic gear assembly task. It gives the LLM only the instruction, initial state, goal state, objects, locations, and robot capability library.

- `go2_z1` opens the drawer, moves the gear tray and screwdriver, then returns the tool.
- `franka1` holds and stabilizes the gearbase.
- `franka2` picks the gear, mounts it, picks the screwdriver, and fastens the screw.

The scenario does not provide an action checklist or fixed task ordering. The LLM must infer the task graph, assignments, synchronization, and BT nodes from capability preconditions/effects. A plan is accepted only when the validator passes and the symbolic simulator reaches every `goal_state` predicate.

## Limitations

This is a simulation-first draft. It does not control real robots, ROS nodes, perception, motion planners, continuous geometry, collision checking, or real-time execution. The symbolic simulator is deliberately small so the generated BT structure stays easy to inspect before upgrading the execution backend.
