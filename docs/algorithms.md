# Algorithms (full LLM-driven planning)

A paper-ready, notation-level description of the method implemented in `mrbtp`. The stance is
**LLM-as-planner**: all plan *structure* is produced by the language model; the deterministic code only
**verifies** (`validation`) and **simulates** (`simulation`). There is no symbolic back-chaining or
deterministic repair on the critical path.

## Problem setup

A **scenario** is `S = (I, G, O, L, R)`:

- `I`, `G` — initial and goal world states, each a set of ground facts `name(arg, …)`;
- `O`, `L` — objects and locations (the constant symbols);
- `R` — robots. Each `r ∈ R` owns a **capability library** `C_r`. A capability is
  `c = (name, params, pre(c), eff(c))`, where `pre(c)` is a list of precondition templates and
  `eff(c) = (add(c), del(c))` are add/delete templates (PDDL-style; delete literals may be prefix/wildcard
  patterns, see `predicates.matches_pattern`).

A **plan** is `P = (T, A, Y, B)`:

- `T` — task graph (nodes `(id, action, params, depends_on)`, must be acyclic);
- `A` — assignments `task → robot`;
- `Y` — synchronization edges `(condition, producer, consumer)` for inter-robot waits;
- `B = {r ↦ tree_r}` — one Behavior Tree per robot (`Sequence`/`Fallback`/`Parallel` composites over
  `Action`/`Condition` leaves).

**World-state transition.** Executing action `a` (capability `c` under binding `θ`) on state `w`:
`w' = (w \ match(del(c), θ)) ∪ ground(add(c), θ)` — delete patterns applied before adds. This is the only
operation that mutates world state; it is used identically by the simulator and the validator's reachability
check.

The planning target: produce `P` such that the per-robot BTs, executed under reactive tick semantics with
blocking inter-robot guards, drive `I` to a state satisfying `G` without deadlock.

---

## Algorithm 1 — LLM-as-planner with self-correction

`planner.run_planner` (single-stage). Budget `C` correction rounds (`C=0` ⇒ single-shot), `N` samples per
round (best-of-N), tick bound `K`.

```
Input:  scenario S, LLM M, budget C, samples N, ticks K
Output: plan P, validation report e_v, simulation report e_s, rounds c

prompt ← BuildPrompt(S)                       # scenario + output schema + rules + back-chaining METHOD
(P, e_v, e_s) ← BestOfN(M, prompt, S, N, K)
c ← 0
while c < C and not (valid(e_v) and success(e_s)):
    c ← c + 1
    prompt ← BuildCorrection(S, e_v, e_s, P)  # typed validator errors + compact sim trace + previous P
    (P, e_v, e_s) ← BestOfN(M, prompt, S, N, K)
return (P, e_v, e_s, c)
```

`BuildPrompt` gives the model only `S`, the JSON output schema, task-agnostic rules, and a **general
back-chaining method** (work backward from `G`; every precondition must be in `I`, produced earlier by the
*same* robot, or produced by another robot and waited on via a `Condition`; robot-scoped predicates such as
`holding(r, x)` are creatable only by `r`'s own action). No task-specific producer chain is supplied.

### Best-of-N selection

```
BestOfN(M, prompt, S, N, K):
    best ← ⊥
    repeat N times:
        P   ← Parse(ExtractJSON(M.complete(SYSTEM, prompt)))
        e_v ← Validate(P, S)                                   # Algorithm 4
        e_s ← valid(e_v) ? Simulate(P, S, K) : ⊥              # Algorithm 3
        score ← ( valid(e_v) ∧ success(e_s),  valid(e_v),  −|e_v| )   # lexicographic
        if best = ⊥ or score > score(best): best ← (P, e_v, e_s, score)
        if valid(e_v) ∧ success(e_s): break                   # early exit on first working plan
    return best
```

Ranking prefers a valid+successful plan, then a merely valid one, then the fewest validation errors.
Diversity across the `N` samples requires a non-zero sampling temperature (OpenAI; Anthropic Opus 4.8 has no
temperature knob).

---

## Algorithm 2 — Two-stage generation

`planner._two_stage_generate`. Empirically, most pure-mode failures are in *choosing and ordering producer
actions*, not in BT syntax. Two-stage isolates that step. The two stages share the round counter (Stage 2 may
use up to `C` further rounds, i.e. `2C` total).

```
# Stage 1 — feasible, ordered action plan (no conditions, no synchronization)
A ← BestOfN_actions(M, BuildActionPlanPrompt(S), S, N, K)     # returns {r ↦ [(action, params), …]}
c ← 0
while c < C and not working(A):
    c ← c + 1
    A ← BestOfN_actions(M, BuildActionPlanCorrection(S, …, A), S, N, K)

# Stage 2 — encode the FIXED action plan as BTs + synchronization
P ← BestOfN(M, BuildEncodePrompt(S, A), S, N, K)
while c < 2C and not (valid ∧ success):
    c ← c + 1
    P ← BestOfN(M, BuildEncodeCorrection(S, …, P, A), S, N, K)
return (P, …, c)
```

**Stage-1 validation without conditions.** `_synthesize_plan(A)` wraps each robot's action list into a
condition-free `Sequence` (plus a matching task graph and assignments) and runs the *same* validator and
simulator. Because an `Action` whose preconditions are unmet returns `RUNNING` (a blocking guard, §Algorithm
3), a feasible action plan simulates to success *without any explicit conditions*, while a missing or
mis-ordered producer surfaces as `unsupported_precondition` or `deadlock`. The synthesis step is pure
bookkeeping — it never plans.

---

## Algorithm 3 — Tick-based symbolic simulation

`simulation.simulate`. Round-robin over robots; each robot's tree is ticked once per global tick. Reactive
composite semantics with memory; inter-robot synchronization is modeled as **blocking guards**.

```
Input:  plan P, scenario S, tick bound K
w ← I;  done[r] ← false for all r;  mem[r] ← ∅
for tick = 1 … K:
    if all done[·]: return Success(goal = G ⊆ w)
    snapshot ← w
    for r in R, in order, with not done[r]:
        budget ← 1                              # actions_per_tick: one action per robot per tick
        st ← Tick(tree_r, r, w, budget, mem[r])
        if st ∈ {SUCCESS, FAILURE}: done[r] ← true
    if all done[·]: return Success(goal = G ⊆ w)
    if w = snapshot: return Deadlock(blocked)   # nothing changed, yet some tree still RUNNING
return Timeout
```

**Leaf semantics** (the synchronization model):

- `Action a`: if `budget = 0` → `RUNNING` (resume next tick); else if any `pre(a)` ∉ `w` → `RUNNING` (wait);
  else apply `w ← w'`, `budget ← budget − 1`, return `SUCCESS`.
- `Condition p`: `p ∈ w` → `SUCCESS`, else `RUNNING` (wait).

**Composite semantics:** `Sequence` advances to the first non-`SUCCESS` child and resumes there next tick
(`mem`); `Fallback` to the first non-`FAILURE`; `Parallel` ticks unfinished children and succeeds at
`success_threshold` successes, **latching** each child that has returned a terminal status so its one-shot
effects are not re-applied and its trace row is not duplicated.

**Deadlock soundness.** Only an executed action mutates `w`, and each robot acts at most once per tick. Hence
a full tick that leaves `w` unchanged while some tree is still `RUNNING` guarantees every subsequent tick is
identical — no progress is possible, so the state is reported as a deadlock rather than a transient wait.
`Timeout` is the separate case of exhausting the tick bound `K`.

---

## Algorithm 4 — Static validator `V` (task-agnostic)

`validation.validate_plan`. Turns "is `P` correct?" into concrete, typed errors — the structured feedback
that makes LLM self-correction tractable. Every check is **task-agnostic** (it generalizes to any scenario),
which is why it stays on by default while planning-style aids do not (see *pure vs assisted*, below).

Let `produced(P, S) = ⋃ add-effects of every Action leaf in B` (the predicates any BT action can create). The
core causal check: a goal / precondition / condition predicate `q` is **supported** iff `q ∈ I` or
`q ∈ produced`.

| Group | Error types |
|---|---|
| **Structure** | `missing_field`, `invalid_bt`, `invalid_task_graph`, `duplicate_task`, `unknown_dependency`, `cyclic_dependency` |
| **Assignment** | `unknown_task`, `unknown_robot`, `invalid_capability`, `duplicate_assignment`, `unassigned_task`, `missing_bt_action`, `unassigned_bt_action` |
| **Causal support** | `unsupported_goal`, `unsupported_precondition`, `unsupported_condition`, `condition_before_producer` |
| **Synchronization** | `invalid_synchronization`, `missing_sync_condition`, `missing_sync_producer` |

`condition_before_producer` enforces temporal ordering inside one robot's BT (a robot may not wait on a
predicate before its own action creates it); the synchronization checks enforce that each `Y`-edge has a real
producer action upstream and the exact `Condition` placed before the consumer. Acyclicity is a white/grey/black
DFS over `T`.

---

## Pure vs assisted (ablation switch)

Two deterministic computations perform planning-style causal reasoning and are therefore **off by default**,
gated behind flags so they do not confound the "LLM alone" claim:

- **`include_hints`** — `prompts.build_dependency_hints` precomputes, per unmet precondition, candidate
  producer capabilities (`domain.candidate_producers`) and injects them into the prompt: partial back-chaining
  handed to the model.
- **`suggest_producers`** — the validator appends candidate-producer names to its `unsupported_*` messages
  instead of only stating *what* is unsupported.

In **pure** mode (default) neither is active: `(prompt + I + schema) → "what is wrong"`. **Assisted** mode
enables both as a baseline to quantify how much the scaffolding helps. The fixed hint/suggestion text would not
generalize across tasks — which is exactly why it is an opt-in baseline, not the default path.

---

## Complexity and termination

Each round costs `N` LLM calls; the loop runs at most `C` (single-stage) or `2C` (two-stage) rounds, so a run
issues `O(N·C)` calls and terminates on the first valid+successful plan or budget exhaustion. `Validate` is
linear in plan size; `Simulate` runs `≤ K` ticks, each `O(|R| · |tree|)`, and always halts (success, deadlock,
or timeout). The reported metrics per trial are `valid`, `goal_success`, and `correction_rounds`; experiments
aggregate these over `scenarios × trials` (`experiments.run_experiment`).
