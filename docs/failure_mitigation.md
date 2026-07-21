# Physical failure mitigation: dropped, missing, or damaged items

The existing `RecoveryController` handles an action returning `FAILURE` through
retry and capability-based reassignment. Physical incidents are different because
the world state itself may have changed.

## Safety invariant

No robot resumes the original BT after an object incident until perception has
refreshed the affected predicates and the remaining plan has been revalidated.
Damaged objects are never automatically reused.

## Response sequence

1. Pause BTs whose future actions depend on the affected object.
2. Command a safe stop/retreat and preserve the scene.
3. Re-localize and inspect the object.
4. Update `object_dropped`, `object_damaged`, `object_unavailable`, and
   `plan_revalidation_required` predicates.
5. Select one bounded mitigation:
   - intact and reachable: reacquire, refresh state, and revalidate;
   - damaged with a verified spare: quarantine, substitute the spare, invalidate
     dependent effects, and regenerate/revalidate the remaining BTs;
   - failed tool with another capable robot/tool: reassign;
   - unreachable, ambiguous, or damaged without a spare: safe abort and escalate.
6. Resume only after perception confirms the recovery postconditions.

`execution.anomalies` implements the incident/decision contracts, conservative
policy, and symbolic invalidation step. Next, connect the contracts to a MuJoCo
contact/drop detector and then ROS perception and safety controllers. Measure
incident recovery, replacement use, safe aborts, human escalation, and latency in
a separate robustness experiment.
