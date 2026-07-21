"""Contracts and policy for object/tool incidents during physical execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..predicates import format_predicate, parse_predicate


class IncidentType(str, Enum):
    GRASP_LOST = "grasp_lost"
    OBJECT_DROPPED = "object_dropped"
    OBJECT_DAMAGED = "object_damaged"
    TOOL_FAILURE = "tool_failure"
    OBJECT_MISSING = "object_missing"


class MitigationAction(str, Enum):
    REACQUIRE = "reacquire"
    QUARANTINE_AND_REPLACE = "quarantine_and_replace"
    REASSIGN_TOOL = "reassign_tool"
    SAFE_ABORT = "safe_abort"
    HUMAN_INSPECTION = "human_inspection"


@dataclass(frozen=True)
class ExecutionIncident:
    incident_type: IncidentType
    object_id: str
    robot_id: str | None = None
    location: str | None = None
    damaged: bool = False
    accessible: bool = True
    replacement_ids: tuple[str, ...] = ()
    alternate_robots: tuple[str, ...] = ()
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MitigationDecision:
    action: MitigationAction
    pause_team: bool
    selected_robot: str | None = None
    replacement_id: str | None = None
    requires_human: bool = False
    reason: str = ""
    recovery_steps: tuple[str, ...] = ()


class IncidentMitigationPolicy:
    """Conservative first response; physical execution must verify recovery."""

    def decide(self, incident: ExecutionIncident) -> MitigationDecision:
        if incident.confidence < 0.5:
            return MitigationDecision(
                MitigationAction.HUMAN_INSPECTION,
                pause_team=True,
                requires_human=True,
                reason="Incident confidence is too low for autonomous recovery.",
                recovery_steps=("safe_stop", "request_perception_recheck", "request_human_confirmation"),
            )

        damaged = incident.damaged or incident.incident_type is IncidentType.OBJECT_DAMAGED
        if damaged:
            if incident.replacement_ids:
                candidates = incident.alternate_robots or (incident.robot_id,)
                return MitigationDecision(
                    MitigationAction.QUARANTINE_AND_REPLACE,
                    pause_team=True,
                    selected_robot=candidates[0],
                    replacement_id=incident.replacement_ids[0],
                    reason="Damaged items must not be reused.",
                    recovery_steps=(
                        "safe_stop", "quarantine_damaged_item", "fetch_replacement",
                        "invalidate_dependent_effects", "revalidate_remaining_plan",
                    ),
                )
            return MitigationDecision(
                MitigationAction.SAFE_ABORT,
                pause_team=True,
                requires_human=True,
                reason="The item is damaged and no verified replacement is available.",
                recovery_steps=("safe_stop", "quarantine_damaged_item", "notify_operator"),
            )

        if incident.incident_type in {IncidentType.GRASP_LOST, IncidentType.OBJECT_DROPPED}:
            if incident.accessible:
                candidates = incident.alternate_robots or (incident.robot_id,)
                return MitigationDecision(
                    MitigationAction.REACQUIRE,
                    pause_team=True,
                    selected_robot=candidates[0],
                    reason="The item appears intact and reachable.",
                    recovery_steps=(
                        "safe_stop", "localize_item", "inspect_item", "reacquire_item",
                        "refresh_world_state", "revalidate_remaining_plan",
                    ),
                )
            return MitigationDecision(
                MitigationAction.SAFE_ABORT,
                pause_team=True,
                requires_human=True,
                reason="The item is outside the certified recovery workspace.",
                recovery_steps=("safe_stop", "mark_item_unreachable", "notify_operator"),
            )

        if incident.incident_type is IncidentType.TOOL_FAILURE and incident.alternate_robots:
            return MitigationDecision(
                MitigationAction.REASSIGN_TOOL,
                pause_team=True,
                selected_robot=incident.alternate_robots[0],
                replacement_id=incident.replacement_ids[0] if incident.replacement_ids else None,
                reason="A capable alternate robot is available.",
                recovery_steps=("safe_stop", "isolate_failed_tool", "reassign_or_replace", "revalidate_remaining_plan"),
            )

        return MitigationDecision(
            MitigationAction.SAFE_ABORT,
            pause_team=True,
            requires_human=True,
            reason="No certified autonomous mitigation is available.",
            recovery_steps=("safe_stop", "preserve_scene", "notify_operator"),
        )


def apply_incident_to_state(state: Iterable[str], incident: ExecutionIncident) -> set[str]:
    """Invalidate stale holding facts and add explicit incident predicates."""
    updated = set(state)
    for fact in list(updated):
        name, args = parse_predicate(fact)
        if name == "holding" and incident.object_id in args:
            updated.remove(fact)
    if incident.incident_type in {IncidentType.GRASP_LOST, IncidentType.OBJECT_DROPPED}:
        updated.add(format_predicate("object_dropped", [incident.object_id, incident.location or "unknown"]))
    if incident.damaged or incident.incident_type is IncidentType.OBJECT_DAMAGED:
        updated.add(format_predicate("object_damaged", [incident.object_id]))
        updated.add(format_predicate("object_unavailable", [incident.object_id]))
    if incident.incident_type is IncidentType.OBJECT_MISSING:
        updated.add(format_predicate("object_missing", [incident.object_id]))
    updated.add(format_predicate("plan_revalidation_required", [incident.object_id]))
    return updated
