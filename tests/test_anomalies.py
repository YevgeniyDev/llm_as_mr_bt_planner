from llm_mr_bt_planner.execution.anomalies import (
    ExecutionIncident,
    IncidentMitigationPolicy,
    IncidentType,
    MitigationAction,
    apply_incident_to_state,
)


def test_dropped_intact_item_is_reacquired():
    decision = IncidentMitigationPolicy().decide(
        ExecutionIncident(IncidentType.OBJECT_DROPPED, "gear", "franka2", "floor_zone")
    )
    assert decision.action is MitigationAction.REACQUIRE
    assert decision.pause_team
    assert "revalidate_remaining_plan" in decision.recovery_steps


def test_damaged_item_uses_replacement():
    decision = IncidentMitigationPolicy().decide(
        ExecutionIncident(
            IncidentType.OBJECT_DAMAGED, "gear", "franka2", damaged=True,
            replacement_ids=("gear_spare_1",), alternate_robots=("go2_z1",),
        )
    )
    assert decision.action is MitigationAction.QUARANTINE_AND_REPLACE
    assert decision.replacement_id == "gear_spare_1"


def test_damaged_item_without_replacement_aborts_safely():
    decision = IncidentMitigationPolicy().decide(
        ExecutionIncident(IncidentType.OBJECT_DAMAGED, "gear", damaged=True)
    )
    assert decision.action is MitigationAction.SAFE_ABORT
    assert decision.requires_human


def test_low_confidence_requires_confirmation():
    decision = IncidentMitigationPolicy().decide(
        ExecutionIncident(IncidentType.OBJECT_DROPPED, "gear", confidence=0.3)
    )
    assert decision.action is MitigationAction.HUMAN_INSPECTION


def test_incident_invalidates_holding_and_marks_revalidation():
    updated = apply_incident_to_state(
        {"holding(franka2, gear)", "drawer_open(parts_drawer)"},
        ExecutionIncident(IncidentType.OBJECT_DROPPED, "gear", "franka2", "floor_zone"),
    )
    assert "holding(franka2, gear)" not in updated
    assert "object_dropped(gear, floor_zone)" in updated
    assert "plan_revalidation_required(gear)" in updated
