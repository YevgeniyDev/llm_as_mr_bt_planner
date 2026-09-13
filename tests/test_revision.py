from __future__ import annotations

import pytest

from llm_mr_bt_planner.artifacts import load_plan_file
from llm_mr_bt_planner.config import PROJECT_ROOT
from llm_mr_bt_planner.revision import (
    RevisionCoordinator,
    StaleRevisionError,
    canonical_plan_sha256,
)


def _plans():
    nominal = load_plan_file(
        PROJECT_ROOT / "examples" / "three_robot_component_installation.bt.json"
    )
    recovery = load_plan_file(
        PROJECT_ROOT
        / "examples"
        / "three_robot_component_installation.expected_recovery.bt.json"
    )
    return nominal, recovery


def test_valid_continuation_commits_and_advances_revision():
    nominal, _ = _plans()
    coordinator = RevisionCoordinator(nominal.mission_id)
    candidate_hash = canonical_plan_sha256(nominal)
    context = coordinator.snapshot("state-a")
    certificate = coordinator.certify(context, candidate_hash)

    coordinator.check(
        certificate,
        current_state_sha256="state-a",
        candidate_sha256=candidate_hash,
    )
    assert coordinator.revision == 0

    committed_revision = coordinator.commit(
        certificate,
        current_state_sha256="state-a",
        candidate_sha256=candidate_hash,
    )

    assert committed_revision == 1
    assert coordinator.revision == 1


def test_stale_physical_state_is_rejected_before_candidate_starts():
    nominal, _ = _plans()
    coordinator = RevisionCoordinator(nominal.mission_id)
    candidate_hash = canonical_plan_sha256(nominal)
    certificate = coordinator.certify(coordinator.snapshot("state-a"), candidate_hash)
    dispatched_actions: list[str] = []

    with pytest.raises(StaleRevisionError, match="execution state changed"):
        coordinator.commit(
            certificate,
            current_state_sha256="state-b",
            candidate_sha256=candidate_hash,
        )
        dispatched_actions.append("candidate-started")

    assert coordinator.revision == 0
    assert dispatched_actions == []


def test_old_revision_is_rejected():
    nominal, recovery = _plans()
    coordinator = RevisionCoordinator(nominal.mission_id)
    old_candidate_hash = canonical_plan_sha256(nominal)
    old_certificate = coordinator.certify(
        coordinator.snapshot("state-a"), old_candidate_hash
    )
    current_candidate_hash = canonical_plan_sha256(recovery)
    current_certificate = coordinator.certify(
        coordinator.snapshot("state-a"), current_candidate_hash
    )
    coordinator.commit(
        current_certificate,
        current_state_sha256="state-a",
        candidate_sha256=current_candidate_hash,
    )

    with pytest.raises(StaleRevisionError, match="forest revision changed"):
        coordinator.commit(
            old_certificate,
            current_state_sha256="state-a",
            candidate_sha256=old_candidate_hash,
        )

    assert coordinator.revision == 1


def test_candidate_modification_after_verification_is_rejected():
    nominal, recovery = _plans()
    coordinator = RevisionCoordinator(nominal.mission_id)
    verified_hash = canonical_plan_sha256(nominal)
    certificate = coordinator.certify(coordinator.snapshot("state-a"), verified_hash)

    with pytest.raises(StaleRevisionError, match="candidate changed after verification"):
        coordinator.commit(
            certificate,
            current_state_sha256="state-a",
            candidate_sha256=canonical_plan_sha256(recovery),
        )

    assert coordinator.revision == 0
