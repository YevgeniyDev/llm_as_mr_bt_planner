"""State-bound authorization for installing verified behavior-tree revisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .plan import Plan


@dataclass(frozen=True)
class RevisionContext:
    """Mission revision and physical state against which a candidate is prepared."""

    mission_id: str
    revision: int
    state_sha256: str


@dataclass(frozen=True)
class RevisionCertificate:
    """Authorization inputs bound to an already verified candidate plan."""

    mission_id: str
    source_revision: int
    source_state_sha256: str
    candidate_sha256: str


class StaleRevisionError(RuntimeError):
    """Raised when a verified candidate is stale or differs at commit time."""


class RevisionCoordinator:
    """Atomically check and advance the installed behavior-tree revision."""

    def __init__(self, mission_id: str):
        self.mission_id = mission_id
        self.revision = 0

    def snapshot(self, state_sha256: str) -> RevisionContext:
        return RevisionContext(
            mission_id=self.mission_id,
            revision=self.revision,
            state_sha256=state_sha256,
        )

    def certify(
        self,
        context: RevisionContext,
        candidate_sha256: str,
    ) -> RevisionCertificate:
        return RevisionCertificate(
            mission_id=context.mission_id,
            source_revision=context.revision,
            source_state_sha256=context.state_sha256,
            candidate_sha256=candidate_sha256,
        )

    def commit(
        self,
        certificate: RevisionCertificate,
        *,
        current_state_sha256: str,
        candidate_sha256: str,
    ) -> int:
        """Authorize an exact candidate and advance the installed revision."""

        self.check(
            certificate,
            current_state_sha256=current_state_sha256,
            candidate_sha256=candidate_sha256,
        )
        self.revision += 1
        return self.revision

    def check(
        self,
        certificate: RevisionCertificate,
        *,
        current_state_sha256: str,
        candidate_sha256: str,
    ) -> None:
        """Check a certificate without changing the installed revision."""

        if certificate.mission_id != self.mission_id:
            raise StaleRevisionError("mission changed")
        if certificate.source_revision != self.revision:
            raise StaleRevisionError("forest revision changed")
        if certificate.source_state_sha256 != current_state_sha256:
            raise StaleRevisionError("execution state changed")
        if certificate.candidate_sha256 != candidate_sha256:
            raise StaleRevisionError("candidate changed after verification")


def canonical_plan_sha256(plan: Plan) -> str:
    """Hash the canonical JSON representation of a behavior-tree plan."""

    payload = json.dumps(
        plan.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
