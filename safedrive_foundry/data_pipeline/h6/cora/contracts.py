"""Typed, self-hashed C2 root/proposal/outcome contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from data_pipeline.h2.contracts import stable_sha256


ROOT_ANCHOR_SCHEMA = "safedrive.cora.root_anchor.v1"
PROPOSAL_SCHEMA = "safedrive.cora.proposal.v1"
BRANCH_OUTCOME_SCHEMA = "safedrive.cora.branch_outcome.v1"
PAIR_INDEX_SCHEMA = "safedrive.cora.pair_index.v1"
FEATURE_SCHEMA = "safedrive.cora.feature_view.v1"


def _finite(name: str, value: float | int | None) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"cora_non_finite:{name}")


@dataclass(frozen=True)
class OutcomeValue:
    value: float | int | bool | str | None
    unit: str
    valid: bool
    derivation_version: str = "cora-c2-labeler-v1"

    def __post_init__(self) -> None:
        if not self.valid and self.value is not None:
            raise ValueError("cora_invalid_outcome_must_be_null")
        if self.valid and isinstance(self.value, (float, int)) and not isinstance(self.value, bool):
            _finite("outcome", self.value)
        if not self.unit:
            raise ValueError("cora_outcome_unit_missing")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoraProposal:
    proposal_id: str
    proposal_sha256: str
    root_id: str
    kind: str
    trajectory: tuple[Mapping[str, Any], ...]
    guard: Mapping[str, Any]
    base_proposal_id: str | None = None
    base_proposal_sha256: str | None = None
    audit_source: str | None = None
    operator: str | None = None
    magnitude: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    auxiliary_only: bool = False
    status: str = "READY"
    schema_version: str = PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.root_id or len(self.proposal_sha256) != 64:
            raise ValueError("cora_proposal_identity")
        if self.kind not in {"nominal", "offline_intervention"}:
            raise ValueError("cora_proposal_kind")
        if self.kind == "offline_intervention" and (
            not self.base_proposal_id or not self.base_proposal_sha256 or not self.operator
        ):
            raise ValueError("cora_intervention_parent")
        if self.kind == "nominal" and self.operator is not None:
            raise ValueError("cora_nominal_operator")

    @property
    def content_sha256(self) -> str:
        return stable_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if include_hash:
            payload["content_sha256"] = stable_sha256(payload)
        return payload


@dataclass(frozen=True)
class CoraBranchOutcome:
    root_id: str
    proposal_id: str
    proposal_sha256: str
    split: str
    guard_verdict: str
    reset: Mapping[str, Any]
    safety_input_id: str | None
    safety_input_sha256: str | None
    pre_repair_id: str | None
    pre_repair_sha256: str | None
    executable_id: str | None
    executable_sha256: str | None
    applied_id: str | None
    applied_sha256: str | None
    decision_kind: str | None
    applied_mode: str | None
    repair_attempted: bool
    repair_success: bool | None
    would_require_cross_candidate_fallback: bool
    ticks_executed: int
    terminal_reason: str
    cleanup_complete: bool
    heads: Mapping[str, OutcomeValue]
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    artifact_sha256: Mapping[str, str] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    auxiliary_only: bool = False
    schema_version: str = BRANCH_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        if not self.root_id or not self.proposal_id or len(self.proposal_sha256) != 64:
            raise ValueError("cora_branch_identity")
        if self.ticks_executed < 0:
            raise ValueError("cora_branch_negative_ticks")
        if self.repair_success is True and not self.repair_attempted:
            raise ValueError("cora_branch_repair_without_attempt")

    @property
    def identity_valid(self) -> bool:
        if self.safety_input_id != self.proposal_id or self.safety_input_sha256 != self.proposal_sha256:
            return False
        if self.pre_repair_id is not None:
            if self.pre_repair_id != self.proposal_id or self.pre_repair_sha256 != self.proposal_sha256:
                return False
        if self.repair_attempted and self.pre_repair_id is None:
            return False
        if self.applied_mode == "TRACK_APPROVED":
            return bool(
                self.executable_id
                and self.executable_sha256
                and self.applied_id == self.executable_id
                and self.applied_sha256 == self.executable_sha256
            )
        return self.applied_mode in {
            "MINIMAL_RISK_BRAKE", "EMERGENCY_BRAKE", "HOLD_NO_EXEC", "EXEC_ID_ORPHAN"
        }

    @property
    def reset_comparable(self) -> bool:
        return bool(self.reset.get("comparable", False))

    @property
    def legal_terminal(self) -> bool:
        if self.terminal_reason == "HORIZON_COMPLETE":
            return self.ticks_executed == 50
        if self.terminal_reason == "MRM_STANDSTILL":
            return 10 <= self.ticks_executed <= 50
        if self.terminal_reason in {"ROUTE_COMPLETE", "COLLISION_TERMINAL"}:
            return 1 <= self.ticks_executed <= 50
        return False

    @property
    def outcome_valid(self) -> bool:
        return bool(
            self.guard_verdict in {"PASS", "REVIEW", "REJECT"}
            and (self.guard_verdict in {"PASS", "REVIEW"} or self.auxiliary_only)
            and self.reset_comparable
            and self.identity_valid
            and self.legal_terminal
            and self.cleanup_complete
            and not self.would_require_cross_candidate_fallback
            and not self.errors
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["heads"] = {key: value.to_dict() for key, value in self.heads.items()}
        payload["identity_valid"] = self.identity_valid
        payload["legal_terminal"] = self.legal_terminal
        payload["outcome_valid"] = self.outcome_valid
        payload["content_sha256"] = stable_sha256(payload)
        return payload


@dataclass(frozen=True)
class CoraPairEdge:
    root_id: str
    left_proposal_id: str
    right_proposal_id: str
    left_proposal_sha256: str
    right_proposal_sha256: str
    edge_kind: str
    pair_outcome_mask: bool
    schema_version: str = PAIR_INDEX_SCHEMA

    def __post_init__(self) -> None:
        if self.left_proposal_id == self.right_proposal_id:
            raise ValueError("cora_pair_same_proposal")
        if self.edge_kind not in {"nominal", "intervention_base"}:
            raise ValueError("cora_pair_kind")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoraRootRecord:
    dataset_id: str
    root_id: str
    split: str
    scenario: Mapping[str, Any]
    matrix_sha256: str
    config_sha256: str
    anchor_path: str
    anchor_sha256: str
    feature_paths: Mapping[str, str]
    feature_sha256: Mapping[str, str]
    proposals: tuple[CoraProposal, ...]
    branches: tuple[CoraBranchOutcome, ...]
    edges: tuple[CoraPairEdge, ...]
    vla_forward_count: int
    terminal_status: str
    missingness: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = ROOT_ANCHOR_SCHEMA

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.root_id:
            raise ValueError("cora_root_identity")
        if self.split == "reserved_formal":
            raise ValueError("cora_formal_root_record_forbidden")
        if self.vla_forward_count < 0:
            raise ValueError("cora_root_negative_forward_count")
        ids = [item.proposal_id for item in self.proposals]
        if len(ids) != len(set(ids)):
            raise ValueError("cora_root_duplicate_proposal")
        branch_by_id = {item.proposal_id: item for item in self.branches}
        proposal_by_id = {item.proposal_id: item for item in self.proposals}
        for edge in self.edges:
            if edge.root_id != self.root_id:
                raise ValueError("cora_root_edge_root_mismatch")
            if not edge.pair_outcome_mask:
                continue
            left = branch_by_id.get(edge.left_proposal_id)
            right = branch_by_id.get(edge.right_proposal_id)
            if left is None or right is None or not left.outcome_valid or not right.outcome_valid:
                raise ValueError("cora_root_masked_edge_invalid_branch")
            if edge.edge_kind == "nominal":
                left_proposal = proposal_by_id.get(edge.left_proposal_id)
                right_proposal = proposal_by_id.get(edge.right_proposal_id)
                if (
                    left_proposal is None
                    or right_proposal is None
                    or left_proposal.kind != "nominal"
                    or right_proposal.kind != "nominal"
                    or left_proposal.guard.get("verdict") not in {"PASS", "REVIEW"}
                    or right_proposal.guard.get("verdict") not in {"PASS", "REVIEW"}
                ):
                    raise ValueError("cora_root_masked_nominal_edge_ineligible")

    @property
    def nominal_pair_outcome_mask(self) -> bool:
        nominal = [edge for edge in self.edges if edge.edge_kind == "nominal"]
        return len(nominal) == 1 and nominal[0].pair_outcome_mask

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "root_id": self.root_id,
            "split": self.split,
            "scenario": dict(self.scenario),
            "matrix_sha256": self.matrix_sha256,
            "config_sha256": self.config_sha256,
            "anchor_path": self.anchor_path,
            "anchor_sha256": self.anchor_sha256,
            "feature_paths": dict(self.feature_paths),
            "feature_sha256": dict(self.feature_sha256),
            "proposals": [item.to_dict() for item in self.proposals],
            "branches": [item.to_dict() for item in self.branches],
            "edges": [item.to_dict() for item in self.edges],
            "vla_forward_count": self.vla_forward_count,
            "terminal_status": self.terminal_status,
            "missingness": [dict(item) for item in self.missingness],
            "nominal_pair_outcome_mask": self.nominal_pair_outcome_mask,
        }
        payload["content_sha256"] = stable_sha256(payload)
        return payload


__all__ = [
    "BRANCH_OUTCOME_SCHEMA",
    "FEATURE_SCHEMA",
    "PAIR_INDEX_SCHEMA",
    "PROPOSAL_SCHEMA",
    "ROOT_ANCHOR_SCHEMA",
    "CoraBranchOutcome",
    "CoraPairEdge",
    "CoraProposal",
    "CoraRootRecord",
    "OutcomeValue",
]
