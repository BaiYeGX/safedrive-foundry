"""R2 runner contracts: frozen registry gate, attempt isolation, idempotent read, ledger.

Pure offline helpers — no CARLA import. Fail-closed before live connect when possible.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from driving_vla.evaluation.paired_contract import (
    ContractError,
    compute_pair_id,
    content_hash,
)
from driving_vla.evaluation.scenario_registry import (
    REGISTRY_SCHEMA,
    REGISTRY_VERSION_DEFAULT,
    REQUIRED_SCENARIO_IDS,
    REQUIRED_SEEDS,
    ScenarioRegistryV1,
    load_scenario_registry,
)

PAIR_STATUS_COMPLETED = "COMPLETED"
PAIR_STATUS_FAILED = "FAILED"
PAIR_STATUS_RUNNING = "RUNNING"
PAIR_STATUS_INCOMPARABLE = "INCOMPARABLE"
PAIR_STATUS_COMPARABLE = "COMPARABLE"

RUN_SET_MANIFEST_SCHEMA = "safedrive.g4a.run_set_manifest.v1"
RUN_SET_REPORT_SCHEMA = "safedrive.g4a.run_set_report.v1"
RUN_SET_CHECKPOINT_SCHEMA = "safedrive.g4a.run_set_checkpoint.v1"

# Optional V2 identity fields.  They are included in the immutable content
# hash only when present, preserving validation of frozen longitudinal V1
# manifests while making every new Spatial K2 run fail-closed on head/config
# changes.
SPATIAL_RUN_IDENTITY_FIELDS = (
    "policy_type",
    "policy_model_id",
    "spatial_head_checkpoint_hash",
    "spatial_k2_config_hash",
)

# Pre-registered continue policies (cannot be chosen from outcomes mid-run).
CONTINUE_POLICY_CONTINUE_ALL = "continue_all"
CONTINUE_POLICY_STOP_ON_FAIL = "stop_on_fail"
ALLOWED_CONTINUE_POLICIES = frozenset(
    {CONTINUE_POLICY_CONTINUE_ALL, CONTINUE_POLICY_STOP_ON_FAIL}
)

# Pre-registered retry policy for R2-D (fixed).
RETRY_POLICY_NO_AUTO_RETRY = "no_auto_retry"
ALLOWED_RETRY_POLICIES = frozenset({RETRY_POLICY_NO_AUTO_RETRY})

DEFAULT_FROZEN_MANIFEST_REL = Path(
    "docs/runtime-evidence/r2-g4a-paired-pilot/registry/registry_manifest.json"
)

_ATTEMPT_RE = re.compile(r"^attempt_(\d+)$")


class RunnerContractError(ContractError):
    """Fail-closed runner preflight / evidence layout violation."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RunnerContractError(f"missing json: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerContractError(f"invalid json {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RunnerContractError(f"json root must be object: {path}")
    return raw


def resolve_frozen_manifest_path(
    registry_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> Path:
    """Locate frozen registry_manifest.json next to evidence or explicit path."""
    if manifest_path is not None:
        p = Path(manifest_path)
        if not p.is_file():
            raise RunnerContractError(f"frozen registry manifest not found: {p}")
        return p

    reg = Path(registry_path)
    candidates: list[Path] = [
        reg.parent / "registry_manifest.json",
        reg.with_name("registry_manifest.json"),
    ]
    if repo_root is not None:
        candidates.append(Path(repo_root) / DEFAULT_FROZEN_MANIFEST_REL)
    # common layout: config/g4a/toml → repo root / docs/runtime-evidence/...
    try:
        candidates.append(
            reg.resolve().parents[3]
            / "docs"
            / "runtime-evidence"
            / "r2-g4a-paired-pilot"
            / "registry"
            / "registry_manifest.json"
        )
    except IndexError:
        pass
    for c in candidates:
        if c.is_file():
            return c
    raise RunnerContractError(
        "frozen registry_manifest.json not found; pass --registry-manifest "
        f"(searched: {[str(c) for c in candidates[:4]]}...)"
    )


def validate_frozen_registry_manifest(
    registry: ScenarioRegistryV1,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail-closed checks against a freeze-time manifest (R2-B)."""
    errors: list[str] = []
    if not bool(manifest.get("frozen", False)):
        errors.append("manifest.frozen must be true")
    schema = str(manifest.get("schema_version", ""))
    if schema != REGISTRY_SCHEMA:
        errors.append(f"schema_version mismatch: {schema!r} != {REGISTRY_SCHEMA!r}")
    if schema != registry.schema_version:
        errors.append(
            f"manifest schema vs loaded registry: {schema!r} != {registry.schema_version!r}"
        )
    reg_hash = registry.compute_registry_sha256()
    man_hash = str(manifest.get("registry_sha256", ""))
    if not man_hash:
        errors.append("manifest missing registry_sha256")
    elif man_hash != reg_hash:
        errors.append(
            f"registry_sha256 mismatch: manifest={man_hash[:16]}… live={reg_hash[:16]}…"
        )
    n_pairs = int(manifest.get("n_pairs", -1))
    if n_pairs != 12 or len(registry.fixtures) != 12:
        errors.append(
            f"n_pairs must be 12 (manifest={n_pairs}, loaded={len(registry.fixtures)})"
        )
    n_sc = int(manifest.get("n_scenarios", -1))
    if n_sc != 6 or len(registry.scenario_ids()) != 6:
        errors.append(
            f"n_scenarios must be 6 (manifest={n_sc}, loaded={len(registry.scenario_ids())})"
        )
    man_pairs = manifest.get("pairs") or []
    if not isinstance(man_pairs, list) or len(man_pairs) != 12:
        errors.append("manifest.pairs must be a list of 12 entries")
    else:
        loaded = {(f.scenario_id, f.seed_id) for f in registry.fixtures}
        man_set = set()
        for i, row in enumerate(man_pairs):
            if not isinstance(row, Mapping):
                errors.append(f"manifest.pairs[{i}] not an object")
                continue
            key = (str(row.get("scenario_id", "")), str(row.get("seed_id", "")))
            man_set.add(key)
        if man_set != loaded:
            errors.append(
                f"manifest.pairs set mismatch loaded fixtures "
                f"(only_manifest={sorted(man_set - loaded)[:3]} "
                f"only_loaded={sorted(loaded - man_set)[:3]})"
            )
    # V1 retains the historical exact IDs. Post-v1 registries are already
    # checked against their own immutable manifest set above.
    if registry.registry_version == REGISTRY_VERSION_DEFAULT:
        for sid in REQUIRED_SCENARIO_IDS:
            for seed in REQUIRED_SEEDS:
                if (sid, seed) not in {
                    (f.scenario_id, f.seed_id) for f in registry.fixtures
                }:
                    errors.append(f"missing fixture {sid}/{seed}")
    if errors:
        raise RunnerContractError(
            "frozen registry validation failed: " + "; ".join(errors)
        )
    return {
        "ok": True,
        "registry_sha256": reg_hash,
        "schema_version": REGISTRY_SCHEMA,
        "n_pairs": 12,
        "n_scenarios": 6,
        "frozen": True,
    }


def require_frozen_registry(
    registry_path: Path | str,
    *,
    manifest_path: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> tuple[ScenarioRegistryV1, dict[str, Any], Path]:
    """Load registry + validate frozen manifest. No CARLA."""
    reg_path = Path(registry_path)
    if not reg_path.is_file():
        raise RunnerContractError(f"registry file not found: {reg_path}")
    man_path = resolve_frozen_manifest_path(
        reg_path, manifest_path=manifest_path, repo_root=repo_root
    )
    registry = load_scenario_registry(reg_path)
    manifest = _read_json(man_path)
    audit = validate_frozen_registry_manifest(registry, manifest)
    audit["manifest_path"] = str(man_path.as_posix())
    audit["registry_path"] = str(reg_path.as_posix())
    return registry, audit, man_path


@dataclass(frozen=True)
class ExpectedPairHashes:
    """Inputs that must match a completed pair for idempotent re-read."""

    pair_id: str
    scenario_id: str
    seed_id: str
    registry_sha256: str
    model_retimer_hash: str
    executor_config_hash: str
    # optional: if set, must match stored artifact hash
    artifact_content_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pair_id": self.pair_id,
            "scenario_id": self.scenario_id,
            "seed_id": self.seed_id,
            "registry_sha256": self.registry_sha256,
            "model_retimer_hash": self.model_retimer_hash,
            "executor_config_hash": self.executor_config_hash,
        }
        if self.artifact_content_hash is not None:
            d["artifact_content_hash"] = self.artifact_content_hash
        return d


def pair_manifest_matches_expected(
    manifest: Mapping[str, Any],
    expected: ExpectedPairHashes,
) -> tuple[bool, tuple[str, ...]]:
    """Return (ok, mismatch_reasons). Requires status COMPLETED for full match."""
    reasons: list[str] = []
    status = str(manifest.get("status", PAIR_STATUS_COMPLETED))
    # legacy manifests without status treated as COMPLETED if comparable key present
    if "status" not in manifest and "comparable" in manifest:
        status = PAIR_STATUS_COMPLETED
    if status != PAIR_STATUS_COMPLETED:
        reasons.append(f"status={status}")
    checks = {
        "pair_id": expected.pair_id,
        "scenario_id": expected.scenario_id,
        "seed_id": expected.seed_id,
        "registry_sha256": expected.registry_sha256,
        "model_retimer_hash": expected.model_retimer_hash,
        "executor_config_hash": expected.executor_config_hash,
    }
    for key, exp in checks.items():
        got = str(manifest.get(key, ""))
        if got != exp:
            reasons.append(f"{key}:{got[:16] if got else 'missing'}!={exp[:16]}")
    if expected.artifact_content_hash is not None:
        got_a = str(manifest.get("artifact_content_hash", ""))
        if got_a != expected.artifact_content_hash:
            reasons.append("artifact_content_hash_mismatch")
    return (len(reasons) == 0, tuple(reasons))


def legacy_top_level_evidence_occupied(pair_root: Path) -> bool:
    """True if R2-C-style top-level pair dir holds Evidence (must not overwrite)."""
    if not pair_root.is_dir():
        return False
    if (pair_root / "pair_manifest.json").is_file():
        return True
    if (pair_root / "pair_oracle.json").is_file():
        return True
    if (pair_root / "pair_comparability.json").is_file():
        return True
    if (pair_root / "anchor").is_dir():
        return True
    if (pair_root / "branch-0").is_dir() or (pair_root / "branch-1").is_dir():
        return True
    return False


def attempt_slot_occupied(pair_root: Path, attempt_id: int) -> bool:
    """Read-only: whether attempt_id already holds Evidence (any completeness)."""
    aid = int(attempt_id)
    modern = pair_root / f"attempt_{aid}"
    if modern.exists():
        return True
    if aid == 0:
        # modern attempt_0 not present — legacy top-level counts as attempt 0
        if (pair_root / "attempt_0").exists():
            return True
        if legacy_top_level_evidence_occupied(pair_root):
            return True
    return False


def first_unoccupied_attempt_id(pair_root: Path, *, max_scan: int = 256) -> int:
    """Freeze-time planner: first free attempt id (read-only; never mutates Evidence)."""
    for n in range(max_scan + 1):
        if not attempt_slot_occupied(pair_root, n):
            return n
    raise RunnerContractError(
        f"no free attempt_id under {pair_root} within 0..{max_scan}"
    )


def list_attempt_dirs(pair_root: Path) -> list[tuple[int, Path]]:
    """Return sorted (attempt_id, path). Legacy top-level Evidence → attempt 0 virtual."""
    found: list[tuple[int, Path]] = []
    if not pair_root.is_dir():
        return found
    for child in pair_root.iterdir():
        if not child.is_dir():
            continue
        m = _ATTEMPT_RE.match(child.name)
        if m:
            found.append((int(m.group(1)), child))
    # legacy top-level as virtual attempt 0 if no real attempt_0 dir
    if legacy_top_level_evidence_occupied(pair_root) and not any(
        i == 0 for i, _ in found
    ):
        if not (pair_root / "attempt_0").is_dir():
            found.append((0, pair_root))
    found.sort(key=lambda x: x[0])
    return found


def next_attempt_id(pair_root: Path) -> int:
    """Next free id after all occupied slots (same as first_unoccupied)."""
    return first_unoccupied_attempt_id(pair_root)


def load_attempt_manifest(attempt_dir: Path) -> dict[str, Any] | None:
    path = attempt_dir / "pair_manifest.json"
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except RunnerContractError:
        return None


@dataclass(frozen=True)
class AttemptPlan:
    mode: str  # "idempotent_read" | "new_run"
    pair_root: Path
    attempt_id: int
    attempt_dir: Path
    existing_manifest: dict[str, Any] | None = None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pair_root": str(self.pair_root.as_posix()),
            "attempt_id": self.attempt_id,
            "attempt_dir": str(self.attempt_dir.as_posix()),
            "reasons": list(self.reasons),
            "has_existing_manifest": self.existing_manifest is not None,
        }


def plan_pair_attempt(
    evidence_root: Path | str,
    expected: ExpectedPairHashes,
) -> AttemptPlan:
    """Idempotent read if completed+hashes match; else allocate new attempt dir."""
    root = Path(evidence_root)
    pair_root = root / expected.pair_id
    for attempt_id, attempt_dir in list_attempt_dirs(pair_root):
        man = load_attempt_manifest(attempt_dir)
        if man is None:
            continue
        ok, reasons = pair_manifest_matches_expected(man, expected)
        if ok:
            return AttemptPlan(
                mode="idempotent_read",
                pair_root=pair_root,
                attempt_id=attempt_id,
                attempt_dir=attempt_dir,
                existing_manifest=man,
                reasons=(),
            )
    # no matching completed → new attempt (never overwrite old)
    aid = next_attempt_id(pair_root)
    # Always use attempt_N directory for new runs (even attempt 0)
    adir = pair_root / f"attempt_{aid}"
    # If legacy top-level is the only attempt_0, next is 1 under attempt_1
    return AttemptPlan(
        mode="new_run",
        pair_root=pair_root,
        attempt_id=aid,
        attempt_dir=adir,
        existing_manifest=None,
        reasons=("no_matching_completed_attempt",),
    )


def ledger_path_for_evidence_root(evidence_root: Path | str) -> Path:
    root = Path(evidence_root)
    if root.name == "pairs":
        return root.parent / "paired_outcomes.jsonl"
    return root / "paired_outcomes.jsonl"


def ledger_has_entry(
    ledger_path: Path,
    *,
    pair_id: str,
    attempt_id: int,
    artifact_content_hash: str | None = None,
) -> bool:
    """True if ledger already contains this pair attempt (optionally hash)."""
    if not ledger_path.is_file():
        return False
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if str(row.get("pair_id", "")) != pair_id:
            continue
        if int(row.get("attempt_id", -1)) != int(attempt_id):
            continue
        if artifact_content_hash is not None:
            if str(row.get("artifact_content_hash", "")) != artifact_content_hash:
                continue
        return True
    return False


def append_ledger_if_new(
    ledger_path: Path,
    row: Mapping[str, Any],
) -> bool:
    """Append one JSONL row if pair_id+attempt_id(+artifact) not already present.

    Returns True if appended, False if skipped (idempotent).
    """
    pair_id = str(row.get("pair_id", ""))
    attempt_id = int(row.get("attempt_id", 0))
    art = row.get("artifact_content_hash")
    art_s = str(art) if art is not None else None
    if not pair_id:
        raise RunnerContractError("ledger row missing pair_id")
    if ledger_has_entry(
        ledger_path,
        pair_id=pair_id,
        attempt_id=attempt_id,
        artifact_content_hash=art_s,
    ):
        return False
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
    return True


def build_completed_manifest(
    *,
    pair_id: str,
    scenario_id: str,
    seed_id: str,
    family: str,
    registry_sha256: str,
    model_retimer_hash: str,
    executor_config_hash: str,
    artifact_content_hash: str,
    attempt_id: int,
    branch_order: Sequence[int],
    forward_count_total: int,
    comparable: bool,
    comparability: Mapping[str, Any],
    oracle: Mapping[str, Any],
    anchor: Mapping[str, Any],
    branch_0: Mapping[str, Any],
    branch_1: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    man: dict[str, Any] = {
        "status": PAIR_STATUS_COMPLETED,
        "pair_id": pair_id,
        "scenario_id": scenario_id,
        "seed_id": seed_id,
        "family": family,
        "registry_sha256": registry_sha256,
        "model_retimer_hash": model_retimer_hash,
        "executor_config_hash": executor_config_hash,
        "artifact_content_hash": artifact_content_hash,
        "attempt_id": int(attempt_id),
        "branch_order": list(branch_order),
        "forward_count_total": int(forward_count_total),
        "comparable": bool(comparable),
        "comparability": dict(comparability),
        "oracle": dict(oracle),
        "anchor": dict(anchor),
        "branch_0": dict(branch_0),
        "branch_1": dict(branch_1),
    }
    if extra:
        man.update(dict(extra))
    return man


def build_failed_manifest(
    *,
    pair_id: str,
    scenario_id: str,
    seed_id: str,
    registry_sha256: str,
    model_retimer_hash: str,
    executor_config_hash: str,
    attempt_id: int,
    error: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    man: dict[str, Any] = {
        "status": PAIR_STATUS_FAILED,
        "pair_id": pair_id,
        "scenario_id": scenario_id,
        "seed_id": seed_id,
        "registry_sha256": registry_sha256,
        "model_retimer_hash": model_retimer_hash,
        "executor_config_hash": executor_config_hash,
        "attempt_id": int(attempt_id),
        "error": error,
        "comparable": False,
    }
    if extra:
        man.update(dict(extra))
    return man


def finalize_branch_failure_codes(
    *,
    cleanup_ok: bool,
    extra_codes: Sequence[str] = (),
) -> tuple[str, ...]:
    """Merge failure codes; CLEANUP_FAILURE when cleanup did not succeed."""
    codes: list[str] = []
    for c in extra_codes:
        if c and c not in codes:
            codes.append(str(c))
    if not cleanup_ok and "CLEANUP_FAILURE" not in codes:
        codes.append("CLEANUP_FAILURE")
    return tuple(codes)


def content_fingerprint(obj: Mapping[str, Any]) -> str:
    """Stable short hash for audit."""
    return content_hash(dict(obj), nibble=16)


def counterbalance_branch_order(seed_id: str) -> tuple[int, int]:
    """R2-D: seed_a → 0 then 1; seed_b → 1 then 0."""
    sid = str(seed_id).strip()
    if sid == "seed_a":
        return (0, 1)
    if sid == "seed_b":
        return (1, 0)
    raise RunnerContractError(
        f"counterbalance_branch_order: seed_id must be seed_a or seed_b, got {sid!r}"
    )


def plan_run_set_pairs(registry: ScenarioRegistryV1) -> list[dict[str, Any]]:
    """Ordered 12-pair plan from frozen registry fixtures (registry sort order)."""
    if len(registry.fixtures) != 12:
        raise RunnerContractError(
            f"run-set requires 12 fixtures, got {len(registry.fixtures)}"
        )
    plan: list[dict[str, Any]] = []
    for index, fx in enumerate(registry.fixtures):
        order = counterbalance_branch_order(fx.seed_id)
        plan.append(
            {
                "index": index,
                "scenario_id": fx.scenario_id,
                "seed_id": fx.seed_id,
                "family": fx.family,
                "branch_order": list(order),
                "requested_initial_state_hash": fx.requested_initial_state_hash(),
            }
        )
    # sanity: 6 scenarios × 2 seeds, counterbalance alternates
    if len(plan) != 12:
        raise RunnerContractError("plan_run_set_pairs internal length error")
    return plan


def summarize_run_set_results(
    pair_results: Sequence[Mapping[str, Any]],
    *,
    n_planned: int | None = None,
) -> dict[str, Any]:
    """Counts for run-set report (no CARLA)."""
    n_exec = len(pair_results)
    planned = int(n_planned) if n_planned is not None else n_exec
    n_ok = 0
    n_fail = 0
    n_idempotent = 0
    n_comparable = 0
    n_incomparable = 0
    for row in pair_results:
        status = str(row.get("status", ""))
        if status == "FAILED" or row.get("error"):
            n_fail += 1
            continue
        n_ok += 1
        if bool(row.get("idempotent_read")):
            n_idempotent += 1
        if "comparable" in row:
            if bool(row.get("comparable")):
                n_comparable += 1
            else:
                n_incomparable += 1
    return {
        "n_planned": planned,
        "n_executed": n_exec,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_idempotent": n_idempotent,
        "n_completed": n_ok,
        "n_comparable": n_comparable,
        "n_incomparable": n_incomparable,
        "comparable_rate": (n_comparable / planned) if planned else 0.0,
    }


def run_set_exit_code(summary: Mapping[str, Any], *, min_comparable: int = 10) -> int:
    """0 = all ok and enough comparable; 4 = completed but pilot gate fail; 1 = hard fail."""
    if int(summary.get("n_fail", 0)) > 0:
        return 1
    if int(summary.get("n_ok", 0)) < int(summary.get("n_planned", 0)):
        return 1
    if int(summary.get("n_comparable", 0)) < int(min_comparable):
        return 4
    return 0


def normalize_continue_policy(policy: str | None) -> str:
    p = str(policy or CONTINUE_POLICY_CONTINUE_ALL).strip()
    if p not in ALLOWED_CONTINUE_POLICIES:
        raise RunnerContractError(
            f"continue_policy must be one of {sorted(ALLOWED_CONTINUE_POLICIES)}, got {p!r}"
        )
    return p


def normalize_retry_policy(policy: str | None) -> str:
    """R2-D freezes no_auto_retry; other values fail-closed."""
    p = str(policy or RETRY_POLICY_NO_AUTO_RETRY).strip()
    if p not in ALLOWED_RETRY_POLICIES:
        raise RunnerContractError(
            f"retry_policy must be one of {sorted(ALLOWED_RETRY_POLICIES)}, got {p!r}"
        )
    return p


def planned_attempt_dir_rel(pair_id: str, attempt_id: int) -> str:
    """Evidence-relative attempt path (portable; used in content hash)."""
    return f"{pair_id}/attempt_{int(attempt_id)}"


def run_set_manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical fields for content hash — no machine-absolute paths."""
    pairs_out: list[dict[str, Any]] = []
    for p in manifest.get("pairs") or []:
        if not isinstance(p, Mapping):
            raise RunnerContractError("manifest.pairs entry must be object")
        aid = int(p["planned_attempt_id"])
        pid = str(p["pair_id"])
        pairs_out.append(
            {
                "index": int(p["index"]),
                "scenario_id": str(p["scenario_id"]),
                "seed_id": str(p["seed_id"]),
                "family": str(p.get("family", "")),
                "pair_id": pid,
                "branch_order": list(p["branch_order"]),
                "planned_attempt_id": aid,
                # planned_mode is diagnostic (depends on disk) — excluded from hash
                "planned_attempt_dir_rel": str(
                    p.get("planned_attempt_dir_rel") or planned_attempt_dir_rel(pid, aid)
                ),
                "requested_initial_state_hash": str(p["requested_initial_state_hash"]),
            }
        )
    payload = {
        "schema_version": str(manifest.get("schema_version")),
        "immutable": bool(manifest.get("immutable")),
        "frozen": bool(manifest.get("frozen")),
        "registry_schema_version": str(manifest.get("registry_schema_version")),
        "registry_sha256": str(manifest.get("registry_sha256")),
        "model_retimer_hash": str(manifest.get("model_retimer_hash")),
        "model_checkpoint_hash": str(manifest.get("model_checkpoint_hash", "")),
        "retimer_hash": str(manifest.get("retimer_hash", "")),
        "executor_config_hash": str(manifest.get("executor_config_hash")),
        "continue_policy": str(manifest.get("continue_policy")),
        "retry_policy": str(manifest.get("retry_policy", RETRY_POLICY_NO_AUTO_RETRY)),
        "n_pairs": int(manifest.get("n_pairs", 0)),
        "pairs": pairs_out,
    }
    for key in SPATIAL_RUN_IDENTITY_FIELDS:
        if key in manifest:
            payload[key] = str(manifest.get(key, ""))
    return payload


def compute_run_set_manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    return content_hash(run_set_manifest_hash_payload(manifest), nibble=64)


def build_run_set_manifest(
    *,
    registry: ScenarioRegistryV1,
    freeze_audit: Mapping[str, Any],
    pairs_root: Path | str,
    model_retimer_hash: str,
    executor_config_hash: str,
    continue_policy: str = CONTINUE_POLICY_CONTINUE_ALL,
    retry_policy: str = RETRY_POLICY_NO_AUTO_RETRY,
    model_checkpoint_hash: str = "",
    retimer_hash: str = "",
    registry_path: str = "",
    registry_manifest_path: str = "",
    policy_type: str | None = None,
    policy_model_id: str | None = None,
    spatial_head_checkpoint_hash: str | None = None,
    spatial_k2_config_hash: str | None = None,
) -> dict[str, Any]:
    """Immutable run-set plan frozen BEFORE any pair outcome is observed.

    For each pair, **read-only** scan existing Evidence and freeze
    ``planned_attempt_id`` to the first unoccupied id (legacy top-level and
    attempt_N dirs count as occupied). After freeze, no_auto_retry forbids
    further increment. Never moves/overwrites existing Evidence.
    """
    policy = normalize_continue_policy(continue_policy)
    retry = normalize_retry_policy(retry_policy)
    reg_hash = str(freeze_audit.get("registry_sha256") or registry.compute_registry_sha256())
    if not freeze_audit.get("frozen", True):
        raise RunnerContractError("run_set_manifest requires frozen registry audit")
    base_plan = plan_run_set_pairs(registry)
    pairs_root_p = Path(pairs_root)
    slots: list[dict[str, Any]] = []
    for item in base_plan:
        pair_id = compute_pair_id(
            scenario_registry_hash=reg_hash,
            scenario_id=str(item["scenario_id"]),
            seed_id=str(item["seed_id"]),
            model_checkpoint_config_retimer_hash=model_retimer_hash,
            executor_config_hash=executor_config_hash,
        )
        pair_root = pairs_root_p / pair_id
        # Read-only: first free attempt id at freeze time (enters content hash).
        planned_attempt_id = first_unoccupied_attempt_id(pair_root)
        rel = planned_attempt_dir_rel(pair_id, planned_attempt_id)
        abs_dir = pairs_root_p / rel
        # Diagnostic planned_mode only (excluded from content hash).
        existing = load_attempt_manifest(abs_dir)
        if existing is None and planned_attempt_id == 0:
            if (pair_root / "pair_manifest.json").is_file():
                existing = load_attempt_manifest(pair_root)
        if existing is not None and str(existing.get("status")) == PAIR_STATUS_COMPLETED:
            planned_mode = "idempotent_read"
        elif existing is not None and str(existing.get("status")) == PAIR_STATUS_FAILED:
            planned_mode = "retain_failed"
        else:
            planned_mode = "new_run"
        slots.append(
            {
                "index": int(item["index"]),
                "scenario_id": str(item["scenario_id"]),
                "seed_id": str(item["seed_id"]),
                "family": str(item["family"]),
                "pair_id": pair_id,
                "branch_order": list(item["branch_order"]),
                "planned_attempt_id": planned_attempt_id,
                "planned_mode": planned_mode,
                "planned_attempt_dir_rel": rel,
                # convenience only — excluded from content hash
                "planned_attempt_dir": str(abs_dir.as_posix()),
                "requested_initial_state_hash": str(item["requested_initial_state_hash"]),
                "occupied_attempts_at_freeze": [
                    i for i, _ in list_attempt_dirs(pair_root)
                ],
            }
        )
    if len(slots) != 12:
        raise RunnerContractError(f"run_set_manifest must have 12 slots, got {len(slots)}")
    payload: dict[str, Any] = {
        "schema_version": RUN_SET_MANIFEST_SCHEMA,
        "immutable": True,
        "frozen": True,
        "registry_schema_version": REGISTRY_SCHEMA,
        "registry_sha256": reg_hash,
        # machine-local convenience paths (excluded from content hash)
        "registry_path": registry_path,
        "registry_manifest_path": registry_manifest_path,
        "model_retimer_hash": model_retimer_hash,
        "model_checkpoint_hash": model_checkpoint_hash,
        "retimer_hash": retimer_hash,
        "executor_config_hash": executor_config_hash,
        "continue_policy": policy,
        "retry_policy": retry,
        "n_pairs": 12,
        "pairs": slots,
    }
    spatial_identity = {
        "policy_type": policy_type,
        "policy_model_id": policy_model_id,
        "spatial_head_checkpoint_hash": spatial_head_checkpoint_hash,
        "spatial_k2_config_hash": spatial_k2_config_hash,
    }
    for key, value in spatial_identity.items():
        if value is not None:
            if not str(value):
                raise RunnerContractError(f"spatial run identity field {key} is empty")
            payload[key] = str(value)
    payload["manifest_content_hash"] = compute_run_set_manifest_content_hash(payload)
    return payload


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, obj: Any) -> None:
    """Atomic replace via temp file + os.replace (for checkpoint/report updates)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_json_exclusive_create(path: Path, obj: Any) -> None:
    """Create path only if it does not exist (O_CREAT|O_EXCL). Fail if exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(path), flags, 0o644)
    except FileExistsError as exc:
        raise RunnerContractError(
            f"refusing to overwrite existing file (exclusive create): {path}"
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        try:
            path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except TypeError:
            if path.is_file():
                path.unlink()
        except OSError:
            pass
        raise


def validate_run_set_manifest_self_integrity(existing: Mapping[str, Any]) -> None:
    """Validate immutable schema, self content hash, and 12 frozen slots.

    Does not rescan Evidence or recompute planned_attempt_id values.
    """
    if not bool(existing.get("immutable", False)):
        raise RunnerContractError("existing run_set_manifest.immutable must be true")
    if str(existing.get("schema_version")) != RUN_SET_MANIFEST_SCHEMA:
        raise RunnerContractError(
            f"existing schema_version mismatch: {existing.get('schema_version')}"
        )
    got_pairs = existing.get("pairs") or []
    if not isinstance(got_pairs, list) or len(got_pairs) != 12:
        raise RunnerContractError("run_set_manifest must have exactly 12 slots")
    if int(existing.get("n_pairs", 0)) != 12:
        raise RunnerContractError(
            f"run_set_manifest.n_pairs must be 12, got {existing.get('n_pairs')!r}"
        )
    got_hash = str(existing.get("manifest_content_hash") or "")
    recomputed = compute_run_set_manifest_content_hash(existing)
    if not got_hash or got_hash != recomputed:
        raise RunnerContractError(
            "existing run_set_manifest content hash does not match payload "
            f"(stored={got_hash[:16]}… recomputed={recomputed[:16]}…)"
        )
    for i, slot in enumerate(got_pairs):
        if not isinstance(slot, Mapping):
            raise RunnerContractError(f"run_set_manifest.pairs[{i}] must be object")
        if int(slot.get("index", -1)) != i:
            raise RunnerContractError(
                f"run_set_manifest.pairs[{i}].index must be {i}, got {slot.get('index')!r}"
            )
        for key in (
            "scenario_id",
            "seed_id",
            "pair_id",
            "planned_attempt_id",
            "branch_order",
            "requested_initial_state_hash",
        ):
            if key not in slot:
                raise RunnerContractError(f"run_set_manifest.pairs[{i}] missing {key}")


def validate_run_set_manifest_stable_identity(
    existing: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Check freeze-time stable identity fields only (never planned slots).

    Stable fields: registry/model/retimer/executor/continue/retry (+ schema).
    """
    for key in (
        "registry_sha256",
        "model_retimer_hash",
        "model_checkpoint_hash",
        "retimer_hash",
        "executor_config_hash",
        "continue_policy",
        "retry_policy",
        "n_pairs",
        "registry_schema_version",
        *SPATIAL_RUN_IDENTITY_FIELDS,
    ):
        if key not in identity:
            continue
        if str(existing.get(key, "")) != str(identity.get(key, "")):
            raise RunnerContractError(
                f"run_set_manifest field mismatch on {key}: "
                f"existing={existing.get(key)!r} expected={identity.get(key)!r}"
            )


def validate_run_set_manifest_identity(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Full identity check including frozen slots (create-time / strict compare).

    Prefer :func:`validate_run_set_manifest_self_integrity` +
    :func:`validate_run_set_manifest_stable_identity` on resume so planned
    attempt IDs are never recomputed from Evidence.
    """
    validate_run_set_manifest_self_integrity(existing)
    validate_run_set_manifest_stable_identity(existing, expected)
    exp_pairs = expected.get("pairs") or []
    got_pairs = existing.get("pairs") or []
    if not isinstance(exp_pairs, list) or len(exp_pairs) != 12:
        raise RunnerContractError("expected run_set_manifest must have exactly 12 slots")
    for i, (e, g) in enumerate(zip(exp_pairs, got_pairs)):
        if int(e["index"]) != int(g["index"]):
            raise RunnerContractError(f"slot[{i}] index mismatch")
        if str(e["scenario_id"]) != str(g["scenario_id"]) or str(e["seed_id"]) != str(g["seed_id"]):
            raise RunnerContractError(f"slot[{i}] scenario/seed mismatch")
        if str(e.get("family", "")) != str(g.get("family", "")):
            raise RunnerContractError(f"slot[{i}] family mismatch")
        if str(e["pair_id"]) != str(g["pair_id"]):
            raise RunnerContractError(f"slot[{i}] pair_id mismatch")
        if int(e["planned_attempt_id"]) != int(g["planned_attempt_id"]):
            raise RunnerContractError(f"slot[{i}] planned_attempt_id mismatch")
        if list(e["branch_order"]) != list(g["branch_order"]):
            raise RunnerContractError(f"slot[{i}] branch_order mismatch")
        if str(e.get("requested_initial_state_hash")) != str(g.get("requested_initial_state_hash")):
            raise RunnerContractError(f"slot[{i}] requested_initial_state_hash mismatch")
        exp_rel = str(
            e.get("planned_attempt_dir_rel")
            or planned_attempt_dir_rel(str(e["pair_id"]), int(e["planned_attempt_id"]))
        )
        got_rel = str(
            g.get("planned_attempt_dir_rel")
            or planned_attempt_dir_rel(str(g["pair_id"]), int(g["planned_attempt_id"]))
        )
        if exp_rel != got_rel:
            raise RunnerContractError(f"slot[{i}] planned_attempt_dir_rel mismatch")


def ensure_run_set_manifest(
    path: Path | str,
    expected: Mapping[str, Any] | None = None,
    *,
    build_fn: Callable[[], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Exclusive create or validate-and-reuse existing immutable manifest.

    When ``path`` already exists:
      - read existing bytes / JSON;
      - validate self content hash, schema, immutable, 12 slots;
      - validate stable identity fields against ``expected`` when provided
        (registry/model/retimer/executor/continue/retry);
      - **reuse existing 12 slots as-is** — never rescan Evidence or recompute
        planned_attempt_id.

    When ``path`` is missing:
      - build via ``build_fn()`` if given, else require ``expected``;
      - exclusive-create only.

    Returns (manifest, mode) where mode is 'created' | 'reused'.
    """
    path = Path(path)
    if path.is_file():
        existing = _read_json(path)
        validate_run_set_manifest_self_integrity(existing)
        if expected is not None:
            validate_run_set_manifest_stable_identity(existing, expected)
        return existing, "reused"

    if expected is None:
        if build_fn is None:
            raise RunnerContractError(
                "ensure_run_set_manifest: need expected or build_fn when file is missing"
            )
        expected = dict(build_fn())
    else:
        expected = dict(expected)
    if "manifest_content_hash" not in expected:
        expected["manifest_content_hash"] = compute_run_set_manifest_content_hash(expected)
    write_json_exclusive_create(path, expected)
    return dict(expected), "created"


FAILURE_INTERRUPTED_PARTIAL_ATTEMPT = "INTERRUPTED_PARTIAL_ATTEMPT"


def _seal_interrupted_partial_attempt(
    *,
    adir: Path,
    man_path: Path,
    pair_id: str,
    planned_attempt_id: int,
    expected: ExpectedPairHashes,
    phase: str,
    corrupt_backup_rel: str | None = None,
) -> dict[str, Any]:
    """Atomically seal planned attempt as FAILED/INTERRUPTED_PARTIAL_ATTEMPT.

    Retains any existing dir contents. Does not mkdir a *new* attempt or bump id.
    """
    if not adir.exists():
        raise RunnerContractError(
            f"cannot seal missing attempt path as partial: {adir}"
        )
    if not adir.is_dir():
        raise RunnerContractError(
            f"planned attempt path exists but is not a directory: {adir}"
        )
    extra: dict[str, Any] = {
        "failure_code": FAILURE_INTERRUPTED_PARTIAL_ATTEMPT,
        "comparable": False,
        "phase": phase,
        "note": "sealed partial attempt; evidence retained; no re-run",
    }
    if corrupt_backup_rel:
        extra["corrupt_manifest_backup"] = corrupt_backup_rel
    fail_man = build_failed_manifest(
        pair_id=pair_id,
        scenario_id=expected.scenario_id,
        seed_id=expected.seed_id,
        registry_sha256=expected.registry_sha256,
        model_retimer_hash=expected.model_retimer_hash,
        executor_config_hash=expected.executor_config_hash,
        attempt_id=int(planned_attempt_id),
        error=FAILURE_INTERRUPTED_PARTIAL_ATTEMPT,
        extra=extra,
    )
    write_json_atomic(man_path, fail_man)
    return {
        "action": "retain_failed",
        "attempt_id": int(planned_attempt_id),
        "attempt_dir": adir,
        "existing_manifest": fail_man,
        "sealed_partial": True,
    }


def backup_corrupt_pair_manifest(man_path: Path) -> str:
    """Preserve original corrupt manifest bytes as a read-only audit copy.

    Writes ``pair_manifest.corrupt.<sha256>.json`` beside the original and
    returns the backup filename (relative name). Never overwrites an existing
    backup with the same content hash.
    """
    man_path = Path(man_path)
    raw = man_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    backup_name = f"pair_manifest.corrupt.{sha}.json"
    backup_path = man_path.parent / backup_name
    if not backup_path.exists():
        backup_path.write_bytes(raw)
        try:
            # Best-effort read-only audit copy (Windows/POSIX).
            os.chmod(backup_path, 0o444)
        except OSError:
            pass
    return backup_name


def resolve_no_auto_retry_action(
    pairs_root: Path | str,
    *,
    pair_id: str,
    planned_attempt_id: int,
    expected: ExpectedPairHashes,
) -> dict[str, Any]:
    """Decide action for a frozen planned attempt under no_auto_retry.

    - planned path absent → new_run (only this planned_attempt_id)
    - path exists but no valid manifest (empty or non-empty dir) → atomically
      seal FAILED/INTERRUPTED_PARTIAL_ATTEMPT; do not mkdir new_run / next attempt
    - valid COMPLETED matching hashes → idempotent_read
    - valid FAILED → retain_failed (no auto-increment)
    - corrupt manifest → backup original bytes, then seal FAILED
    """
    adir = attempt_dir_for(pairs_root, pair_id, planned_attempt_id)
    man_path = adir / "pair_manifest.json"
    if not man_path.is_file():
        # Path missing entirely → only then open a new run at planned id.
        if not adir.exists():
            return {
                "action": "new_run",
                "attempt_id": int(planned_attempt_id),
                "attempt_dir": adir,
                "existing_manifest": None,
            }
        # Path exists (empty or with residual files) without a valid manifest:
        # seal as interrupted partial; never promote to a new attempt id.
        return _seal_interrupted_partial_attempt(
            adir=adir,
            man_path=man_path,
            pair_id=pair_id,
            planned_attempt_id=planned_attempt_id,
            expected=expected,
            phase="interrupted_partial_attempt",
        )
    try:
        man = _read_json(man_path)
    except RunnerContractError:
        # Corrupt manifest: preserve original bytes, then seal overwrite.
        backup_name = backup_corrupt_pair_manifest(man_path)
        return _seal_interrupted_partial_attempt(
            adir=adir,
            man_path=man_path,
            pair_id=pair_id,
            planned_attempt_id=planned_attempt_id,
            expected=expected,
            phase="corrupt_pair_manifest",
            corrupt_backup_rel=backup_name,
        )
    status = str(man.get("status", ""))
    # legacy R2-C manifests without status but with comparable
    if not status and "comparable" in man:
        status = PAIR_STATUS_COMPLETED
    if status == PAIR_STATUS_COMPLETED:
        ok, reasons = pair_manifest_matches_expected(man, expected)
        if ok:
            return {
                "action": "idempotent_read",
                "attempt_id": int(planned_attempt_id),
                "attempt_dir": adir,
                "existing_manifest": man,
            }
        raise RunnerContractError(
            f"no_auto_retry: completed attempt {planned_attempt_id} for {pair_id} "
            f"hash mismatch ({','.join(reasons)}); refuse auto new attempt"
        )
    if status == PAIR_STATUS_FAILED:
        return {
            "action": "retain_failed",
            "attempt_id": int(planned_attempt_id),
            "attempt_dir": adir,
            "existing_manifest": man,
        }
    raise RunnerContractError(
        f"no_auto_retry: unknown status {status!r} for {pair_id}/attempt_{planned_attempt_id}"
    )


def write_run_set_checkpoint(
    path: Path | str,
    *,
    run_set_manifest: Mapping[str, Any],
    pair_results: Sequence[Mapping[str, Any]],
    last_completed_index: int,
    status: str = "IN_PROGRESS",
    started_wall_time: float | None = None,
) -> dict[str, Any]:
    """Atomic checkpoint after each pair for interrupt resume."""
    ckpt = {
        "schema_version": RUN_SET_CHECKPOINT_SCHEMA,
        "run_set_manifest_content_hash": run_set_manifest.get("manifest_content_hash"),
        "registry_sha256": run_set_manifest.get("registry_sha256"),
        "model_retimer_hash": run_set_manifest.get("model_retimer_hash"),
        "executor_config_hash": run_set_manifest.get("executor_config_hash"),
        "continue_policy": run_set_manifest.get("continue_policy"),
        "retry_policy": run_set_manifest.get("retry_policy"),
        "last_completed_index": int(last_completed_index),
        "n_planned": 12,
        "status": status,
        "pair_results": list(pair_results),
        "started_wall_time": float(started_wall_time if started_wall_time is not None else time.time()),
        "updated_wall_time": time.time(),
    }
    write_json_atomic(Path(path), ckpt)
    return ckpt


def load_run_set_checkpoint(
    path: Path | str,
    *,
    run_set_manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load checkpoint; require pair_results is a continuous prefix of manifest slots."""
    path = Path(path)
    if not path.is_file():
        return None
    ckpt = _read_json(path)
    if str(ckpt.get("schema_version")) != RUN_SET_CHECKPOINT_SCHEMA:
        raise RunnerContractError(f"bad checkpoint schema: {ckpt.get('schema_version')}")
    if str(ckpt.get("run_set_manifest_content_hash")) != str(
        run_set_manifest.get("manifest_content_hash")
    ):
        raise RunnerContractError(
            "checkpoint manifest hash mismatch; refuse resume with different immutable plan"
        )
    for key in ("registry_sha256", "model_retimer_hash", "executor_config_hash", "retry_policy"):
        if str(ckpt.get(key, "")) != str(run_set_manifest.get(key, "")):
            raise RunnerContractError(f"checkpoint field mismatch on {key}")
    slots = run_set_manifest.get("pairs") or []
    if not isinstance(slots, list) or len(slots) != 12:
        raise RunnerContractError("manifest.pairs must be length 12 for checkpoint load")
    results = ckpt.get("pair_results") or []
    if not isinstance(results, list):
        raise RunnerContractError("checkpoint.pair_results must be a list")
    if len(results) > 12:
        raise RunnerContractError("checkpoint.pair_results longer than 12")
    last = int(ckpt.get("last_completed_index", -1))
    if len(results) == 0:
        if last != -1:
            raise RunnerContractError(
                f"empty checkpoint pair_results but last_completed_index={last}"
            )
        return ckpt
    if last != len(results) - 1:
        raise RunnerContractError(
            f"checkpoint last_completed_index={last} != len(pair_results)-1={len(results) - 1}"
        )
    for i, row in enumerate(results):
        if not isinstance(row, Mapping):
            raise RunnerContractError(f"checkpoint.pair_results[{i}] not an object")
        slot = slots[i]
        if int(row.get("index", -1)) != int(slot["index"]) or int(row.get("index", -1)) != i:
            raise RunnerContractError(
                f"checkpoint non-contiguous or wrong index at {i}: row={row.get('index')}"
            )
        if str(row.get("pair_id")) != str(slot["pair_id"]):
            raise RunnerContractError(f"checkpoint pair_id mismatch at {i}")
        if str(row.get("scenario_id")) != str(slot["scenario_id"]):
            raise RunnerContractError(f"checkpoint scenario_id mismatch at {i}")
        if str(row.get("seed_id")) != str(slot["seed_id"]):
            raise RunnerContractError(f"checkpoint seed_id mismatch at {i}")
        if int(row.get("attempt_id", -1)) != int(slot["planned_attempt_id"]):
            raise RunnerContractError(
                f"checkpoint attempt_id mismatch at {i}: "
                f"{row.get('attempt_id')} != planned {slot['planned_attempt_id']}"
            )
        if list(row.get("branch_order") or []) != list(slot["branch_order"]):
            raise RunnerContractError(f"checkpoint branch_order mismatch at {i}")
    return ckpt


def validate_report_against_manifest(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Strict alignment when aggregate reads both report and manifest."""
    if str(report.get("run_set_manifest_content_hash")) != str(
        manifest.get("manifest_content_hash")
    ):
        raise RunnerContractError("report.run_set_manifest_content_hash mismatch vs manifest")
    for key in ("registry_sha256", "model_retimer_hash", "executor_config_hash"):
        if str(report.get(key, "")) != str(manifest.get(key, "")):
            raise RunnerContractError(f"report/manifest mismatch on {key}")
    if str(report.get("retry_policy", RETRY_POLICY_NO_AUTO_RETRY)) != str(
        manifest.get("retry_policy", RETRY_POLICY_NO_AUTO_RETRY)
    ):
        raise RunnerContractError("report/manifest retry_policy mismatch")
    results = report.get("pair_results")
    slots = manifest.get("pairs")
    if not isinstance(results, list) or len(results) != 12:
        raise RunnerContractError("report.pair_results must be length 12")
    if not isinstance(slots, list) or len(slots) != 12:
        raise RunnerContractError("manifest.pairs must be length 12")
    retry = normalize_retry_policy(str(manifest.get("retry_policy")))
    for i, (row, slot) in enumerate(zip(results, slots)):
        if int(row.get("index", -1)) != int(slot.get("index", -2)):
            raise RunnerContractError(f"report/manifest index mismatch at {i}")
        if str(row.get("pair_id")) != str(slot.get("pair_id")):
            raise RunnerContractError(f"report/manifest pair_id mismatch at {i}")
        if str(row.get("scenario_id")) != str(slot.get("scenario_id")):
            raise RunnerContractError(f"report/manifest scenario_id mismatch at {i}")
        if str(row.get("seed_id")) != str(slot.get("seed_id")):
            raise RunnerContractError(f"report/manifest seed_id mismatch at {i}")
        planned = int(slot["planned_attempt_id"])
        resolved = int(row["attempt_id"])
        if retry == RETRY_POLICY_NO_AUTO_RETRY and resolved != planned:
            raise RunnerContractError(
                f"resolved attempt_id {resolved} != planned {planned} at index {i} "
                f"(no_auto_retry)"
            )


def attempt_dir_for(
    pairs_root: Path | str,
    pair_id: str,
    attempt_id: int,
) -> Path:
    root = Path(pairs_root)
    modern = root / pair_id / f"attempt_{int(attempt_id)}"
    if modern.is_dir() or not (root / pair_id / "pair_manifest.json").is_file():
        return modern
    # legacy flat only valid as attempt 0
    if int(attempt_id) == 0:
        return root / pair_id
    return modern


def load_pair_manifest_for_slot(
    pairs_root: Path | str,
    *,
    pair_id: str,
    attempt_id: int,
) -> dict[str, Any]:
    """Load pair_manifest.json for exact pair_id+attempt_id (fail-closed if missing)."""
    adir = attempt_dir_for(pairs_root, pair_id, attempt_id)
    path = adir / "pair_manifest.json"
    if not path.is_file():
        raise RunnerContractError(
            f"missing pair_manifest for pair_id={pair_id} attempt_id={attempt_id} path={path}"
        )
    return _read_json(path)


def normalize_aggregate_row(man: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize pair_manifest / oracle row for pilot aggregation."""
    status = str(man.get("status", PAIR_STATUS_COMPLETED))
    if status == PAIR_STATUS_FAILED or man.get("error"):
        top1_index = int(man.get("top1_candidate_index", 0))
        candidate_ids = man.get("candidate_ids")
        top1_id = str(man.get("top1_candidate_id", "")).strip()
        if (
            not top1_id
            and isinstance(candidate_ids, Sequence)
            and not isinstance(candidate_ids, (str, bytes))
            and 0 <= top1_index < len(candidate_ids)
        ):
            top1_id = str(candidate_ids[top1_index])
        if not top1_id:
            top1_id = "UNKNOWN_CANDIDATE"
        return {
            "pair_id": str(man.get("pair_id", "")),
            "scenario_id": str(man.get("scenario_id", "")),
            "seed_id": str(man.get("seed_id", "")),
            "family": str(man.get("family", "")),
            "attempt_id": int(man.get("attempt_id", 0)),
            "status": PAIR_STATUS_FAILED,
            "comparable": False,
            "pair_label": "INCOMPARABLE",
            "top1_candidate_id": top1_id,
            "top1_candidate_index": top1_index,
            "oracle_candidate_id": None,
            "oracle_candidate_index": None,
            "oracle_decision_level": None,
            "decision_reason": str(man.get("error", "failed")),
            "both_bad": False,
            "outcome_delta": {},
            "failure_reasons": (str(man.get("error", "failed")),),
            "artifact_content_hash": man.get("artifact_content_hash"),
            "idempotent_read": bool(man.get("idempotent_read", False)),
            "error": man.get("error"),
        }
    oracle = man.get("oracle") if isinstance(man.get("oracle"), dict) else None
    if oracle is not None:
        row = dict(oracle)
        for k in (
            "pair_id",
            "scenario_id",
            "seed_id",
            "family",
            "attempt_id",
            "artifact_content_hash",
            "comparable",
        ):
            if k in man and (k not in row or row.get(k) is None):
                row[k] = man[k]
    else:
        row = dict(man)
    row["attempt_id"] = int(row.get("attempt_id", man.get("attempt_id", 0)))
    row["status"] = status if status else (
        PAIR_STATUS_COMPARABLE if bool(row.get("comparable")) else PAIR_STATUS_INCOMPARABLE
    )
    if not bool(row.get("comparable", False)):
        row.setdefault("pair_label", "INCOMPARABLE")
    row["idempotent_read"] = bool(man.get("idempotent_read", False))
    return row


def aggregate_from_run_set_spec(
    *,
    pairs_root: Path | str,
    slots: Sequence[Mapping[str, Any]],
    require_n: int = 12,
) -> list[dict[str, Any]]:
    """Load exactly require_n rows for specified pair_id+attempt_id only.

    - Reads only pair_manifest.json per slot (not ledger, no directory scan).
    - Dedupes on pair_id+attempt_id (fail-closed on duplicates in slots).
    - failed/incomparable rows are kept for denominator.
    """
    if len(slots) != int(require_n):
        raise RunnerContractError(
            f"aggregate slots must be exactly {require_n}, got {len(slots)}"
        )
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for i, slot in enumerate(slots):
        pair_id = str(slot.get("pair_id", "")).strip()
        if not pair_id:
            raise RunnerContractError(f"slot[{i}] missing pair_id")
        if "attempt_id" in slot:
            attempt_id = int(slot["attempt_id"])
        elif "resolved_attempt_id" in slot:
            attempt_id = int(slot["resolved_attempt_id"])
        elif "planned_attempt_id" in slot:
            attempt_id = int(slot["planned_attempt_id"])
        else:
            raise RunnerContractError(f"slot[{i}] missing attempt_id")
        key = (pair_id, attempt_id)
        if key in seen:
            raise RunnerContractError(
                f"duplicate pair_id+attempt_id in aggregate slots: {pair_id}/{attempt_id}"
            )
        seen.add(key)
        man = load_pair_manifest_for_slot(pairs_root, pair_id=pair_id, attempt_id=attempt_id)
        # identity check
        if str(man.get("pair_id", "")) != pair_id:
            raise RunnerContractError(
                f"manifest pair_id mismatch for {pair_id}: {man.get('pair_id')}"
            )
        if int(man.get("attempt_id", -1)) != attempt_id:
            raise RunnerContractError(
                f"manifest attempt_id mismatch for {pair_id}: "
                f"expected {attempt_id} got {man.get('attempt_id')}"
            )
        rows.append(normalize_aggregate_row(man))
    if len(rows) != int(require_n):
        raise RunnerContractError(f"aggregate must yield {require_n} rows, got {len(rows)}")
    return rows


def slots_from_run_set_report_or_manifest(
    *,
    run_set_manifest: Mapping[str, Any] | None = None,
    run_set_report: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Prefer report resolved attempt_ids; fall back to manifest planned ids.

    When both are provided, strictly validate report against immutable manifest.
    """
    if run_set_report is not None and run_set_manifest is not None:
        validate_report_against_manifest(run_set_report, run_set_manifest)
    if run_set_report is not None:
        results = run_set_report.get("pair_results")
        if not isinstance(results, list) or len(results) != 12:
            raise RunnerContractError(
                f"run_set_report.pair_results must be length 12, got "
                f"{len(results) if isinstance(results, list) else type(results)}"
            )
        slots: list[dict[str, Any]] = []
        for i, row in enumerate(results):
            if not isinstance(row, Mapping):
                raise RunnerContractError(f"pair_results[{i}] not an object")
            pair_id = str(row.get("pair_id", ""))
            if not pair_id:
                raise RunnerContractError(f"pair_results[{i}] missing pair_id")
            if "attempt_id" not in row:
                raise RunnerContractError(f"pair_results[{i}] missing attempt_id")
            slots.append(
                {
                    "index": int(row.get("index", i)),
                    "pair_id": pair_id,
                    "attempt_id": int(row["attempt_id"]),
                    "scenario_id": row.get("scenario_id"),
                    "seed_id": row.get("seed_id"),
                }
            )
        return slots
    if run_set_manifest is None:
        raise RunnerContractError("need run_set_manifest or run_set_report for aggregate")
    pairs = run_set_manifest.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 12:
        raise RunnerContractError("run_set_manifest.pairs must be length 12")
    slots = []
    for i, p in enumerate(pairs):
        if not isinstance(p, Mapping):
            raise RunnerContractError(f"manifest.pairs[{i}] not an object")
        slots.append(
            {
                "index": int(p.get("index", i)),
                "pair_id": str(p["pair_id"]),
                "attempt_id": int(p.get("resolved_attempt_id", p["planned_attempt_id"])),
                "scenario_id": p.get("scenario_id"),
                "seed_id": p.get("seed_id"),
            }
        )
    return slots


def pair_result_row_from_run_pair_output(
    *,
    index: int,
    scenario_id: str,
    seed_id: str,
    family: str,
    branch_order: Sequence[int],
    planned_attempt_id: int,
    pair_id: str,
    man: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Uniform run_set_report pair row (success / idempotent / failed)."""
    if error is not None or man is None:
        return {
            "index": index,
            "scenario_id": scenario_id,
            "seed_id": seed_id,
            "family": family,
            "branch_order": list(branch_order),
            "pair_id": pair_id,
            "planned_attempt_id": planned_attempt_id,
            "attempt_id": int(man.get("attempt_id", planned_attempt_id)) if man else planned_attempt_id,
            "status": PAIR_STATUS_FAILED,
            "comparable": False,
            "idempotent_read": False,
            "oracle": None,
            "pair_label": None,
            "failure": str(error or "unknown"),
            "artifact_content_hash": None,
            "attempt_dir": man.get("attempt_dir") if man else None,
        }
    oracle = man.get("oracle") if isinstance(man.get("oracle"), dict) else None
    comparable = bool(man.get("comparable", False))
    status = str(man.get("status", PAIR_STATUS_COMPLETED))
    if status == PAIR_STATUS_FAILED:
        row_status = PAIR_STATUS_FAILED
    elif comparable:
        row_status = PAIR_STATUS_COMPARABLE
    else:
        row_status = PAIR_STATUS_INCOMPARABLE
    return {
        "index": index,
        "scenario_id": scenario_id,
        "seed_id": seed_id,
        "family": family,
        "branch_order": list(branch_order),
        "pair_id": str(man.get("pair_id", pair_id)),
        "planned_attempt_id": planned_attempt_id,
        "attempt_id": int(man.get("attempt_id", planned_attempt_id)),
        "status": row_status,
        "run_status": status,
        "comparable": comparable,
        "idempotent_read": bool(man.get("idempotent_read", False)),
        "oracle": oracle,
        "pair_label": (oracle or {}).get("pair_label") if oracle else man.get("pair_label"),
        "oracle_candidate_id": (oracle or {}).get("oracle_candidate_id") if oracle else None,
        "failure": man.get("error"),
        "artifact_content_hash": man.get("artifact_content_hash"),
        "attempt_dir": man.get("attempt_dir"),
        "ledger_appended": man.get("ledger_appended"),
    }


def execute_run_set_orchestration(
    *,
    run_set_manifest: Mapping[str, Any],
    run_pair_fn: Callable[..., Mapping[str, Any]],
    pairs_root: Path | str,
    run_pair_kwargs: Mapping[str, Any] | None = None,
    checkpoint_path: Path | str | None = None,
    report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Drive 12 pairs via injected run_pair_fn (fake or live). No CARLA here.

    continue_policy / retry_policy come only from the immutable manifest.
    After each pair, atomically updates checkpoint (+ optional partial report).
    """
    if not bool(run_set_manifest.get("immutable", False)):
        raise RunnerContractError("run_set_manifest must be immutable=true")
    if str(run_set_manifest.get("schema_version")) != RUN_SET_MANIFEST_SCHEMA:
        raise RunnerContractError(
            f"bad run_set_manifest schema: {run_set_manifest.get('schema_version')}"
        )
    slots = run_set_manifest.get("pairs")
    if not isinstance(slots, list) or len(slots) != 12:
        raise RunnerContractError("run_set_manifest.pairs must be length 12")
    policy = normalize_continue_policy(str(run_set_manifest.get("continue_policy")))
    retry = normalize_retry_policy(str(run_set_manifest.get("retry_policy")))
    kwargs = dict(run_pair_kwargs or {})
    pairs_root_p = Path(pairs_root)
    ckpt_path = Path(checkpoint_path) if checkpoint_path else pairs_root_p.parent / "run_set_checkpoint.json"
    rep_path = Path(report_path) if report_path else pairs_root_p.parent / "run_set_report.json"

    # Resume from checkpoint if same immutable manifest
    pair_results: list[dict[str, Any]] = []
    start_index = 0
    started = time.time()
    existing_ckpt = load_run_set_checkpoint(ckpt_path, run_set_manifest=run_set_manifest)
    if existing_ckpt is not None:
        prev = existing_ckpt.get("pair_results") or []
        if not isinstance(prev, list):
            raise RunnerContractError("checkpoint.pair_results must be a list")
        pair_results = [dict(x) for x in prev]
        start_index = int(existing_ckpt.get("last_completed_index", -1)) + 1
        started = float(existing_ckpt.get("started_wall_time", started))

    def _pad_not_run(from_index: int) -> None:
        for rest in slots[from_index:]:
            pair_results.append(
                pair_result_row_from_run_pair_output(
                    index=int(rest["index"]),
                    scenario_id=str(rest["scenario_id"]),
                    seed_id=str(rest["seed_id"]),
                    family=str(rest.get("family", "")),
                    branch_order=list(rest["branch_order"]),
                    planned_attempt_id=int(rest["planned_attempt_id"]),
                    pair_id=str(rest["pair_id"]),
                    man=None,
                    error="not_run_stop_on_fail",
                )
            )

    def _persist(last_idx: int, *, status: str) -> None:
        write_run_set_checkpoint(
            ckpt_path,
            run_set_manifest=run_set_manifest,
            pair_results=pair_results,
            last_completed_index=last_idx,
            status=status,
            started_wall_time=started,
        )
        # partial report for recovery readers
        partial = {
            "schema_version": RUN_SET_REPORT_SCHEMA,
            "started_wall_time": started,
            "ended_wall_time": time.time(),
            "status": status,
            "run_set_manifest_content_hash": run_set_manifest.get("manifest_content_hash"),
            "registry_sha256": run_set_manifest.get("registry_sha256"),
            "model_retimer_hash": run_set_manifest.get("model_retimer_hash"),
            "executor_config_hash": run_set_manifest.get("executor_config_hash"),
            "continue_policy": policy,
            "retry_policy": retry,
            "n_planned": 12,
            "pair_results": list(pair_results),
            "summary": summarize_run_set_results(pair_results, n_planned=12),
            "pairs_root": str(pairs_root_p.as_posix()),
            "last_completed_index": last_idx,
        }
        write_json_atomic(rep_path, partial)

    for slot in slots:
        index = int(slot["index"])
        if index < start_index:
            continue
        scenario_id = str(slot["scenario_id"])
        seed_id = str(slot["seed_id"])
        family = str(slot.get("family", ""))
        pair_id = str(slot["pair_id"])
        branch_order = tuple(int(x) for x in slot["branch_order"])
        planned_attempt_id = int(slot["planned_attempt_id"])
        try:
            man = dict(
                run_pair_fn(
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    branch_order=branch_order,
                    force_attempt_id=planned_attempt_id,
                    retry_policy=retry,
                    **kwargs,
                )
            )
            if str(man.get("pair_id", pair_id)) != pair_id:
                if str(man.get("status")) != PAIR_STATUS_FAILED:
                    raise RunnerContractError(
                        f"run_pair returned pair_id {man.get('pair_id')} != planned {pair_id}"
                    )
            returned_attempt = int(man.get("attempt_id", -1))
            if returned_attempt != planned_attempt_id:
                raise RunnerContractError(
                    f"run_pair attempt_id {returned_attempt} != planned {planned_attempt_id} "
                    f"for {pair_id} (no_auto_retry)"
                )
            row = pair_result_row_from_run_pair_output(
                index=index,
                scenario_id=scenario_id,
                seed_id=seed_id,
                family=family,
                branch_order=branch_order,
                planned_attempt_id=planned_attempt_id,
                pair_id=pair_id,
                man=man,
            )
        except RunnerContractError:
            # Contract violations (attempt mismatch, wrong pair_id) fail-closed.
            raise
        except Exception as exc:  # noqa: BLE001 — pair runtime failure
            row = pair_result_row_from_run_pair_output(
                index=index,
                scenario_id=scenario_id,
                seed_id=seed_id,
                family=family,
                branch_order=branch_order,
                planned_attempt_id=planned_attempt_id,
                pair_id=pair_id,
                man=None,
                error=str(exc),
            )
            # failed rows still use planned attempt id
            row["attempt_id"] = planned_attempt_id
            pair_results.append(row)
            _persist(index, status="IN_PROGRESS")
            if policy == CONTINUE_POLICY_STOP_ON_FAIL:
                _pad_not_run(index + 1)
                break
            continue
        pair_results.append(row)
        _persist(index, status="IN_PROGRESS")
        if row["status"] == PAIR_STATUS_FAILED and policy == CONTINUE_POLICY_STOP_ON_FAIL:
            _pad_not_run(index + 1)
            break

    if len(pair_results) != 12:
        raise RunnerContractError(
            f"orchestration must produce 12 pair_results, got {len(pair_results)}"
        )
    summary = summarize_run_set_results(pair_results, n_planned=12)
    report = {
        "schema_version": RUN_SET_REPORT_SCHEMA,
        "started_wall_time": started,
        "ended_wall_time": time.time(),
        "status": "COMPLETED",
        "run_set_manifest_content_hash": run_set_manifest.get("manifest_content_hash"),
        "registry_sha256": run_set_manifest.get("registry_sha256"),
        "model_retimer_hash": run_set_manifest.get("model_retimer_hash"),
        "executor_config_hash": run_set_manifest.get("executor_config_hash"),
        "continue_policy": policy,
        "retry_policy": retry,
        "n_planned": 12,
        "pair_results": pair_results,
        "summary": summary,
        "pairs_root": str(pairs_root_p.as_posix()),
        "last_completed_index": 11,
    }
    write_run_set_checkpoint(
        ckpt_path,
        run_set_manifest=run_set_manifest,
        pair_results=pair_results,
        last_completed_index=11,
        status="COMPLETED",
        started_wall_time=started,
    )
    write_json_atomic(rep_path, report)
    return report


# ---------------------------------------------------------------------------
# R2-E: immutable repeat audit plan (frozen BEFORE any R2-D outcomes)
# ---------------------------------------------------------------------------

REPEAT_AUDIT_PLAN_SCHEMA = "safedrive.g4a.repeat_audit_plan.v1"
REPEAT_AUDIT_REPORT_SCHEMA = "safedrive.g4a.repeat_audit_report.v1"
R2_CLOSURE_REPORT_SCHEMA = "safedrive.g4a.r2_closure_report.v1"
# Fixed audit seed — selection depends only on registry hash + this seed.
REPEAT_AUDIT_SEED = "safedrive.g4a.r2e.repeat_audit.v1"
N_REPEAT_PAIRS = 2


def select_repeat_pair_indices(
    slots: Sequence[Mapping[str, Any]],
    *,
    registry_sha256: str,
    audit_seed: str = REPEAT_AUDIT_SEED,
    n: int = N_REPEAT_PAIRS,
) -> list[int]:
    """Deterministically pick n indices from frozen 12 slots.

    Selection uses only ``registry_sha256`` + fixed ``audit_seed`` + slot
    identity fields (scenario/seed/family/index). Never uses candidate or
    outcome. Prefers covering different families when possible.
    """
    if len(slots) != 12:
        raise RunnerContractError(
            f"select_repeat_pair_indices requires 12 slots, got {len(slots)}"
        )
    if n < 1 or n > 12:
        raise RunnerContractError(f"n must be in 1..12, got {n}")
    base = f"{registry_sha256}|{audit_seed}".encode("utf-8")
    ranked: list[tuple[str, int]] = []
    for i, s in enumerate(slots):
        if not isinstance(s, Mapping):
            raise RunnerContractError(f"slot[{i}] must be object")
        digest = hashlib.sha256(
            base
            + (
                f"|idx={i}|scenario={s.get('scenario_id')}"
                f"|seed={s.get('seed_id')}|family={s.get('family')}"
            ).encode("utf-8")
        ).hexdigest()
        ranked.append((digest, i))
    ranked.sort(key=lambda t: (t[0], t[1]))
    chosen: list[int] = []
    used_families: set[str] = set()
    # Pass 1: prefer new families
    for _digest, i in ranked:
        if len(chosen) >= n:
            break
        fam = str(slots[i].get("family") or "")
        if fam and fam in used_families and len(used_families) < 3:
            continue
        chosen.append(i)
        if fam:
            used_families.add(fam)
    # Pass 2: fill remaining by rank
    if len(chosen) < n:
        for _digest, i in ranked:
            if i in chosen:
                continue
            chosen.append(i)
            if len(chosen) >= n:
                break
    if len(chosen) != n:
        raise RunnerContractError(f"could not select {n} repeat pairs")
    return sorted(chosen)


def _next_free_attempt_after(pair_root: Path, reserved_id: int) -> int:
    """First free attempt id strictly greater than reserved D attempt (or any free > reserved)."""
    candidate = int(reserved_id) + 1
    while attempt_slot_occupied(pair_root, candidate):
        candidate += 1
        if candidate > 512:
            raise RunnerContractError(f"no free repeat attempt under {pair_root}")
    return candidate


def build_repeat_audit_plan(
    *,
    registry: ScenarioRegistryV1,
    freeze_audit: Mapping[str, Any],
    pairs_root: Path | str,
    model_retimer_hash: str,
    executor_config_hash: str,
    model_checkpoint_hash: str = "",
    retimer_hash: str = "",
    audit_seed: str = REPEAT_AUDIT_SEED,
    n_repeat: int = N_REPEAT_PAIRS,
) -> dict[str, Any]:
    """Build immutable R2-E repeat plan WITHOUT reading pair outcomes.

    D planned attempt IDs are derived from current Evidence occupancy only
    (same rule as run_set freeze). Repeat attempt = first free after D planned.
    Selection is deterministic from registry hash + audit_seed only.
    """
    reg_hash = str(freeze_audit.get("registry_sha256") or registry.compute_registry_sha256())
    if not freeze_audit.get("frozen", True):
        raise RunnerContractError("repeat_audit_plan requires frozen registry")
    base = plan_run_set_pairs(registry)
    pairs_root_p = Path(pairs_root)
    pool: list[dict[str, Any]] = []
    for item in base:
        pair_id = compute_pair_id(
            scenario_registry_hash=reg_hash,
            scenario_id=str(item["scenario_id"]),
            seed_id=str(item["seed_id"]),
            model_checkpoint_config_retimer_hash=model_retimer_hash,
            executor_config_hash=executor_config_hash,
        )
        pair_root = pairs_root_p / pair_id
        d_attempt = first_unoccupied_attempt_id(pair_root)
        # Reserve D slot when computing repeat id (D not yet run may still free d_attempt).
        # After D freezes the same d_attempt, repeat must not collide: use d_attempt+1 free.
        # Temporarily treat d_attempt as occupied for repeat selection.
        occupied = {i for i, _ in list_attempt_dirs(pair_root)}
        occupied.add(int(d_attempt))
        r_attempt = int(d_attempt) + 1
        while r_attempt in occupied:
            r_attempt += 1
        pool.append(
            {
                "index": int(item["index"]),
                "scenario_id": str(item["scenario_id"]),
                "seed_id": str(item["seed_id"]),
                "family": str(item["family"]),
                "pair_id": pair_id,
                "branch_order": list(item["branch_order"]),
                "d_planned_attempt_id": int(d_attempt),
                "repeat_attempt_id": int(r_attempt),
                "requested_initial_state_hash": str(item["requested_initial_state_hash"]),
                "d_attempt_dir_rel": planned_attempt_dir_rel(pair_id, d_attempt),
                "repeat_attempt_dir_rel": planned_attempt_dir_rel(pair_id, r_attempt),
            }
        )
    chosen_idx = select_repeat_pair_indices(
        pool, registry_sha256=reg_hash, audit_seed=audit_seed, n=n_repeat
    )
    selected = [dict(pool[i]) for i in chosen_idx]
    for j, s in enumerate(selected):
        s["repeat_slot"] = j
        s["in_d_denominator"] = False
        s["note"] = "repeat independent attempt; not counted in D 12-pair denominator"
    families = sorted({str(s["family"]) for s in selected})
    payload: dict[str, Any] = {
        "schema_version": REPEAT_AUDIT_PLAN_SCHEMA,
        "immutable": True,
        "frozen_before_outcomes": True,
        "audit_seed": audit_seed,
        "registry_schema_version": REGISTRY_SCHEMA,
        "registry_sha256": reg_hash,
        "model_retimer_hash": model_retimer_hash,
        "model_checkpoint_hash": model_checkpoint_hash,
        "retimer_hash": retimer_hash,
        "executor_config_hash": executor_config_hash,
        "n_repeat_pairs": n_repeat,
        "selection_rule": (
            "sha256(registry_sha256|audit_seed|idx|scenario|seed|family) rank; "
            "prefer distinct families; never uses candidate/outcome"
        ),
        "selected_indices": list(chosen_idx),
        "selected_families": families,
        "pairs": selected,
        "pool_n": 12,
        "role": "label_consistency_audit_only",
    }
    payload["plan_content_hash"] = content_hash(
        {
            "schema_version": payload["schema_version"],
            "immutable": True,
            "audit_seed": audit_seed,
            "registry_sha256": reg_hash,
            "model_retimer_hash": model_retimer_hash,
            "model_checkpoint_hash": model_checkpoint_hash,
            "retimer_hash": retimer_hash,
            "executor_config_hash": executor_config_hash,
            "n_repeat_pairs": n_repeat,
            "selected_indices": list(chosen_idx),
            "pairs": [
                {
                    "index": p["index"],
                    "scenario_id": p["scenario_id"],
                    "seed_id": p["seed_id"],
                    "family": p["family"],
                    "pair_id": p["pair_id"],
                    "branch_order": p["branch_order"],
                    "d_planned_attempt_id": p["d_planned_attempt_id"],
                    "repeat_attempt_id": p["repeat_attempt_id"],
                    "d_attempt_dir_rel": p["d_attempt_dir_rel"],
                    "repeat_attempt_dir_rel": p["repeat_attempt_dir_rel"],
                    "requested_initial_state_hash": p["requested_initial_state_hash"],
                }
                for p in selected
            ],
        },
        nibble=64,
    )
    return payload


def validate_repeat_audit_plan_self(plan: Mapping[str, Any]) -> None:
    if not bool(plan.get("immutable", False)):
        raise RunnerContractError("repeat_audit_plan.immutable must be true")
    if str(plan.get("schema_version")) != REPEAT_AUDIT_PLAN_SCHEMA:
        raise RunnerContractError(
            f"bad repeat_audit_plan schema: {plan.get('schema_version')}"
        )
    pairs = plan.get("pairs") or []
    if not isinstance(pairs, list) or len(pairs) != int(plan.get("n_repeat_pairs", N_REPEAT_PAIRS)):
        raise RunnerContractError("repeat_audit_plan.pairs length mismatch")
    recomputed = content_hash(
        {
            "schema_version": plan.get("schema_version"),
            "immutable": True,
            "audit_seed": plan.get("audit_seed"),
            "registry_sha256": plan.get("registry_sha256"),
            "model_retimer_hash": plan.get("model_retimer_hash"),
            "model_checkpoint_hash": plan.get("model_checkpoint_hash", ""),
            "retimer_hash": plan.get("retimer_hash", ""),
            "executor_config_hash": plan.get("executor_config_hash"),
            "n_repeat_pairs": int(plan.get("n_repeat_pairs", 0)),
            "selected_indices": list(plan.get("selected_indices") or []),
            "pairs": [
                {
                    "index": p["index"],
                    "scenario_id": p["scenario_id"],
                    "seed_id": p["seed_id"],
                    "family": p["family"],
                    "pair_id": p["pair_id"],
                    "branch_order": p["branch_order"],
                    "d_planned_attempt_id": p["d_planned_attempt_id"],
                    "repeat_attempt_id": p["repeat_attempt_id"],
                    "d_attempt_dir_rel": p["d_attempt_dir_rel"],
                    "repeat_attempt_dir_rel": p["repeat_attempt_dir_rel"],
                    "requested_initial_state_hash": p["requested_initial_state_hash"],
                }
                for p in pairs
            ],
        },
        nibble=64,
    )
    got = str(plan.get("plan_content_hash") or "")
    if got != recomputed:
        raise RunnerContractError(
            f"repeat_audit_plan content hash mismatch stored={got[:16]}… "
            f"recomputed={recomputed[:16]}…"
        )


def validate_repeat_audit_plan_stable_identity(
    plan: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    for key in (
        "registry_sha256",
        "model_retimer_hash",
        "model_checkpoint_hash",
        "retimer_hash",
        "executor_config_hash",
        "audit_seed",
    ):
        if key not in identity:
            continue
        if str(plan.get(key, "")) != str(identity.get(key, "")):
            raise RunnerContractError(
                f"repeat_audit_plan field mismatch on {key}: "
                f"existing={plan.get(key)!r} expected={identity.get(key)!r}"
            )


def ensure_repeat_audit_plan(
    path: Path | str,
    expected: Mapping[str, Any] | None = None,
    *,
    build_fn: Callable[[], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Exclusive create or validate-and-reuse immutable repeat_audit_plan.

    When existing: self-hash + stable identity only — **never reselect pairs**.
    """
    path = Path(path)
    if path.is_file():
        existing = _read_json(path)
        validate_repeat_audit_plan_self(existing)
        if expected is not None:
            validate_repeat_audit_plan_stable_identity(existing, expected)
        return existing, "reused"
    if expected is None:
        if build_fn is None:
            raise RunnerContractError(
                "ensure_repeat_audit_plan: need expected or build_fn when missing"
            )
        expected = dict(build_fn())
    else:
        expected = dict(expected)
    if "plan_content_hash" not in expected:
        raise RunnerContractError("repeat_audit_plan missing plan_content_hash")
    write_json_exclusive_create(path, expected)
    return dict(expected), "created"


def load_pair_oracle_label(
    pairs_root: Path | str,
    *,
    pair_id: str,
    attempt_id: int,
) -> dict[str, Any]:
    """Load oracle label fields from an attempt (or legacy top-level for attempt 0)."""
    pairs_root = Path(pairs_root)
    adir = attempt_dir_for(pairs_root, pair_id, attempt_id)
    man_path = adir / "pair_manifest.json"
    oracle_path = adir / "pair_oracle.json"
    # legacy R2-C top-level for attempt_0
    if not man_path.is_file() and attempt_id == 0:
        legacy = pairs_root / pair_id
        if (legacy / "pair_manifest.json").is_file():
            adir = legacy
            man_path = adir / "pair_manifest.json"
            oracle_path = adir / "pair_oracle.json"
    out: dict[str, Any] = {
        "pair_id": pair_id,
        "attempt_id": int(attempt_id),
        "attempt_dir": str(adir.as_posix()),
        "found": False,
        "status": None,
        "comparable": False,
        "pair_label": None,
        "oracle_candidate_index": None,
        "both_bad": False,
        "error": None,
    }
    if not man_path.is_file():
        out["error"] = "missing_pair_manifest"
        return out
    try:
        man = _read_json(man_path)
    except RunnerContractError as exc:
        out["error"] = f"corrupt_manifest:{exc}"
        return out
    out["found"] = True
    out["status"] = man.get("status")
    out["comparable"] = bool(man.get("comparable", False))
    if oracle_path.is_file():
        try:
            ora = _read_json(oracle_path)
            out["pair_label"] = ora.get("pair_label") or man.get("pair_label")
            out["oracle_candidate_index"] = ora.get("oracle_candidate_index")
            out["both_bad"] = bool(ora.get("both_bad", False))
            out["oracle"] = ora
        except RunnerContractError:
            out["pair_label"] = man.get("pair_label")
    else:
        out["pair_label"] = man.get("pair_label")
        out["oracle"] = man.get("oracle")
    out["manifest"] = {
        k: man.get(k)
        for k in (
            "status",
            "comparable",
            "pair_label",
            "error",
            "artifact_content_hash",
            "model_retimer_hash",
            "registry_sha256",
        )
    }
    return out


def compare_repeat_label_consistency(
    *,
    original: Mapping[str, Any],
    repeat: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare oracle labels; retain all differences; never reselect for consistency."""
    o_label = original.get("pair_label")
    r_label = repeat.get("pair_label")
    o_comp = bool(original.get("comparable"))
    r_comp = bool(repeat.get("comparable"))
    consistent = (
        o_label is not None
        and r_label is not None
        and str(o_label) == str(r_label)
        and o_comp == r_comp
    )
    return {
        "label_consistent": bool(consistent),
        "original_label": o_label,
        "repeat_label": r_label,
        "original_comparable": o_comp,
        "repeat_comparable": r_comp,
        "original_oracle_candidate_index": original.get("oracle_candidate_index"),
        "repeat_oracle_candidate_index": repeat.get("oracle_candidate_index"),
        "original_both_bad": bool(original.get("both_bad", False)),
        "repeat_both_bad": bool(repeat.get("both_bad", False)),
        "original_status": original.get("status"),
        "repeat_status": repeat.get("status"),
        "original_error": original.get("error"),
        "repeat_error": repeat.get("error"),
    }


def classify_r2_closure(
    *,
    n_comparable: int,
    n_pairs: int = 12,
    min_comparable: int = 10,
    repeat_labels_all_consistent: bool | None,
    n_repeat_done: int,
    n_repeat_planned: int = N_REPEAT_PAIRS,
    pilot_label: str,
    dominant_failure_class: str | None = None,
) -> dict[str, Any]:
    """Map measured facts to R2 terminal status (honest; not force-pass)."""
    reasons: list[str] = []
    default_pilot = pilot_label or "PILOT_INCONCLUSIVE"
    if n_comparable < min_comparable:
        if dominant_failure_class in {"carla", "sync", "server", "external"}:
            status = "BLOCKED_EXTERNAL"
            reasons.append(
                f"comparable={n_comparable}/{n_pairs}<{min_comparable}; "
                f"class={dominant_failure_class}"
            )
        else:
            status = "REPAIR_REQUIRED"
            reasons.append(
                f"comparable={n_comparable}/{n_pairs}<{min_comparable}; "
                f"class={dominant_failure_class or 'fixture_or_runner'}"
            )
        return {
            "r2_status": status,
            "pilot_label": default_pilot,
            "completed_with_limits": False,
            "reasons": reasons,
            "comparable": n_comparable,
            "n_pairs": n_pairs,
            "repeat_consistent": repeat_labels_all_consistent,
        }

    if n_repeat_done < n_repeat_planned:
        return {
            "r2_status": "REPAIR_REQUIRED",
            "pilot_label": default_pilot,
            "completed_with_limits": False,
            "reasons": [
                f"repeat incomplete: {n_repeat_done}/{n_repeat_planned}"
            ],
            "comparable": n_comparable,
            "n_pairs": n_pairs,
            "repeat_consistent": repeat_labels_all_consistent,
        }

    if repeat_labels_all_consistent is False:
        return {
            "r2_status": "PILOT_INCONCLUSIVE",
            "pilot_label": default_pilot,
            "completed_with_limits": False,
            "reasons": ["repeat oracle labels not all consistent"],
            "comparable": n_comparable,
            "n_pairs": n_pairs,
            "repeat_consistent": False,
        }

    if repeat_labels_all_consistent is not True:
        return {
            "r2_status": "REPAIR_REQUIRED",
            "pilot_label": default_pilot,
            "completed_with_limits": False,
            "reasons": ["repeat consistency unknown"],
            "comparable": n_comparable,
            "n_pairs": n_pairs,
            "repeat_consistent": repeat_labels_all_consistent,
        }

    # Closure criteria met (WITH_LIMITS always for pilot scale)
    return {
        "r2_status": "COMPLETED_WITH_LIMITS",
        "pilot_label": pilot_label,
        "completed_with_limits": True,
        "reasons": [
            f"comparable={n_comparable}/{n_pairs}>={min_comparable}",
            f"repeat_labels_consistent n={n_repeat_done}",
            f"pilot={pilot_label}",
        ],
        "comparable": n_comparable,
        "n_pairs": n_pairs,
        "repeat_consistent": True,
    }
