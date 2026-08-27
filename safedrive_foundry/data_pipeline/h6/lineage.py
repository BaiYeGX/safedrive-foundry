"""Fail-closed state tracking for the pre-registered H6 VLA75 lineages.

The formal lineage rules are an evidence contract, not merely a convention in
the handoff document.  This module keeps one state record per lineage so a
failed pilot/full cannot be silently retried, and so a full run cannot be
started before its pilot has passed.  The records live under the ignored
runtime-evidence tree and are deliberately not part of the scoped runtime hash
used by a run lock.  A passing pilot is the one allowed transition: its state
is retained in ``history`` when the later full result is recorded.  Terminal
failure/success records cannot be changed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from data_pipeline.h3.contracts import stable_sha256
from data_pipeline.h6.config import H6_VLA75_FORMAL_LINEAGES, _validate_lineage_id


LINEAGE_STATE_SCHEMA = "safedrive.h6.vla75.lineage_state.v1"
LINEAGE_STATES = frozenset(
    {
        "PILOT_PASSED",
        "PILOT_FAILED",
        "FULL_FAILED",
        "COMPLETED",
    }
)
LINEAGE_FAILURE_STATES = frozenset({"PILOT_FAILED", "FULL_FAILED"})
LINEAGE_TERMINAL_STATES = frozenset({*LINEAGE_FAILURE_STATES, "COMPLETED"})


def frozen_run_lock_identity(lock: Mapping[str, Any] | None) -> str:
    """Hash lock fields that must stay fixed from pilot to full.

    Pilot and full locks intentionally have different matrix scopes (12 vs
    108 pairs), and may use separate dataset ids.  Comparing their complete
    ``lock_sha256`` would therefore make the mandated full continuation
    impossible.  This identity binds the model/config/calibration/code and
    seed contract while excluding only scope/dataset bookkeeping fields.
    """

    if not isinstance(lock, Mapping):
        return ""
    excluded = {
        "dataset_id",
        "matrix_sha256",
        "matrix_pairs",
        "matrix_scope",
        "matrix_pair_ids",
        "lock_sha256",
    }
    return stable_sha256(
        {str(key): value for key, value in lock.items() if str(key) not in excluded}
    )


def formal_lineage_state_path(root: Path, lineage_id: str) -> Path:
    """Return the immutable state path for one formal lineage."""

    lineage = _validate_lineage_id(lineage_id)
    return (
        Path(root)
        / "docs"
        / "runtime-evidence"
        / "h6"
        / "formal-lineages"
        / f"{lineage}.json"
    )


def read_formal_lineage_state(root: Path, lineage_id: str) -> dict[str, Any] | None:
    """Read and validate a lineage state, or return ``None`` before first use.

    A malformed existing record is an error rather than an invitation to
    overwrite it.  This is important because deleting/correcting a failed
    formal record would destroy the negative evidence contract.
    """

    path = formal_lineage_state_path(root, lineage_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"h6_vla75_lineage_state_unreadable:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("h6_vla75_lineage_state_not_mapping")
    state = dict(payload)
    lineage = _validate_lineage_id(lineage_id)
    if state.get("schema_version") != LINEAGE_STATE_SCHEMA:
        raise ValueError("h6_vla75_lineage_state_schema_mismatch")
    if str(state.get("lineage_id", "")).lower() != lineage:
        raise ValueError("h6_vla75_lineage_state_lineage_mismatch")
    if tuple(int(item) for item in state.get("formal_seeds", ())) != H6_VLA75_FORMAL_LINEAGES[lineage]:
        raise ValueError("h6_vla75_lineage_state_seed_mismatch")
    if state.get("status") not in LINEAGE_STATES:
        raise ValueError("h6_vla75_lineage_state_status_invalid")
    expected_hash = stable_sha256(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )
    if state.get("state_sha256") != expected_hash:
        raise ValueError("h6_vla75_lineage_state_hash_mismatch")
    history = state.get("history", [])
    if not isinstance(history, list):
        raise ValueError("h6_vla75_lineage_state_history_invalid")
    for event in history:
        if not isinstance(event, Mapping):
            raise ValueError("h6_vla75_lineage_state_history_event_invalid")
        if event.get("lineage_id") not in (None, lineage):
            raise ValueError("h6_vla75_lineage_state_history_lineage_mismatch")
    return state


def assert_formal_lineage_available(
    root: Path,
    lineage_id: str,
    *,
    scope: str,
    run_lock_sha256: str | None = None,
    run_lock_identity: str | None = None,
) -> dict[str, Any] | None:
    """Check that ``scope`` is the next legal operation for a lineage.

    Missing state is the only valid starting point for a pilot.  A full run
    requires a previously recorded passing pilot and the same frozen run-lock
    identity (the pilot/full matrix scope and dataset bookkeeping may differ).
    Terminal success/failure and repeated pilots are never reusable.
    """

    if scope not in {"pilot", "full"}:
        raise ValueError("h6_vla75_lineage_scope_invalid")
    lineage = _validate_lineage_id(lineage_id)
    state = read_formal_lineage_state(root, lineage)
    if state is None:
        if scope == "full":
            raise RuntimeError("h6_vla75_full_requires_passing_pilot")
        return None
    status = str(state["status"])
    if status in LINEAGE_FAILURE_STATES:
        raise RuntimeError(f"h6_vla75_lineage_frozen:{lineage}:{status}")
    if status == "COMPLETED":
        raise RuntimeError(f"h6_vla75_lineage_completed:{lineage}")
    if status != "PILOT_PASSED":
        raise RuntimeError(f"h6_vla75_lineage_state_not_ready:{lineage}:{status}")
    if scope == "pilot":
        raise RuntimeError(f"h6_vla75_pilot_already_passed:{lineage}")
    locked_identity = str(
        state.get("run_lock_identity") or state.get("run_lock_sha256") or ""
    )
    observed_identity = str(run_lock_identity or run_lock_sha256 or "")
    if locked_identity and observed_identity != locked_identity:
        raise RuntimeError("h6_vla75_full_run_lock_mismatch")
    return state


def _write_state(
    path: Path,
    payload: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> None:
    """Persist a state, allowing only the passing-pilot -> full transition.

    Initial pilot records and all terminal records use ``O_EXCL``.  When a
    passing pilot is advanced to full, the current file is atomically replaced
    with a payload that includes the prior event and hash; this is a state
    transition, not deletion of the pilot evidence.  A small exclusive lease
    prevents two full writers from racing.  A stale lease is intentionally
    never removed automatically, because recovering from it by overwriting a
    formal record would be less safe than stopping for inspection.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    next_payload = dict(payload)
    if previous is not None:
        prior_status = str(previous.get("status"))
        if prior_status != "PILOT_PASSED" or next_payload.get("scope") != "full":
            raise RuntimeError(f"h6_vla75_lineage_state_already_exists:{path}")
        prior_event = {
            key: value
            for key, value in previous.items()
            if key not in {"state_sha256", "history"}
        }
        prior_event["state_sha256"] = previous.get("state_sha256")
        next_payload["history"] = [
            *list(previous.get("history") or []),
            prior_event,
        ]
    else:
        next_payload["history"] = []
    next_payload["state_sha256"] = stable_sha256(next_payload)
    encoded = json.dumps(next_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    lease = path.with_name(f".{path.name}.lease")
    descriptor = None
    try:
        try:
            descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise RuntimeError(f"h6_vla75_lineage_state_lease_exists:{lease}") from exc
        if previous is None:
            try:
                descriptor_state = os.open(
                    path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o644,
                )
            except FileExistsError as exc:
                raise RuntimeError(f"h6_vla75_lineage_state_already_exists:{path}") from exc
            with os.fdopen(descriptor_state, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        # If the process dies before this point, the stale lease blocks an
        # unsafe retry and requires inspection.
        if lease.exists():
            lease.unlink()


def record_formal_lineage_result(
    root: Path,
    lineage_id: str,
    *,
    scope: str,
    passed: bool,
    dataset_id: str,
    run_lock_sha256: str | None,
    run_lock_identity: str | None = None,
    evidence_path: str | None = None,
    evidence_sha256: str | None = None,
    gate_result: Mapping[str, Any] | None = None,
    recorded_wall_time_s: float | None = None,
) -> dict[str, Any]:
    """Record exactly one pilot/full result and return the persisted payload.

    A failed pilot/full becomes terminal for that lineage.  A passing pilot is
    the sole non-terminal state and permits exactly one later full run.
    """

    if scope not in {"pilot", "full"}:
        raise ValueError("h6_vla75_lineage_scope_invalid")
    lineage = _validate_lineage_id(lineage_id)
    if not str(dataset_id).startswith("h6-vla75-"):
        raise ValueError("h6_vla75_dataset_id_required")
    previous = assert_formal_lineage_available(
        root,
        lineage,
        scope=scope,
        run_lock_sha256=run_lock_sha256,
        run_lock_identity=run_lock_identity,
    )
    if scope == "full" and previous is None:
        raise RuntimeError("h6_vla75_full_requires_passing_pilot")
    status = (
        "PILOT_PASSED"
        if scope == "pilot" and bool(passed)
        else "PILOT_FAILED"
        if scope == "pilot"
        else "COMPLETED"
        if bool(passed)
        else "FULL_FAILED"
    )
    payload: dict[str, Any] = {
        "schema_version": LINEAGE_STATE_SCHEMA,
        "lineage_id": lineage,
        "formal_seeds": list(H6_VLA75_FORMAL_LINEAGES[lineage]),
        "status": status,
        "scope": scope,
        "passed": bool(passed),
        "dataset_id": str(dataset_id),
        "run_lock_sha256": str(run_lock_sha256 or ""),
        "run_lock_identity": str(run_lock_identity or run_lock_sha256 or ""),
        "evidence_path": str(evidence_path or ""),
        "evidence_sha256": str(evidence_sha256 or ""),
        "gate_result": dict(gate_result or {}),
        "recorded_wall_time_s": float(
            time.time() if recorded_wall_time_s is None else recorded_wall_time_s
        ),
    }
    path = formal_lineage_state_path(root, lineage)
    _write_state(path, payload, previous)
    persisted = read_formal_lineage_state(root, lineage)
    if persisted is None:  # pragma: no cover - guarded by _write_state
        raise RuntimeError("h6_vla75_lineage_state_write_missing")
    return persisted


def all_formal_lineages_failed(root: Path) -> bool:
    """Return true only after immutable failure records exist for A, B and C."""

    states = []
    for lineage in H6_VLA75_FORMAL_LINEAGES:
        state = read_formal_lineage_state(root, lineage)
        if state is None:
            return False
        states.append(str(state.get("status")))
    return all(status in LINEAGE_FAILURE_STATES for status in states)


__all__ = [
    "LINEAGE_FAILURE_STATES",
    "LINEAGE_STATE_SCHEMA",
    "LINEAGE_STATES",
    "LINEAGE_TERMINAL_STATES",
    "all_formal_lineages_failed",
    "assert_formal_lineage_available",
    "formal_lineage_state_path",
    "frozen_run_lock_identity",
    "read_formal_lineage_state",
    "record_formal_lineage_result",
]
