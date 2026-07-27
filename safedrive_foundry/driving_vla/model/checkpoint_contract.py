"""Checkpoint use-case contract (R2-X A2).

Fail-closed before torch.load / CARLA preflight for formal uses.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class CheckpointUse(str, Enum):
    HISTORICAL_COMPARISON = "historical_comparison"
    OFFLINE_DIAGNOSTIC = "offline_diagnostic"
    DEVELOPMENT_LIVE_SMOKE = "development_live_smoke"
    FORMAL_OFFLINE = "formal_offline"
    X5H_ACCEPTANCE = "x5h_acceptance"
    R2K_PILOT = "r2k_pilot"


FORMAL_USES = frozenset(
    {
        CheckpointUse.FORMAL_OFFLINE,
        CheckpointUse.X5H_ACCEPTANCE,
        CheckpointUse.R2K_PILOT,
    }
)

STATUS_BOOTSTRAP_INVALID = "BOOTSTRAP_INVALID_TEACHER"
STATUS_OK = "OK"
STATUS_HISTORICAL = "HISTORICAL_ONLY"
STATUS_TRAINED_NOT_FORMAL = "HEAD_TRAINED_NOT_FORMAL"

DEFAULT_STATUS_NAMES = (
    "CHECKPOINT_STATUS.json",
    "checkpoint_status.json",
)


class CheckpointContractError(RuntimeError):
    """Fail-closed checkpoint use rejection."""


def file_sha256(path: Path | str, *, full: bool = True) -> str:
    p = Path(path)
    if not p.is_file():
        raise CheckpointContractError(f"checkpoint_file_missing:{p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return digest if full else digest[:16]


def _find_manifest_path(checkpoint_path: Path) -> Path | None:
    # same dir as pt
    for name in DEFAULT_STATUS_NAMES:
        cand = checkpoint_path.parent / name
        if cand.is_file():
            return cand
    # parent checkpoints root
    root = checkpoint_path.parent.parent if checkpoint_path.parent.name else checkpoint_path.parent
    for name in DEFAULT_STATUS_NAMES:
        cand = root / name
        if cand.is_file():
            return cand
    return None


def load_checkpoint_manifest(
    checkpoint_path: Path | str | None = None,
    *,
    manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load status manifest next to checkpoint or explicit path."""
    if manifest_path is not None:
        mp = Path(manifest_path)
        if not mp.is_file():
            raise CheckpointContractError(f"manifest_missing:{mp}")
        return json.loads(mp.read_text(encoding="utf-8"))
    if checkpoint_path is None:
        raise CheckpointContractError("checkpoint_path_or_manifest_required")
    cp = Path(checkpoint_path)
    mp2 = _find_manifest_path(cp)
    if mp2 is None:
        raise CheckpointContractError(
            f"checkpoint_status_manifest_missing_for:{cp.as_posix()}"
        )
    return json.loads(mp2.read_text(encoding="utf-8"))


def validate_checkpoint_for_use(
    checkpoint_path: Path | str,
    use: CheckpointUse | str,
    *,
    manifest: Mapping[str, Any] | None = None,
    require_file: bool = True,
) -> dict[str, Any]:
    """Validate checkpoint may be used for ``use``. Does not torch.load."""
    use_e = CheckpointUse(use) if not isinstance(use, CheckpointUse) else use
    cp = Path(checkpoint_path)
    if require_file and not cp.is_file():
        raise CheckpointContractError(f"checkpoint_file_missing:{cp}")

    man = dict(manifest) if manifest is not None else load_checkpoint_manifest(cp)
    status = str(man.get("status") or "").strip()
    sha_expected = str(man.get("checkpoint_sha256") or man.get("sha256") or "").strip()
    if not sha_expected:
        raise CheckpointContractError(
            "manifest_missing_checkpoint_sha256_binding"
        )
    if cp.is_file():
        sha_actual = file_sha256(cp, full=True)
        if sha_actual.lower() != sha_expected.lower():
            raise CheckpointContractError(
                f"checkpoint_sha256_mismatch:expected={sha_expected[:16]}…:"
                f"actual={sha_actual[:16]}…"
            )

    allowed = {str(x) for x in (man.get("allowed_uses") or [])}
    forbidden = {str(x) for x in (man.get("forbidden_uses") or [])}
    use_s = use_e.value

    if use_e in FORMAL_USES:
        if status == STATUS_BOOTSTRAP_INVALID:
            raise CheckpointContractError(
                f"checkpoint_status_blocks_use:{status}:{use_s}"
            )
        if status != STATUS_OK:
            raise CheckpointContractError(
                f"checkpoint_not_formal_ok:{status}:{use_s}"
            )
        if use_s in forbidden or "formal_offline" in forbidden:
            raise CheckpointContractError(
                f"checkpoint_forbidden_for_use:{status}:{use_s}"
            )
        if allowed and use_s not in allowed and "formal_offline" not in allowed:
            raise CheckpointContractError(
                f"checkpoint_not_allowed_use:{use_s}:allowed={sorted(allowed)}"
            )
    elif use_e == CheckpointUse.HISTORICAL_COMPARISON:
        # allowed when hash binds; optional allow list
        if allowed and "historical_comparison" not in allowed and "historical_comparison_only" not in allowed:
            if status == STATUS_BOOTSTRAP_INVALID:
                pass  # still allow historical on invalid bootstrap by default
    elif use_e == CheckpointUse.OFFLINE_DIAGNOSTIC:
        if status == STATUS_BOOTSTRAP_INVALID:
            if (
                "offline_diagnostic" not in allowed
                and "code_path_smoke" not in allowed
            ):
                raise CheckpointContractError(
                    f"checkpoint_blocks_offline_diagnostic:{status}"
                )
    elif use_e == CheckpointUse.DEVELOPMENT_LIVE_SMOKE:
        if status == STATUS_BOOTSTRAP_INVALID:
            raise CheckpointContractError(
                f"checkpoint_status_blocks_use:{status}:{use_s}"
            )
        if use_s not in allowed:
            raise CheckpointContractError(
                f"checkpoint_not_allowed_use:{use_s}:allowed={sorted(allowed)}"
            )
    if use_s in forbidden and use_e in FORMAL_USES:
        raise CheckpointContractError(f"checkpoint_forbidden_for_use:{use_s}")

    return {
        "ok": True,
        "use": use_s,
        "status": status,
        "checkpoint_sha256": sha_expected,
        "path": str(cp.as_posix()),
    }


def require_checkpoint_for_use(
    checkpoint_path: Path | str,
    use: CheckpointUse | str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Alias of validate; raises CheckpointContractError on failure."""
    return validate_checkpoint_for_use(checkpoint_path, use, **kwargs)


def require_checkpoint_blind_registry(
    checkpoint_path: Path | str,
    expected_registry_sha256: str,
) -> dict[str, Any]:
    """Require a formal checkpoint to be bound to this exact blind registry."""
    man = load_checkpoint_manifest(checkpoint_path)
    bound = str(man.get("blind_registry_sha256") or "").strip()
    expected = str(expected_registry_sha256 or "").strip()
    if not bound:
        raise CheckpointContractError("checkpoint_missing_blind_registry_binding")
    if not expected or bound.lower() != expected.lower():
        raise CheckpointContractError(
            "checkpoint_blind_registry_mismatch:"
            f"bound={bound[:16]}…:expected={expected[:16]}…"
        )
    if not bool(man.get("blind_pair_overlap_zero")):
        raise CheckpointContractError("checkpoint_blind_overlap_audit_not_ok")
    return {
        "ok": True,
        "blind_registry_sha256": bound,
        "blind_registry_version": str(man.get("blind_registry_version") or ""),
    }


def write_checkpoint_manifest(
    path: Path | str,
    *,
    checkpoint_path: Path | str,
    status: str,
    allowed_uses: list[str],
    forbidden_uses: list[str],
    reasons: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cp = Path(checkpoint_path)
    sha = file_sha256(cp, full=True) if cp.is_file() else ""
    payload = {
        "schema_version": "safedrive.r2x.checkpoint_status.v2",
        "checkpoint": str(cp.as_posix()),
        "checkpoint_sha256": sha,
        "status": status,
        "allowed_uses": list(allowed_uses),
        "forbidden_uses": list(forbidden_uses),
        "reasons": list(reasons or []),
        "retain": True,
        "delete": False,
    }
    if extra:
        payload.update(dict(extra))
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
