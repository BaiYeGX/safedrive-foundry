"""C2 collection run-lock creation and verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_pipeline.h2.contracts import stable_sha256
from data_pipeline.h2.store import file_sha256

from .config import CORA_C2_CONFIG, CORA_C2_CONFIG_SHA256
from .matrix import CORA_DATA_MATRIX, CORA_FORMAL_MATRIX, CORA_MATRIX_SHA256


RUN_LOCK_SCHEMA = "safedrive.cora.run_lock.v1"
SOURCE_AMENDMENT_SCHEMA = "safedrive.cora.source_amendment.v1"
SOURCE_AMENDMENT_SCOPE = "resume_resource_projection_only"
SOURCE_AMENDMENT_FILES = frozenset(
    {
        "safedrive_foundry/data_pipeline/h6/cora/live.py",
        "safedrive_foundry/data_pipeline/h6/cora/run_lock.py",
        "safedrive_foundry/data_pipeline/h6/cora/store.py",
        "tests/hybrid/test_cora_counterfactual_data.py",
    }
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return (result.stdout or "").strip()


def source_identity(root: Path) -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"], cwd=root, capture_output=True, check=False
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    ).stdout
    included = []
    excluded = []
    excluded_prefixes = ("generated/", "docs/runtime-evidence/")
    for encoded in sorted(item for item in untracked_raw.split(b"\0") if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        if relative == "test_registry.sqlite3" or relative.startswith(excluded_prefixes):
            excluded.append(relative)
            continue
        path = root / relative
        if path.is_file():
            included.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_source": included,
        "untracked_source_sha256": stable_sha256(included),
        "excluded_preexisting_or_outputs": excluded,
    }


def build_run_lock(
    root: Path,
    *,
    environment: Mapping[str, Any],
    model: Mapping[str, Any],
    component_hashes: Mapping[str, str],
    disk: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_LOCK_SCHEMA,
        "dataset_id": CORA_C2_CONFIG["dataset_id"],
        "config_sha256": CORA_C2_CONFIG_SHA256,
        "matrix_sha256": CORA_MATRIX_SHA256,
        "matrix_root_count": len(CORA_DATA_MATRIX),
        "formal_reserved_root_count": len(CORA_FORMAL_MATRIX),
        "splits": dict(CORA_C2_CONFIG["splits"]),
        "resources": dict(CORA_C2_CONFIG["resources"]),
        "source": source_identity(root),
        "environment": dict(environment),
        "model": dict(model),
        "component_hashes": dict(component_hashes),
        "disk": dict(disk),
        "formal_collected": False,
        "c3_authorized": False,
    }
    payload["run_lock_sha256"] = stable_sha256(payload)
    return payload


def verify_run_lock(payload: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    failures = []
    data = dict(payload)
    expected = data.pop("run_lock_sha256", None)
    if expected != stable_sha256(data):
        failures.append("run_lock_self_hash")
    if data.get("schema_version") != RUN_LOCK_SCHEMA:
        failures.append("run_lock_schema")
    if data.get("dataset_id") != CORA_C2_CONFIG["dataset_id"]:
        failures.append("run_lock_dataset")
    if data.get("config_sha256") != CORA_C2_CONFIG_SHA256:
        failures.append("run_lock_config")
    if data.get("matrix_sha256") != CORA_MATRIX_SHA256:
        failures.append("run_lock_matrix")
    if int(data.get("matrix_root_count", -1)) != 351:
        failures.append("run_lock_root_count")
    if int(data.get("formal_reserved_root_count", -1)) != 108:
        failures.append("run_lock_formal_count")
    if bool(data.get("formal_collected")):
        failures.append("run_lock_formal_collected")
    source = data.get("source")
    if not isinstance(source, Mapping) or "test_registry.sqlite3" in {
        str(item.get("path")) for item in source.get("untracked_source", ()) if isinstance(item, Mapping)
    }:
        failures.append("run_lock_source_identity")
    for name in ("environment", "model", "component_hashes", "disk"):
        if not isinstance(data.get(name), Mapping) or not data.get(name):
            failures.append(f"run_lock_{name}")
    return not failures, tuple(failures)


def verify_source_amendment(
    root: Path,
    run_lock: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    """Verify a narrowly scoped, evidence-bound post-freeze source repair.

    C2 permits evidence-driven code repair followed by resume under the same
    immutable data run-lock.  The exception is fail-closed: it binds both the
    original and current complete source identities, the original run-lock,
    and exact hashes for a small allow-list of implementation/test files.
    """

    failures: list[str] = []
    data = dict(amendment)
    expected = data.pop("evidence_sha256", None)
    if expected != stable_sha256(data):
        failures.append("source_amendment_self_hash")
    if data.get("schema_version") != SOURCE_AMENDMENT_SCHEMA:
        failures.append("source_amendment_schema")
    if data.get("authorized_scope") != SOURCE_AMENDMENT_SCOPE:
        failures.append("source_amendment_scope")
    if data.get("run_lock_sha256") != run_lock.get("run_lock_sha256"):
        failures.append("source_amendment_run_lock")

    locked_source = run_lock.get("source")
    if not isinstance(locked_source, Mapping):
        failures.append("source_amendment_locked_source")
    elif data.get("base_source_identity_sha256") != stable_sha256(locked_source):
        failures.append("source_amendment_base_identity")

    current_source = source_identity(root)
    if data.get("amended_source_identity_sha256") != stable_sha256(current_source):
        failures.append("source_amendment_current_identity")

    files = data.get("files")
    if not isinstance(files, Mapping) or not files:
        failures.append("source_amendment_files")
        files = {}
    declared = {str(path) for path in files}
    if not declared.issubset(SOURCE_AMENDMENT_FILES):
        failures.append("source_amendment_file_allowlist")
    for relative, identity in files.items():
        relative = str(relative)
        if not isinstance(identity, Mapping):
            failures.append(f"source_amendment_file_identity:{relative}")
            continue
        before = str(identity.get("before_sha256", ""))
        after = str(identity.get("after_sha256", ""))
        if len(before) != 64 or len(after) != 64 or before == after:
            failures.append(f"source_amendment_file_hash_pair:{relative}")
            continue
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            failures.append(f"source_amendment_file_escape:{relative}")
            continue
        if not path.is_file() or file_sha256(path) != after:
            failures.append(f"source_amendment_file_current_hash:{relative}")
    return not failures, tuple(failures)


__all__ = [
    "RUN_LOCK_SCHEMA",
    "SOURCE_AMENDMENT_FILES",
    "SOURCE_AMENDMENT_SCHEMA",
    "SOURCE_AMENDMENT_SCOPE",
    "build_run_lock",
    "source_identity",
    "verify_run_lock",
    "verify_source_amendment",
]
