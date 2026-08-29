"""Immutable H6 VLA75 run-lock and scoped provenance helpers.

The lock deliberately hashes runtime/source/config/checkpoint inputs but not
the documents that are updated after a run.  A full worktree identity is still
recorded separately so an Evidence reader can recover the exact dirty tree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from data_pipeline.h3.contracts import stable_sha256
from data_pipeline.h6.config import (
    H6_VLA75_FORMAL_LINEAGES,
    build_h6_vla75_config,
    h6_vla75_config_sha256,
)
from data_pipeline.h6.matrix import h6_vla75_matrix_sha256, load_h6_vla75_matrix


RUN_LOCK_SCHEMA_V1 = "safedrive.h6.vla75.run_lock.v1"
RUN_LOCK_SCHEMA_V2 = "safedrive.h6.vla75.run_lock.v2"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scoped_file_hashes(root: Path, paths: Sequence[str | Path]) -> dict[str, str]:
    """Hash exactly the supplied runtime/source paths, excluding documents."""

    output: dict[str, str] = {}
    for item in sorted({str(path) for path in paths}):
        path = root / item
        if not path.is_file():
            raise FileNotFoundError(f"run_lock_scoped_file_missing:{item}")
        output[item] = _file_sha256(path)
    return output


def checkpoint_hashes(paths: Sequence[str | Path]) -> dict[str, str]:
    return {
        str(path): _file_sha256(Path(path))
        for path in sorted({str(item) for item in paths})
    }


def verify_summary_checkpoints_against_lock(
    summary_models: object,
    lock: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Bind a training summary's ordered ensemble to an immutable run lock.

    ``verify_run_lock`` checks the lock itself and its files.  This companion
    check closes the other side of the contract: a collector/readiness caller
    must pass the same ordered checkpoint files that the lock hashed, and each
    summary row must carry the matching digest.  Returning structured failures
    lets readiness report all mismatches, while collectors can fail closed with
    one stable error token.
    """

    failures: list[str] = []
    if not isinstance(summary_models, list):
        return {"valid": False, "failures": ["summary_models_missing"]}
    locked_order = lock.get("checkpoint_order")
    if not isinstance(locked_order, Sequence) or isinstance(locked_order, (str, bytes)):
        return {"valid": False, "failures": ["checkpoint_order_missing"]}
    if len(summary_models) != len(locked_order):
        failures.append("checkpoint_count_mismatch")
    summary_entries: list[tuple[Mapping[str, Any], Path]] = []
    for index, item in enumerate(summary_models):
        if not isinstance(item, Mapping) or "checkpoint_path" not in item:
            failures.append(f"summary_checkpoint_path_missing:{index}")
            continue
        raw_path = Path(str(item["checkpoint_path"]))
        path = raw_path if raw_path.is_absolute() else root / raw_path
        summary_entries.append((item, path))
    summary_paths = [path for _item, path in summary_entries]
    locked_paths = []
    for raw_path in locked_order:
        path = Path(str(raw_path))
        locked_paths.append(path if path.is_absolute() else root / path)
    if len(summary_paths) == len(locked_paths) and [
        path.resolve() for path in summary_paths
    ] != [path.resolve() for path in locked_paths]:
        failures.append("checkpoint_order_mismatch")
    observed_hashes: list[str] = []
    for item, path in summary_entries:
        if not path.is_file():
            failures.append(f"summary_checkpoint_missing:{path}")
            continue
        actual = _file_sha256(path)
        observed_hashes.append(actual)
        if item.get("checkpoint_sha256") != actual:
            failures.append(f"summary_checkpoint_hash:{path}")
    if len(observed_hashes) == len(summary_paths) == len(locked_paths):
        model_hash = hashlib.sha256("|".join(observed_hashes).encode("ascii")).hexdigest()
        if model_hash != lock.get("model_hash"):
            failures.append("model_hash_mismatch")
    return {"valid": not failures, "failures": failures}


def worktree_identity(root: Path) -> dict[str, Any]:
    """Capture HEAD/branch, binary diff hash and untracked manifest hash."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return (result.stdout or "").strip()

    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=root,
        capture_output=True,
        check=False,
    ).stdout
    raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    ).stdout
    files = []
    for encoded in sorted(item for item in raw.split(b"\0") if item):
        relative = encoded.decode("utf-8", errors="surrogateescape")
        # Local unittest discovery state is neither a runtime input nor
        # reproducibility evidence and is explicitly excluded from C1 locks.
        if relative == "test_registry.sqlite3":
            continue
        path = root / relative
        if path.is_file():
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "worktree_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_manifest_sha256": stable_sha256(files),
        "untracked_files": files,
        "full_worktree_hash": stable_sha256(
            {
                "head": git("rev-parse", "HEAD"),
                "diff_sha256": hashlib.sha256(diff).hexdigest(),
                "untracked": files,
            }
        ),
    }


def build_h6_vla75_run_lock(
    root: Path,
    *,
    lineage_id: str,
    dataset_id: str,
    matrix_rows: Sequence[Any],
    checkpoint_paths: Sequence[str | Path],
    calibration: Mapping[str, Any],
    training_roots: Sequence[str | Path] = (),
    seeds: Mapping[str, Sequence[int]] | None = None,
    versions: Mapping[str, Any] | None = None,
    scoped_paths: Sequence[str | Path] | None = None,
    worktree: Mapping[str, Any] | None = None,
    schema_version: str = RUN_LOCK_SCHEMA_V2,
) -> dict[str, Any]:
    """Create a lock payload; call :func:`write_run_lock` to persist it."""

    lineage = str(lineage_id).lower()
    if schema_version not in {RUN_LOCK_SCHEMA_V1, RUN_LOCK_SCHEMA_V2}:
        raise ValueError(f"unsupported_h6_vla75_run_lock_schema:{schema_version}")
    if lineage not in H6_VLA75_FORMAL_LINEAGES:
        raise ValueError(f"unknown_h6_vla75_formal_lineage:{lineage_id}")
    config = build_h6_vla75_config(lineage)
    if not str(dataset_id).startswith("h6-vla75-"):
        raise ValueError("h6_vla75_dataset_id_required")
    if len(tuple(checkpoint_paths)) != 3:
        raise ValueError("h6_vla75_requires_three_checkpoints")
    if len(tuple(matrix_rows)) not in {12, 108}:
        raise ValueError("h6_vla75_matrix_cardinality")
    # Normalize the explicit ensemble order once.  The manifest is keyed by
    # path for human/audit readability, while ``checkpoint_order`` preserves
    # the exact order used by the runtime model hash.  Formal acceptance can
    # therefore detect a checkpoint reorder as well as a byte mutation.
    normalized_checkpoint_paths = tuple(
        Path(path) if Path(path).is_absolute() else root / Path(path)
        for path in checkpoint_paths
    )
    checkpoint_digest = {
        str(path): _file_sha256(path)
        for path in sorted(normalized_checkpoint_paths, key=lambda item: str(item))
    }
    ordered_checkpoint_hashes = [_file_sha256(path) for path in normalized_checkpoint_paths]
    default_scoped = (
        "safedrive_foundry/data_pipeline/h6/__init__.py",
        "safedrive_foundry/data_pipeline/h6/config.py",
        "safedrive_foundry/data_pipeline/h6/contracts.py",
        "safedrive_foundry/data_pipeline/h6/acceptance.py",
        "safedrive_foundry/data_pipeline/h6/calibration.py",
        "safedrive_foundry/data_pipeline/h6/dataset.py",
        "safedrive_foundry/data_pipeline/h6/evaluator.py",
        "safedrive_foundry/data_pipeline/h6/matrix.py",
        "safedrive_foundry/data_pipeline/h6/model.py",
        "safedrive_foundry/data_pipeline/h6/runtime.py",
        "safedrive_foundry/data_pipeline/h6/temporal.py",
        "safedrive_foundry/data_pipeline/h6/run_lock.py",
        "safedrive_foundry/data_pipeline/h6/lineage.py",
        "safedrive_foundry/data_pipeline/h5/config.py",
        "safedrive_foundry/data_pipeline/h5/contracts.py",
        "safedrive_foundry/data_pipeline/h5/runtime.py",
        "scripts/h5_collect.py",
        "scripts/h6_collect.py",
        "scripts/h6_acceptance.py",
        "scripts/h6_readiness.py",
        "scripts/h6_run.py",
        "scripts/h6_run_lock.py",
        "scripts/h6_retrain.py",
        "scripts/train_world_v3.py",
        "safedrive_foundry/data_pipeline/h3/live_features.py",
        "safedrive_foundry/data_pipeline/h3/contracts.py",
        "safedrive_foundry/driving_vla/hybrid/contracts.py",
        "safedrive_foundry/driving_vla/hybrid/pipeline.py",
        "safedrive_foundry/driving_vla/hybrid/guard.py",
        "safedrive_foundry/driving_vla/hybrid/router.py",
        "safedrive_foundry/driving_vla/hybrid/generators.py",
        "safedrive_foundry/driving_vla/hybrid/carla_anchor.py",
        "safedrive_foundry/driving_vla/hybrid/vla_smoother.py",
        "safedrive_foundry/driving_vla/hybrid/__init__.py",
        "safedrive_foundry/driving_vla/model/canonicalizer.py",
        "safedrive_foundry/driving_vla/model/lineage.py",
        "safedrive_foundry/driving_vla/model/nominal_policy.py",
        "safedrive_foundry/driving_vla/model/simlingo_runtime.py",
        "safedrive_foundry/driving_vla/adapter/policy_adapter.py",
        "safedrive_foundry/driving_vla/schema/trajectory_contract.py",
        "safedrive_foundry/driving_vla/runtime/safety_control_bind.py",
        "safedrive_foundry/classic_stack/planning/frenet/__init__.py",
        "safedrive_foundry/classic_stack/planning/frenet/planner.py",
        "safedrive_foundry/classic_stack/planning/frenet/config.py",
        "safedrive_foundry/classic_stack/planning/speed/__init__.py",
        "safedrive_foundry/classic_stack/planning/speed/prediction.py",
        "safedrive_foundry/classic_stack/planning/speed/st_dp.py",
        "safedrive_foundry/classic_stack/geometry/__init__.py",
        "safedrive_foundry/classic_stack/geometry/frenet_frame.py",
        "safedrive_foundry/classic_stack/geometry/vehicle.py",
        "safedrive_foundry/classic_stack/control/controller.py",
        "safedrive_foundry/classic_stack/control/config.py",
        "safedrive_foundry/safety_kernel/arbitration/pipeline.py",
        "safedrive_foundry/safety_kernel/arbitration/types.py",
        "safedrive_foundry/safety_kernel/arbitration/degradation.py",
        "safedrive_foundry/safety_kernel/config.py",
        "safedrive_foundry/safety_kernel/kernel.py",
        "safedrive_foundry/safety_kernel/repair/corridor.py",
        "safedrive_foundry/safety_kernel/repair/longitudinal_qp.py",
        "safedrive_foundry/safety_kernel/repair/rato_scp.py",
        "safedrive_foundry/safety_kernel/validator/checks.py",
        "safedrive_foundry/safety_kernel/validator/engine.py",
        "safedrive_foundry/safety_kernel/contracts/types.py",
        "safedrive_foundry/safety_kernel/contracts/serialize.py",
    )
    lock = {
        "schema_version": schema_version,
        "contract": "vla75-v2",
        "lineage_id": lineage,
        "formal_seeds": list(H6_VLA75_FORMAL_LINEAGES[lineage]),
        "dataset_id": str(dataset_id),
        "config": config,
        "config_sha256": h6_vla75_config_sha256(lineage),
        "matrix_sha256": h6_vla75_matrix_sha256(lineage, matrix_rows),
        "matrix_pairs": len(matrix_rows),
        "matrix_scope": "full" if len(matrix_rows) == 108 else "pilot",
        "matrix_pair_ids": sorted(str(row.pair_id) for row in matrix_rows),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_order": [str(path) for path in normalized_checkpoint_paths],
        "ensemble_sha256": stable_sha256(checkpoint_digest),
        # Runtime scorers retain the caller's ensemble order when deriving
        # their model hash.  Record that exact order in addition to the
        # order-independent checkpoint manifest hash above.
        "model_hash": hashlib.sha256(
            "|".join(ordered_checkpoint_hashes).encode("ascii")
        ).hexdigest(),
        "calibration": dict(calibration),
        "training_roots": [str(Path(item)) for item in training_roots],
        "seeds": {
            "development": [89, 97],
            **{str(key): [int(value) for value in values] for key, values in (seeds or {}).items()},
        },
        "versions": dict(versions or {}),
        "feature_schema": "safedrive.h3.world_scorer.v2",
        "scoped_runtime_sha256": scoped_file_hashes(
            root, default_scoped if scoped_paths is None else scoped_paths
        ),
        "worktree": dict(worktree or worktree_identity(root)),
    }
    lock["lock_sha256"] = stable_sha256(lock)
    return lock


def write_run_lock(path: Path, lock: Mapping[str, Any]) -> str:
    """Atomically persist a lock and return its stable payload hash."""

    payload = dict(lock)
    expected = stable_sha256({key: value for key, value in payload.items() if key != "lock_sha256"})
    if payload.get("lock_sha256") not in (None, expected):
        raise ValueError("run_lock_hash_mismatch_before_write")
    payload["lock_sha256"] = expected
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return expected


def verify_run_lock(lock: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Verify lock self-hash and optionally every scoped source/checkpoint hash."""

    payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    expected = stable_sha256(payload)
    failures: list[str] = []
    if lock.get("lock_sha256") != expected:
        failures.append("lock_hash")
    schema_version = lock.get("schema_version")
    if schema_version not in {RUN_LOCK_SCHEMA_V1, RUN_LOCK_SCHEMA_V2}:
        failures.append("schema_version")
    calibration = lock.get("calibration")
    if not isinstance(calibration, Mapping):
        failures.append("calibration")
    elif schema_version == RUN_LOCK_SCHEMA_V2:
        deployment = calibration.get("deployment")
        bindings = deployment.get("c1_bindings") if isinstance(deployment, Mapping) else None
        if not isinstance(bindings, Mapping):
            failures.append("c1_calibration_bindings_missing")
        else:
            evaluator_hashes = bindings.get("evaluator_sha256")
            if (
                not isinstance(evaluator_hashes, list)
                or len(evaluator_hashes) != 3
                or any(
                    not isinstance(value, str) or len(value) != 64
                    for value in evaluator_hashes
                )
            ):
                failures.append("c1_evaluator_bindings")
            for name in ("validation_lineage_sha256", "training_input_sha256"):
                value = bindings.get(name)
                if not isinstance(value, str) or len(value) != 64:
                    failures.append(f"c1_binding:{name}")
    lineage = str(lock.get("lineage_id") or "").lower()
    if lineage not in H6_VLA75_FORMAL_LINEAGES:
        failures.append("lineage")
    else:
        expected_config = h6_vla75_config_sha256(lineage)
        if lock.get("config_sha256") != expected_config:
            failures.append("config_hash")
        config = lock.get("config")
        if not isinstance(config, Mapping) or stable_sha256(dict(config)) != expected_config:
            failures.append("config_payload")
        expected_seeds = list(H6_VLA75_FORMAL_LINEAGES[lineage])
        if [int(item) for item in lock.get("formal_seeds", ())] != expected_seeds:
            failures.append("formal_seeds")
        seed_manifest = lock.get("seeds")
        if not isinstance(seed_manifest, Mapping):
            failures.append("seed_manifest")
        else:
            development = seed_manifest.get("development")
            try:
                development_values = [int(item) for item in (development or ())]
            except (TypeError, ValueError):
                development_values = []
                failures.append("seed_manifest_values:development")
            if development_values != [89, 97]:
                failures.append("development_seeds")
            forbidden = {101, *expected_seeds}
            for name, values in seed_manifest.items():
                if name in {"formal", "formal_pilot", "formal_full"}:
                    continue
                try:
                    if forbidden.intersection(int(item) for item in (values or ())):
                        failures.append(f"seed_isolation:{name}")
                except (TypeError, ValueError):
                    failures.append(f"seed_manifest_values:{name}")
            for key in ("formal", "formal_pilot", "formal_full"):
                if key in seed_manifest:
                    try:
                        observed_formal = [int(item) for item in seed_manifest[key]]
                    except (TypeError, ValueError):
                        failures.append(f"seed_manifest_values:{key}")
                    else:
                        if key == "formal_pilot" and observed_formal != [expected_seeds[0]]:
                            failures.append("formal_pilot_seed")
                        if key == "formal_full" and observed_formal != expected_seeds:
                            failures.append("formal_full_seeds")
                        if key == "formal" and observed_formal != expected_seeds:
                            failures.append("formal_seeds_manifest")
    if not str(lock.get("dataset_id") or "").startswith("h6-vla75-"):
        failures.append("dataset_id")
    if int(lock.get("matrix_pairs", 0)) not in {12, 108}:
        failures.append("matrix_cardinality")
    elif lineage in H6_VLA75_FORMAL_LINEAGES:
        expected_scope = "full" if int(lock.get("matrix_pairs", 0)) == 108 else "pilot"
        if lock.get("matrix_scope") not in {None, expected_scope}:
            failures.append("matrix_scope")
        # The matrix hash is not just a self-consistency checksum: compare it
        # with the immutable pre-registered lineage matrix.  Otherwise a
        # caller could rewrite both ``matrix_pair_ids`` and ``matrix_sha256``
        # and still present a self-hashed lock for a different set of rows.
        expected_rows = load_h6_vla75_matrix(
            lineage, full=int(lock.get("matrix_pairs", 0)) == 108
        )
        expected_matrix_hash = h6_vla75_matrix_sha256(lineage, expected_rows)
        if lock.get("matrix_sha256") != expected_matrix_hash:
            failures.append("matrix_hash")
        expected_pair_ids = sorted(str(row.pair_id) for row in expected_rows)
        if [str(item) for item in (lock.get("matrix_pair_ids") or ())] != expected_pair_ids:
            failures.append("matrix_pair_ids_content")
    pair_ids = lock.get("matrix_pair_ids")
    if pair_ids is not None:
        if not isinstance(pair_ids, Sequence) or len(pair_ids) != int(lock.get("matrix_pairs", 0)):
            failures.append("matrix_pair_ids")
        elif len(set(str(item) for item in pair_ids)) != len(pair_ids):
            failures.append("matrix_pair_ids_duplicate")

    checkpoint_map = lock.get("checkpoint_sha256")
    if not isinstance(checkpoint_map, Mapping) or len(checkpoint_map) != 3:
        failures.append("checkpoint_manifest")
    else:
        observed_checkpoint_hashes: dict[str, str] = {}
        for raw_path, digest in sorted(checkpoint_map.items(), key=lambda item: str(item[0])):
            path = Path(str(raw_path))
            if not path.is_absolute() and root is not None:
                path = root / path
            if not path.is_file():
                failures.append(f"checkpoint_missing:{raw_path}")
                continue
            actual = _file_sha256(path)
            observed_checkpoint_hashes[str(raw_path)] = actual
            if actual != digest:
                failures.append(f"checkpoint_hash:{raw_path}")
        if stable_sha256(observed_checkpoint_hashes) != lock.get("ensemble_sha256"):
            failures.append("ensemble_hash")
        checkpoint_order = lock.get("checkpoint_order")
        if not isinstance(checkpoint_order, Sequence) or isinstance(checkpoint_order, (str, bytes)):
            failures.append("checkpoint_order")
        elif len(checkpoint_order) != len(checkpoint_map) or set(str(item) for item in checkpoint_order) != set(str(item) for item in checkpoint_map):
            failures.append("checkpoint_order_content")
        else:
            ordered_hashes: list[str] = []
            for raw_path in checkpoint_order:
                key = str(raw_path)
                if key not in observed_checkpoint_hashes:
                    failures.append(f"checkpoint_order_missing:{key}")
                    continue
                ordered_hashes.append(observed_checkpoint_hashes[key])
            if len(ordered_hashes) == len(checkpoint_map):
                expected_model_hash = hashlib.sha256(
                    "|".join(ordered_hashes).encode("ascii")
                ).hexdigest()
                if lock.get("model_hash") != expected_model_hash:
                    failures.append("model_hash")
    if root is not None:
        for relative, digest in dict(lock.get("scoped_runtime_sha256") or {}).items():
            path = root / relative
            if not path.is_file():
                failures.append(f"scoped_missing:{relative}")
            elif _file_sha256(path) != digest:
                failures.append(f"scoped_hash:{relative}")
    worktree = lock.get("worktree")
    if isinstance(worktree, Mapping) and worktree.get("full_worktree_hash"):
        expected_worktree = stable_sha256(
            {
                "head": worktree.get("head", worktree.get("commit", "")),
                "diff_sha256": worktree.get("worktree_diff_sha256", ""),
                "untracked": list(worktree.get("untracked_files", ())),
            }
        )
        if expected_worktree != worktree.get("full_worktree_hash"):
            failures.append("worktree_hash")
    return {
        "valid": not failures,
        "failures": failures,
        "lock_sha256": lock.get("lock_sha256"),
    }


__all__ = [
    "build_h6_vla75_run_lock",
    "checkpoint_hashes",
    "scoped_file_hashes",
    "verify_run_lock",
    "verify_summary_checkpoints_against_lock",
    "worktree_identity",
    "write_run_lock",
]
