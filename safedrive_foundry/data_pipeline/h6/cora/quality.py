"""C2 pilot/development integrity, coverage and leakage gates."""

from __future__ import annotations

import collections
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data_pipeline.h2.contracts import stable_sha256

from .config import CORA_C2_CONFIG, CORA_C2_CONFIG_SHA256
from .feature import build_cora_feature_view, validate_cora_feature_view
from .matrix import CORA_DATA_MATRIX, CORA_MATRIX_SHA256
from .outcomes import (
    PUBLIC_OUTCOME_HEADS,
    materialize_public_labels,
    read_public_label,
)
from .store import CoraDataStore


QUALITY_SCHEMA = "safedrive.cora.data_quality.v1"
ALL_OPERATORS = frozenset(
    {
        "speed_scale_up",
        "delayed_brake",
        "shortened_stopping_margin",
        "stop_line_crossing",
        "lateral_offset_toward_conflict",
        "curvature_bump",
        "obstacle_envelope_approach",
    }
)
DEVELOPMENT_SPLITS = ("train", "validation", "calibration", "locked_development")


def _branch_valid(branch: Mapping[str, Any]) -> bool:
    return bool(branch.get("outcome_valid", False))


def _head_value(branch: Mapping[str, Any], key: str) -> Any:
    item = branch.get("heads", {}).get(key)
    if not isinstance(item, Mapping) or not bool(item.get("valid")):
        return None
    return item.get("value")


def _trajectory_vector(proposal: Mapping[str, Any]) -> tuple[float, ...] | None:
    rows = proposal.get("trajectory")
    if not isinstance(rows, Sequence) or len(rows) != 10:
        return None
    try:
        first_x, first_y = float(rows[0]["x"]), float(rows[0]["y"])
        output = []
        for row in rows:
            output.extend(
                (
                    float(row["x"]) - first_x,
                    float(row["y"]) - first_y,
                    float(row["v"]),
                    float(row["a"]),
                    float(row["kappa"]),
                )
            )
        if not all(math.isfinite(value) for value in output):
            return None
        return tuple(output)
    except (KeyError, TypeError, ValueError):
        return None


def trajectory_source_probe(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    train: dict[str, list[tuple[float, ...]]] = {"expert": [], "vla": []}
    test: list[tuple[str, tuple[float, ...]]] = []
    for record in records:
        root_id = str(record.get("root_id", ""))
        held_out = int(stable_sha256({"trajectory_source_probe": root_id}), 16) % 3 == 0
        for proposal in record.get("proposals", ()):
            if proposal.get("kind") != "nominal":
                continue
            source = str(proposal.get("audit_source", ""))
            vector = _trajectory_vector(proposal)
            if source not in train or vector is None:
                continue
            if held_out:
                test.append((source, vector))
            else:
                train[source].append(vector)
    if not test or any(not rows for rows in train.values()):
        return {"status": "NOT_MEASURED", "count": 0, "accuracy": None}
    centroids = {
        source: tuple(sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0])))
        for source, rows in train.items()
    }
    correct = 0
    for source, vector in test:
        distances = {
            key: sum((left - right) ** 2 for left, right in zip(vector, centroid))
            for key, centroid in centroids.items()
        }
        predicted = min(distances, key=lambda key: (distances[key], key))
        correct += predicted == source
    return {
        "status": "MEASURED",
        "count": len(test),
        "correct": correct,
        "accuracy": correct / len(test),
        "interpretation": "diagnostic_only_trajectory_style_is_not_metadata_leakage",
    }


def _read_feature_audit(store: CoraDataStore, records: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    failures = []
    for record in records:
        root_id = str(record.get("root_id", ""))
        try:
            anchor = json.loads(
                (store.root / str(record.get("anchor_path", ""))).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{root_id}:anchor:{type(exc).__name__}:{exc}")
            continue
        proposal_by_digest = {
            str(proposal.get("proposal_sha256")): proposal
            for proposal in record.get("proposals", ())
        }
        for key, relative in record.get("feature_paths", {}).items():
            path = store.root / str(relative)
            try:
                feature = json.loads(path.read_text(encoding="utf-8"))
                validate_cora_feature_view(feature)
                proposal = proposal_by_digest.get(str(key))
                if proposal is None:
                    raise ValueError("feature_proposal_missing")
                rebuilt = build_cora_feature_view(anchor, proposal.get("trajectory", ()))
                if rebuilt != feature:
                    raise ValueError("feature_not_reproducible_from_allow_list")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"{record.get('root_id')}:{key}:{type(exc).__name__}:{exc}")
    return not failures, failures


def _attach_public_heads(
    store: CoraDataStore, records: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    output: list[dict[str, Any]] = []
    failures: list[str] = []
    for original in records:
        record = copy.deepcopy(dict(original))
        root_id = str(record.get("root_id", ""))
        for branch in record.get("branches", ()):
            proposal_digest = str(branch.get("proposal_sha256", ""))
            try:
                public = read_public_label(store, root_id, proposal_digest)
                paths = branch.get("artifact_paths", {})
                identities = branch.get("artifact_sha256", {})
                if (
                    public.get("proposal_id") != branch.get("proposal_id")
                    or public.get("source_label_path") != paths.get("label")
                    or public.get("source_label_sha256") != identities.get("label")
                ):
                    raise ValueError("public_label_source_binding")
                branch["heads"] = public["heads"]
                branch["public_label_sha256"] = public["public_label_sha256"]
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                failures.append(
                    f"{root_id}:{branch.get('proposal_id')}:{type(exc).__name__}:{exc}"
                )
        output.append(record)
    return output, failures


def _resource_audit(dataset_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    evidence = root / "docs" / "runtime-evidence" / "h6" / dataset_path.name
    rows = []
    failure_rows = []
    for path in sorted(evidence.glob("collect-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == "safedrive.cora.collection_summary.v1":
            rows.append(payload)
        elif payload.get("schema_version") == "safedrive.cora.collection_failure.v1":
            failure_rows.append(payload)
    usage = shutil.disk_usage(dataset_path)
    limits = CORA_C2_CONFIG["resources"]
    wall_rows = rows + failure_rows
    wall_s = sum(max(0.0, float(row.get("elapsed_s", 0.0))) for row in wall_rows)
    measured_peaks = [float(row.get("whole_gpu_peak_gib", 0.0)) for row in rows]
    measured_peaks.extend(
        float(row["gpu"]["value"])
        for row in failure_rows
        if isinstance(row.get("gpu"), Mapping)
        and bool(row["gpu"].get("valid"))
        and row["gpu"].get("value") is not None
    )
    peak_gib = max(measured_peaks, default=0.0)
    free_observations = [
        float(row["resource"]["free_gib"])
        for row in wall_rows
        if isinstance(row.get("resource"), Mapping)
        and row["resource"].get("free_gib") is not None
    ]
    current_free_gib = usage.free / 1024**3
    minimum_free_gib = min([current_free_gib, *free_observations])
    failures = []
    if not rows:
        failures.append("collection_resource_evidence_missing")
    if wall_s > float(limits["aggregate_wall_limit_hours"]) * 3600.0:
        failures.append("aggregate_wall_limit")
    if peak_gib > float(limits["whole_gpu_peak_limit_gib"]):
        failures.append("whole_gpu_peak_limit")
    if minimum_free_gib < float(limits["free_disk_floor_gib"]):
        failures.append("free_disk_floor")
    return {
        "passed": not failures,
        "failures": failures,
        "collection_evidence_count": len(rows),
        "collection_failure_evidence_count": len(failure_rows),
        "aggregate_collector_wall_s": wall_s,
        "aggregate_wall_measurement": "success_elapsed_plus_preserved_failure_elapsed",
        "whole_gpu_peak_gib": peak_gib,
        "free_disk_gib": current_free_gib,
        "minimum_observed_free_disk_gib": minimum_free_gib,
        "collector_failures": [
            {
                "map_name": row.get("map_name"),
                "scope": row.get("scope"),
                "error": row.get("error"),
                "elapsed_s": row.get("elapsed_s"),
                "failed_root_id": row.get("failed_root_id"),
            }
            for row in failure_rows
        ],
    }


def _audit_records(
    records: Sequence[Mapping[str, Any]], *, scope: str, manifest_ok: bool, dataset_bytes: int
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if (scope == "pilot" and record.get("split") == "coverage_pilot")
        or (scope == "development" and record.get("split") in DEVELOPMENT_SPLITS)
    ]
    terminal = [record for record in selected if record.get("terminal_status")]
    valid_nominal = [record for record in selected if bool(record.get("nominal_pair_outcome_mask"))]
    valid_by_map = collections.Counter(str(item.get("scenario", {}).get("map_name")) for item in valid_nominal)
    valid_by_family = collections.Counter(str(item.get("scenario", {}).get("family")) for item in valid_nominal)
    valid_by_weather = collections.Counter(str(item.get("scenario", {}).get("weather")) for item in valid_nominal)
    valid_by_split = collections.Counter(str(item.get("split")) for item in valid_nominal)
    branch_by_source = collections.Counter()
    guard_verdicts = collections.Counter()
    branch_orders = collections.Counter()
    missingness = collections.Counter()
    missingness_dimensions = {
        name: collections.Counter()
        for name in (
            "source",
            "guard",
            "operator",
            "risk_family",
            "safety_stage",
            "branch_order",
            "map",
            "family",
            "weather",
            "split",
        )
    }
    operator_proposals = collections.Counter()
    operator_terminal = collections.Counter()
    operator_eligible_valid = collections.Counter()
    hazard = {
        name: {split: 0 for split in DEVELOPMENT_SPLITS}
        for name in ("collision", "red_light", "offroad")
    }
    binary = {
        name: {split: {True: 0, False: 0} for split in DEVELOPMENT_SPLITS}
        for name in ("repair_success", "executable")
    }
    safety_decisions = collections.Counter()
    terminal_reasons = collections.Counter()
    branch_guard = collections.Counter()
    public_head_counts = {
        name: {
            "valid": 0,
            "invalid": 0,
            "positive": 0,
            "negative": 0,
            "by_split": {
                split: {"valid": 0, "invalid": 0, "positive": 0, "negative": 0}
                for split in DEVELOPMENT_SPLITS
            },
        }
        for name in PUBLIC_OUTCOME_HEADS
    }
    identity_failures = []
    contract_failures = []
    reset_failures = []
    cleanup_failures = []
    cross_fallback_failures = []
    core_intervention_valid = 0
    expected_by_id = {row.root_id: row for row in CORA_DATA_MATRIX}
    for record in selected:
        root_id = str(record.get("root_id"))
        expected = expected_by_id.get(root_id)
        if expected is None:
            contract_failures.append(f"{root_id}:not_in_frozen_matrix")
        else:
            scenario = record.get("scenario", {})
            if str(record.get("split")) != expected.split:
                contract_failures.append(f"{root_id}:split_mismatch")
            for name, value in expected.scenario.to_dict().items():
                if scenario.get(name) != value:
                    contract_failures.append(f"{root_id}:scenario_{name}_mismatch")
            order = tuple(str(item) for item in scenario.get("branch_order", ()))
            if order != expected.branch_order or int(scenario.get("expert_slot", -1)) != expected.expert_slot:
                contract_failures.append(f"{root_id}:permutation_mismatch")
            branch_orders["->".join(order)] += 1
        if record.get("config_sha256") != CORA_C2_CONFIG_SHA256:
            contract_failures.append(f"{root_id}:config_sha256")
        if record.get("matrix_sha256") != CORA_MATRIX_SHA256:
            contract_failures.append(f"{root_id}:matrix_sha256")
        if int(record.get("vla_forward_count", 0)) != 1:
            contract_failures.append(f"{root_id}:vla_forward_count_not_one")
        if len(record.get("proposals", ())) > 4 or len(record.get("branches", ())) > 4:
            contract_failures.append(f"{root_id}:per_root_limit")
        proposal_by_id = {str(item.get("proposal_id")): item for item in record.get("proposals", ())}
        branches_by_id = {str(item.get("proposal_id")): item for item in record.get("branches", ())}
        scenario = record.get("scenario", {})
        order_text = "->".join(str(item) for item in scenario.get("branch_order", ()))
        for row in record.get("missingness", ()):
            reason = str(row.get("reason", "unknown"))
            missingness[reason] += 1
            dimensions = {
                "source": row.get("source", "UNSPECIFIED"),
                "guard": "REJECT" if reason == "guard_reject" else row.get("guard", "UNSPECIFIED"),
                "operator": row.get("operator", "UNSPECIFIED"),
                "risk_family": scenario.get("family", "UNSPECIFIED"),
                "safety_stage": row.get("stage", "UNSPECIFIED"),
                "branch_order": order_text or "UNSPECIFIED",
                "map": scenario.get("map_name", "UNSPECIFIED"),
                "family": scenario.get("family", "UNSPECIFIED"),
                "weather": scenario.get("weather", "UNSPECIFIED"),
                "split": record.get("split", "UNSPECIFIED"),
            }
            for dimension, value in dimensions.items():
                missingness_dimensions[dimension][f"{value}:{reason}"] += 1
        for proposal in record.get("proposals", ()):
            guard_verdicts[str(proposal.get("guard", {}).get("verdict", "MISSING"))] += 1
            if proposal.get("kind") != "offline_intervention":
                continue
            operator = str(proposal.get("operator", ""))
            operator_proposals[operator] += 1
            branch = branches_by_id.get(str(proposal.get("proposal_id")))
            if branch is not None:
                operator_terminal[operator] += 1
                if (
                    not bool(proposal.get("auxiliary_only"))
                    and str(proposal.get("guard", {}).get("verdict")) in {"PASS", "REVIEW"}
                    and _branch_valid(branch)
                ):
                    operator_eligible_valid[operator] += 1
                    core_intervention_valid += 1
        for edge in record.get("edges", ()):
            left_proposal = proposal_by_id.get(str(edge.get("left_proposal_id")))
            right_proposal = proposal_by_id.get(str(edge.get("right_proposal_id")))
            if (
                left_proposal is None
                or right_proposal is None
                or edge.get("left_proposal_sha256") != left_proposal.get("proposal_sha256")
                or edge.get("right_proposal_sha256") != right_proposal.get("proposal_sha256")
            ):
                identity_failures.append(f"{root_id}:edge_proposal_binding")
            if edge.get("edge_kind") == "intervention_base" and edge.get("pair_outcome_mask"):
                if bool((left_proposal or {}).get("auxiliary_only")):
                    contract_failures.append(f"{root_id}:auxiliary_edge_in_core")
            if not edge.get("pair_outcome_mask"):
                continue
            for proposal_id in (edge.get("left_proposal_id"), edge.get("right_proposal_id")):
                branch = branches_by_id.get(str(proposal_id))
                if branch is None or not _branch_valid(branch):
                    identity_failures.append(f"{root_id}:masked_edge_invalid_branch:{proposal_id}")
        split = str(record.get("split"))
        for branch in record.get("branches", ()):
            proposal = proposal_by_id.get(str(branch.get("proposal_id")), {})
            source = str(proposal.get("audit_source", "unknown"))
            branch_by_source[source] += 1
            safety_decisions[str(branch.get("decision_kind", "MISSING"))] += 1
            terminal_reasons[str(branch.get("terminal_reason", "MISSING"))] += 1
            branch_guard[str(branch.get("guard_verdict", "MISSING"))] += 1
            if set(branch.get("heads", {})) != set(PUBLIC_OUTCOME_HEADS):
                contract_failures.append(f"{root_id}:public_outcome_heads:{branch.get('proposal_id')}")
            for name in PUBLIC_OUTCOME_HEADS:
                item = branch.get("heads", {}).get(name)
                valid = isinstance(item, Mapping) and bool(item.get("valid"))
                bucket = public_head_counts[name]
                bucket["valid" if valid else "invalid"] += 1
                split_bucket = bucket["by_split"].get(split)
                if split_bucket is not None:
                    split_bucket["valid" if valid else "invalid"] += 1
                value = item.get("value") if isinstance(item, Mapping) else None
                if valid and isinstance(value, bool):
                    bucket["positive" if value else "negative"] += 1
                    if split_bucket is not None:
                        split_bucket["positive" if value else "negative"] += 1
            if not bool(branch.get("cleanup_complete")):
                cleanup_failures.append(f"{root_id}:{branch.get('proposal_id')}")
            if bool(branch.get("would_require_cross_candidate_fallback")):
                cross_fallback_failures.append(f"{root_id}:{branch.get('proposal_id')}")
            if not bool(proposal.get("auxiliary_only")) and str(branch.get("guard_verdict")) in {"PASS", "REVIEW"}:
                if not bool(branch.get("reset", {}).get("comparable")):
                    reset_failures.append(f"{root_id}:{branch.get('proposal_id')}")
                if not bool(branch.get("identity_valid")):
                    identity_failures.append(f"{root_id}:branch_identity:{branch.get('proposal_id')}")
            if split not in DEVELOPMENT_SPLITS:
                continue
            if bool(proposal.get("auxiliary_only")) or not _branch_valid(branch):
                continue
            if str(branch.get("guard_verdict")) not in {"PASS", "REVIEW"}:
                continue
            collision_count = _head_value(branch, "collision_count")
            if isinstance(collision_count, (int, float)) and collision_count > 0:
                hazard["collision"][split] += 1
            if bool(_head_value(branch, "red_light_violation")):
                hazard["red_light"][split] += 1
            off_duration = _head_value(branch, "off_corridor_duration_s")
            if isinstance(off_duration, (int, float)) and off_duration > 0.0:
                hazard["offroad"][split] += 1
            for name in binary:
                value = _head_value(branch, name)
                if isinstance(value, bool):
                    binary[name][split][value] += 1

    metrics: dict[str, Any] = {
        "scope": scope,
        "terminal_roots": len(terminal),
        "valid_nominal_pairs": len(valid_nominal),
        "valid_by_map": dict(valid_by_map),
        "valid_by_family": dict(valid_by_family),
        "valid_by_weather": dict(valid_by_weather),
        "valid_by_split": dict(valid_by_split),
        "branch_by_source": dict(branch_by_source),
        "guard_verdicts": dict(guard_verdicts),
        "branch_guard_verdicts": dict(branch_guard),
        "safety_decisions": dict(safety_decisions),
        "terminal_reasons": dict(terminal_reasons),
        "branch_orders": dict(branch_orders),
        "vla_forward_count": sum(int(item.get("vla_forward_count", 0)) for item in selected),
        "operator_finite_proposals": dict(operator_proposals),
        "operator_terminal": dict(operator_terminal),
        "operator_guard_eligible_valid": dict(operator_eligible_valid),
        "guard_eligible_valid_intervention_branches": core_intervention_valid,
        "hazard_positives": hazard,
        "binary_class_counts": binary,
        "public_outcome_head_counts": public_head_counts,
        "missingness": dict(missingness),
        "missingness_by": {
            name: dict(counter) for name, counter in missingness_dimensions.items()
        },
        "identity_failures": identity_failures,
        "contract_failures": contract_failures,
        "reset_failures": reset_failures,
        "cleanup_failures": cleanup_failures,
        "cross_candidate_fallback_failures": cross_fallback_failures,
        "root_attempts": len(selected),
        "branch_attempts": sum(len(item.get("branches", ())) for item in selected),
        "intervention_proposals": sum(
            proposal.get("kind") == "offline_intervention"
            for item in selected
            for proposal in item.get("proposals", ())
        ),
        "pair_edges": sum(len(item.get("edges", ())) for item in selected),
        "ticks_recorded": sum(
            int(branch.get("ticks_executed", 0))
            for item in selected
            for branch in item.get("branches", ())
        ),
        "manifest_valid": manifest_ok,
        "dataset_bytes": dataset_bytes,
        "config_sha256": CORA_C2_CONFIG_SHA256,
        "matrix_sha256": CORA_MATRIX_SHA256,
    }
    failures: list[str] = []
    if not manifest_ok:
        failures.append("manifest")
    if dataset_bytes > float(CORA_C2_CONFIG["resources"]["dataset_limit_gib"]) * 1024**3:
        failures.append("dataset_size")
    if identity_failures:
        failures.append("identity")
    if contract_failures:
        failures.append("frozen_contract")
    if reset_failures:
        failures.append("reset")
    if cleanup_failures:
        failures.append("cleanup")
    if cross_fallback_failures:
        failures.append("cross_candidate_fallback")
    if scope == "pilot":
        gate = CORA_C2_CONFIG["pilot_gate"]
        if len(terminal) != int(gate["terminal_roots"]):
            failures.append("pilot_terminal_roots")
        if len(valid_nominal) < int(gate["min_valid_nominal_pairs"]):
            failures.append("pilot_valid_nominal_pairs")
        for map_name in CORA_C2_CONFIG["maps"]:
            if valid_by_map[str(map_name)] < int(gate["min_valid_per_map"]):
                failures.append(f"pilot_map:{map_name}")
        for family in CORA_C2_CONFIG["families"]:
            if valid_by_family[str(family)] < int(gate["min_valid_per_family"]):
                failures.append(f"pilot_family:{family}")
        if metrics["vla_forward_count"] != int(gate["expected_vla_forwards"]):
            failures.append("pilot_vla_forward_count")
        for operator in ALL_OPERATORS:
            if operator_proposals[operator] < int(gate["min_finite_per_operator"]):
                failures.append(f"pilot_operator:{operator}")
        if core_intervention_valid < int(gate["min_guard_eligible_intervention_branches"]):
            failures.append("pilot_intervention_eligible")
    elif scope == "development":
        gate = CORA_C2_CONFIG["development_gate"]
        if len(terminal) != int(gate["terminal_roots"]):
            failures.append("development_terminal_roots")
        if len(valid_nominal) < int(gate["min_valid_nominal_pairs"]):
            failures.append("development_valid_nominal_pairs")
        split_thresholds = {
            "train": "min_train",
            "validation": "min_validation",
            "calibration": "min_calibration",
            "locked_development": "min_locked_development",
        }
        for split, threshold in split_thresholds.items():
            if valid_by_split[split] < int(gate[threshold]):
                failures.append(f"development_split:{split}")
        for map_name in CORA_C2_CONFIG["maps"]:
            if valid_by_map[str(map_name)] < int(gate["min_per_map"]):
                failures.append(f"development_map:{map_name}")
        for family in CORA_C2_CONFIG["families"]:
            if valid_by_family[str(family)] < int(gate["min_per_family"]):
                failures.append(f"development_family:{family}")
        for weather in CORA_C2_CONFIG["weather"]:
            if valid_by_weather[str(weather)] < int(gate["min_per_weather"]):
                failures.append(f"development_weather:{weather}")
        for operator in ALL_OPERATORS:
            if operator_terminal[operator] < int(gate["min_operator_terminal"]):
                failures.append(f"development_operator_terminal:{operator}")
            if operator_eligible_valid[operator] < int(gate["min_operator_guard_eligible"]):
                failures.append(f"development_operator_eligible:{operator}")
        for name, rows in hazard.items():
            if rows["train"] < int(gate["min_hazard_train"]):
                failures.append(f"hazard_train:{name}")
            for split in ("validation", "calibration", "locked_development"):
                if rows[split] < int(gate["min_hazard_other_split"]):
                    failures.append(f"hazard_{split}:{name}")
        for name, rows in binary.items():
            for value in (True, False):
                if rows["train"][value] < int(gate["min_binary_class_train"]):
                    failures.append(f"binary_train:{name}:{value}")
                for split in ("validation", "calibration", "locked_development"):
                    if rows[split][value] < int(gate["min_binary_class_other_split"]):
                        failures.append(f"binary_{split}:{name}:{value}")
    else:
        raise ValueError(f"cora_quality_scope:{scope}")
    return {"metrics": metrics, "failures": list(dict.fromkeys(failures))}


def audit_cora_dataset(
    dataset_root: Path | str,
    *,
    scope: str,
    trust_pilot_manifest: bool = False,
) -> dict[str, Any]:
    path = Path(dataset_root)
    store = CoraDataStore(path.parent, path.name)
    if trust_pilot_manifest:
        if scope != "development":
            raise ValueError("cora_trusted_manifest_scope")
        public_count = sum(1 for _ in store.labels_dir.glob("*.public.json"))
        public_materialization = {
            "schema_version": "safedrive.cora.outcome_labels.v2",
            "root_count": sum(1 for _ in store.pairs_dir.glob("*.json")),
            "written": 0,
            "reused": public_count,
            "label_count": public_count,
            "mode": "REUSED_AFTER_PILOT_MANIFEST_VERIFIED",
        }
        manifest_ok, manifest_failures = True, ()
        manifest_mode = "REUSED_PILOT_VERIFICATION"
    else:
        public_materialization = materialize_public_labels(
            store,
            dt_s=float(CORA_C2_CONFIG["timing"]["fixed_delta_seconds"]),
        )
        manifest_ok, manifest_failures = store.verify_manifest()
        manifest_mode = "VERIFIED_IN_THIS_AUDIT"
    raw_records = list(store.iter_roots())
    records, public_label_failures = _attach_public_heads(store, raw_records)
    feature_records = [
        record
        for record in records
        if (scope == "pilot" and record.get("split") == "coverage_pilot")
        or (scope == "development" and record.get("split") in DEVELOPMENT_SPLITS)
    ]
    feature_ok, feature_failures = _read_feature_audit(store, feature_records)
    resource = _resource_audit(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8")) if (path / "manifest.json").is_file() else {}
    audited = _audit_records(
        records,
        scope=scope,
        manifest_ok=manifest_ok,
        dataset_bytes=int(manifest.get("total_bytes", 0)),
    )
    failures = list(audited["failures"])
    actual_ids = {str(record.get("root_id")) for record in records}
    expected_ids = {row.root_id for row in CORA_DATA_MATRIX}
    inventory_failures = []
    if actual_ids != expected_ids:
        inventory_failures.append(
            {
                "missing": sorted(expected_ids - actual_ids),
                "unexpected": sorted(actual_ids - expected_ids),
            }
        )
        failures.append("frozen_root_inventory")
    if len(records) > int(CORA_C2_CONFIG["resources"]["root_attempt_limit"]):
        failures.append("root_attempt_limit")
    total_branches = sum(len(record.get("branches", ())) for record in records)
    if total_branches > int(CORA_C2_CONFIG["resources"]["branch_attempt_limit"]):
        failures.append("branch_attempt_limit")
    if public_label_failures:
        failures.append("public_outcome_labels")
    if not feature_ok:
        failures.append("feature_leakage_or_schema")
    if not resource["passed"]:
        failures.append("resource")
    payload: dict[str, Any] = {
        "schema_version": QUALITY_SCHEMA,
        "dataset_id": path.name,
        "scope": scope,
        "passed": not failures,
        "metrics": audited["metrics"],
        "public_label_audit": {
            "passed": not public_label_failures,
            "failures": public_label_failures,
            "materialization": public_materialization,
        },
        "inventory_audit": {
            "passed": not inventory_failures,
            "root_count": len(records),
            "branch_count": total_branches,
            "formal_collected": any(
                str(record.get("split")) == "reserved_formal" for record in records
            ),
            "failures": inventory_failures,
        },
        "feature_audit": {"passed": feature_ok, "failures": feature_failures},
        "trajectory_to_source_probe": trajectory_source_probe(raw_records),
        "resource_audit": resource,
        "manifest_verification_mode": manifest_mode,
        "manifest_failures": list(manifest_failures),
        "failures": list(dict.fromkeys(failures)),
    }
    payload["data_quality_sha256"] = stable_sha256(payload)
    return payload


__all__ = [
    "ALL_OPERATORS",
    "DEVELOPMENT_SPLITS",
    "QUALITY_SCHEMA",
    "audit_cora_dataset",
    "trajectory_source_probe",
]
