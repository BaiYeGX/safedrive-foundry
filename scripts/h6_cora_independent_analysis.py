"""Independently recompute C3 group metrics and simple-baseline comparisons.

This script intentionally does not call the production aggregation function.
It reads the frozen manifest and saved prediction arrays, recomputes masked
errors with NumPy, and then checks the published aggregate.  It is meant for
the final evidence pass and for exposing improvements that are concentrated in
one scene or in a label-template artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
from driving_vla.model.canonicalizer import (  # noqa: E402
    CanonicalizationError, TrajectoryCanonicalizer, UpstreamPathSpeed,
)
from driving_vla.model.speed_convert import speed_wps_to_planner_samples  # noqa: E402


ROUTE_STEPS = 20
SPEED_STEPS = 10
BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 71


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_vector_errors(values: Any, target: Any, mask: Iterable[bool], steps: int) -> tuple[np.ndarray, np.ndarray, bool]:
    truth = np.asarray(target, dtype=np.float64)
    valid_mask = np.asarray(tuple(mask), dtype=bool)
    if truth.shape != (steps, 2) or valid_mask.shape != (steps,) or not np.isfinite(truth[valid_mask]).all():
        raise ValueError("independent_analysis_invalid_frozen_truth")
    try:
        prediction = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError):
        return np.empty(0, dtype=np.float64), np.zeros(steps, dtype=bool), False
    shape_valid = prediction.shape == (steps, 2)
    if not shape_valid:
        return np.empty(0, dtype=np.float64), np.zeros(steps, dtype=bool), False
    finite = np.isfinite(prediction).all(axis=1)
    errors = np.linalg.norm(prediction - truth, axis=1)
    return errors, valid_mask & finite, True


def _score(row: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    route_errors, route_valid, route_shape = _finite_vector_errors(
        row.get("route", []), sample.get("route_target", []), sample.get("route_mask", []), ROUTE_STEPS
    )
    speed_errors, speed_valid, speed_shape = _finite_vector_errors(
        row.get("speed", []), sample.get("speed_target", []), sample.get("speed_mask", []), SPEED_STEPS
    )
    route_mask = np.asarray(sample.get("route_mask", []), dtype=bool)
    speed_mask = np.asarray(sample.get("speed_mask", []), dtype=bool)
    # Target coverage is unrelated to whether the model's full output is valid.
    native_finite = bool(
        route_shape and speed_shape
        and np.isfinite(np.asarray(row["route"], dtype=float)).all()
        and np.isfinite(np.asarray(row["speed"], dtype=float)).all()
    )
    reasons = list(row.get("failure_reasons", []))
    canonical_valid = False
    if native_finite:
        route = np.asarray(row["route"], dtype=float)
        yaw = float(sample.get("ego_yaw", 0.0))
        c, s = math.cos(yaw), math.sin(yaw)
        path_map = tuple((float(sample.get("ego_x", 0.0)) + c*x - s*y,
                          float(sample.get("ego_y", 0.0)) + s*x + c*y) for x, y in route)
        try:
            converted = TrajectoryCanonicalizer().canonicalize_with_report(
                UpstreamPathSpeed(path_xy=path_map,
                                  speed_mps=speed_wps_to_planner_samples(row["speed"], use_official_scalar=True),
                                  frame="map"), to_map=False,
            )
            canonical_valid = bool(
                len(converted.trajectory.points_xy_yaw_v_a_kappa) == SPEED_STEPS
                and np.isfinite(converted.trajectory.points_xy_yaw_v_a_kappa).all()
            )
        except CanonicalizationError as exc:
            reasons.append(f"canonical_rejected:{exc}")
    else:
        reasons.append("native_shape_or_nonfinite")
    if row.get("prediction_exception"):
        canonical_valid = False
        reasons.append(str(row["prediction_exception"]))
    route_target_available = bool(route_mask.shape == (ROUTE_STEPS,) and route_mask.any())
    speed_target_available = bool(speed_mask.shape == (SPEED_STEPS,) and speed_mask.any())
    route_ade = float(route_errors[route_mask].mean()) if route_shape and route_target_available and route_valid[route_mask].all() else None
    speed_ade = float(speed_errors[speed_mask].mean()) if speed_shape and speed_target_available and speed_valid[speed_mask].all() else None
    fde_available = bool(route_mask.shape == (ROUTE_STEPS,) and route_mask[-1])
    fde = float(route_errors[-1]) if fde_available and route_valid.shape == (ROUTE_STEPS,) and route_valid[-1] else None
    return {
        "root_id": str(sample["root_id"]),
        "route_ade_m": route_ade,
        "route_fde_m": fde,
        "speed_wp_ade_m": speed_ade,
        "route_target_available": route_target_available,
        "route_fde_target_available": fde_available,
        "speed_target_available": speed_target_available,
        "prediction_valid": canonical_valid,
        "native_finite": native_finite,
        "route_valid_points": int(route_mask.sum()),
        "speed_valid_points": int(speed_mask.sum()),
        "route_full_valid": bool(route_target_available and route_mask.all() and route_valid.all()),
        "speed_full_valid": bool(speed_target_available and speed_mask.all() and speed_valid.all()),
        "failure_reasons": sorted(set(reasons)),
    }


def _mean(rows: list[dict[str, Any]], key: str, availability_key: str) -> float | None:
    supported = [row for row in rows if row.get(availability_key)]
    values = [float(row[key]) for row in supported if row.get(key) is not None and math.isfinite(float(row[key]))]
    return float(np.mean(values)) if values and len(values) == len(supported) else None


def _bootstrap(differences: list[float], rng: np.random.Generator) -> list[float] | None:
    if len(differences) < 2:
        return None
    values = np.asarray(differences, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_ROUNDS, len(values)))
    boot = values[draws].mean(axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def _paired_summary(scores: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    for model, rows in scores.items():
        _unique_rows(rows, label=model)
    scores = {model: sorted(rows, key=lambda row: row["root_id"]) for model, rows in scores.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by_root: dict[str, dict[str, dict[str, Any]]] = {}
    for model, rows in scores.items():
        for row in rows:
            by_root.setdefault(str(row["root_id"]), {})[model] = row
    paired = [pair for pair in by_root.values() if "m0" in pair and "m1" in pair]
    metrics: dict[str, Any] = {}
    for key, availability in (
        ("route_ade_m", "route_target_available"),
        ("route_fde_m", "route_fde_target_available"),
        ("speed_wp_ade_m", "speed_target_available"),
    ):
        base = _mean(scores.get("m0", []), key, availability)
        method = _mean(scores.get("m1", []), key, availability)
        differences = [
            float(pair["m0"][key]) - float(pair["m1"][key])
            for pair in paired
            if pair["m0"].get(key) is not None and pair["m1"].get(key) is not None
        ]
        delta = None if base is None or method is None else base - method
        metrics[key] = {
            "m0": base,
            "m1": method,
            "delta_m0_minus_m1": delta,
            "relative_improvement_percent": None if delta is None or not base else 100.0 * delta / abs(base),
            "paired_root_count": len(differences),
            "bootstrap_ci95": _bootstrap(differences, rng) if base is not None and method is not None else None,
            "improved_root_count": sum(value > 1e-5 for value in differences),
            "degraded_root_count": sum(value < -1e-5 for value in differences),
            "target_root_count": {
                "m0": sum(bool(row.get(availability)) for row in scores.get("m0", [])),
                "m1": sum(bool(row.get(availability)) for row in scores.get("m1", [])),
            },
        }
    all_rows = {model: rows for model, rows in scores.items()}
    metrics["root_count"] = {model: len(rows) for model, rows in all_rows.items()}
    metrics["prediction_valid_rate_percent"] = {
        model: (100.0 * sum(bool(row.get("prediction_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in all_rows.items()
    }
    metrics["prediction_fail_rate_percent"] = {
        model: (100.0 * sum(not bool(row.get("prediction_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in all_rows.items()
    }
    metrics["native_finite_rate_percent"] = {
        model: 100.0 * sum(bool(row.get("native_finite")) for row in rows) / len(rows) if rows else None
        for model, rows in all_rows.items()
    }
    metrics["route_full_valid_rate_percent"] = {
        model: (100.0 * sum(bool(row.get("route_full_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in all_rows.items()
    }
    metrics["speed_full_valid_rate_percent"] = {
        model: (100.0 * sum(bool(row.get("speed_full_valid")) for row in rows) / len(rows) if rows else None)
        for model, rows in all_rows.items()
    }
    metrics["failure_reason_counts"] = {
        model: {
            reason: sum(reason in row.get("failure_reasons", ()) for row in rows)
            for reason in sorted({str(reason) for row in rows for reason in row.get("failure_reasons", ())})
        }
        for model, rows in all_rows.items()
    }
    return metrics


def _support_bin(value: float) -> str:
    if value <= 1e-9:
        return "zero_support"
    if value < 10.0:
        return "partial_0_to_10m"
    if value < 19.0:
        return "partial_10_to_19m"
    return "full_19m_or_more"


def _group_summary(
    samples: dict[str, dict[str, Any]],
    m0_rows: list[dict[str, Any]],
    m1_rows: list[dict[str, Any]],
    dimension: str,
) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model, rows in (("m0", m0_rows), ("m1", m1_rows)):
        for row in rows:
            sample = samples[str(row["root_id"])]
            if dimension == "cohort":
                label = str(sample.get("cohort", "unknown"))
            elif dimension == "label_source":
                label = f"{sample.get('route_source', 'unknown')}|{sample.get('speed_source', 'unknown')}"
            elif dimension == "map":
                label = str(sample.get("map_name", "unknown"))
            elif dimension == "scene":
                label = str(sample.get("family", "unknown"))
            elif dimension == "weather":
                label = str(sample.get("weather", "unknown"))
            elif dimension == "support_bin":
                label = _support_bin(float(sample.get("route_support_m", 0.0)))
            elif dimension == "speed_input":
                speed = float(sample.get("ego_speed_mps", 0.0))
                label = "zero" if abs(speed) <= 1e-9 else "nonzero"
            else:
                raise ValueError(f"unknown_group_dimension:{dimension}")
            groups.setdefault(label, {}).setdefault(model, []).append(row)
    return {
        label: _paired_summary(values)
        for label, values in sorted(groups.items())
    }


def _simple_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        availability = "route_target_available" if key == "route_ade_m" else "speed_target_available"
        return _mean(rows, key, availability)

    return {
        "root_count": len(rows),
        "route_ade_root_equal_m": mean("route_ade_m"),
        "speed_wp_ade_root_equal_m": mean("speed_wp_ade_m"),
        "route_valid_points": sum(int(row.get("route_valid_points", 0)) for row in rows),
        "route_fde_supported_roots": sum(row.get("route_fde_m") is not None for row in rows),
        "prediction_valid_rate_percent": 100.0 * sum(bool(row.get("prediction_valid")) for row in rows) / len(rows) if rows else None,
        "route_full_valid_rate_percent": 100.0 * sum(bool(row.get("route_full_valid")) for row in rows) / len(rows) if rows else None,
    }


def _close(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def _unique_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    """The frozen C3 release has one anchor per root; reject ambiguous repeats."""
    result = {}
    for row in rows:
        root = str(row.get("root_id", ""))
        if not root or root in result:
            raise ValueError(f"independent_analysis_duplicate_or_empty_root:{label}:{root}")
        result[root] = row
    return result


def _recompute_diagnostics(diagnostics, samples, predictions):
    summaries, shared_scores = {}, {name: [] for name in ("navigation_geometry", "m0", "m1")}
    for model, rows in diagnostics.get("rows", {}).items():
        indexed = _unique_rows(rows, label=model)
        if set(indexed) != set(samples):
            raise ValueError(f"independent_analysis_diagnostic_root_set:{model}")
        scores = []
        for root, sample in samples.items():
            row = indexed[root]
            if model == "navigation_geometry":
                geometry_mask = np.asarray(row.get("geometry_route_mask", []), dtype=bool)
                if geometry_mask.shape != (ROUTE_STEPS,):
                    raise ValueError("independent_analysis_navigation_mask_shape")
                sample = {**sample, "route_mask": (np.asarray(sample["route_mask"], dtype=bool) & geometry_mask).tolist()}
                shared_scores[model].append(_score(row, sample))
                for name in ("m0", "m1"):
                    shared_scores[name].append(_score(predictions[name][root], sample))
            scores.append(_score(row, sample))
        summaries[model] = _simple_baseline(scores)
    return summaries, {name: _simple_baseline(rows) for name, rows in shared_scores.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New report path; existing reports are never overwritten.")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output = (args.output or run_dir / "independent-analysis.json").expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"independent_analysis_output_exists:{output}")
    manifest = _read(run_dir / "manifest.json")
    m0 = _read(run_dir / "m0_predictions.json")
    m1 = _read(run_dir / "m1_predictions.json")
    published = _read(run_dir / "metrics.json")
    diagnostics = _read(run_dir / "diagnostic-baselines.json")
    samples = _unique_rows([sample for sample in manifest["samples"] if sample.get("split") == "validation"], label="manifest")
    if not samples:
        raise ValueError("independent_analysis_empty_validation")
    m0_rows = list(m0.get("rows", []))
    m1_rows = list(m1.get("rows", []))
    predictions = {"m0": _unique_rows(m0_rows, label="m0"), "m1": _unique_rows(m1_rows, label="m1")}
    if set(row.get("root_id") for row in m0_rows) != set(samples) or set(row.get("root_id") for row in m1_rows) != set(samples):
        raise ValueError("independent_analysis_validation_root_set")
    scores = {
        "m0": [_score(row, samples[str(row["root_id"])]) for row in m0_rows],
        "m1": [_score(row, samples[str(row["root_id"])]) for row in m1_rows],
    }
    independent_metrics = _paired_summary(scores)
    errors: list[str] = []
    for key in ("route_ade_m", "route_fde_m", "speed_wp_ade_m"):
        for field in ("m0", "m1", "delta_m0_minus_m1", "paired_root_count", "bootstrap_ci95"):
            if not _close(independent_metrics[key].get(field), published.get("metrics", {}).get(key, {}).get(field)):
                errors.append(f"published_metric_mismatch:{key}:{field}")
    for key in ("root_count", "prediction_valid_rate_percent", "prediction_fail_rate_percent"):
        if independent_metrics[key] != published.get("metrics", {}).get(key):
            errors.append(f"published_metric_mismatch:{key}")
    dimensions = ("cohort", "label_source", "map", "scene", "weather", "support_bin", "speed_input")
    groups = {
        dimension: _group_summary(samples, scores["m0"], scores["m1"], dimension)
        for dimension in dimensions
    }
    per_root = []
    by_model = {
        model: {row["root_id"]: row for row in rows}
        for model, rows in scores.items()
    }
    for root_id in sorted(samples):
        left, right = by_model["m0"][root_id], by_model["m1"][root_id]
        per_root.append({
            "root_id": root_id,
            "route_delta_m0_minus_m1_m": None if left["route_ade_m"] is None or right["route_ade_m"] is None else left["route_ade_m"] - right["route_ade_m"],
            "speed_delta_m0_minus_m1_m": None if left["speed_wp_ade_m"] is None or right["speed_wp_ade_m"] is None else left["speed_wp_ade_m"] - right["speed_wp_ade_m"],
            "route_support_m": float(samples[root_id].get("route_support_m", 0.0)),
            "family": samples[root_id].get("family"),
            "map": samples[root_id].get("map_name"),
        })
    per_root.sort(key=lambda row: (float("inf") if row["route_delta_m0_minus_m1_m"] is None else row["route_delta_m0_minus_m1_m"], row["root_id"]))
    baseline_summary, matched_navigation = _recompute_diagnostics(diagnostics, samples, predictions)
    payload: dict[str, Any] = {
        "schema_version": "safedrive.c3.independent_analysis.v2",
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_artifact_sha256": {name: hashlib.sha256((run_dir/name).read_bytes()).hexdigest()
                                   for name in ("manifest.json", "m0_predictions.json", "m1_predictions.json", "metrics.json", "diagnostic-baselines.json")},
        "canonical_contract": "deployment_official_scalar_speed_and_T10_dt025_canonicalizer",
        "performance_based_exclusions": [],
        "comparison_scope": "frozen_C3_development_set; one_anchor_per_root; partial_native_support",
        "run_id": run_dir.name,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "m0_content_sha256": m0.get("content_sha256"),
        "m1_content_sha256": m1.get("content_sha256"),
        "published_metrics_content_sha256": published.get("content_sha256"),
        "validation_root_count": len(samples),
        "independent_metrics": independent_metrics,
        "published_metric_match_errors": errors,
        "group_metrics": groups,
        "simple_baselines": baseline_summary,
        "navigation_matched_support": matched_navigation,
        "per_root_deltas_sorted_worst_first": per_root,
        "prediction_failures": {
            model: sum(not bool(row.get("prediction_valid")) for row in rows)
            for model, rows in scores.items()
        },
    }
    payload["status"] = "PASSED" if not errors else "FAILED"
    payload["content_sha256"] = _sha(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "validation_root_count": len(samples),
        "published_metric_match_errors": errors,
        "route_ade": independent_metrics["route_ade_m"],
        "speed_wp_ade": independent_metrics["speed_wp_ade_m"],
        "worst_route_deltas": per_root[:5],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
