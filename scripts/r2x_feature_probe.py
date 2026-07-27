#!/usr/bin/env python3
"""X5A feature signal probe on collected anchors (offline).

Expects Evidence from r2x_feature_collect.py:
  docs/runtime-evidence/r2x-feature-probe/anchors/*/feature.json

Reports:
  - stability (duplicate hashes / per-anchor repeat)
  - finite/non-empty
  - inter-sample variance
  - linear probe left/right/empty (or family labels)
  - real vs synthetic eligibility for X5A PASS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.driving_feature import linear_probe_labels_accuracy  # noqa: E402

DEFAULT_EV = ROOT / "docs/runtime-evidence/r2x-feature-probe"


def _load_anchors(ev: Path) -> list[dict]:
    rows = []
    ad = ev / "anchors"
    if not ad.is_dir():
        return rows
    for p in sorted(ad.glob("*/feature.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=str, default=str(DEFAULT_EV))
    ap.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="if set, synthetic anchors may yield PIPELINE_PASS (never X5A_REAL_PASS)",
    )
    args = ap.parse_args()
    ev = Path(args.evidence)
    rows = _load_anchors(ev)
    if not rows:
        report = {
            "schema_version": "safedrive.r2x.feature_probe.v1",
            "status": "NO_DATA",
            "n": 0,
            "note": "no anchors; run r2x_feature_collect.py first",
        }
        ev.mkdir(parents=True, exist_ok=True)
        (ev / "feature_probe_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 2

    n = len(rows)
    n_ok = sum(1 for r in rows if r.get("driving_feature_ok"))
    n_real = sum(1 for r in rows if r.get("is_real_simlingo_feature"))
    n_syn = n - n_real
    mean64 = [r.get("mean64") or r.get("driving_feature") or [] for r in rows]
    full = [r.get("full_pool") or [] for r in rows]
    hashes = [r.get("mean64_hash") or r.get("driving_feature_hash") or "" for r in rows]
    raw_hashes = [r.get("raw_content_hash") or r.get("driving_feature_raw_hash") or "" for r in rows]
    modes = sorted({str(r.get("simlingo_mode") or "unknown") for r in rows})

    def _var_report(vecs: list) -> dict:
        xs = [np.asarray(v, dtype=np.float64) for v in vecs if v]
        if len(xs) < 2:
            return {"n": float(len(xs)), "mean_pairwise_l2": 0.0}
        d = []
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                a, b = xs[i], xs[j]
                m = min(a.size, b.size)
                d.append(float(np.linalg.norm(a[:m] - b[:m])))
        return {
            "n": float(len(xs)),
            "mean_pairwise_l2": float(np.mean(d)) if d else 0.0,
            "p50_pairwise_l2": float(np.median(d)) if d else 0.0,
        }

    labels = []
    label_names = []
    for r in rows:
        side = str(r.get("conflict_side") or r.get("label_side") or "").lower()
        if side in {"left", "right", "none", "empty", "center"}:
            name = "empty" if side in {"none", "empty"} else side
        else:
            fam = str(r.get("scenario_family") or "unk").lower()
            if "left" in fam:
                name = "left"
            elif "right" in fam:
                name = "right"
            elif "empty" in fam or "clear" in fam:
                name = "empty"
            else:
                name = fam[:12]
        label_names.append(name)
    uniq = sorted(set(label_names))
    name_to_i = {n: i for i, n in enumerate(uniq)}
    labels = [name_to_i[n] for n in label_names]

    probe_mean64 = linear_probe_labels_accuracy(mean64, labels, seed=7)
    probe_full = (
        linear_probe_labels_accuracy(full, labels, seed=7)
        if all(len(v) > 0 for v in full)
        else {"ok": 0.0, "accuracy": 0.0, "note": "full_pool_missing"}
    )

    by_obs: dict[str, set[str]] = {}
    for r in rows:
        oh = str(r.get("observation_hash") or r.get("image_hash") or "")
        if oh:
            by_obs.setdefault(oh, set()).add(str(r.get("mean64_hash") or ""))
    stability_conflicts = sum(1 for s in by_obs.values() if len(s) > 1)

    # per-anchor repeat stability from collect
    repeat_fail = 0
    for r in rows:
        st = r.get("stability") or {}
        if st and st.get("repeat_match") is False:
            repeat_fail += 1

    finite_ok = all(
        all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for v in (mean64[i],) if v)
        for i in range(n)
    )
    dims_ok = all(len(v) == 64 for v in mean64 if v)

    signal_ok = (
        n_ok == n
        and n >= 8
        and stability_conflicts == 0
        and repeat_fail == 0
        and finite_ok
        and dims_ok
        and float(probe_mean64.get("ok", 0)) >= 1.0
        and float(probe_mean64.get("accuracy", 0))
        >= float(probe_mean64.get("chance", 0.25)) + 0.25
    )

    families = sorted({str(r.get("scenario_family") or "") for r in rows})
    n_classes = int(probe_mean64.get("n_classes") or len(uniq))
    # Full X5A wants spatial diversity: left/right or left/empty-like multi-class
    has_lr = ("left" in uniq and "right" in uniq) or (
        "left_cut_in" in families and "right_cut_in" in families
    )
    has_empty = "empty" in uniq or any("empty" in f for f in families)
    coverage_full = n >= 24 and n_classes >= 3 and (has_lr or (has_empty and n_classes >= 3))
    coverage_partial = n >= 8 and n_classes >= 2

    all_real = n_real == n and n >= 8
    if signal_ok and all_real and coverage_full:
        status = "X5A_REAL_PASS"
        exit_code = 0
    elif signal_ok and all_real and coverage_partial:
        # Real SimLingo signal confirmed, but family inventory incomplete
        status = "X5A_REAL_PARTIAL_PASS"
        exit_code = 0
    elif signal_ok and n_syn == n and args.allow_synthetic:
        status = "PIPELINE_PASS_SYNTHETIC_ONLY"
        exit_code = 0
    elif signal_ok and not all_real:
        status = "PIPELINE_PASS_NOT_REAL"
        exit_code = 4  # not X5A real
    elif not signal_ok:
        status = "FAIL"
        exit_code = 3
    else:
        status = "FAIL"
        exit_code = 3

    report = {
        "schema_version": "safedrive.r2x.feature_probe.v1",
        "status": status,
        "n": n,
        "n_feature_ok": n_ok,
        "n_real_simlingo": n_real,
        "n_synthetic": n_syn,
        "simlingo_modes": modes,
        "finite_ok": finite_ok,
        "dims_ok": dims_ok,
        "stability_conflicts": stability_conflicts,
        "repeat_stability_fail": repeat_fail,
        "variance_mean64": _var_report(mean64),
        "variance_full_pool": _var_report(full),
        "linear_probe_mean64": probe_mean64,
        "linear_probe_full_pool": probe_full,
        "label_names": uniq,
        "unique_mean64_hashes": len(set(hashes)),
        "unique_raw_hashes": len(set(raw_hashes)),
        "families": families,
        "coverage": {
            "n_classes": n_classes,
            "has_left_right": has_lr,
            "has_empty": has_empty,
            "coverage_full": coverage_full,
            "coverage_partial": coverage_partial,
        },
        "gates": {
            "all_feature_ok": n_ok == n,
            "min_n": 8,
            "stability_conflicts_max": 0,
            "repeat_stability_fail_max": 0,
            "probe_accuracy_min": "chance+0.25",
            "x5a_real_requires_all_real_simlingo": True,
            "x5a_full_requires_n24_classes3_lr_or_empty": True,
        },
        "x5a_real_pass": status == "X5A_REAL_PASS",
        "x5a_real_partial_pass": status == "X5A_REAL_PARTIAL_PASS",
        "note": (
            "X5A_REAL_PASS enables full X5B expand; "
            "X5A_REAL_PARTIAL_PASS = real SimLingo signal but incomplete family coverage "
            "(may pilot teacher tooling, not freeze train registry); "
            "PIPELINE_PASS_SYNTHETIC_ONLY only validates probe plumbing; "
            "FAIL → representation ablation (mean64 vs full_pool vs tokens+queries)"
        ),
    }
    ev.mkdir(parents=True, exist_ok=True)
    (ev / "feature_probe_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
