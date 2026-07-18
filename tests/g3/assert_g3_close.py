#!/usr/bin/env python3
"""G3 stage-close gate: neural F0 + honest neural CARLA live (v2 schema).

Rejects legacy evidence that only checked steps/all_ok without Safety lineage.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F0 = ROOT / "docs/architecture/evidence/g3-03/f0_neural/report.json"
# Prefer v2 live evidence; fall back only if env explicitly allows legacy path.
LIVE_V2 = ROOT / "docs/architecture/evidence/g3-05/neural_live_v2/latest_live_summary.json"
LIVE_LEGACY = ROOT / "docs/architecture/evidence/g3-05/neural_live/latest_live_summary.json"

MIN_LATENCY_N = int(os.environ.get("SDF_G3_CLOSE_MIN_LATENCY_N", "20"))
REQUIRE_V2 = os.environ.get("SDF_G3_CLOSE_REQUIRE_V2", "1") not in {"0", "false", "False"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_f0(errors: list[str]) -> None:
    if not F0.is_file():
        errors.append(f"missing F0 neural report: {F0}")
        return
    f0 = _load(F0)
    if not f0.get("f0_pass"):
        errors.append(f"f0_pass false asserts={f0.get('asserts')}")
    if f0.get("source") not in {"neural_simlingo", "neural_ivl"}:
        errors.append(f"bad f0 source {f0.get('source')}")
    gpu = (f0.get("stages") or {}).get("gpu") or {}
    if float(gpu.get("peak_vram_mb") or 0) < 1024:
        errors.append(f"H5 vram too low: {gpu}")
    lat = (f0.get("stages") or {}).get("latency") or {}
    if float(lat.get("p50_ms") or 0) < 5:
        errors.append(f"H6 p50 too low (geom?): {lat}")
    n_lat = int(lat.get("n") or 0)
    if n_lat < MIN_LATENCY_N:
        errors.append(f"F0 latency n={n_lat} < {MIN_LATENCY_N}")
    # SHA required when present in schema; empty/deferred fails when key exists as deferred
    lineage = f0.get("lineage") or {}
    sha = str(lineage.get("checkpoint_sha256") or f0.get("checkpoint_sha256") or "")
    if sha.startswith("deferred") or sha == "deferred_full_sha_use_SDF_G3_FULL_HASH":
        errors.append(f"checkpoint sha deferred: {sha[:40]}")


def _pick_live() -> Path | None:
    if LIVE_V2.is_file():
        return LIVE_V2
    if REQUIRE_V2:
        return None
    if LIVE_LEGACY.is_file():
        return LIVE_LEGACY
    return None


def _check_live(errors: list[str]) -> None:
    live_path = _pick_live()
    if live_path is None:
        errors.append(
            f"missing neural live summary (require v2={REQUIRE_V2}): "
            f"expected {LIVE_V2}"
        )
        return
    if live_path == LIVE_LEGACY:
        errors.append(
            "legacy neural_live evidence is INVALID for stage close; "
            "use neural_live_v2 after safety-bind fix"
        )
        return

    live = _load(live_path)
    if live.get("force_throttle") is True:
        errors.append("force_throttle true — safety bypass")
    if live.get("evidence_schema") not in {"g3_live_v2", "g3_live_v2_fault"}:
        # Allow missing only if rich fields present (partial migration)
        if "n_track_approved" not in str(live.get("results")):
            errors.append(
                f"live evidence_schema not g3_live_v2: {live.get('evidence_schema')}"
            )

    backend = str(live.get("policy_backend") or "")
    if not backend.startswith("neural"):
        errors.append(f"live backend not neural: {backend}")
    if live.get("classic_current_frame") is True and live.get("mode") == "VLA_SAFETY":
        errors.append("classic_current_frame true under VLA_SAFETY")
    if not live.get("camera"):
        errors.append("live camera false")
    if not live.get("all_ok"):
        errors.append(f"live all_ok false results={live.get('results')}")

    results = live.get("results") or []
    seeds = live.get("seeds") or []
    if len(seeds) < 2 and live.get("mode") == "VLA_SAFETY":
        if len(results) < 2:
            errors.append(
                f"need >=2 seeds/results for VLA_SAFETY, got seeds={seeds} n_results={len(results)}"
            )

    # sources_seen must be honest union of per-result sources
    top_sources = set(live.get("sources_seen") or [])
    union: set[str] = set()
    total_track = 0
    for r in results:
        if r.get("policy_backend") and not str(r["policy_backend"]).startswith("neural"):
            errors.append(f"result backend {r.get('policy_backend')}")
        if int(r.get("camera_frames") or 0) < 1:
            errors.append(f"seed {r.get('seed')} camera_frames=0")
        if r.get("status") != "COMPLETED":
            errors.append(f"seed {r.get('seed')} status={r.get('status')} reasons={r.get('fail_reasons')}")
        # New schema fields required
        if "n_track_approved" not in r:
            errors.append(f"seed {r.get('seed')} missing n_track_approved (legacy/invalid evidence)")
        else:
            total_track += int(r.get("n_track_approved") or 0)
        if r.get("force_throttle") is True:
            errors.append(f"seed {r.get('seed')} force_throttle true")
        for s in r.get("sources_seen") or []:
            union.add(str(s))
        # Reject the original false success pattern
        tail = r.get("decision_tail") or []
        dist = float(r.get("distance_m") or 0.0)
        n_em = int(r.get("n_emergency") or 0)
        steps = int(r.get("steps") or 0)
        if dist > 20.0 and tail and all(str(d) == "EMERGENCY" for d in tail):
            if int(r.get("n_track_approved") or 0) == 0:
                errors.append(
                    f"seed {r.get('seed')} all-EMERGENCY tail with distance={dist:.1f}m and no track"
                )
        if steps > 0 and n_em >= int(0.25 * steps) and dist > 20.0:
            errors.append(
                f"seed {r.get('seed')} high emergency ratio {n_em}/{steps} with distance={dist:.1f}m"
            )

    if live.get("mode") == "VLA_SAFETY" and total_track < 1:
        errors.append(f"total n_track_approved={total_track} < 1")

    if top_sources != union:
        errors.append(
            f"sources_seen mismatch: top={sorted(top_sources)} union={sorted(union)}"
        )


def main() -> int:
    errors: list[str] = []
    _check_f0(errors)
    _check_live(errors)

    if errors:
        print("G3 CLOSE FAIL:")
        for e in errors:
            print(" -", e)
        return 1
    print("G3 CLOSE PASS: neural F0 + neural live v2 evidence OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
