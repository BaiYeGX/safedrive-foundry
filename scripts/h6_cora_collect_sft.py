#!/usr/bin/env python3
"""Pre-register and fail-closed collect the bounded C3 supplement cohort.

The plan is intentionally usable without importing CARLA, which makes it
possible to freeze physical identities before a server is started.  Collection
does one real connection preflight through the repository's ConnectionResolver.
If the Windows server cannot reach READY, every planned root remains a failed
or pending record and the resulting empty supplement is still a valid input to
the offline C3 audit.  No synthetic image, trajectory, or accepted root is
written to make the count look complete.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "safedrive_foundry", REPO / "simlingo-main"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from runtime.carla_connection import ConnectionResolver, READY  # noqa: E402


PLAN_SCHEMA = "safedrive.c3.supplement_plan.v1"
MANIFEST_SCHEMA = "safedrive.c3.supplement_manifest.v1"
MAX_ROOTS = 48
MAX_ATTEMPTS = 96
MAX_CARLA_SECONDS = 4.0 * 3600.0
MAX_ATTEMPT_SECONDS = 60.0
GROUPS = (
    ("straight_nonzero_initial", "free_flow"),
    ("turn_nonzero_initial", "cut_in"),
    ("follow_decel", "slow_lead"),
    ("stop_start", "red_light_dilemma"),
)
MAPS = ("Town01", "Town03", "Town05")
WEATHER = "ClearNoon"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing_to_overwrite:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["content_sha256"] = _sha({key: value for key, value in result.items() if key not in {"content_sha256", "created_at_utc"}})
    return result


def _run_dir(args: argparse.Namespace) -> Path:
    run_id = str(args.run_id)
    if not run_id.startswith("c3-repair-"):
        raise ValueError("c3_supplement_requires_repair_run_id")
    return Path(args.output_root).expanduser().resolve() / run_id


def _old_identity(args: argparse.Namespace) -> dict[str, Any]:
    release_path = Path(args.release_root).resolve() / "release-index.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    rows = release.get("samples", ())
    roots = sorted(str(row.get("pair_id", "")) for row in rows if row.get("pair_id"))
    return {
        "release_index": str(release_path),
        "release_index_sha256": _file_sha(release_path),
        "root_count": len(roots),
        "root_ids_sha256": _sha(roots),
    }


def _plan(args: argparse.Namespace) -> Path:
    run_dir = _run_dir(args)
    output = run_dir / "supplement" / "manifest-plan.json"
    if output.exists():
        raise FileExistsError(f"supplement_plan_already_exists:{output}")
    old = _old_identity(args)
    rows: list[dict[str, Any]] = []
    group_rows = {group: {"train": 0, "validation": 0} for group, _family in GROUPS}
    index = 0
    for group_index, (group, family) in enumerate(GROUPS):
        for split, count in (("train", 8), ("validation", 4)):
            for local_index in range(count):
                seed = 300000 + group_index * 1000 + (0 if split == "train" else 500) + local_index
                map_name = MAPS[index % len(MAPS)]
                root_id = f"{map_name}__{family}__s{seed}__{WEATHER}__c3supp"
                rows.append({
                    "root_id": root_id,
                    "pair_id": root_id,
                    "split": split,
                    "case_group": group,
                    "family": family,
                    "map": map_name,
                    "weather": WEATHER,
                    "seed": seed,
                    "priority": index,
                    "attempt_ids": [f"{root_id}__attempt0", f"{root_id}__attempt1"],
                    "max_attempts": 2,
                    "max_attempt_seconds": MAX_ATTEMPT_SECONDS,
                    "acceptance": {
                        "native_route_support_m_at_least": 19.0,
                        "speed_support_s_at_least": 2.5,
                        "must_have_same_frame_anchor": True,
                        "must_have_measured_history": True,
                        "must_have_no_unresolved_collision_or_offroad": True,
                    },
                    "status": "PLANNED",
                })
                group_rows[group][split] += 1
                index += 1
    if len(rows) != MAX_ROOTS or len({row["root_id"] for row in rows}) != MAX_ROOTS:
        raise AssertionError("c3_supplement_plan_count_or_identity")
    plan = _with_digest({
        "schema_version": PLAN_SCHEMA,
        "status": "PRE_REGISTERED",
        "run_id": str(args.run_id),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "limits": {
            "max_roots": MAX_ROOTS,
            "max_attempts": MAX_ATTEMPTS,
            "max_carla_seconds": MAX_CARLA_SECONDS,
            "max_attempt_seconds": MAX_ATTEMPT_SECONDS,
            "train_roots": 32,
            "validation_roots": 16,
        },
        "group_counts": group_rows,
        "old_identity": old,
        "excluded_identity_policy": "all_old_release_roots_and_reserved_or_locked_lineage_by_identity;_no_renamed_or_adjacent_frame_substitute",
        "excluded_root_ids_sha256": old["root_ids_sha256"],
        "rows": rows,
    })
    _write(output, plan)
    print(json.dumps({"status": plan["status"], "output": str(output), "root_count": len(rows), "content_sha256": plan["content_sha256"]}, indent=2))
    return output


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("c3_supplement_plan_schema")
    digest = payload.get("content_sha256")
    check = {key: value for key, value in payload.items() if key not in {"content_sha256", "created_at_utc"}}
    # ``created_at_utc`` is omitted from the semantic plan digest by design.
    if digest != _sha(check):
        raise ValueError("c3_supplement_plan_digest")
    rows = payload.get("rows", ())
    if len(rows) != MAX_ROOTS or len({str(row.get("root_id", "")) for row in rows}) != MAX_ROOTS:
        raise ValueError("c3_supplement_plan_root_count")
    if sum(len(row.get("attempt_ids", ())) for row in rows) != MAX_ATTEMPTS:
        raise ValueError("c3_supplement_plan_attempt_count")
    return payload


def _collect(args: argparse.Namespace) -> Path:
    run_dir = _run_dir(args)
    plan_path = Path(args.plan).resolve() if args.plan else run_dir / "supplement" / "manifest-plan.json"
    plan = _load_plan(plan_path)
    output = run_dir / "supplement" / "supplement-manifest.json"
    if output.exists():
        raise FileExistsError(f"supplement_manifest_already_exists:{output}")
    started = time.perf_counter()
    resolver = ConnectionResolver(REPO, expected_version="0.9.16", timeout_seconds=10.0)
    report = resolver.preflight()
    report_payload = report.to_dict() if hasattr(report, "to_dict") else {"status": str(getattr(report, "status", "UNKNOWN")), "repr": repr(report)}
    ready = str(getattr(report, "status", "")) == READY
    rows: list[dict[str, Any]] = []
    if not ready:
        for planned in plan["rows"]:
            rows.append({
                "root_id": planned["root_id"],
                "pair_id": planned["pair_id"],
                "split": planned["split"],
                "accepted": False,
                "status": "NOT_STARTED_CARLA_BLOCKED",
                "failure_reason": f"CARLA_NOT_READY:{getattr(report, 'error_code', 'UNKNOWN')}:{getattr(report, 'error_message', '')}",
                "attempts_started": 0,
                "attempts_reserved": len(planned.get("attempt_ids", ())),
                "case_group": planned["case_group"],
            })
        status = "CARLA_BLOCKED_EXTERNAL"
        collection_note = "No root was accepted because the unified preflight did not reach READY; no image or label was fabricated."
    else:
        # A READY server must enter the shared ScenarioRuntime capture path.
        # This guard prevents a future caller from accidentally collecting with
        # a direct world.tick or an unbound sensor loop.  The live implementation
        # is kept as a separate follow-up once the server is actually available.
        for planned in plan["rows"]:
            rows.append({
                "root_id": planned["root_id"],
                "pair_id": planned["pair_id"],
                "split": planned["split"],
                "accepted": False,
                "status": "NOT_STARTED_LIVE_CAPTURE_GUARD",
                "failure_reason": "CARLA_READY_BUT_SHARED_CAPTURE_PATH_NOT_ARMED",
                "attempts_started": 0,
                "attempts_reserved": len(planned.get("attempt_ids", ())),
                "case_group": planned["case_group"],
            })
        status = "CARLA_READY_COLLECTION_NOT_ARMED"
        collection_note = "READY was observed, but the bounded collector refused to write data until the shared ScenarioRuntime capture adapter is explicitly armed."
    elapsed = time.perf_counter() - started
    payload = _with_digest({
        "schema_version": MANIFEST_SCHEMA,
        "status": status,
        "run_id": str(args.run_id),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_path": str(plan_path),
        "plan_sha256": plan["content_sha256"],
        "carla_installation": r"E:\CARLA_0.9.16\CarlaUE4.exe",
        "preflight": report_payload,
        "ready": ready,
        "collection_note": collection_note,
        "attempts_started": 0,
        "attempts_reserved": MAX_ATTEMPTS,
        "accepted_count": 0,
        "accepted_counts": {"train": 0, "validation": 0},
        "rows": rows,
        "resource": {
            "carla_wall_time_s": elapsed,
            "gpu_optimization_wall_time_s": 0.0,
            "attempt_budget_seconds": MAX_CARLA_SECONDS,
            "evidence_basis": "single_unified_connection_preflight",
        },
    })
    _write(output, payload)
    print(json.dumps({"status": status, "output": str(output), "accepted_count": 0, "preflight_status": report_payload.get("status")}, ensure_ascii=False, indent=2))
    return output


def _audit(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args)
    plan_path = Path(args.plan).resolve() if args.plan else run_dir / "supplement" / "manifest-plan.json"
    manifest_path = Path(args.manifest).resolve() if args.manifest else run_dir / "supplement" / "supplement-manifest.json"
    plan = _load_plan(plan_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA or payload.get("plan_sha256") != plan.get("content_sha256"):
        raise ValueError("c3_supplement_audit_identity")
    check = {key: value for key, value in payload.items() if key not in {"content_sha256", "created_at_utc"}}
    errors: list[str] = []
    if payload.get("content_sha256") != _sha(check):
        errors.append("manifest_digest")
    rows = payload.get("rows", ())
    if len(rows) != MAX_ROOTS:
        errors.append("planned_row_count")
    if any(row.get("accepted") is True for row in rows):
        errors.append("accepted_rows_require_live_schema_audit")
    if payload.get("status") not in {"CARLA_BLOCKED_EXTERNAL", "CARLA_READY_COLLECTION_NOT_ARMED"}:
        errors.append("status")
    result = {
        "schema_version": "safedrive.c3.supplement_audit.v1",
        "status": "PASSED" if not errors else "FAILED",
        "run_id": str(args.run_id),
        "plan_sha256": plan.get("content_sha256"),
        "manifest_sha256": payload.get("content_sha256"),
        "planned_roots": len(rows),
        "accepted_roots": sum(row.get("accepted") is True for row in rows),
        "errors": errors,
        "carla_blocked_is_explicit": payload.get("status") == "CARLA_BLOCKED_EXTERNAL",
    }
    result["content_sha256"] = _sha(result)
    _write(run_dir / "supplement" / "audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "collect", "audit"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "generated/h6/cora")
    parser.add_argument("--release-root", type=Path, default=REPO / "generated/h6/cora/h6-cora-c2-devbaseline-20260907-v1")
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "plan":
        _plan(args)
        return 0
    if args.command == "collect":
        _collect(args)
        return 0
    return _audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
