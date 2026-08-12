#!/usr/bin/env python3
"""Build the V4 semantic-head JSONL from frozen anchor evidence.

Labels come only from the pre-frozen counterfactual manifest.  The script does
not inspect outcomes or choose examples after collection; missing anchors,
token hashes, or split coverage fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.v4_token_features import DrivingTokenBundleV4  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_attempts(root: Path) -> list[Path]:
    return sorted(
        {
            path.parent.parent
            for path in root.rglob("formal_trace_v4.json")
            if (path.parent / "driving_tokens_v4.npy").is_file()
        }
    )


def build_rows(evidence_roots: list[Path], campaign: dict[str, Any]) -> list[dict[str, Any]]:
    slot_by_scenario = {
        str(slot["scenario_id"]): dict(slot) for slot in campaign.get("slots", [])
    }
    attempts = sorted({attempt for root in evidence_roots for attempt in _walk_attempts(root)})
    if not attempts:
        raise ValueError("no V4 formal traces with raw token dumps found")
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        anchor_dir = attempt / "anchor"
        trace = json.loads((anchor_dir / "formal_trace_v4.json").read_text(encoding="utf-8"))
        pair = json.loads((attempt / "pair_report.json").read_text(encoding="utf-8"))
        scenario_id = str(pair.get("scenario_id") or "")
        slot = slot_by_scenario.get(scenario_id)
        if slot is None:
            raise ValueError(f"collected scenario not present in frozen manifest: {scenario_id}")
        if str(trace.get("namespace")) not in {
            "r2_v4_calibration",
            "r2_v4_formal",
            "r2_v4_pilot",
        }:
            raise ValueError(f"unexpected V4 trace namespace for {scenario_id}")
        if int(trace.get("semantic_rescue_count", 0)) != 0:
            raise ValueError(f"semantic rescue present in {scenario_id}")
        if int(trace.get("scenario_family_runtime_use", 0)) != 0:
            raise ValueError(f"scenario family runtime use present in {scenario_id}")
        if not isinstance(trace.get("raw_head_output"), dict) or not str(
            trace.get("raw_head_output_hash") or ""
        ):
            raise ValueError(f"raw V4 head output missing for {scenario_id}")
        token_path = anchor_dir / "driving_tokens_v4.npy"
        token_bundle = DrivingTokenBundleV4.from_adaptor_output(
            np.load(token_path, allow_pickle=False),
            raw_tensor_path=token_path,
        )
        declared_hash = str(
            (json.loads((anchor_dir / "driving_tokens_v4.json").read_text(encoding="utf-8")))
            .get("raw_content_hash")
            or ""
        ).lower()
        if declared_hash and declared_hash != token_bundle.raw_content_hash.lower():
            raise ValueError(f"token metadata hash mismatch in {scenario_id}")
        identity = str(pair.get("pair_id") or attempt.name)
        aux = (json.loads((anchor_dir / "anchor_bundle_v3.json").read_text(encoding="utf-8"))
               .get("bundle", {})
               .get("observation_identity", {})
               .get("v4_aux"))
        if not isinstance(aux, list) or len(aux) != 178:
            raise ValueError(f"V4 aux vector missing for {scenario_id}")
        branch_summaries = list((pair.get("branches") or {}).values())
        executable = bool(branch_summaries) and all(
            bool((summary.get("metrics") or {}).get("completed_primary_horizon", False))
            and int((summary.get("metrics") or {}).get("mpc_timeout_count", 0)) == 0
            and int((summary.get("metrics") or {}).get("mpc_fallback_count", 0)) == 0
            for summary in branch_summaries
        )
        legal_route_target = str(trace.get("guard_status") or "") == "OK"
        if "expected_maneuver_params" not in slot:
            raise ValueError(f"frozen V4 slot lacks maneuver parameter labels: {scenario_id}")
        params = slot.get("expected_maneuver_params")
        if not isinstance(params, list) or len(params) != 6:
            raise ValueError(f"frozen V4 slot maneuver parameter labels malformed: {scenario_id}")
        rows.append(
            {
                "row_id": identity,
                "pair_id": identity,
                "scenario_id": scenario_id,
                "lineage_id": str(slot["lineage_id"]),
                "group_key": str(slot["root_group"]),
                "split": str(slot["split"]),
                "map_name": str(slot["map_name"]),
                "family": str(slot["family"]),
                "condition_variant": str(slot["condition_variant"]),
                "v4_raw_tensor_path": str(token_path.resolve()),
                "v4_token_raw_content_hash": token_bundle.raw_content_hash,
                "v4_aux": aux,
                "target_kind": str(slot["expected_kind"]),
                "target_side": str(slot["expected_side"]),
                "available": bool(slot["expected_available"]),
                "maneuver_params": [float(value) for value in params],
                "anchor_artifact_hash": str(pair.get("artifact_content_hash") or ""),
                "raw_head_output_hash": str(trace.get("raw_head_output_hash") or ""),
                "checkpoint_sha256": str(trace.get("checkpoint_sha256") or ""),
                "guard_mpc_executable": executable,
                "legal_route_target": legal_route_target,
                "artifact_schema": str(
                    (json.loads((anchor_dir / "anchor_bundle_v3.json").read_text(encoding="utf-8"))).get("schema_version")
                    or ""
                ),
            }
        )
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate V4 row identity")
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row["group_key"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("V4 root-lineage split overlap")
    return sorted(rows, key=lambda row: row["row_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", action="append", required=True)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign = json.loads(Path(args.campaign_manifest).read_text(encoding="utf-8"))
    rows = build_rows([Path(value) for value in args.evidence_root], campaign)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite V4 dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": len(rows), "sha256": _sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
