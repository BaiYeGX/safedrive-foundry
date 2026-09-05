"""Train-only, no-CARLA intervention screening for the C2 repair protocol."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from driving_vla.adapter.policy_adapter import ObservationBundle
from driving_vla.hybrid.contracts import (
    CandidateProvenance, HybridCandidate, HybridCandidateSet, HybridSource,
    ObservableAnchor,
)
from driving_vla.hybrid.guard import CandidateGuard
from safety_kernel.contracts.types import (
    CandidateSource, ComponentAvailability, ObservableSnapshot, ObservationPrivilege,
    PolicyCandidate, PolicyCandidateSet, TrafficLightObs, TrackedObject, TrajectoryPoint,
)
from safety_kernel.kernel import SafetyKernel

from data_pipeline.h2.contracts import stable_sha256
from data_pipeline.h2.live_contract import route_projection

from .interventions import FAMILY_OPERATORS, InterventionNotApplicable, derive_scaled_intervention
from .repair import read_json


def _anchor(payload: Mapping[str, Any]) -> ObservableAnchor:
    snap = payload["observable_snapshot"]
    meta = payload["observable_anchor"]
    actors = tuple(TrackedObject(**dict(row)) for row in snap.get("actors", ()))
    lights = tuple(TrafficLightObs(**dict(row)) for row in snap.get("traffic_lights", ()))
    safety = ObservableSnapshot(
        run_id=snap["run_id"], frame_id=snap["frame_id"], scenario_id=snap["scenario_id"],
        simulation_time_s=float(snap["simulation_time_s"]), wall_time_s=float(snap["wall_time_s"]),
        ego_x=float(snap["ego_x"]), ego_y=float(snap["ego_y"]), ego_yaw=float(snap["ego_yaw"]),
        ego_v=float(snap["ego_v"]), ego_a=float(snap.get("ego_a", 0.0)),
        observed_time_s=float(snap.get("observed_time_s", snap["simulation_time_s"])),
        freshness_s=float(snap.get("freshness_s", 0.0)), speed_limit_mps=snap.get("speed_limit_mps"),
        actors=actors, traffic_lights=lights,
        corridor_centerline=tuple(tuple(map(float, row)) for row in snap.get("corridor_centerline", ())),
        corridor_half_width_m=float(snap.get("corridor_half_width_m", 1.75)),
        privilege=ObservationPrivilege.OBSERVABLE, oracle_fields={},
        schema_version=snap.get("schema_version", "safedrive.safety.contracts.v1"),
        coordinate_frame="map",
    )
    history = tuple((float(row["ego_x"]), float(row["ego_y"]), float(row["ego_yaw"]),
                     float(row["ego_speed_mps"])) for row in payload.get("observable_history", ()))
    bundle = ObservationBundle(
        run_id=meta["run_id"], frame_id=meta["frame_id"], scenario_id=meta["scenario_id"],
        simulation_time_s=float(meta["simulation_time_s"]), wall_time_s=float(meta["wall_time_s"]),
        carla_frame=int(meta["carla_frame"]), ego_x=float(snap["ego_x"]), ego_y=float(snap["ego_y"]),
        ego_yaw=float(snap["ego_yaw"]), ego_v=float(snap["ego_v"]),
        route_xy=tuple(tuple(map(float, row)) for row in payload["route"]),
        ego_history=history, meta={"cora_screen_replay": True},
    )
    return ObservableAnchor(
        observation_id=meta["observation_id"], bundle=bundle, safety_snapshot=safety,
        route_revision=meta["route_revision"], sensor_frames={k: int(v) for k, v in meta["sensor_frames"].items()},
        sensor_timestamps_s={k: float(v) for k, v in meta["sensor_timestamps_s"].items()},
    )


def _candidate(root_id: str, payload: Mapping[str, Any]) -> HybridCandidate:
    source_name = str(payload["audit_source"])
    prov = payload["provenance"]
    source = HybridSource.EXPERT if source_name == "expert" else HybridSource.VLA
    policy_source = CandidateSource.CLASSIC if source_name == "expert" else CandidateSource.VLA_FAST
    points = tuple(TrajectoryPoint(**dict(row)) for row in payload["trajectory"])
    candidate = PolicyCandidate(
        candidate_id=payload["proposal_id"], source=policy_source,
        generated_time_s=float(prov["simulation_time_s"]), valid_until_s=float(prov["simulation_time_s"])+0.25,
        probability=1.0, points=points, behavior="nominal", intended_action=source_name,
        dynamics_meta={"cora_screen_replay": True},
    )
    provenance = CandidateProvenance(
        source=source, candidate_id=payload["proposal_id"], observation_id=prov["observation_id"],
        frame_id=prov["frame_id"], carla_frame=int(prov["carla_frame"]), simulation_time_s=float(prov["simulation_time_s"]),
        route_revision=prov["route_revision"], generator_id=prov["generator_id"], generator_hash=prov["generator_hash"],
        raw_sha256=prov["raw_sha256"], canonical_sha256=prov["canonical_sha256"],
        canonicalizer_version=prov["canonicalizer_version"], canonicalization_error_m=float(prov["canonicalization_error_m"]),
        coverage_shortfall_m=float(prov["coverage_shortfall_m"]), generation_latency_s=float(prov["generation_latency_s"]),
        generated_wall_time_s=float(prov["generated_wall_time_s"]), freshness_s=float(prov["freshness_s"]), coordinate_frame="map",
    )
    return HybridCandidate(candidate=candidate, provenance=provenance)


def _safety(anchor: ObservableAnchor, item: HybridCandidate) -> dict[str, Any]:
    guarded = CandidateGuard().evaluate_candidate(HybridCandidateSet(anchor=anchor, candidates=(item,)), item)
    item = item.with_guard(guarded)
    if not guarded.passed:
        return {"guard": guarded.to_dict(), "safety": None, "repair_failure": False, "offroad_risk": False}
    candidate = item.candidate
    cset = PolicyCandidateSet(
        run_id=anchor.bundle.run_id, frame_id=anchor.bundle.frame_id, scenario_id=anchor.bundle.scenario_id,
        model_id="cora-screen@1.0.0", carla_frame=anchor.bundle.carla_frame,
        simulation_time_s=anchor.bundle.simulation_time_s, wall_time_s=anchor.bundle.wall_time_s,
        candidates=(candidate,), schema_version="safedrive.safety.contracts.v1", coordinate_frame="map")
    source = str(getattr(candidate.source, "value", candidate.source))
    kernel = SafetyKernel()
    result = kernel.tick(anchor.safety_snapshot, cset, now_s=anchor.bundle.simulation_time_s,
                         availability=ComponentAvailability(classic=source == "classic", vla=source.startswith("vla"), world=False, safety=True))
    route = anchor.bundle.route_xy
    offroad = any(route_projection(p.x, p.y, route)[1] > anchor.safety_snapshot.corridor_half_width_m
                  for p in candidate.points)
    traces = [t for t in kernel.repair_traces if t.get("mode") in {"longitudinal", "rato"}]
    # A repair-failure arm means an observed attempt with no successful
    # executable, not merely a failed first stage followed by RATO success.
    failure = bool(traces) and not any(bool(t.get("success")) for t in traces)
    return {"guard": guarded.to_dict(), "safety": {
        "decision_kind": str(getattr(result.decision.decision_kind, "value", result.decision.decision_kind)),
        "executed": result.decision.executed_trajectory_id, "repair_traces": list(kernel.repair_traces)},
        "repair_failure": failure, "offroad_risk": offroad}


def screen_train(base_dataset: Path, *, multipliers=(1.0, 2.0, 3.0)) -> dict[str, Any]:
    rows = []
    for path in sorted((base_dataset / "pairs").glob("*.json")):
        record = read_json(path)
        if record.get("split") != "train" or record.get("scenario", {}).get("map_name") != "Town03":
            continue
        anchor_payload = read_json(base_dataset / record["anchor_path"])
        anchor = _anchor(anchor_payload)
        nominal = {str(p.get("audit_source")): p for p in record.get("proposals", ()) if p.get("kind") == "nominal"}
        family = str(record["scenario"]["family"])
        for source, proposal in sorted(nominal.items()):
            base = _candidate(record["root_id"], proposal)
            for operator in FAMILY_OPERATORS[family]:
                for multiplier in multipliers:
                    try:
                        result = derive_scaled_intervention(record["root_id"], anchor, base, operator, float(multiplier))
                        measured = _safety(anchor, result.candidate)
                    except InterventionNotApplicable as exc:
                        measured = {"guard": None, "safety": None, "repair_failure": False, "offroad_risk": False, "status": str(exc)}
                    rows.append({"root_id": record["root_id"], "family": family, "source": source,
                                 "operator": operator, "multiplier": multiplier, **measured})
    targets = {"repair_failure": [], "offroad_risk": []}
    for key in targets:
        counts = {}
        for row in rows:
            if row.get(key):
                ident = (row["operator"], float(row["multiplier"]), row["source"])
                counts[ident] = counts.get(ident, 0) + 1
        targets[key] = [dict(operator=k[0], multiplier=k[1], source=k[2], appearances=v)
                        for k, v in sorted(counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0], item[0][2]))]
    return {"schema_version": "safedrive.cora.screening.v1", "scope": "train_town03_only",
            "multipliers": list(multipliers), "rows": rows, "ranked": targets,
            "screening_sha256": stable_sha256({"rows": rows, "ranked": targets})}


__all__ = ["screen_train"]
