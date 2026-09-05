"""Trace-backed C2 correction labels; never rewrite legacy artifacts."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from data_pipeline.h2.live_contract import route_projection
from safety_kernel.contracts.serialize import decision_to_dict
from .outcomes import derive_public_outcome_heads, validate_public_outcome_heads

SCHEMA = "safedrive.cora.outcome_labels.v3"
VERSION = "cora-c2-trace-and-root-correction-v3"
TRACE_SCHEMA = "safedrive.cora.safety_trace.v1"


def safety_trace(result: Any, attempts: Sequence[Mapping[str, Any]], proposal_id: str) -> dict[str, Any]:
    rows = copy.deepcopy(list(attempts))
    if not rows and result.repair_result is not None:
        rows = [result.repair_result.to_dict()]
    for row in rows:
        if row.get("pre_repair_id") != proposal_id:
            raise ValueError("repair_trace_parent_mismatch")
    decision = result.decision
    kind = str(getattr(decision.decision_kind, "value", decision.decision_kind))
    succeeded = bool(rows and kind in {"QP", "RATO"} and decision.accepted_candidate is not None)
    selected = next((row for row in reversed(rows) if row.get("success") and
                     row.get("post_repair_id") == decision.post_repair_trajectory_id), None)
    if succeeded and selected is None:
        raise ValueError("repair_trace_selected_executable_missing")
    return {
        "schema_version": TRACE_SCHEMA, "proposal_id": proposal_id,
        "attempts": rows, "observed": True, "repair_attempted": bool(rows),
        "repair_success": succeeded if rows else None,
        "repair_mode": selected.get("mode") if selected else (rows[-1].get("mode") if rows else None),
        "decision": decision_to_dict(decision),
    }


def _head(value: Any, unit: str, valid: bool = True) -> dict[str, Any]:
    return {"value": value if valid else None, "unit": unit, "valid": bool(valid),
            "derivation_version": VERSION}


def route_labels(timeline: Sequence[Mapping[str, Any]], physical: Mapping[str, Any],
                 initial_xy: Sequence[float] | None, initial_light: str | None = None) -> dict[str, Any]:
    route = physical.get("route", ())
    if len(route) < 2 or not timeline or initial_xy is None:
        return {"route_completed": _head(None, "bool", False),
                "local_goal_completed": _head(None, "bool", False),
                "red_light_violation": _head(None, "bool", False)}
    total = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(route, route[1:]))
    start = route_projection(float(initial_xy[0]), float(initial_xy[1]), route)[0]
    previous = start
    crossed_red = False
    unknown = False
    stop = physical.get("red_light")
    script = physical.get("script", {})
    for row in timeline:
        absolute = route_projection(float(row["x"]), float(row["y"]), route)[0]
        if stop and previous <= float(stop["stop_progress_m"]) + 1.0 < absolute:
            light = row.get("traffic_light_state")
            if light is None:
                red_tick = script.get("red_after_tick")
                light = ("Red" if script.get("dynamic_traffic_light_timing") and red_tick is not None
                         and int(row["tick"]) >= int(red_tick) else initial_light)
            if light is None:
                unknown = True
            elif str(light).split(".")[-1].lower() == "red":
                crossed_red = True
        previous = absolute
    completed = previous >= total - 2.0
    return {"route_completed": _head(completed, "bool"),
            "local_goal_completed": _head(completed, "bool"),
            "red_light_violation": _head(crossed_red, "bool", crossed_red or not unknown)}


def derive_v3(branch: Mapping[str, Any], *, timeline: Sequence[Mapping[str, Any]],
              events: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any],
              physical: Mapping[str, Any], trace: Mapping[str, Any] | None = None) -> dict[str, Any]:
    heads = derive_public_outcome_heads(branch, timeline=timeline, events=events,
        corridor_half_width_m=anchor.get("observable_snapshot", {}).get("corridor_half_width_m"))
    for head in heads.values():
        head["derivation_version"] = VERSION
    if trace is not None:
        if trace.get("schema_version") != TRACE_SCHEMA or trace.get("proposal_id") != branch["proposal_id"]:
            raise ValueError("repair_trace_binding")
        observed = trace.get("observed") is True
        attempted = trace.get("repair_attempted") is True
        heads["repair_attempted"] = _head(attempted, "bool", observed)
        heads["repair_success"] = _head(trace.get("repair_success"), "bool",
            observed and attempted and isinstance(trace.get("repair_success"), bool))
        heads["repair_mode"] = _head(trace.get("repair_mode"), "enum",
            observed and attempted and trace.get("repair_mode") is not None)
    elif not (branch.get("decision_kind") in {"QP", "RATO"} and branch.get("identity_valid")
              and branch.get("executable_id") and branch.get("repair_success") is True):
        for name, unit in (("repair_attempted", "bool"), ("repair_success", "bool"), ("repair_mode", "enum")):
            heads[name] = _head(None, unit, False)
    snap = anchor.get("observable_snapshot", {})
    initial = ((snap["ego_x"], snap["ego_y"]) if "ego_x" in snap else None)
    if trace and trace.get("initial_xy") is not None:
        initial = trace["initial_xy"]
    lights = [x for x in snap.get("traffic_lights", ()) if x.get("controls_ego_lane") is not False]
    states = {str(x.get("state")) for x in lights if x.get("state") is not None}
    light = next(iter(states)) if len(states) == 1 else None
    heads.update(route_labels(timeline, physical, initial, light))
    validate_public_outcome_heads(heads, derivation_version=VERSION)
    return heads
