"""X5C Spatial K2 defensive teacher — executable production contract (B2).

Production default: require_execution_filters=true and structured stages.
Unit tests must pass TeacherConfig(require_execution_filters=False) explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_TEACHER_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "vla" / "k2_spatial_teacher.toml"
)

PRODUCTION_STAGE_ORDER = (
    "guard_v2",
    "path_manager_densify_accepted",
    "steer_curvature_gate",
    "mpc_rollout_30tick",
)


@dataclass(frozen=True)
class TeacherFilterResult:
    stage_id: str
    passed: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "pass": self.passed,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


TeacherStageFn = Callable[[dict[str, Any]], TeacherFilterResult]


@dataclass(frozen=True)
class TeacherConfig:
    schema_version: str = "safedrive.k2_spatial_teacher.v8"
    teacher_id: str = "spatial_defensive_lattice_v8_topology_authorized"
    d_peaks_m: tuple[float, ...] = (-1.0, -0.7, -0.4, 0.0, 0.4, 0.7, 1.0)
    speed_scales: tuple[float, ...] = (0.70, 0.85, 1.00)
    side_shift_start_s: tuple[float, ...] = (0.30, 0.40)
    side_shift_recover_s: tuple[float, ...] = (0.75, 0.90)
    min_lateral_sep_m: float = 0.5
    forbid_empty_road_forced_swerve: bool = True
    require_strictly_better_than_nominal: bool = True
    require_execution_filters: bool = True  # production default
    horizon_ticks: int = 30
    better_margin_ttc_s: float = 0.25
    better_margin_clearance_m: float = 0.30
    max_comfort_degradation: float = 2.0
    min_progress_ratio: float = 0.85

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_teacher_config(path: str | Path | None = None) -> TeacherConfig:
    p = Path(path) if path else DEFAULT_TEACHER_CONFIG
    if not p.is_file():
        return TeacherConfig()
    try:
        try:
            import tomllib
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore
        data = tomllib.loads(p.read_text(encoding="utf-8"))
        avail = data.get("availability") or {}
        mpc = data.get("mpc_rollout") or {}
        return TeacherConfig(
            schema_version=str(data.get("schema_version") or TeacherConfig.schema_version),
            teacher_id=str(data.get("teacher_id") or TeacherConfig.teacher_id),
            d_peaks_m=tuple(
                float(x) for x in (data.get("d_peaks_m") or TeacherConfig.d_peaks_m)
            ),
            speed_scales=tuple(
                float(x) for x in (data.get("speed_scales") or TeacherConfig.speed_scales)
            ),
            side_shift_start_s=tuple(
                float(x)
                for x in (
                    data.get("side_shift_start_s") or TeacherConfig.side_shift_start_s
                )
            ),
            side_shift_recover_s=tuple(
                float(x)
                for x in (
                    data.get("side_shift_recover_s")
                    or TeacherConfig.side_shift_recover_s
                )
            ),
            min_lateral_sep_m=float(
                avail.get("min_lateral_sep_m", TeacherConfig.min_lateral_sep_m)
            ),
            forbid_empty_road_forced_swerve=bool(
                avail.get(
                    "forbid_empty_road_forced_swerve",
                    TeacherConfig.forbid_empty_road_forced_swerve,
                )
            ),
            require_strictly_better_than_nominal=bool(
                avail.get(
                    "require_strictly_better_than_nominal",
                    TeacherConfig.require_strictly_better_than_nominal,
                )
            ),
            require_execution_filters=bool(
                data.get("require_execution_filters", True)
            ),
            horizon_ticks=int(mpc.get("horizon_ticks", TeacherConfig.horizon_ticks)),
            better_margin_ttc_s=float(
                avail.get("better_margin_ttc_s", TeacherConfig.better_margin_ttc_s)
            ),
            better_margin_clearance_m=float(
                avail.get(
                    "better_margin_clearance_m",
                    TeacherConfig.better_margin_clearance_m,
                )
            ),
            max_comfort_degradation=float(
                avail.get(
                    "max_comfort_degradation", TeacherConfig.max_comfort_degradation
                )
            ),
            min_progress_ratio=float(
                avail.get("min_progress_ratio", TeacherConfig.min_progress_ratio)
            ),
        )
    except Exception:  # noqa: BLE001
        return TeacherConfig()


@dataclass
class LatticeCandidate:
    candidate_id: str
    d_peak_m: float
    speed_scale: float
    shift_start_s: float
    shift_recover_s: float
    raw_d: tuple[float, ...] = ()
    raw_delta_s: tuple[float, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


def _lateral_profile(
    n: int, d_peak: float, start_s: float, recover_s: float
) -> tuple[float, ...]:
    if n <= 0:
        return ()
    out: list[float] = []
    for i in range(n):
        u = i / max(n - 1, 1)
        if i < 2 or u < start_s:
            d = 0.0
        elif u < recover_s:
            # A complete avoidance manoeuvre: smoothly leave the native line
            # and smoothly rejoin it.  The previous half-cosine reached the
            # peak at recover_s and then jumped directly to zero.
            t = (u - start_s) / max(recover_s - start_s, 1e-6)
            # sin² has zero lateral slope at departure, peak, and rejoin.
            d = float(d_peak) * math.sin(math.pi * t) ** 2
        else:
            d = 0.0
        out.append(d)
    return tuple(out)


def generate_lattice(
    *, n_path: int = 20, config: TeacherConfig | None = None
) -> list[LatticeCandidate]:
    cfg = config or TeacherConfig()
    cands: list[LatticeCandidate] = []
    idx = 0
    for d in cfg.d_peaks_m:
        for sc in cfg.speed_scales:
            for st in cfg.side_shift_start_s:
                for rc in cfg.side_shift_recover_s:
                    if rc <= st:
                        continue
                    prof = _lateral_profile(n_path, float(d), float(st), float(rc))
                    cands.append(
                        LatticeCandidate(
                            candidate_id=f"lat_{idx:04d}",
                            d_peak_m=float(d),
                            speed_scale=float(sc),
                            shift_start_s=float(st),
                            shift_recover_s=float(rc),
                            raw_d=prof,
                            raw_delta_s=tuple(0.0 for _ in range(n_path)),
                            meta={},
                        )
                    )
                    idx += 1
    return cands


def native_horizon_raw_delta_s(
    native_path_xy: Sequence[Sequence[float]], n_path: int
) -> tuple[float, ...]:
    """Encode native segment lengths for the softplus Frenet decoder.

    A zero raw value decodes to 0.693 m and silently shortens a roughly
    1 m-spaced native path.  That produced the measured "side path then stop"
    failure.  The teacher must supervise a full-horizon spatial route.
    """
    pts = [(float(p[0]), float(p[1])) for p in native_path_xy]
    out = [0.0]
    for index in range(1, int(n_path)):
        if index < len(pts):
            ds = math.hypot(
                pts[index][0] - pts[index - 1][0],
                pts[index][1] - pts[index - 1][1],
            )
        elif len(pts) >= 2:
            ds = math.hypot(
                pts[-1][0] - pts[-2][0],
                pts[-1][1] - pts[-2][1],
            )
        else:
            ds = 1.0
        ds = max(1.0e-4, min(float(ds), 3.0))
        out.append(math.log(math.expm1(ds)))
    return tuple(out)


def away_from_conflict_sign(conflict_side: str) -> float:
    side = str(conflict_side or "").lower()
    if side == "left":
        return -1.0
    if side == "right":
        return 1.0
    return 0.0


def filter_direction_away_from_conflict(
    payload: dict[str, Any], *, conflict_side: str
) -> TeacherFilterResult:
    prefer = away_from_conflict_sign(conflict_side)
    if prefer == 0.0:
        return TeacherFilterResult("direction_away", True, "no_side_constraint")
    d_peak = float(payload.get("d_peak_m", 0.0))
    # Zero lateral is legal for nominal identity and pure-speed defensive lattice.
    # Only reject motion *toward* the conflict side.
    if abs(d_peak) < 1e-9:
        return TeacherFilterResult(
            "direction_away", True, "zero_lateral_ok", {"d_peak_m": 0.0}
        )
    if d_peak * prefer < 0:
        return TeacherFilterResult("direction_away", False, "toward_conflict_side")
    return TeacherFilterResult(
        "direction_away", True, "away_from_conflict", {"d_peak_m": d_peak}
    )


def filter_lead_no_forced_lateral(
    payload: dict[str, Any], *, scenario_family: str
) -> TeacherFilterResult:
    fam = str(scenario_family or "").lower()
    if "lead" not in fam and "brake" not in fam:
        return TeacherFilterResult("lead_gate", True, "not_lead")
    d_peak = abs(float(payload.get("d_peak_m", 0.0)))
    # B2: abs(d)>0.45 always reject for lead (no "slow+swerve" loophole)
    if d_peak > 0.45:
        return TeacherFilterResult(
            "lead_gate", False, "lead_forced_swerve", {"d_peak_m": d_peak}
        )
    return TeacherFilterResult("lead_gate", True, "lead_ok")


def validate_production_stages(
    stages: Sequence[TeacherStageFn] | Mapping[str, TeacherStageFn],
) -> list[tuple[str, TeacherStageFn]]:
    """Require exact production stage set and order; reject dummy ids."""
    if isinstance(stages, Mapping):
        ordered = [(sid, stages[sid]) for sid in PRODUCTION_STAGE_ORDER if sid in stages]
        if len(ordered) != len(PRODUCTION_STAGE_ORDER):
            missing = [s for s in PRODUCTION_STAGE_ORDER if s not in stages]
            raise RuntimeError(f"teacher_missing_stages:{missing}")
        extras = [k for k in stages if k not in PRODUCTION_STAGE_ORDER]
        if extras:
            raise RuntimeError(f"teacher_unknown_stages:{extras}")
        return ordered
    # sequence of callables that must set stage_id via result — probe once
    raise RuntimeError(
        "production_requires_mapping_stage_id_to_callable:"
        f"expected_keys={list(PRODUCTION_STAGE_ORDER)}"
    )


def unit_test_stub_privileged_metrics(
    payload: dict[str, Any],
    *,
    conflict_side: str,
) -> dict[str, float]:
    """UNIT TEST ONLY — synthetic TTC/clearance. Forbidden in production path."""
    d = float(payload.get("d_peak_m", 0.0))
    prefer = away_from_conflict_sign(conflict_side)
    away = 1.0 if prefer != 0.0 and d * prefer > 0 else 0.0
    toward = 1.0 if prefer != 0.0 and d * prefer < 0 else 0.0
    return {
        "ttc_s": 4.0 + 0.8 * away * abs(d) - 1.2 * toward * abs(d),
        "clearance_m": 6.0 + 0.5 * away * abs(d) - 0.8 * toward * abs(d),
        "progress_m": 10.0 * float(payload.get("speed_scale", 1.0)),
        "comfort_cost": abs(d) * 0.3 + abs(float(payload.get("speed_scale", 1.0)) - 1.0),
        "collision": 0.0,
        "metrics_source": 0.0,  # 0=stub
    }


def privileged_cost_from_metrics(metrics: Mapping[str, Any]) -> tuple:
    """Lexicographic cost from real/stub metrics (lower better)."""
    if float(metrics.get("collision", 0.0)) > 0.5:
        return (1, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        0,
        -float(metrics.get("ttc_s", 0.0)),
        -float(metrics.get("clearance_m", 0.0)),
        -float(metrics.get("progress_m", 0.0)),
        float(metrics.get("comfort_cost", 0.0)),
        abs(float(metrics.get("speed_scale", 1.0)) - 1.0),
    )


@dataclass
class TeacherLabel:
    alternative_available: bool
    availability_reason: str
    nominal_residual: dict[str, Any]
    defensive_residual: dict[str, Any] | None
    selected_candidate_id: str | None
    config_hash: str
    filter_log: list[dict[str, Any]] = field(default_factory=list)
    cost_log: list[dict[str, Any]] = field(default_factory=list)
    privileged_used_for_selection_only: bool = True


def teacher_label_to_dict(label: TeacherLabel) -> dict[str, Any]:
    return {
        "alternative_available": label.alternative_available,
        "availability_reason": label.availability_reason,
        "nominal": label.nominal_residual,
        "defensive": label.defensive_residual,
        "selected_candidate_id": label.selected_candidate_id,
        "config_hash": label.config_hash,
        "filter_log_n": len(label.filter_log),
        "cost_log_n": len(label.cost_log),
        "filter_log": label.filter_log[:80],
        "cost_log": label.cost_log[:40],
        "privileged_used_for_selection_only": label.privileged_used_for_selection_only,
    }


def select_defensive_teacher(
    *,
    scenario_family: str,
    conflict_side: str,
    n_path: int = 20,
    config: TeacherConfig | None = None,
    execution_stages: Mapping[str, TeacherStageFn] | None = None,
    allow_unit_test_stub_metrics: bool = False,
    privileged_scene: Mapping[str, Any] | None = None,
    native_path_xy: Sequence[Sequence[float]] | None = None,
    ego_v: float = 5.0,
) -> TeacherLabel:
    """Production teacher selection with structured stages and nominal cost compare."""
    cfg = config or TeacherConfig()
    fam = str(scenario_family or "").lower()
    side = str(conflict_side or "").lower()
    # Missing lateral side does not mean an empty scene: lead/obstruction and
    # crossing actors can be centered in the ego frame.  Only explicit clear
    # families are NO_ALTERNATIVE by construction.
    empty = fam in {"clear", "clear_no_alternative", "empty"} or "empty" in fam
    native_list: list[list[float]] | None = None
    if native_path_xy is not None and len(native_path_xy) >= 2:
        native_list = [[float(p[0]), float(p[1])] for p in native_path_xy]
        n_path = max(n_path, len(native_list))

    nominal = {
        "raw_delta_s": [0.0] * n_path,
        "raw_d": [0.0] * n_path,
        "speed_scale": 1.0,
        "head_lineage": "native_anchor",
        "d_peak_m": 0.0,
    }

    lane_change_authorized = bool(
        (privileged_scene or {}).get("adjacent_lane_authorized", False)
    )
    if (
        ("obstruction" in fam or "narrow" in fam)
        and not lane_change_authorized
    ):
        # A ±1 m within-lane residual cannot clear a full vehicle centred in
        # the ego lane.  Such a sample requires a separately authorized
        # adjacent-lane manoeuvre; never teach the head to sideswipe it.
        return TeacherLabel(
            alternative_available=False,
            availability_reason="obstruction_requires_topology_authorization",
            nominal_residual=nominal,
            defensive_residual=None,
            selected_candidate_id=None,
            config_hash=cfg.config_hash(),
            filter_log=[
                {
                    "stage_id": "topology_gate",
                    "pass": False,
                    "reason": "obstruction_layout_not_topology_authorized",
                }
            ],
        )

    if empty and cfg.forbid_empty_road_forced_swerve:
        return TeacherLabel(
            alternative_available=False,
            availability_reason="empty_road_no_forced_swerve",
            nominal_residual=nominal,
            defensive_residual=None,
            selected_candidate_id=None,
            config_hash=cfg.config_hash(),
            filter_log=[{"stage_id": "empty_gate", "pass": False, "reason": "empty"}],
        )

    if cfg.require_execution_filters:
        if not execution_stages:
            raise RuntimeError("teacher_requires_execution_stages_mapping")
        stages = validate_production_stages(execution_stages)
    else:
        stages = []

    def _run_pipeline(payload: dict[str, Any]) -> tuple[bool, list[dict], dict[str, Any]]:
        log: list[dict] = []
        # geometry stages always
        for geo in (
            filter_direction_away_from_conflict(payload, conflict_side=side),
            filter_lead_no_forced_lateral(payload, scenario_family=fam),
        ):
            log.append(geo.to_dict())
            if not geo.passed:
                return False, log, dict(payload)
            payload.update(geo.metrics)
        for stage_id, fn in stages:
            res = fn(payload)
            if res.stage_id != stage_id:
                raise RuntimeError(
                    f"teacher_stage_id_mismatch:expected={stage_id}:got={res.stage_id}"
                )
            log.append(res.to_dict())
            if not res.passed:
                return False, log, dict(payload)
            # persist metrics into payload for cost
            for k, v in res.metrics.items():
                payload[k] = v
        # metrics for cost
        if allow_unit_test_stub_metrics and "ttc_s" not in payload:
            stub = unit_test_stub_privileged_metrics(payload, conflict_side=side)
            payload.update(stub)
            payload["metrics_source"] = "unit_test_stub"
        elif "ttc_s" not in payload and cfg.require_execution_filters:
            raise RuntimeError(
                "production_teacher_missing_rollout_metrics:ttc_s"
            )
        return True, log, payload

    # Nominal through same filters (retime identity)
    base_ctx: dict[str, Any] = {
        "scenario_family": fam,
        "conflict_side": side,
        "privileged_scene": dict(privileged_scene or {}),
        "ego_v": float(ego_v),
    }
    if native_list is not None:
        base_ctx["native_path_xy"] = native_list
    nom_payload = {
        "d_peak_m": 0.0,
        "speed_scale": 1.0,
        "raw_d": nominal["raw_d"],
        "raw_delta_s": nominal["raw_delta_s"],
        "candidate_id": "nominal",
        **base_ctx,
    }
    nom_ok, nom_log, nom_payload = _run_pipeline(nom_payload)
    if not nom_ok and cfg.require_execution_filters:
        # nominal illegal → no alternative labeling
        return TeacherLabel(
            alternative_available=False,
            availability_reason="nominal_failed_execution_filters",
            nominal_residual=nominal,
            defensive_residual=None,
            selected_candidate_id=None,
            config_hash=cfg.config_hash(),
            filter_log=nom_log,
        )
    nom_cost = privileged_cost_from_metrics(
        {**nom_payload, "speed_scale": 1.0}
    )

    lattice = generate_lattice(n_path=n_path, config=cfg)
    if native_list is not None:
        from dataclasses import replace

        full_horizon_ds = native_horizon_raw_delta_s(native_list, n_path)
        lattice = [
            replace(candidate, raw_delta_s=full_horizon_ds)
            for candidate in lattice
        ]
    survivors: list[tuple[LatticeCandidate, dict[str, Any], list[dict]]] = []
    filter_log: list[dict[str, Any]] = list(nom_log)
    for cand in lattice:
        payload = {
            "d_peak_m": cand.d_peak_m,
            "speed_scale": cand.speed_scale,
            "raw_d": cand.raw_d,
            "raw_delta_s": cand.raw_delta_s,
            "candidate_id": cand.candidate_id,
            **base_ctx,
        }
        ok, clog, payload = _run_pipeline(payload)
        filter_log.extend(clog)
        if ok:
            survivors.append((cand, payload, clog))

    if not survivors:
        return TeacherLabel(
            alternative_available=False,
            availability_reason="no_legal_lattice_survivor",
            nominal_residual=nominal,
            defensive_residual=None,
            selected_candidate_id=None,
            config_hash=cfg.config_hash(),
            filter_log=filter_log,
            cost_log=[{"candidate_id": "nominal", "cost": list(nom_cost)}],
        )

    cost_log: list[dict[str, Any]] = [
        {"candidate_id": "nominal", "cost": list(nom_cost), "metrics": {
            k: nom_payload.get(k) for k in ("ttc_s", "clearance_m", "progress_m", "comfort_cost", "collision")
        }}
    ]
    scored: list[tuple[tuple, LatticeCandidate, dict[str, Any]]] = []
    for cand, payload, _ in survivors:
        if float(payload.get("collision", 0.0)) > 0.5:
            cost_log.append(
                {
                    "candidate_id": cand.candidate_id,
                    "cost": [1],
                    "rejected": "collision",
                }
            )
            continue
        ctuple = privileged_cost_from_metrics(payload)
        rejection: str | None = None
        ttc_b = float(payload.get("ttc_s", 0.0))
        ttc_n = float(nom_payload.get("ttc_s", 0.0))
        cl_b = float(payload.get("clearance_m", 0.0))
        cl_n = float(nom_payload.get("clearance_m", 0.0))
        prog_b = float(payload.get("progress_m", 0.0))
        prog_n = float(nom_payload.get("progress_m", 0.0))
        com_b = float(payload.get("comfort_cost", 0.0))
        com_n = float(nom_payload.get("comfort_cost", 0.0))
        prefer = away_from_conflict_sign(side)
        # Availability constraints define the legal pool. Applying them after
        # argmin lets an illegal d=0/slow candidate eclipse a valid spatial one.
        if abs(cand.d_peak_m) + 1e-9 < cfg.min_lateral_sep_m:
            rejection = "sep_below_min"
        elif prog_n > 1e-6 and prog_b < cfg.min_progress_ratio * prog_n:
            rejection = "progress_guardrail"
        elif com_b > com_n + cfg.max_comfort_degradation:
            rejection = "comfort_guardrail"
        elif prefer != 0.0 and cand.d_peak_m * prefer <= 0:
            rejection = "not_away_from_conflict"
        elif cfg.require_strictly_better_than_nominal and not (
            ttc_b >= ttc_n + cfg.better_margin_ttc_s
            or cl_b >= cl_n + cfg.better_margin_clearance_m
            or ctuple < nom_cost
        ):
            rejection = "not_better_than_nominal_margin"
        cost_log.append(
            {
                "candidate_id": cand.candidate_id,
                "cost": list(ctuple),
                **({"rejected": rejection} if rejection else {}),
                "metrics": {
                    k: payload.get(k)
                    for k in (
                        "ttc_s",
                        "clearance_m",
                        "progress_m",
                        "comfort_cost",
                        "collision",
                        "metrics_source",
                    )
                },
            }
        )
        if rejection is None:
            scored.append((ctuple, cand, payload))

    if not scored:
        return TeacherLabel(
            alternative_available=False,
            availability_reason="no_candidate_meets_availability_contract",
            nominal_residual=nominal,
            defensive_residual=None,
            selected_candidate_id=None,
            config_hash=cfg.config_hash(),
            filter_log=filter_log,
            cost_log=cost_log,
        )

    scored.sort(key=lambda x: x[0])
    _best_cost, best, _best_payload = scored[0]
    reason = "lattice_selected"

    defensive = {
        "raw_delta_s": list(best.raw_delta_s),
        "raw_d": list(best.raw_d),
        "speed_scale": float(best.speed_scale),
        "head_lineage": "teacher_lattice",
        "d_peak_m": float(best.d_peak_m),
        "candidate_id": best.candidate_id,
    }
    return TeacherLabel(
        alternative_available=True,
        availability_reason=reason,
        nominal_residual=nominal,
        defensive_residual=defensive,
        selected_candidate_id=best.candidate_id,
        config_hash=cfg.config_hash(),
        filter_log=filter_log,
        cost_log=cost_log,
    )
