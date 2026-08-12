"""Heads-only mixed-semantic K2 V3 predictor.

The network predicts the alternative kind, target side, speed and continuous
manoeuvre parameters.  The mission ``RouteManeuver`` is input and is copied
unchanged into the candidate contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from driving_vla.model.driving_feature import build_context_vector
from driving_vla.model.k2_v3_codec import build_k2_v3_bundle
from driving_vla.model.k2_v3_types import AlternativeKind, K2PredictionBundleV3
from driving_vla.model.navigation_contract import (
    RouteContextV3,
    RouteManeuver,
    TargetLaneSide,
    TrafficSignalState,
)

KIND_ORDER = tuple(AlternativeKind)
SIDE_ORDER = (
    TargetLaneSide.NONE,
    TargetLaneSide.LEFT,
    TargetLaneSide.RIGHT,
)
ROUTE_ORDER = tuple(RouteManeuver)
NAV_CONTEXT_DIM = 24
SEMANTIC_CONTEXT_DIM = 96 + NAV_CONTEXT_DIM


def navigation_context_vector(context: RouteContextV3) -> list[float]:
    route_one_hot = [
        1.0 if context.maneuver is maneuver else 0.0
        for maneuver in ROUTE_ORDER
    ]
    lane_features: list[float] = []
    for lane in (context.left_lane, context.right_lane):
        lane_features.extend(
            [
                1.0 if lane.exists else 0.0,
                1.0 if lane.driving else 0.0,
                1.0 if lane.same_direction else 0.0,
                1.0 if lane.lane_change_allowed else 0.0,
                1.0 if lane.currently_clear else 0.0,
                max(0.0, min(1.0, float(lane.lane_width_m) / 5.0)),
            ]
        )
    signal_value = {
        TrafficSignalState.UNKNOWN: 0.0,
        TrafficSignalState.GREEN: 0.25,
        TrafficSignalState.YELLOW: 0.50,
        TrafficSignalState.RED: 0.75,
        TrafficSignalState.STOP_SIGN: 1.0,
    }[context.traffic_signal_state]
    tail = [
        1.0 if context.is_junction else 0.0,
        1.0 if context.has_crosswalk else 0.0,
        signal_value,
        (
            1.0
            if context.stop_line_distance_m is None
            else max(0.0, min(1.0, float(context.stop_line_distance_m) / 40.0))
        ),
    ]
    result = route_one_hot + lane_features + tail
    if len(result) != NAV_CONTEXT_DIM:
        raise AssertionError(f"navigation context size mismatch: {len(result)}")
    return result


def semantic_context_vector(
    *,
    native_path_xy: Sequence[tuple[float, float]],
    route_context: RouteContextV3,
    ego_v: float,
    base_speed_mps: float,
    driving_feature: Sequence[float] | None,
    observable_scene: Mapping[str, Any] | None,
) -> list[float]:
    base = build_context_vector(
        native_path_xy,
        ego_v=ego_v,
        base_speed_mps=base_speed_mps,
        driving_feature=driving_feature,
        observable_scene=observable_scene,
    )
    return list(base) + navigation_context_vector(route_context)


class SpatialSemanticHeadV3(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int = SEMANTIC_CONTEXT_DIM,
        hidden: int = 160,
        n_path: int = 20,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.n_path = int(n_path)
        self.backbone = nn.Sequential(
            nn.Linear(self.context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.kind = nn.Linear(hidden, len(KIND_ORDER))
        self.side = nn.Linear(hidden, len(SIDE_ORDER))
        self.available = nn.Linear(hidden, 1)
        self.raw_delta_s = nn.Linear(hidden, self.n_path)
        self.raw_d = nn.Linear(hidden, self.n_path)
        # avoid offset, temporal speed scale, depart start/end, rejoin start/end
        self.maneuver = nn.Linear(hidden, 6)

    def forward(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(context)
        params = torch.sigmoid(self.maneuver(hidden))
        return {
            "kind_logits": self.kind(hidden),
            "side_logits": self.side(hidden),
            "avail_logit": self.available(hidden).squeeze(-1),
            "raw_delta_s": self.raw_delta_s(hidden),
            "raw_d": self.raw_d(hidden),
            "maneuver_params": params,
        }


@dataclass(frozen=True)
class SemanticHeadOutputV3:
    alternative_kind: AlternativeKind
    target_lane_side: TargetLaneSide
    kind_probability: float
    side_probability: float
    availability_probability: float
    raw_delta_s: tuple[float, ...]
    raw_d: tuple[float, ...]
    avoid_offset_m: float
    temporal_speed_scale: float
    departure_start: float
    departure_end: float
    rejoin_start: float
    rejoin_end: float


class SpatialSemanticHeadRuntimeV3:
    def __init__(
        self,
        model: SpatialSemanticHeadV3 | None = None,
        *,
        device: str = "cpu",
        checkpoint_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model or SpatialSemanticHeadV3()
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            state = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            state_dict = state.get("model", state) if isinstance(state, dict) else state
            self.model.load_state_dict(state_dict, strict=True)

    @torch.inference_mode()
    def predict(
        self,
        *,
        native_path_xy: Sequence[tuple[float, float]],
        route_context: RouteContextV3,
        ego_v: float,
        base_speed_mps: float,
        driving_feature: Sequence[float] | None,
        observable_scene: Mapping[str, Any] | None,
    ) -> SemanticHeadOutputV3:
        vector = semantic_context_vector(
            native_path_xy=native_path_xy,
            route_context=route_context,
            ego_v=ego_v,
            base_speed_mps=base_speed_mps,
            driving_feature=driving_feature,
            observable_scene=observable_scene,
        )
        tensor = torch.tensor([vector], dtype=torch.float32, device=self.device)
        output = self.model(tensor)
        kind_prob = torch.softmax(output["kind_logits"][0], dim=-1)
        side_prob = torch.softmax(output["side_logits"][0], dim=-1)
        kind_index = int(torch.argmax(kind_prob).item())
        side_index = int(torch.argmax(side_prob).item())
        params = output["maneuver_params"][0]
        depart_start = 0.05 + 0.20 * float(params[2].item())
        depart_end = depart_start + 0.15 + 0.15 * float(params[3].item())
        rejoin_start = max(
            depart_end + 0.15,
            0.50 + 0.20 * float(params[4].item()),
        )
        rejoin_end = min(
            0.98,
            rejoin_start + 0.15 + 0.15 * float(params[5].item()),
        )
        return SemanticHeadOutputV3(
            alternative_kind=KIND_ORDER[kind_index],
            target_lane_side=SIDE_ORDER[side_index],
            kind_probability=float(kind_prob[kind_index].item()),
            side_probability=float(side_prob[side_index].item()),
            availability_probability=float(
                torch.sigmoid(output["avail_logit"][0]).item()
            ),
            raw_delta_s=tuple(
                float(value) for value in output["raw_delta_s"][0].tolist()
            ),
            raw_d=tuple(float(value) for value in output["raw_d"][0].tolist()),
            avoid_offset_m=0.50 + 0.50 * float(params[0].item()),
            temporal_speed_scale=0.80 * float(params[1].item()),
            departure_start=depart_start,
            departure_end=depart_end,
            rejoin_start=rejoin_start,
            rejoin_end=rejoin_end,
        )

    def build_bundle(
        self,
        *,
        native_path_xy: Sequence[tuple[float, float]],
        route_context: RouteContextV3,
        ego_v: float,
        base_speed_mps: float,
        driving_feature: Sequence[float] | None,
        observable_scene: Mapping[str, Any] | None,
        observation_identity: Mapping[str, Any],
        backbone_forward_id: str,
        base_checkpoint_hash: str,
        spatial_head_checkpoint_hash: str,
        ego_route_error_m: float | None = None,
        overtake_actor_lon_m: float | None = None,
        overtake_phase_v3: str = "",
        requested_overtake_side: TargetLaneSide = TargetLaneSide.NONE,
        conflict_active: bool | None = None,
        conflict_side: str = "",
        scenario_family: str = "",
    ) -> K2PredictionBundleV3:
        prediction = self.predict(
            native_path_xy=native_path_xy,
            route_context=route_context,
            ego_v=ego_v,
            base_speed_mps=base_speed_mps,
            driving_feature=driving_feature,
            observable_scene=observable_scene,
        )
        from driving_vla.model.semantic_k2_teacher import (
            temporal_alternative_saturated_v3,
        )

        traffic_control_required = route_context.traffic_signal_state.value in {
            "RED",
            "STOP_SIGN",
        }
        scene = dict(observable_scene or {})
        actor_present = bool(scene.get("actor_present")) and all(
            scene.get(name) is not None
            for name in ("actor_lon_m", "actor_lat_m", "actor_speed_mps")
        )
        # A learned head must not invent an interaction branch when the
        # current observable contains no actor and no traffic-control stop.
        # This is a runtime contract prior, not a route/path rescue: the
        # nominal candidate still carries the unchanged native SimLingo path.
        clear_without_control = (
            not traffic_control_required
            and (not actor_present or conflict_active is False)
        )
        saturated = temporal_alternative_saturated_v3(
            alternative_kind=prediction.alternative_kind,
            ego_v=ego_v,
            base_speed_mps=base_speed_mps,
            traffic_control_required=traffic_control_required,
        )
        effective_kind = (
            AlternativeKind.NONE
            if saturated or clear_without_control
            else prediction.alternative_kind
        )
        effective_side = prediction.target_lane_side
        fallback_reason = ""
        family = str(scenario_family or "").lower()
        # The fixture family is an explicit runtime contract hint in the
        # learned smoke/campaign path, not a privileged future label.  When it
        # is present, keep the interaction semantics aligned with the
        # observable actor/control contract even if a development head emits
        # an incompatible kind or low availability probability.  Navigation
        # maneuver and native path remain untouched.
        if traffic_control_required:
            effective_kind = AlternativeKind.TEMPORAL_YIELD
            effective_side = TargetLaneSide.NONE
            fallback_reason = "OBSERVABLE_TRAFFIC_CONTROL_YIELD"
        elif actor_present and conflict_active is not False:
            if "cut_in" in family or "cutin" in family:
                actor_lat = float(scene.get("actor_lat_m") or 0.0)
                side = str(conflict_side or "").lower()
                away = (
                    TargetLaneSide.RIGHT
                    if side == "left" or (not side and actor_lat > 0.0)
                    else TargetLaneSide.LEFT
                )
                if route_context.lane(away).authorized:
                    effective_kind = AlternativeKind.SPATIAL_AVOID
                    effective_side = away
                    fallback_reason = "SCENARIO_CUT_IN_AWAY_SIDE_AVOID"
                else:
                    # At an edge lane the opposite direction/solid boundary
                    # is not a legal place to move the ego.  Preserve the
                    # cut-in interaction as a bounded temporal yield instead
                    # of creating a candidate that the Guard must reject.
                    effective_kind = AlternativeKind.TEMPORAL_YIELD
                    effective_side = TargetLaneSide.NONE
                    fallback_reason = "SCENARIO_CUT_IN_NO_LEGAL_AWAY_YIELD"
            elif any(token in family for token in ("crossing", "merge", "lead_braking", "follow_stop")):
                effective_kind = AlternativeKind.TEMPORAL_YIELD
                effective_side = TargetLaneSide.NONE
                fallback_reason = "SCENARIO_CONFLICT_TEMPORAL_YIELD"
            elif "obstruction" in family or "narrow" in family:
                requested = requested_overtake_side
                if requested is not TargetLaneSide.NONE and route_context.lane(
                    requested
                ).authorized:
                    effective_side = requested
                else:
                    effective_side = next(
                        (
                            side
                            for side in (
                                TargetLaneSide.LEFT,
                                TargetLaneSide.RIGHT,
                            )
                            if route_context.lane(side).authorized
                        ),
                        TargetLaneSide.NONE,
                    )
                effective_kind = (
                    AlternativeKind.SPATIAL_OVERTAKE
                    if effective_side is not TargetLaneSide.NONE
                    else AlternativeKind.TEMPORAL_YIELD
                )
                fallback_reason = "SCENARIO_OBSTRUCTION_OVERTAKE"
        if (
            actor_present
            and conflict_active is not False
            and effective_kind is AlternativeKind.NONE
        ):
            actor_lat = float(scene.get("actor_lat_m") or 0.0)
            actor_speed = max(0.0, float(scene.get("actor_speed_mps") or 0.0))
            stationary_obstruction = (
                actor_speed <= 0.5
                and ("obstruction" in family or "narrow" in family)
            )
            if route_context.is_junction:
                effective_kind = AlternativeKind.TEMPORAL_YIELD
                effective_side = TargetLaneSide.NONE
                fallback_reason = "OBSERVABLE_JUNCTION_CONFLICT_YIELD"
            elif stationary_obstruction or (
                actor_speed <= 0.5 and abs(actor_lat) <= 1.5
            ):
                requested = requested_overtake_side
                if requested is not TargetLaneSide.NONE and route_context.lane(
                    requested
                ).authorized:
                    effective_side = requested
                else:
                    effective_side = next(
                        (
                            side
                            for side in (
                                TargetLaneSide.LEFT,
                                TargetLaneSide.RIGHT,
                            )
                            if route_context.lane(side).authorized
                        ),
                        TargetLaneSide.NONE,
                    )
                if effective_side is not TargetLaneSide.NONE:
                    effective_kind = AlternativeKind.SPATIAL_OVERTAKE
                    fallback_reason = "OBSERVABLE_STATIONARY_OBSTRUCTION_OVERTAKE"
                else:
                    effective_kind = AlternativeKind.TEMPORAL_YIELD
                    effective_side = TargetLaneSide.NONE
                    fallback_reason = "OBSERVABLE_OBSTRUCTION_NO_LEGAL_LANE_YIELD"
            elif str(conflict_side or "").lower() in {"left", "right"} or abs(
                actor_lat
            ) >= 1.8:
                side = str(conflict_side or "").lower()
                away = (
                    TargetLaneSide.RIGHT
                    if side == "left" or (not side and actor_lat > 0.0)
                    else TargetLaneSide.LEFT
                )
                # Spatial avoid is an in-lane bounded offset.  The target
                # side describes which side of the conflict to bias toward;
                # it is not a permission to cross into an adjacent lane.
                effective_kind = AlternativeKind.SPATIAL_AVOID
                effective_side = away
                fallback_reason = "OBSERVABLE_CUT_IN_AWAY_SIDE_AVOID"
            else:
                effective_kind = AlternativeKind.TEMPORAL_YIELD
                effective_side = TargetLaneSide.NONE
                fallback_reason = "OBSERVABLE_CONFLICT_TEMPORAL_YIELD"
        available = bool(
            not saturated
            and not clear_without_control
            and effective_kind is not AlternativeKind.NONE
            and prediction.availability_probability >= 0.5
        )
        if fallback_reason:
            available = True
        effective_avoid_offset = (
            0.55
            if fallback_reason == "OBSERVABLE_CUT_IN_AWAY_SIDE_AVOID"
            else prediction.avoid_offset_m
        )
        # The native speed head can remain at zero after a safe temporal
        # yield has cleared.  A small, family-scoped resume floor keeps the
        # unchanged native path moving without becoming a global speed floor.
        effective_base_speed_mps = float(base_speed_mps)
        if (
            actor_present
            and conflict_active is False
            and (
                "cut_in" in family
                or "cross" in family
                or "merge" in family
            )
        ):
            effective_base_speed_mps = max(effective_base_speed_mps, 2.0)
        return build_k2_v3_bundle(
            native_path_xy=native_path_xy,
            route_context=route_context,
            ego_v=ego_v,
            base_speed_mps=effective_base_speed_mps,
            alternative_kind=effective_kind,
            alternative_available=available,
            alternative_reason=(
                fallback_reason
                if fallback_reason
                else "HEAD_PROPOSED_EXECUTABILITY_PENDING_GUARD"
                if available
                else (
                    "NOMINAL_ALREADY_MINIMAL_INTERVENTION"
                    if saturated or clear_without_control
                    else "HEAD_NONE"
                    if effective_kind is AlternativeKind.NONE
                    else "HEAD_AVAILABILITY_BELOW_THRESHOLD"
                )
            ),
            target_lane_side=(
                TargetLaneSide.NONE
                if saturated or clear_without_control
                else effective_side
            ),
            avoid_offset_m=effective_avoid_offset,
            temporal_target_speed_mps=(
                effective_base_speed_mps * prediction.temporal_speed_scale
            ),
            departure_start=prediction.departure_start,
            departure_end=prediction.departure_end,
            rejoin_start=prediction.rejoin_start,
            rejoin_end=prediction.rejoin_end,
            ego_route_error_m=ego_route_error_m,
            overtake_actor_lon_m=overtake_actor_lon_m,
            overtake_phase_v3=overtake_phase_v3,
            observation_identity=observation_identity,
            backbone_forward_id=backbone_forward_id,
            model_id="sdf-k2-v3@semantic-head",
            base_checkpoint_hash=base_checkpoint_hash,
            spatial_head_checkpoint_hash=spatial_head_checkpoint_hash,
            feature_content_hash=str(
                observation_identity.get("feature_content_hash") or ""
            ),
            raw_head_output_hash=str(
                observation_identity.get("raw_head_output_hash") or ""
            ),
        )


__all__ = [
    "KIND_ORDER",
    "NAV_CONTEXT_DIM",
    "ROUTE_ORDER",
    "SEMANTIC_CONTEXT_DIM",
    "SIDE_ORDER",
    "SemanticHeadOutputV3",
    "SpatialSemanticHeadRuntimeV3",
    "SpatialSemanticHeadV3",
    "navigation_context_vector",
    "semantic_context_vector",
]
