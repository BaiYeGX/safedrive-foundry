"""VLA sample identity and four-layer field schema (G3-01)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "safedrive.vla.sample.v1"

# Restricted language labels allowed in policy_input / privileged layers.
ALLOWED_LANGUAGE_KEYS = frozenset(
    {
        "behavior",
        "critical_actor",
        "conflict",
        "risk_horizon",
        "intended_action",
    }
)


class FieldLayer(str, Enum):
    POLICY_INPUT = "policy_input"
    PRIVILEGED_LABEL = "privileged_label"
    EVALUATION_ONLY = "evaluation_only"
    REGRESSION_FROZEN = "regression_frozen"


@dataclass(frozen=True)
class FrameIdentity:
    run_id: str
    frame_id: str
    scenario_id: str
    attempt_id: str = "0"
    carla_frame: int = 0
    simulation_time_s: float = 0.0
    town: str = ""
    route_id: str = ""
    scenario_family: str = ""
    weather: str = ""
    failure_cluster: str = ""

    def key(self) -> str:
        return f"{self.run_id}|{self.frame_id}|{self.scenario_id}|{self.attempt_id}"


@dataclass
class LayerBundle:
    """Layered fields for one sample."""

    policy_input: dict[str, Any] = field(default_factory=dict)
    privileged_label: dict[str, Any] = field(default_factory=dict)
    evaluation_only: dict[str, Any] = field(default_factory=dict)
    regression_frozen: dict[str, Any] = field(default_factory=dict)

    def layer(self, name: FieldLayer) -> dict[str, Any]:
        return getattr(self, name.value)


@dataclass
class SampleRecord:
    identity: FrameIdentity
    layers: LayerBundle
    schema_version: str = SCHEMA_VERSION
    parameter_hash: str = ""
    content_hash: str = ""

    def recompute_parameter_hash(self) -> str:
        payload = {
            "town": self.identity.town,
            "route_id": self.identity.route_id,
            "scenario_family": self.identity.scenario_family,
            "weather": self.identity.weather,
            "scenario_id": self.identity.scenario_id,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.parameter_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return self.parameter_hash


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def payload_hash(sample: SampleRecord) -> str:
    """Hash over layered fields only (identity excluded) — used for near-dup / cross-split leak."""
    body = {
        "schema_version": sample.schema_version,
        "layers": {
            "policy_input": sample.layers.policy_input,
            "privileged_label": sample.layers.privileged_label,
            "evaluation_only": sample.layers.evaluation_only,
            "regression_frozen": sample.layers.regression_frozen,
        },
    }
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def content_hash(sample: SampleRecord) -> str:
    """Stable full-sample hash over identity + layered fields (excludes content_hash itself)."""
    body = {
        "schema_version": sample.schema_version,
        "identity": asdict(sample.identity),
        "layers": {
            "policy_input": sample.layers.policy_input,
            "privileged_label": sample.layers.privileged_label,
            "evaluation_only": sample.layers.evaluation_only,
            "regression_frozen": sample.layers.regression_frozen,
        },
        "parameter_hash": sample.parameter_hash,
        "payload_hash": payload_hash(sample),
    }
    digest = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return digest


def sample_to_dict(sample: SampleRecord) -> dict[str, Any]:
    if not sample.parameter_hash:
        sample.recompute_parameter_hash()
    if not sample.content_hash:
        sample.content_hash = content_hash(sample)
    return {
        "schema_version": sample.schema_version,
        "identity": asdict(sample.identity),
        "layers": {
            "policy_input": sample.layers.policy_input,
            "privileged_label": sample.layers.privileged_label,
            "evaluation_only": sample.layers.evaluation_only,
            "regression_frozen": sample.layers.regression_frozen,
        },
        "parameter_hash": sample.parameter_hash,
        "content_hash": sample.content_hash,
    }


def sample_from_dict(data: Mapping[str, Any]) -> SampleRecord:
    ident = data["identity"]
    layers = data.get("layers") or {}
    sample = SampleRecord(
        identity=FrameIdentity(
            run_id=str(ident["run_id"]),
            frame_id=str(ident["frame_id"]),
            scenario_id=str(ident["scenario_id"]),
            attempt_id=str(ident.get("attempt_id", "0")),
            carla_frame=int(ident.get("carla_frame", 0)),
            simulation_time_s=float(ident.get("simulation_time_s", 0.0)),
            town=str(ident.get("town", "")),
            route_id=str(ident.get("route_id", "")),
            scenario_family=str(ident.get("scenario_family", "")),
            weather=str(ident.get("weather", "")),
            failure_cluster=str(ident.get("failure_cluster", "")),
        ),
        layers=LayerBundle(
            policy_input=dict(layers.get("policy_input") or {}),
            privileged_label=dict(layers.get("privileged_label") or {}),
            evaluation_only=dict(layers.get("evaluation_only") or {}),
            regression_frozen=dict(layers.get("regression_frozen") or {}),
        ),
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        parameter_hash=str(data.get("parameter_hash", "")),
        content_hash=str(data.get("content_hash", "")),
    )
    if not sample.parameter_hash:
        sample.recompute_parameter_hash()
    return sample


def validate_language_labels(labels: Mapping[str, Any]) -> None:
    bad = set(labels) - ALLOWED_LANGUAGE_KEYS
    if bad:
        raise ValueError(f"language labels outside allowed set: {sorted(bad)}")
