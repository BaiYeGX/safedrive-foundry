"""Deterministic synchronization contracts for SafeDrive G0-05.

The module deliberately has no ROS or CARLA dependency.  It is shared by the
offline validation harness, the CARLA tick-master node, and the read-only
status bridge so that frame validation is implemented once.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
BLOCKED = "BLOCKED"


class ContractViolation(ValueError):
    """A synchronization contract or ownership rule was violated."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _finite_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractViolation("invalid_number", f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ContractViolation("invalid_number", f"{field_name} must be finite")
    return number


@dataclass(frozen=True)
class SyncConfig:
    """The only settings that may control the G0 CARLA clock."""

    synchronous_mode: bool = True
    tick_master: str = "sdf.g0-05.sync"
    fixed_delta_seconds: float = 0.05
    substepping: bool = True
    max_substep_delta_time: float = 0.01
    max_substeps: int = 5
    clock_topic: str = "/clock"
    snapshot_topic: str = "/safedrive/carla/status"
    frame_contract: str = "episode_id+carla_frame"
    timestamp_tolerance_seconds: float = 1e-6
    max_frame_gap: int = 1

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SyncConfig":
        values = dict(mapping.get("sync", mapping))
        return cls(
            synchronous_mode=bool(values.get("synchronous_mode", True)),
            tick_master=str(values.get("tick_master", cls.tick_master)),
            fixed_delta_seconds=float(values.get("fixed_delta_seconds", cls.fixed_delta_seconds)),
            substepping=bool(values.get("substepping", cls.substepping)),
            max_substep_delta_time=float(
                values.get("max_substep_delta_time", cls.max_substep_delta_time)
            ),
            max_substeps=int(values.get("max_substeps", cls.max_substeps)),
            clock_topic=str(values.get("clock_topic", cls.clock_topic)),
            snapshot_topic=str(values.get("snapshot_topic", cls.snapshot_topic)),
            frame_contract=str(values.get("frame_contract", cls.frame_contract)),
            timestamp_tolerance_seconds=float(
                values.get("timestamp_tolerance_seconds", cls.timestamp_tolerance_seconds)
            ),
            max_frame_gap=int(values.get("max_frame_gap", cls.max_frame_gap)),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.synchronous_mode:
            errors.append("synchronous_mode must be true")
        if not self.tick_master.strip():
            errors.append("tick_master must be non-empty")
        if not math.isfinite(self.fixed_delta_seconds) or self.fixed_delta_seconds <= 0:
            errors.append("fixed_delta_seconds must be finite and greater than zero")
        if self.substepping:
            if (
                not math.isfinite(self.max_substep_delta_time)
                or self.max_substep_delta_time <= 0
            ):
                errors.append("max_substep_delta_time must be finite and greater than zero")
            if self.max_substep_delta_time > 0.01 + 1e-12:
                errors.append("max_substep_delta_time must be <= 0.01 for CARLA substepping")
            if not 1 <= self.max_substeps <= 10:
                errors.append("max_substeps must be an integer in [1, 10]")
            if (
                self.fixed_delta_seconds > 0
                and self.max_substep_delta_time > 0
                and self.fixed_delta_seconds
                > self.max_substep_delta_time * self.max_substeps + 1e-12
            ):
                errors.append(
                    "fixed_delta_seconds must be <= max_substep_delta_time * max_substeps"
                )
        for name, topic in (
            ("clock_topic", self.clock_topic),
            ("snapshot_topic", self.snapshot_topic),
        ):
            if not topic.startswith("/"):
                errors.append(f"{name} must be an absolute ROS topic")
        if self.clock_topic == self.snapshot_topic:
            errors.append("clock_topic and snapshot_topic must be distinct")
        if self.frame_contract != "episode_id+carla_frame":
            errors.append("frame_contract must be episode_id+carla_frame")
        if (
            not math.isfinite(self.timestamp_tolerance_seconds)
            or self.timestamp_tolerance_seconds < 0
        ):
            errors.append("timestamp_tolerance_seconds must be finite and non-negative")
        if self.max_frame_gap < 1:
            errors.append("max_frame_gap must be at least one")
        return errors

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ContractViolation("invalid_sync_config", "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "synchronous_mode": self.synchronous_mode,
            "tick_master": self.tick_master,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "substepping": self.substepping,
            "max_substep_delta_time": self.max_substep_delta_time,
            "max_substeps": self.max_substeps,
            "clock_topic": self.clock_topic,
            "snapshot_topic": self.snapshot_topic,
            "frame_contract": self.frame_contract,
            "timestamp_tolerance_seconds": self.timestamp_tolerance_seconds,
            "max_frame_gap": self.max_frame_gap,
        }


def load_sync_config(path: Path) -> SyncConfig:
    """Load the ``[sync]`` table from a TOML configuration file."""

    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - only Python < 3.11
        raise ContractViolation(
            "toml_parser_missing", "Python 3.11+ (tomllib) is required for G0 configuration"
        ) from exc
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractViolation("config_unreadable", f"cannot read {path}: {exc}") from exc
    config = SyncConfig.from_mapping(data)
    config.assert_valid()
    return config


@dataclass(frozen=True)
class FrameKey:
    episode_id: str
    carla_frame: int

    def __post_init__(self) -> None:
        if not self.episode_id or not self.episode_id.strip():
            raise ContractViolation("invalid_episode_id", "episode_id must be non-empty")
        if ":" in self.episode_id:
            raise ContractViolation("invalid_episode_id", "episode_id cannot contain ':'")
        if isinstance(self.carla_frame, bool) or not isinstance(self.carla_frame, int):
            raise ContractViolation("invalid_frame", "carla_frame must be an integer")
        if self.carla_frame < 0:
            raise ContractViolation("invalid_frame", "carla_frame must be non-negative")

    def as_string(self) -> str:
        return f"{self.episode_id}:{self.carla_frame}"


@dataclass(frozen=True)
class CommandFrames:
    """Frame lineage for a command.

    ``generated`` is the frame that produced the command, ``planned`` is the
    frame whose snapshot was used for planning, and ``executed`` is the frame
    in which the command was applied.  A G0 smoke has all three equal to the
    tick being exercised; later planning tasks may use a larger execution
    frame while retaining the same ordering rule.
    """

    generated: int
    planned: int
    executed: int

    def validate(self) -> list[str]:
        errors: list[str] = []
        values = (self.generated, self.planned, self.executed)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            errors.append("command frame fields must be integers")
        if any(isinstance(value, int) and value < 0 for value in values):
            errors.append("command frame fields must be non-negative")
        if not errors and not (self.generated <= self.planned <= self.executed):
            errors.append("command frames must satisfy generated <= planned <= executed")
        return errors

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise ContractViolation("invalid_command_frames", "; ".join(errors))

    def to_dict(self) -> dict[str, int]:
        return {
            "generated": self.generated,
            "planned": self.planned,
            "executed": self.executed,
        }


@dataclass(frozen=True)
class FrameEnvelope:
    """One frame-aligned snapshot/message/clock contract record."""

    episode_id: str
    carla_frame: int
    simulation_seconds: float
    delta_seconds: float
    snapshot_frame: int
    message_frame: int
    clock_frame: int
    clock_seconds: float
    event_seq: int
    state_hash: str
    command_frames: CommandFrames | None = None
    timestamp_tolerance_seconds: float = 1e-6

    def __post_init__(self) -> None:
        FrameKey(self.episode_id, self.carla_frame)
        for name, value in (
            ("simulation_seconds", self.simulation_seconds),
            ("delta_seconds", self.delta_seconds),
            ("clock_seconds", self.clock_seconds),
            ("timestamp_tolerance_seconds", self.timestamp_tolerance_seconds),
        ):
            _finite_number(value, name)
        if self.simulation_seconds < 0 or self.clock_seconds < 0:
            raise ContractViolation("invalid_time", "simulation and clock time must be non-negative")
        if self.delta_seconds < 0:
            raise ContractViolation("invalid_delta", "delta_seconds must be non-negative")
        for name, value in (
            ("snapshot_frame", self.snapshot_frame),
            ("message_frame", self.message_frame),
            ("clock_frame", self.clock_frame),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractViolation("invalid_frame", f"{name} must be an integer")
            if value != self.carla_frame:
                raise ContractViolation(
                    "frame_mismatch",
                    f"{name}={value} does not match carla_frame={self.carla_frame}",
                )
        if isinstance(self.event_seq, bool) or not isinstance(self.event_seq, int) or self.event_seq <= 0:
            raise ContractViolation("invalid_event_sequence", "event_seq must be a positive integer")
        if not self.state_hash or len(self.state_hash) != 64:
            raise ContractViolation("invalid_state_hash", "state_hash must be a SHA-256 hex digest")
        try:
            int(self.state_hash, 16)
        except ValueError as exc:
            raise ContractViolation("invalid_state_hash", "state_hash must be hexadecimal") from exc
        if abs(self.simulation_seconds - self.clock_seconds) > self.timestamp_tolerance_seconds:
            raise ContractViolation(
                "clock_time_mismatch",
                "simulation_seconds and clock_seconds exceed configured tolerance",
            )
        if self.command_frames is not None:
            self.command_frames.assert_valid()

    @property
    def key(self) -> FrameKey:
        return FrameKey(self.episode_id, self.carla_frame)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "episode_id": self.episode_id,
            "frame_key": self.key.as_string(),
            "carla_frame": self.carla_frame,
            "simulation_seconds": self.simulation_seconds,
            "delta_seconds": self.delta_seconds,
            "snapshot_frame": self.snapshot_frame,
            "message_frame": self.message_frame,
            "clock_frame": self.clock_frame,
            "clock_seconds": self.clock_seconds,
            "event_seq": self.event_seq,
            "state_hash": self.state_hash,
        }
        if self.command_frames is not None:
            result["command_generated_frame"] = self.command_frames.generated
            result["command_planned_frame"] = self.command_frames.planned
            result["command_executed_frame"] = self.command_frames.executed
        else:
            result["command_generated_frame"] = None
            result["command_planned_frame"] = None
            result["command_executed_frame"] = None
        return result

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        timestamp_tolerance_seconds: float = 1e-6,
    ) -> "FrameEnvelope":
        nested = mapping.get("command_frames")
        generated = mapping.get("command_generated_frame")
        planned = mapping.get("command_planned_frame")
        executed = mapping.get("command_executed_frame")
        if isinstance(nested, Mapping):
            generated = nested.get("generated")
            planned = nested.get("planned")
            executed = nested.get("executed")
        command_frames = None
        if generated is not None or planned is not None or executed is not None:
            if generated is None or planned is None or executed is None:
                raise ContractViolation("invalid_command_frames", "command frame fields must be complete")
            command_frames = CommandFrames(int(generated), int(planned), int(executed))
        clock_seconds = mapping.get("clock_seconds", mapping.get("simulation_seconds"))
        return cls(
            episode_id=str(mapping["episode_id"]),
            carla_frame=int(mapping["carla_frame"]),
            simulation_seconds=float(mapping["simulation_seconds"]),
            delta_seconds=float(mapping.get("delta_seconds", 0.0)),
            snapshot_frame=int(mapping.get("snapshot_frame", mapping["carla_frame"])),
            message_frame=int(mapping.get("message_frame", mapping["carla_frame"])),
            clock_frame=int(mapping.get("clock_frame", mapping["carla_frame"])),
            clock_seconds=float(clock_seconds),
            event_seq=int(mapping.get("event_seq", 1)),
            state_hash=str(mapping["state_hash"]),
            command_frames=command_frames,
            timestamp_tolerance_seconds=timestamp_tolerance_seconds,
        )


@dataclass(frozen=True)
class IngestResult:
    status: str
    code: str
    accepted: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "code": self.code,
            "accepted": self.accepted,
            "message": self.message,
        }


class FrameLedger:
    """Reject duplicate, missing, stale, out-of-order and cross-episode frames."""

    def __init__(self, *, max_stale_frames: int = 0) -> None:
        if max_stale_frames < 0:
            raise ValueError("max_stale_frames must be non-negative")
        self.max_stale_frames = max_stale_frames
        self.episode_id: str | None = None
        self.last_frame: int | None = None
        self.accepted_frames: list[int] = []

    def _reject(self, code: str, message: str) -> IngestResult:
        return IngestResult(FAIL, code, False, message)

    def ingest(self, envelope: FrameEnvelope, *, source: str = "message") -> IngestResult:
        if self.episode_id is None:
            self.episode_id = envelope.episode_id
            self.last_frame = envelope.carla_frame
            self.accepted_frames.append(envelope.carla_frame)
            return IngestResult(PASS, "accepted", True, "first frame accepted")
        if envelope.episode_id != self.episode_id:
            return self._reject(
                "episode_mismatch",
                f"received episode {envelope.episode_id}, expected {self.episode_id}",
            )
        assert self.last_frame is not None
        frame = envelope.carla_frame
        if frame == self.last_frame:
            code = "duplicate_tick" if source == "tick" else "duplicate_message"
            return self._reject(code, f"frame {frame} was already accepted")
        if frame < self.last_frame:
            if self.last_frame - frame > self.max_stale_frames:
                return self._reject(
                    "stale_message" if source != "tick" else "out_of_order_tick",
                    f"frame {frame} is older than last accepted frame {self.last_frame}",
                )
            return self._reject("out_of_order", f"frame {frame} is behind {self.last_frame}")
        if frame > self.last_frame + 1:
            return self._reject(
                "missing_frame",
                f"frame gap detected: expected {self.last_frame + 1}, received {frame}",
            )
        self.last_frame = frame
        self.accepted_frames.append(frame)
        return IngestResult(PASS, "accepted", True, f"frame {frame} accepted")


class TickMasterRegistry:
    """Process-local guard against two components claiming the CARLA tick."""

    def __init__(self, configured_owner: str) -> None:
        if not configured_owner.strip():
            raise ValueError("configured_owner must be non-empty")
        self.configured_owner = configured_owner
        self.owner: str | None = None

    def claim(self, owner: str) -> None:
        if self.owner is not None and self.owner != owner:
            raise ContractViolation(
                "multiple_tick_masters",
                f"tick master already claimed by {self.owner!r}",
            )
        if owner != self.configured_owner:
            raise ContractViolation(
                "unexpected_tick_master",
                f"{owner!r} is not configured tick master {self.configured_owner!r}",
            )
        self.owner = owner

    def assert_owner(self, owner: str) -> None:
        if self.owner != owner:
            raise ContractViolation("tick_master_not_claimed", f"{owner!r} does not own the tick")

    def release(self, owner: str) -> None:
        self.assert_owner(owner)
        self.owner = None


def validate_tick_masters(owners: Sequence[str], expected_owner: str) -> list[str]:
    distinct = {owner for owner in owners if owner}
    errors: list[str] = []
    if distinct != {expected_owner}:
        errors.append(
            f"expected exactly one tick master {expected_owner!r}, observed {sorted(distinct)!r}"
        )
    return errors


def validate_carla_settings(settings: Any, config: SyncConfig) -> list[str]:
    """Validate a CARLA ``WorldSettings``-like object without importing CARLA."""

    errors = config.validate()
    expected = {
        "synchronous_mode": config.synchronous_mode,
        "fixed_delta_seconds": config.fixed_delta_seconds,
        "substepping": config.substepping,
        "max_substep_delta_time": config.max_substep_delta_time,
        "max_substeps": config.max_substeps,
    }
    for name, expected_value in expected.items():
        if not hasattr(settings, name):
            errors.append(f"CARLA settings missing {name}")
            continue
        actual = getattr(settings, name)
        if isinstance(expected_value, float):
            try:
                if abs(float(actual) - expected_value) > 1e-12:
                    errors.append(f"CARLA settings {name}={actual!r}, expected {expected_value!r}")
            except (TypeError, ValueError):
                errors.append(f"CARLA settings {name} is not numeric")
        elif actual != expected_value:
            errors.append(f"CARLA settings {name}={actual!r}, expected {expected_value!r}")
    return errors


def apply_sync_settings(world: Any, config: SyncConfig) -> Any:
    """Apply G0 settings and return a copy suitable for restoration."""

    config.assert_valid()
    original = copy.copy(world.get_settings())
    settings = copy.copy(original)
    settings.synchronous_mode = config.synchronous_mode
    settings.fixed_delta_seconds = config.fixed_delta_seconds
    settings.substepping = config.substepping
    settings.max_substep_delta_time = config.max_substep_delta_time
    settings.max_substeps = config.max_substeps
    world.apply_settings(settings)
    applied = world.get_settings()
    errors = validate_carla_settings(applied, config)
    if errors:
        raise ContractViolation("carla_settings_rejected", "; ".join(errors))
    return original


def restore_carla_settings(world: Any, original_settings: Any) -> None:
    world.apply_settings(original_settings)


def canonical_state_hash(state: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        state,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_episode_id(seed: int, config: SyncConfig) -> str:
    material = json.dumps(
        {"seed": seed, "config": config.to_dict()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def _deterministic_state(seed: int, frame: int) -> dict[str, float | int]:
    value = (seed * 1103515245 + frame * 12345 + 1013904223) & 0xFFFFFFFF
    return {
        "x": round(frame * 0.5 + (value % 1000) / 1_000_000.0, 9),
        "y": round((value % 3600) / 100.0, 9),
        "velocity": round(5.0 + ((value >> 8) % 1000) / 10_000.0, 9),
    }


def _make_deterministic_record(
    *, seed: int, frame: int, episode_id: str, config: SyncConfig
) -> dict[str, Any]:
    simulation_seconds = round(frame * config.fixed_delta_seconds, 9)
    state = _deterministic_state(seed, frame)
    state_hash = canonical_state_hash({"frame": frame, "state": state})
    envelope = FrameEnvelope(
        episode_id=episode_id,
        carla_frame=frame,
        simulation_seconds=simulation_seconds,
        delta_seconds=config.fixed_delta_seconds,
        snapshot_frame=frame,
        message_frame=frame,
        clock_frame=frame,
        clock_seconds=simulation_seconds,
        event_seq=frame,
        state_hash=state_hash,
        command_frames=CommandFrames(frame, frame, frame),
        timestamp_tolerance_seconds=config.timestamp_tolerance_seconds,
    )
    result = envelope.to_dict()
    result["state"] = state
    return result


def create_deterministic_checkpoint(seed: int, steps: int, config: SyncConfig) -> dict[str, Any]:
    config.assert_valid()
    if steps <= 0:
        raise ValueError("steps must be positive")
    return {
        "schema": "safedrive.g0.deterministic_checkpoint.v1",
        "status": "RUNNING",
        "seed": seed,
        "steps": steps,
        "episode_id": deterministic_episode_id(seed, config),
        "config": config.to_dict(),
        "records": [],
        "events": [],
    }


def append_deterministic_frame(state: MutableMapping[str, Any], config: SyncConfig) -> None:
    frame = len(state["records"]) + 1
    record = _make_deterministic_record(
        seed=int(state["seed"]),
        frame=frame,
        episode_id=str(state["episode_id"]),
        config=config,
    )
    state["records"].append(record)
    state["events"].extend(
        {"frame": frame, "event": event}
        for event in ("tick_request", "snapshot", "clock_publish", "status_publish")
    )


def checkpoint_to_trace(state: Mapping[str, Any]) -> dict[str, Any]:
    records = list(state["records"])
    return {
        "schema": "safedrive.g0.deterministic_trace.v1",
        "mode": "deterministic",
        "seed": state["seed"],
        "episode_id": state["episode_id"],
        "config": state["config"],
        "frames": [record["carla_frame"] for record in records],
        "timestamps": [record["simulation_seconds"] for record in records],
        "state_hashes": [record["state_hash"] for record in records],
        "events": list(state["events"]),
        "records": records,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def run_deterministic_smoke(
    *,
    seed: int,
    steps: int,
    config: SyncConfig,
    checkpoint_path: Path,
    trace_path: Path,
    resume: bool = False,
    interrupt_after: int | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Run or resume a deterministic smoke with an atomic checkpoint."""

    if resume:
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ContractViolation("checkpoint_unreadable", f"cannot resume {checkpoint_path}: {exc}") from exc
        expected = create_deterministic_checkpoint(seed, steps, config)
        for field in ("schema", "seed", "steps", "episode_id", "config"):
            if state.get(field) != expected[field]:
                raise ContractViolation(
                    "checkpoint_mismatch",
                    f"checkpoint field {field!r} does not match requested run",
                )
    else:
        state = create_deterministic_checkpoint(seed, steps, config)

    new_frames = 0
    while len(state["records"]) < steps:
        append_deterministic_frame(state, config)
        new_frames += 1
        state["status"] = "RUNNING"
        _write_json_atomic(checkpoint_path, state)
        if interrupt_after is not None and new_frames >= interrupt_after:
            state["status"] = "INTERRUPTED"
            _write_json_atomic(checkpoint_path, state)
            return "INTERRUPTED", None

    state["status"] = "COMPLETED"
    _write_json_atomic(checkpoint_path, state)
    trace = checkpoint_to_trace(state)
    _write_json_atomic(trace_path, trace)
    return "COMPLETED", trace


def build_deterministic_trace(seed: int, steps: int, config: SyncConfig) -> dict[str, Any]:
    state = create_deterministic_checkpoint(seed, steps, config)
    for _ in range(steps):
        append_deterministic_frame(state, config)
    state["status"] = "COMPLETED"
    return checkpoint_to_trace(state)


def compare_traces(
    first: Mapping[str, Any], second: Mapping[str, Any], *, tolerance_seconds: float
) -> dict[str, Any]:
    differences: list[str] = []
    if first.get("schema") != second.get("schema"):
        differences.append("schema differs")
    if first.get("seed") != second.get("seed"):
        differences.append("seed differs")
    if first.get("frames") != second.get("frames"):
        differences.append("frame sequence differs")
    if first.get("events") != second.get("events"):
        differences.append("event order differs")
    if first.get("state_hashes") != second.get("state_hashes"):
        differences.append("critical state hash sequence differs")
    first_times = list(first.get("timestamps", []))
    second_times = list(second.get("timestamps", []))
    if len(first_times) != len(second_times):
        differences.append("timestamp length differs")
    else:
        for index, (left, right) in enumerate(zip(first_times, second_times)):
            if abs(float(left) - float(right)) > tolerance_seconds:
                differences.append(f"timestamp differs at index {index}")
                break
    return {
        "pass": not differences,
        "differences": differences,
        "tolerance_seconds": tolerance_seconds,
        "frames_compared": min(len(first_times), len(second_times)),
    }


def run_contract_fault_injection(config: SyncConfig) -> list[dict[str, Any]]:
    """Exercise the required duplicate/missing/stale and ownership failures."""

    trace = build_deterministic_trace(seed=17, steps=4, config=config)
    envelopes = [
        FrameEnvelope.from_mapping(
            record, timestamp_tolerance_seconds=config.timestamp_tolerance_seconds
        )
        for record in trace["records"]
    ]
    results: list[dict[str, Any]] = []

    duplicate_ledger = FrameLedger()
    duplicate_ledger.ingest(envelopes[0], source="tick")
    duplicate = duplicate_ledger.ingest(envelopes[0], source="tick")
    results.append({"id": "duplicate_tick", "observed": duplicate.to_dict()})

    missing_ledger = FrameLedger()
    missing_ledger.ingest(envelopes[0], source="tick")
    missing = missing_ledger.ingest(envelopes[2], source="tick")
    results.append({"id": "missing_frame", "observed": missing.to_dict()})

    stale_ledger = FrameLedger()
    for envelope in envelopes[:3]:
        stale_ledger.ingest(envelope, source="message")
    stale = stale_ledger.ingest(envelopes[0], source="message")
    results.append({"id": "stale_message", "observed": stale.to_dict()})

    registry = TickMasterRegistry(config.tick_master)
    registry.claim(config.tick_master)
    try:
        registry.claim("unexpected.second.tick.master")
    except ContractViolation as exc:
        results.append(
            {
                "id": "multiple_tick_masters",
                "observed": {
                    "status": FAIL,
                    "code": exc.code,
                    "accepted": False,
                    "message": exc.message,
                },
            }
        )
    return results


def run_carla_trace(
    *,
    host: str,
    port: int,
    timeout_seconds: float,
    steps: int,
    config: SyncConfig,
) -> dict[str, Any]:
    """Run a live CARLA tick smoke; ROS publication is provided by sync_driver."""

    try:
        import carla
    except ImportError as exc:  # pragma: no cover - depends on external CARLA install
        raise ContractViolation("carla_client_missing", "CARLA Python API is not importable") from exc

    config.assert_valid()
    if steps <= 0:
        raise ValueError("steps must be positive")
    client = carla.Client(host, port)
    client.set_timeout(timeout_seconds)
    try:
        client_version = client.get_client_version()
        server_version = client.get_server_version()
        if client_version != server_version:
            raise ContractViolation(
                "carla_version_mismatch",
                f"client={client_version}, server={server_version}",
            )
        world = client.get_world()
        original_settings = apply_sync_settings(world, config)
    except ContractViolation:
        raise
    except Exception as exc:  # pragma: no cover - depends on external CARLA server
        raise ContractViolation(
            "carla_connection_failed",
            f"CARLA connection failed at {host}:{port}: {type(exc).__name__}: {exc}",
        ) from exc

    registry = TickMasterRegistry(config.tick_master)
    registry.claim(config.tick_master)
    ledger = FrameLedger()
    episode_id = uuid.uuid4().hex
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    try:
        for event_seq in range(1, steps + 1):
            events.append({"event": "tick_request", "sequence": event_seq})
            frame = int(world.tick())
            events.append({"event": "snapshot", "sequence": event_seq})
            snapshot = world.get_snapshot()
            simulation_seconds = float(snapshot.timestamp.elapsed_seconds)
            state_hash = canonical_state_hash(
                {
                    "frame": frame,
                    "simulation_seconds": simulation_seconds,
                    "map": world.get_map().name,
                }
            )
            envelope = FrameEnvelope(
                episode_id=episode_id,
                carla_frame=frame,
                simulation_seconds=simulation_seconds,
                delta_seconds=float(snapshot.timestamp.delta_seconds),
                snapshot_frame=int(snapshot.frame),
                message_frame=frame,
                clock_frame=frame,
                clock_seconds=simulation_seconds,
                event_seq=event_seq,
                state_hash=state_hash,
                timestamp_tolerance_seconds=config.timestamp_tolerance_seconds,
            )
            if snapshot.frame != frame:
                raise ContractViolation("snapshot_frame_mismatch", "world.tick and snapshot disagree")
            result = ledger.ingest(envelope, source="tick")
            if not result.accepted:
                raise ContractViolation(result.code, result.message)
            events.extend(
                {"event": name, "sequence": event_seq}
                for name in ("clock_publish", "status_publish")
            )
            records.append(envelope.to_dict())
    finally:
        try:
            restore_carla_settings(world, original_settings)
        finally:
            registry.release(config.tick_master)
    return {
        "schema": "safedrive.g0.carla_trace.v1",
        "mode": "carla",
        "episode_id": episode_id,
        "host": host,
        "port": port,
        "client_version": client_version,
        "server_version": server_version,
        "map": world.get_map().name,
        "config": config.to_dict(),
        "frames": [record["carla_frame"] for record in records],
        "timestamps": [record["simulation_seconds"] for record in records],
        "state_hashes": [record["state_hash"] for record in records],
        "events": events,
        "records": records,
    }


__all__ = [
    "BLOCKED",
    "CommandFrames",
    "ContractViolation",
    "FAIL",
    "FrameEnvelope",
    "FrameKey",
    "FrameLedger",
    "IngestResult",
    "PASS",
    "SyncConfig",
    "TickMasterRegistry",
    "WARN",
    "append_deterministic_frame",
    "apply_sync_settings",
    "build_deterministic_trace",
    "canonical_state_hash",
    "checkpoint_to_trace",
    "compare_traces",
    "create_deterministic_checkpoint",
    "deterministic_episode_id",
    "load_sync_config",
    "restore_carla_settings",
    "run_carla_trace",
    "run_contract_fault_injection",
    "run_deterministic_smoke",
    "validate_carla_settings",
    "validate_tick_masters",
]
