"""SafeDrive runtime contracts independent of ROS and CARLA clients."""

from .contracts import ContractViolation, StatusJsonAdapter
from .identity import RunIdentity, RuntimeIdentityFactory
from .profiles import RuntimeProfile, load_runtime_profiles
from .carla_connection import ConnectionReport, ConnectionResolver, exit_code
from .scenario_runtime import (
    ActorSpec,
    CleanupFailed,
    FrameHeader,
    RunRegistry,
    RuntimeViolation,
    ScenarioRuntime,
    ScenarioSpec,
    SensorSpec,
    TickLeaseUnavailable,
)

__all__ = [
    "ContractViolation",
    "RunIdentity",
    "RuntimeIdentityFactory",
    "RuntimeProfile",
    "StatusJsonAdapter",
    "load_runtime_profiles",
    "ConnectionReport",
    "ConnectionResolver",
    "exit_code",
    "ActorSpec",
    "CleanupFailed",
    "FrameHeader",
    "RunRegistry",
    "RuntimeViolation",
    "ScenarioRuntime",
    "ScenarioSpec",
    "SensorSpec",
    "TickLeaseUnavailable",
]
