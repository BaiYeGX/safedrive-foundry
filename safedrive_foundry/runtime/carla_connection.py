"""Unified, bounded CARLA endpoint discovery and preflight checks.

This module is deliberately independent of ROS and ScenarioRuntime.  It owns
host/port resolution, TCP-vs-RPC classification, CARLA version/map/settings
queries, process observation, tick-owner observation, and the one-shot recovery
path used by ``sdf sim``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


EXPECTED_CARLA_VERSION = "0.9.16"
DEFAULT_RPC_PORT = 2000
READY = "READY"
RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"
FAILED_FINAL = "FAILED_FINAL"

EXIT_READY = 0
EXIT_FAILED_FINAL = 1
EXIT_BLOCKED_EXTERNAL = 2
EXIT_RETRYABLE_FAILURE = 75

SERVER_NOT_RUNNING = "SERVER_NOT_RUNNING"
HOST_RESOLUTION_FAILED = "HOST_RESOLUTION_FAILED"
TCP_UNREACHABLE = "TCP_UNREACHABLE"
RPC_HANDSHAKE_FAILED = "RPC_HANDSHAKE_FAILED"
CLIENT_VERSION_MISMATCH = "CLIENT_VERSION_MISMATCH"
SERVER_VERSION_MISMATCH = "SERVER_VERSION_MISMATCH"
MAP_QUERY_FAILED = "MAP_QUERY_FAILED"
WORLD_SETTINGS_QUERY_FAILED = "WORLD_SETTINGS_QUERY_FAILED"
TICK_OWNER_CONFLICT = "TICK_OWNER_CONFLICT"
STARTUP_TIMEOUT = "STARTUP_TIMEOUT"
UNKNOWN_CARLA_PATH = "UNKNOWN_CARLA_PATH"
NEEDS_USER_ACTION = "NEEDS_USER_ACTION"

RETRYABLE_CODES = frozenset(
    {SERVER_NOT_RUNNING, HOST_RESOLUTION_FAILED, TCP_UNREACHABLE, RPC_HANDSHAKE_FAILED, STARTUP_TIMEOUT}
)
BLOCKING_CODES = frozenset(
    {CLIENT_VERSION_MISMATCH, SERVER_VERSION_MISMATCH, MAP_QUERY_FAILED, WORLD_SETTINGS_QUERY_FAILED, TICK_OWNER_CONFLICT}
)


@dataclass(frozen=True)
class Endpoint:
    host: str
    port: int
    source: str


@dataclass
class ConnectionReport:
    status: str
    host: str | None = None
    resolved_host: str | None = None
    port: int = DEFAULT_RPC_PORT
    host_source: str | None = None
    previous_host: str | None = None
    tcp_reachable: bool = False
    rpc_reachable: bool = False
    client_version: str | None = None
    server_version: str | None = None
    map: str | None = None
    process_state: str = "UNKNOWN"
    synchronous_mode: bool | None = None
    fixed_delta_seconds: float | None = None
    tick_owner: str | None = None
    needs_user_action: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    task_action: str = "IN_PROGRESS"
    recovery_action: str | None = None
    recovery_result: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "host": self.host,
            "resolved_host": self.resolved_host or self.host,
            "port": self.port,
            "host_source": self.host_source,
            "previous_host": self.previous_host,
            "tcp_reachable": self.tcp_reachable,
            "rpc_reachable": self.rpc_reachable,
            "client_version": self.client_version,
            "server_version": self.server_version,
            "map": self.map,
            "process_state": self.process_state,
            "synchronous_mode": self.synchronous_mode,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "tick_owner": self.tick_owner,
            "needs_user_action": self.needs_user_action,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "task_action": self.task_action,
            "recovery_action": self.recovery_action,
            "recovery_result": self.recovery_result,
            "details": self.details,
        }


def _default_route_host() -> str:
    """Resolve the current WSL Windows gateway without embedding an address."""

    ip = shutil.which("ip")
    if ip is None:
        raise RuntimeError("ip command is unavailable")
    completed = subprocess.run(
        [ip, "route", "show", "default"],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    for line in completed.stdout.splitlines():
        tokens = line.split()
        if tokens and tokens[0] == "default" and "via" in tokens:
            gateway = tokens[tokens.index("via") + 1]
            if gateway:
                return gateway
    raise RuntimeError("WSL default gateway was not found")


def _default_tcp_probe(host: str, port: int, timeout_seconds: float) -> tuple[bool, str | None]:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True, None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _default_client_factory(host: str, port: int) -> Any:
    try:
        import carla
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError(f"CARLA Python API import failed: {exc}") from exc
    return carla.Client(host, port)


def _default_process_query() -> str:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        return "UNKNOWN"
    command = (
        "Get-Process -Name CarlaUE4,CarlaUE4-Win64-Shipping "
        "-ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Id"
    )
    try:
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"
    return "RUNNING" if completed.returncode == 0 and completed.stdout.strip() else "NOT_RUNNING"


def _json_settings(settings: Any) -> dict[str, Any]:
    return {
        "synchronous_mode": getattr(settings, "synchronous_mode", None),
        "fixed_delta_seconds": getattr(settings, "fixed_delta_seconds", None),
        "substepping": getattr(settings, "substepping", None),
        "max_substep_delta_time": getattr(settings, "max_substep_delta_time", None),
        "max_substeps": getattr(settings, "max_substeps", None),
    }


class ConnectionResolver:
    """Single CARLA connection implementation used by diagnostics and CLI."""

    def __init__(
        self,
        root: Path,
        *,
        expected_version: str = EXPECTED_CARLA_VERSION,
        timeout_seconds: float = 3.0,
        client_factory: Callable[[str, int], Any] | None = None,
        tcp_probe: Callable[[str, int, float], tuple[bool, str | None]] | None = None,
        host_discoverer: Callable[[], str] | None = None,
        process_query: Callable[[], str] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        start_process: Callable[[Mapping[str, Any]], bool] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.expected_version = expected_version
        self.timeout_seconds = timeout_seconds
        self.client_factory = client_factory or _default_client_factory
        self.tcp_probe = tcp_probe or _default_tcp_probe
        self.host_discoverer = host_discoverer or _default_route_host
        self.process_query = process_query or _default_process_query
        self.sleeper = sleeper
        self.start_process = start_process

    def resolve_port(self, explicit_port: int | None = None) -> int:
        value = explicit_port if explicit_port is not None else os.environ.get("CARLA_PORT", DEFAULT_RPC_PORT)
        port = int(value)
        if not 1 <= port <= 65535:
            raise ValueError(f"CARLA port out of range: {port}")
        return port

    def resolve_host(self, explicit_host: str | None = None, *, force_dynamic: bool = False) -> Endpoint:
        if explicit_host:
            return Endpoint(explicit_host, self.resolve_port(), "explicit")
        if not force_dynamic:
            configured = os.environ.get("CARLA_HOST")
            if configured:
                return Endpoint(configured, self.resolve_port(), "environment")
        try:
            return Endpoint(self.host_discoverer(), self.resolve_port(), "wsl_default_gateway" if not force_dynamic else "wsl_default_gateway_retry")
        except Exception as exc:
            if platform.system() == "Windows" and os.environ.get("CARLA_ALLOW_LOCALHOST", "").lower() in {"1", "true", "yes"}:
                return Endpoint("localhost", self.resolve_port(), "verified_localhost")
            raise RuntimeError(str(exc)) from exc

    def _tick_owner(self) -> tuple[str | None, str | None]:
        scenario_path = self.root / "safedrive_foundry" / "config" / "runtime" / "scenario_runtime.toml"
        sync_path = self.root / "safedrive_foundry" / "config" / "carla_ros.toml"
        configured = "sdf.g1-02.runtime"
        lease_value = ".runtime/tick-lease.lock"
        try:
            if scenario_path.exists():
                data = tomllib.loads(scenario_path.read_text(encoding="utf-8"))
                scenario_data = data.get("scenario_runtime", {})
                configured = str(scenario_data.get("owner", configured))
                lease_value = str(scenario_data.get("lease_file", lease_value))
            if sync_path.exists():
                data = tomllib.loads(sync_path.read_text(encoding="utf-8"))
                configured = str(data.get("sync", {}).get("tick_master", configured))
        except (OSError, ValueError, TypeError):
            pass
        lease_path = Path(lease_value)
        if not lease_path.is_absolute():
            lease_path = self.root / lease_path
        # The configured runtime writes its owner into the lease file. Reading
        # it is observational only; this path never acquires a lease.
        if lease_path.exists():
            try:
                payload = json.loads(lease_path.read_text(encoding="utf-8"))
                owner = str(payload.get("owner", ""))
                if owner:
                    return owner, configured
            except (OSError, ValueError, TypeError):
                pass
        return f"free (configured={configured})", configured

    def _start_spec(self) -> dict[str, Any] | None:
        path = self.root / "safedrive_foundry" / "config" / "runtime" / "carla_start.toml"
        if not path.exists():
            return None
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            spec = dict(data.get("carla_start", {}))
            if spec.get("expected_version") != self.expected_version:
                return None
            return spec
        except (OSError, ValueError, TypeError):
            return None

    def _probe(self, endpoint: Endpoint, *, process_state: str, retry_count: int = 0, previous_host: str | None = None) -> ConnectionReport:
        tick_owner, configured_owner = self._tick_owner()
        report = ConnectionReport(
            status=FAILED_FINAL,
            host=endpoint.host,
            resolved_host=endpoint.host,
            port=endpoint.port,
            host_source=endpoint.source,
            previous_host=previous_host,
            process_state=process_state,
            tick_owner=tick_owner,
            retry_count=retry_count,
        )
        tcp_ok, tcp_error = self.tcp_probe(endpoint.host, endpoint.port, self.timeout_seconds)
        report.tcp_reachable = tcp_ok
        if not tcp_ok:
            report.status = RETRYABLE_FAILURE
            report.task_action = "IN_PROGRESS"
            report.error_code = SERVER_NOT_RUNNING if process_state == "NOT_RUNNING" else TCP_UNREACHABLE
            report.error_message = tcp_error or "CARLA RPC TCP endpoint is unreachable"
            return report
        try:
            client = self.client_factory(endpoint.host, endpoint.port)
            if hasattr(client, "set_timeout"):
                client.set_timeout(self.timeout_seconds)
            report.client_version = str(client.get_client_version())
            report.rpc_reachable = True
            if report.client_version != self.expected_version:
                report.error_code = CLIENT_VERSION_MISMATCH
                report.error_message = f"client={report.client_version}, expected={self.expected_version}"
                report.task_action = "BLOCKED"
                return report
            report.server_version = str(client.get_server_version())
            if report.server_version != self.expected_version:
                report.error_code = SERVER_VERSION_MISMATCH
                report.error_message = f"server={report.server_version}, expected={self.expected_version}"
                report.task_action = "BLOCKED"
                return report
            world = client.get_world()
            try:
                report.map = str(world.get_map().name)
            except Exception as exc:
                report.error_code = MAP_QUERY_FAILED
                report.error_message = f"{type(exc).__name__}: {exc}"
                report.task_action = "BLOCKED"
                return report
            try:
                settings = world.get_settings()
                report.synchronous_mode = getattr(settings, "synchronous_mode", None)
                report.fixed_delta_seconds = getattr(settings, "fixed_delta_seconds", None)
                report.details["world_settings"] = _json_settings(settings)
            except Exception as exc:
                report.error_code = WORLD_SETTINGS_QUERY_FAILED
                report.error_message = f"{type(exc).__name__}: {exc}"
                report.task_action = "BLOCKED"
                return report
            if tick_owner and configured_owner and not tick_owner.startswith("free") and tick_owner != configured_owner:
                report.error_code = TICK_OWNER_CONFLICT
                report.error_message = f"active={tick_owner}, configured={configured_owner}"
                report.task_action = "BLOCKED"
                return report
            report.status = READY
            report.task_action = "IN_PROGRESS"
            return report
        except Exception as exc:
            report.status = RETRYABLE_FAILURE
            report.task_action = "IN_PROGRESS"
            report.error_code = RPC_HANDSHAKE_FAILED
            report.error_message = f"{type(exc).__name__}: {exc}"
            return report

    def preflight(self, *, host: str | None = None, port: int | None = None, retry_host: bool = True) -> ConnectionReport:
        try:
            endpoint = self.resolve_host(host)
            endpoint = Endpoint(endpoint.host, self.resolve_port(port), endpoint.source)
        except Exception as exc:
            return ConnectionReport(
                status=RETRYABLE_FAILURE,
                port=self.resolve_port(port),
                needs_user_action=False,
                error_code=HOST_RESOLUTION_FAILED,
                error_message=str(exc),
                task_action="IN_PROGRESS",
            )
        process_state = self.process_query()
        first = self._probe(endpoint, process_state=process_state)
        if first.status == READY or not retry_host or host is not None:
            return first
        if first.error_code not in RETRYABLE_CODES:
            return first
        try:
            refreshed = self.resolve_host(None, force_dynamic=True)
        except Exception:
            return first
        if refreshed.host == endpoint.host:
            return first
        second = self._probe(refreshed, process_state=self.process_query(), retry_count=1, previous_host=endpoint.host)
        return second

    def status(self, *, host: str | None = None, port: int | None = None) -> ConnectionReport:
        return self.preflight(host=host, port=port, retry_host=True)

    def connect(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        report: ConnectionReport | None = None,
    ) -> tuple[Any, ConnectionReport]:
        """Return a CARLA client only after this resolver's READY gate passes."""

        checked = report or self.preflight(host=host, port=port)
        if checked.status != READY or not checked.host:
            raise RuntimeError(f"{checked.error_code or 'CARLA_NOT_READY'}: {checked.error_message or checked.status}")
        client = self.client_factory(checked.host, checked.port)
        if hasattr(client, "set_timeout"):
            client.set_timeout(self.timeout_seconds)
        return client, checked

    def _launch_known(self, spec: Mapping[str, Any]) -> bool:
        if self.start_process is not None:
            return bool(self.start_process(spec))
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
        executable = str(spec.get("windows_executable", ""))
        if powershell is None or not executable:
            return False
        arguments = str(spec.get("arguments", ""))
        command = f"Start-Process -FilePath '{executable}' -ArgumentList '{arguments}'"
        try:
            subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", command], timeout=5.0, check=True)
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def ensure(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        startup_timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.5,
    ) -> ConnectionReport:
        report = self.preflight(host=host, port=port)
        if report.status == READY:
            report.recovery_action = "already_running"
            report.recovery_result = "compatible_instance_present; no second process started"
            return report
        if report.error_code not in RETRYABLE_CODES:
            return report
        if report.process_state != "NOT_RUNNING":
            report.status = BLOCKED_EXTERNAL
            report.needs_user_action = True
            report.error_code = NEEDS_USER_ACTION
            report.error_message = "CARLA process state is unknown; refusing to start a possibly duplicate instance"
            report.task_action = "BLOCKED_EXTERNAL"
            return report
        spec = self._start_spec()
        if spec is None:
            report.status = BLOCKED_EXTERNAL
            report.needs_user_action = True
            report.error_code = NEEDS_USER_ACTION
            report.error_message = "No verified CARLA 0.9.16 startup specification is available"
            report.task_action = "BLOCKED_EXTERNAL"
            return report
        if not self._launch_known(spec):
            report.status = BLOCKED_EXTERNAL
            report.needs_user_action = True
            report.error_code = NEEDS_USER_ACTION
            report.error_message = "Verified CARLA startup could not be invoked without external interaction"
            report.task_action = "BLOCKED_EXTERNAL"
            return report
        deadline = time.monotonic() + max(0.0, startup_timeout_seconds)
        while time.monotonic() < deadline:
            self.sleeper(max(0.0, poll_interval_seconds))
            ready = self.preflight(host=host, port=port, retry_host=False)
            if ready.status == READY:
                ready.recovery_action = "start_known_carla"
                ready.recovery_result = "started_and_handshake_ready"
                return ready
        report.status = RETRYABLE_FAILURE
        report.error_code = STARTUP_TIMEOUT
        report.error_message = f"CARLA did not become ready within {startup_timeout_seconds:.1f}s"
        report.recovery_action = "start_known_carla"
        report.recovery_result = "timeout"
        report.task_action = "IN_PROGRESS"
        return report


def exit_code(report: ConnectionReport) -> int:
    if report.status == READY:
        return EXIT_READY
    if report.status == RETRYABLE_FAILURE:
        return EXIT_RETRYABLE_FAILURE
    if report.status == BLOCKED_EXTERNAL:
        return EXIT_BLOCKED_EXTERNAL
    return EXIT_FAILED_FINAL
