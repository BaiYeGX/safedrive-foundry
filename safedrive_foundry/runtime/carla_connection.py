"""Unified, bounded CARLA endpoint discovery and preflight checks.

This module is deliberately independent of ROS and ScenarioRuntime.  It owns
host/port resolution, TCP-vs-RPC classification, CARLA version/map/settings
queries, process observation, tick-owner observation, and the one-shot recovery
path used by ``sdf sim``.
"""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


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

# Transparent proxy / TUN defaults (e.g. Mihomo 198.18.0.0/15) accept TCP and
# look like gateways but are not the Windows CARLA host.
_PROXY_LIKE_NETWORKS = (
    ipaddress.ip_network("198.18.0.0/15"),
)


def running_in_wsl() -> bool:
    """Return True when this process is already inside WSL (not on bare Windows)."""

    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        version = ""
    if "microsoft" in version or "wsl" in version:
        return True
    return Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()


def _is_proxy_like_host(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return any(address in network for network in _PROXY_LIKE_NETWORKS)


def _parse_default_routes(route_text: str) -> list[tuple[int, str]]:
    """Parse ``ip route`` default lines into (metric, gateway) pairs."""

    routes: list[tuple[int, str]] = []
    for line in route_text.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] != "default" or "via" not in tokens:
            continue
        try:
            gateway = tokens[tokens.index("via") + 1]
        except (ValueError, IndexError):
            continue
        if not gateway:
            continue
        metric = 1000
        if "metric" in tokens:
            try:
                metric = int(tokens[tokens.index("metric") + 1])
            except (ValueError, IndexError):
                metric = 1000
        routes.append((metric, gateway))
    routes.sort(key=lambda item: item[0])
    return routes


def _rank_host_candidates(gateways: Sequence[tuple[int, str]]) -> list[tuple[str, str]]:
    """Rank gateways: non-proxy preferred, then lower metric. Deduplicate hosts."""

    preferred: list[tuple[int, int, str, str]] = []
    for metric, gateway in gateways:
        proxy = _is_proxy_like_host(gateway)
        # proxy-like last (priority 1), real hosts first (priority 0)
        preferred.append((1 if proxy else 0, metric, gateway, "wsl_default_gateway_proxy" if proxy else "wsl_default_gateway"))
    preferred.sort(key=lambda item: (item[0], item[1]))
    ordered: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, _, gateway, source in preferred:
        if gateway in seen:
            continue
        seen.add(gateway)
        ordered.append((gateway, source))
    return ordered


def discover_route_hosts() -> list[tuple[str, str]]:
    """Discover WSL→Windows host candidates without embedding a single address."""

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
    routes = _parse_default_routes(completed.stdout)
    if not routes:
        raise RuntimeError("WSL default gateway was not found")
    ranked = _rank_host_candidates(routes)
    if not ranked:
        raise RuntimeError("WSL default gateway was not found")
    return ranked


def _default_route_host() -> str:
    """Resolve a preferred WSL Windows gateway (first ranked non-proxy when possible)."""

    return discover_route_hosts()[0][0]


def windows_path_to_wsl(path: str) -> Path | None:
    """Convert ``E:\\foo`` / ``E:/foo`` to ``/mnt/e/foo`` when running under WSL."""

    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path.strip())
    if not match:
        return None
    drive, rest = match.group(1).lower(), match.group(2).replace("\\", "/")
    return Path("/mnt") / drive / rest


def wsl_path_to_windows(path: str | Path) -> str | None:
    """Convert ``/mnt/e/foo`` to ``E:\\foo`` for PowerShell Start-Process."""

    text = str(path).replace("\\", "/")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if not match:
        return None
    drive, rest = match.group(1).upper(), match.group(2).replace("/", "\\")
    return f"{drive}:\\{rest}"


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


def _windows_carla_process_running() -> str | None:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        return None
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
        return None
    if completed.returncode == 0 and completed.stdout.strip():
        return "RUNNING"
    return "NOT_RUNNING"


def _linux_carla_process_running() -> str | None:
    """Observe native Linux/WSL CARLA processes when present."""

    try:
        completed = subprocess.run(
            ["pgrep", "-f", r"CarlaUE4(\.sh)?|CarlaUE4-Linux"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return "RUNNING" if completed.returncode == 0 and completed.stdout.strip() else "NOT_RUNNING"


def _default_process_query() -> str:
    """Return RUNNING if either Windows or Linux CARLA process is observed."""

    observations = [state for state in (_windows_carla_process_running(), _linux_carla_process_running()) if state is not None]
    if not observations:
        return "UNKNOWN"
    if any(state == "RUNNING" for state in observations):
        return "RUNNING"
    return "NOT_RUNNING"


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

    def discover_host_candidates(self, explicit_host: str | None = None, *, force_dynamic: bool = False) -> list[Endpoint]:
        """Return ordered endpoints to try; READY is decided only by RPC handshake.

        WSL mirrored networking often exposes Windows CARLA on 127.0.0.1; classic
        NAT mode fails localhost quickly and falls through to ranked gateways.
        Proxy-like gateways (198.18.0.0/15) are always last.
        """

        port = self.resolve_port()
        if explicit_host:
            return [Endpoint(explicit_host, port, "explicit")]
        endpoints: list[Endpoint] = []
        seen: set[str] = set()

        def add(host: str, source: str) -> None:
            if host in seen:
                return
            seen.add(host)
            endpoints.append(Endpoint(host, port, source))

        if not force_dynamic:
            configured = os.environ.get("CARLA_HOST")
            if configured:
                add(configured, "environment")

        in_wsl = running_in_wsl()
        allow_local = os.environ.get("CARLA_ALLOW_LOCALHOST", "").lower() in {"1", "true", "yes"}
        injected = self.host_discoverer is not _default_route_host and self.host_discoverer is not None

        try:
            if injected:
                # Injected single-host discoverer (unit tests): keep one primary + retry label.
                primary = self.host_discoverer()
                source = "wsl_default_gateway_retry" if force_dynamic else "wsl_default_gateway"
                add(primary, source)
            else:
                # Prefer localhost early inside WSL (mirrored networking) and on Windows.
                if in_wsl or platform.system() == "Windows" or allow_local:
                    add("127.0.0.1", "loopback_local")
                for host, source in discover_route_hosts():
                    labeled = source if not force_dynamic else f"{source}_retry"
                    add(host, labeled)
                if allow_local:
                    add("localhost", "verified_localhost")
        except Exception:
            if not endpoints:
                raise
        if not endpoints:
            raise RuntimeError("no CARLA host candidates available")
        return endpoints

    def resolve_host(self, explicit_host: str | None = None, *, force_dynamic: bool = False) -> Endpoint:
        return self.discover_host_candidates(explicit_host, force_dynamic=force_dynamic)[0]

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
            # get_client_version() is local to the Python API; it does not prove
            # the server is reachable. Only a successful server RPC sets READY.
            report.client_version = str(client.get_client_version())
            if report.client_version != self.expected_version:
                report.error_code = CLIENT_VERSION_MISMATCH
                report.error_message = f"client={report.client_version}, expected={self.expected_version}"
                report.task_action = "BLOCKED"
                return report
            report.server_version = str(client.get_server_version())
            report.rpc_reachable = True
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
            candidates = self.discover_host_candidates(host)
            if port is not None:
                resolved_port = self.resolve_port(port)
                candidates = [Endpoint(item.host, resolved_port, item.source) for item in candidates]
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
        # Explicit host: single probe. Otherwise walk ranked candidates; READY
        # requires RPC handshake (TCP alone is not sufficient under proxies).
        if host is not None or not retry_host:
            return self._probe(candidates[0], process_state=process_state)

        last = self._probe(candidates[0], process_state=process_state)
        if last.status == READY or last.error_code not in RETRYABLE_CODES:
            return last
        previous = candidates[0].host
        remaining = list(candidates[1:])
        # Injected single-host discoverers (tests) yield the next host only on a
        # second force_dynamic discovery call.
        if not remaining and self.host_discoverer is not _default_route_host:
            try:
                refreshed = self.resolve_host(None, force_dynamic=True)
                if refreshed.host != previous:
                    remaining.append(refreshed)
            except Exception:
                pass
        for index, endpoint in enumerate(remaining, start=1):
            if endpoint.host == previous:
                continue
            report = self._probe(
                endpoint,
                process_state=self.process_query(),
                retry_count=index,
                previous_host=previous,
            )
            last = report
            if report.status == READY or report.error_code not in RETRYABLE_CODES:
                return report
            previous = endpoint.host
        return last

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

    def _resolve_start_paths(self, spec: Mapping[str, Any]) -> tuple[str | None, Path | None]:
        """Return (windows_executable, wsl_path) for existence checks and launch."""

        windows_executable = str(spec.get("windows_executable", "") or "").strip() or None
        wsl_raw = str(spec.get("wsl_path", "") or spec.get("linux_path", "") or "").strip()
        wsl_path = Path(wsl_raw) if wsl_raw else None
        if wsl_path is None and windows_executable:
            wsl_path = windows_path_to_wsl(windows_executable)
        if windows_executable is None and wsl_path is not None:
            windows_executable = wsl_path_to_windows(wsl_path)
        return windows_executable, wsl_path

    def _launch_known(self, spec: Mapping[str, Any]) -> bool:
        if self.start_process is not None:
            return bool(self.start_process(spec))
        windows_executable, wsl_path = self._resolve_start_paths(spec)
        if wsl_path is not None and not wsl_path.exists():
            return False
        arguments = str(spec.get("arguments", ""))
        # Preferred: Windows CARLA via PowerShell (works from WSL interop).
        # ArgumentList must be a comma-separated quoted list, not one giant string,
        # or Unreal may ignore the map path and reopen the previous map.
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if powershell is not None and windows_executable:
            arg_parts = [p for p in arguments.split() if p]
            if arg_parts:
                ps_list = ",".join(f"'{p}'" for p in arg_parts)
                command = f"Start-Process -FilePath '{windows_executable}' -ArgumentList {ps_list}"
            else:
                command = f"Start-Process -FilePath '{windows_executable}'"
            try:
                subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
                    timeout=5.0,
                    check=True,
                )
                return True
            except (OSError, subprocess.SubprocessError):
                pass
        # Optional native Linux/WSL CARLA binary (CarlaUE4.sh) when present.
        linux_launcher = str(spec.get("linux_executable", "") or "").strip()
        if not linux_launcher and wsl_path is not None:
            sibling = wsl_path.parent / "CarlaUE4.sh"
            if sibling.exists():
                linux_launcher = str(sibling)
        if linux_launcher and Path(linux_launcher).exists():
            try:
                args = [linux_launcher, *arguments.split()] if arguments else [linux_launcher]
                subprocess.Popen(
                    args,
                    cwd=str(Path(linux_launcher).parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return True
            except OSError:
                return False
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
