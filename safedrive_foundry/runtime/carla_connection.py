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
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180.0
DEFAULT_LAUNCH_MODE = "default_engine"
VALID_LAUNCH_MODES = frozenset({"default_engine", "explicit_arguments"})
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
LAUNCH_PARAM_MISMATCH = "LAUNCH_PARAM_MISMATCH"
MAP_OR_RHI_CONFIG_MISMATCH = "MAP_OR_RHI_CONFIG_MISMATCH"
MAP_MISMATCH = "MAP_MISMATCH"
CARLA_SHADER_PIPELINE_FATAL = "CARLA_SHADER_PIPELINE_FATAL"
CARLA_PROCESS_EXITED_EARLY = "CARLA_PROCESS_EXITED_EARLY"
WORKING_DIRECTORY_MISSING = "WORKING_DIRECTORY_MISSING"
INVALID_LAUNCH_CONFIG = "INVALID_LAUNCH_CONFIG"

RETRYABLE_CODES = frozenset(
    {
        SERVER_NOT_RUNNING,
        HOST_RESOLUTION_FAILED,
        TCP_UNREACHABLE,
        RPC_HANDSHAKE_FAILED,
        STARTUP_TIMEOUT,
        LAUNCH_PARAM_MISMATCH,
        CARLA_PROCESS_EXITED_EARLY,
    }
)

VALID_RHI = frozenset({"dx11", "dx12"})
_RHI_ALIASES = {
    "dx11": "dx11",
    "d3d11": "dx11",
    "11": "dx11",
    "dx12": "dx12",
    "d3d12": "dx12",
    "12": "dx12",
}
_RHI_CLI_TOKENS = frozenset({"-dx11", "-dx12", "dx11", "dx12"})
_OFFSCREEN_CLI_TOKENS = frozenset({"-renderoffscreen", "renderoffscreen"})
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


def normalize_rhi(value: str | None, *, default: str = "dx11") -> str:
    """Normalize RHI tokens to exactly one of ``dx11`` / ``dx12``."""

    if value is None or str(value).strip() == "":
        return default
    token = str(value).strip().lower().lstrip("-")
    token = _RHI_ALIASES.get(token, token)
    if token not in VALID_RHI:
        raise ValueError(f"unsupported RHI {value!r}; expected one of {sorted(VALID_RHI)}")
    return token


def parse_rhi_from_command_line(arguments: str | None) -> str | None:
    """Return the sole RHI flag from a CARLA command line, or None if absent."""

    if not arguments:
        return None
    found: list[str] = []
    for token in str(arguments).split():
        lowered = token.lower().lstrip("-")
        if lowered in VALID_RHI or lowered in _RHI_ALIASES:
            found.append(normalize_rhi(token))
    if not found:
        return None
    # Last wins for malformed lines, but callers should still reject dual flags.
    return found[-1]


def parse_offscreen_from_command_line(arguments: str | None) -> bool:
    """Return True when ``-RenderOffScreen`` is present (case-insensitive)."""

    if not arguments:
        return False
    return any(token.lower() in _OFFSCREEN_CLI_TOKENS for token in str(arguments).split())


def count_rhi_flags(arguments: str | None) -> int:
    if not arguments:
        return 0
    count = 0
    for token in str(arguments).split():
        lowered = token.lower().lstrip("-")
        if lowered in VALID_RHI or lowered in {"d3d11", "d3d12", "11", "12"}:
            count += 1
    return count


def count_offscreen_flags(arguments: str | None) -> int:
    if not arguments:
        return 0
    return sum(1 for token in str(arguments).split() if token.lower() in _OFFSCREEN_CLI_TOKENS)


def rewrite_arguments_rhi_offscreen(
    arguments: str,
    *,
    rhi: str | None = None,
    render_offscreen: bool | None = None,
) -> str:
    """Strip existing RHI/offscreen flags and re-apply exactly one RHI (+ optional offscreen)."""

    tokens = [token for token in str(arguments).split() if token]
    kept: list[str] = []
    for token in tokens:
        lowered = token.lower()
        bare = lowered.lstrip("-")
        if lowered in _RHI_CLI_TOKENS or bare in VALID_RHI or bare in _RHI_ALIASES:
            continue
        if lowered in _OFFSCREEN_CLI_TOKENS:
            continue
        kept.append(token)

    effective_rhi = normalize_rhi(
        rhi if rhi is not None else parse_rhi_from_command_line(arguments) or "dx11"
    )
    effective_offscreen = (
        bool(render_offscreen)
        if render_offscreen is not None
        else parse_offscreen_from_command_line(arguments)
    )

    # Keep relative order: inject RHI just before -carla-rpc-port when present.
    rhi_flag = f"-{effective_rhi}"
    offscreen_flag = "-RenderOffScreen"
    insert_at = len(kept)
    for index, token in enumerate(kept):
        if token.lower().startswith("-carla-rpc-port"):
            insert_at = index
            break
    injection = [rhi_flag]
    if effective_offscreen:
        injection.append(offscreen_flag)
    kept[insert_at:insert_at] = injection
    return " ".join(kept)


def build_carla_launch_arguments(
    *,
    map_content: str = "/Game/Carla/Maps/Town04",
    res_x: int = 640,
    res_y: int = 360,
    quality_level: str = "Low",
    rhi: str = "dx11",
    render_offscreen: bool = False,
    rpc_port: int = DEFAULT_RPC_PORT,
    windowed: bool = True,
    nosound: bool = True,
) -> str:
    """Build a single-process CARLA argument string with exactly one RHI flag."""

    rhi_norm = normalize_rhi(rhi)
    parts: list[str] = [str(map_content)]
    if windowed:
        parts.append("-windowed")
    parts.extend(
        [
            f"-ResX={int(res_x)}",
            f"-ResY={int(res_y)}",
            f"-quality-level={quality_level}",
        ]
    )
    if nosound:
        parts.append("-nosound")
    parts.append(f"-{rhi_norm}")
    if render_offscreen:
        parts.append("-RenderOffScreen")
    parts.append(f"-carla-rpc-port={int(rpc_port)}")
    command = " ".join(parts)
    assert count_rhi_flags(command) == 1, command
    assert count_offscreen_flags(command) == (1 if render_offscreen else 0), command
    return command


def rewrite_arguments_map(arguments: str, map_name: str) -> str:
    """Replace the leading map token with the packaged Unreal map asset path.

    The Windows packaged CARLA 0.9.16 executable accepts the first argument as
    an Unreal asset identifier.  A bare content package such as
    ``/Game/Carla/Maps/Town05`` can be ignored and cause the executable to fall
    back to ``DefaultEngine.ini``.  The explicit cold-start path therefore uses
    the same ``<package>.<asset>`` form as the map settings in that file.
    """

    from runtime.carla_engine_config import map_asset_path

    asset = map_asset_path(map_name)
    tokens = [token for token in str(arguments).split() if token]
    if not tokens:
        return asset
    if tokens[0].startswith("/Game/") or tokens[0].startswith("Game/"):
        tokens[0] = asset
    else:
        tokens.insert(0, asset)
    return " ".join(tokens)


def parse_map_from_command_line(arguments: str | None) -> str | None:
    """Best-effort town token from a CARLA argument string."""

    from runtime.carla_engine_config import normalize_map_token

    if not arguments:
        return None
    for token in str(arguments).split():
        if "/Maps/" in token or token.startswith("/Game/"):
            return normalize_map_token(token)
    return None


@dataclass(frozen=True)
class LaunchResult:
    """Result of a single Windows/Linux CARLA process start attempt."""

    ok: bool
    pid: int | None = None
    executable: str | None = None
    working_directory: str | None = None
    launch_arguments: str | None = None
    launch_mode: str | None = None
    powershell_command: str | None = None
    error: str | None = None
    started_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "pid": self.pid,
            "executable": self.executable,
            "working_directory": self.working_directory,
            "launch_arguments": self.launch_arguments,
            "launch_mode": self.launch_mode,
            "powershell_command": self.powershell_command,
            "error": self.error,
            "started_at": self.started_at,
        }


def build_powershell_start_command(
    *,
    windows_executable: str,
    working_directory: str,
    argument_parts: Sequence[str] | None = None,
) -> str:
    """Build a PowerShell Start-Process command with explicit WorkingDirectory and -PassThru."""

    exe = windows_executable.replace("'", "''")
    cwd = working_directory.replace("'", "''")
    base = (
        f"$p = Start-Process -FilePath '{exe}' -WorkingDirectory '{cwd}' "
        f"-PassThru"
    )
    parts = [p for p in (argument_parts or ()) if p]
    if parts:
        # Escape single quotes inside each argument for PowerShell single-quoted strings.
        escaped = [p.replace("'", "''") for p in parts]
        ps_list = ",".join(f"'{p}'" for p in escaped)
        base += f" -ArgumentList {ps_list}"
    base += "; Write-Output $p.Id"
    return base


def windows_path_exists(windows_path: str) -> bool:
    """Return True when a Windows path is visible from this process (WSL mount or native)."""

    text = str(windows_path).strip()
    if not text:
        return False
    wsl = windows_path_to_wsl(text)
    if wsl is not None and wsl.exists():
        return True
    native = Path(text)
    if native.exists():
        return True
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        return False
    safe = text.replace("'", "''")
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"if (Test-Path -LiteralPath '{safe}') {{ '1' }} else {{ '0' }}",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return (completed.stdout or "").strip() == "1"


def query_windows_pid_alive(pid: int) -> bool | None:
    """Return True/False when PowerShell can answer; None if unqueryable."""

    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        return None
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f"if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{ '1' }} else {{ '0' }}",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (completed.stdout or "").strip()
    if out == "1":
        return True
    if out == "0":
        return False
    return None


def _xml_find_text_recursive(root: Any, tag: str) -> str:
    import xml.etree.ElementTree as ET

    wanted = tag
    if root.tag == wanted or root.tag.endswith(wanted):
        return (root.text or "").strip()
    for child in list(root):
        text = _xml_find_text_recursive(child, wanted)
        if text:
            return text
        # Dotted Unreal tags are literal element names.
        if child.tag == wanted or child.tag.endswith(wanted):
            return (child.text or "").strip()
    # Fallback XPath-style.
    try:
        found = root.find(f".//{wanted}")
        if found is not None and found.text:
            return found.text.strip()
    except Exception:
        pass
    return ""


def find_recent_crash_context(
    *,
    since_unix_s: float,
    crash_root: Path | None = None,
) -> dict[str, Any] | None:
    """Locate a CrashContext.runtime-xml written at/after the launch wall time."""

    import xml.etree.ElementTree as ET

    roots: list[Path] = []
    if crash_root is not None:
        roots.append(crash_root)
    else:
        # WSL view of Windows user profiles.
        users = Path("/mnt/c/Users")
        if users.is_dir():
            roots.append(users)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            roots.append(Path(local) / "CarlaUE4" / "Saved" / "Crashes")

    candidates: list[Path] = []
    for root in roots:
        try:
            if root.name == "Crashes" or (root / "CrashContext.runtime-xml").is_file():
                pattern_roots = [root]
            else:
                pattern_roots = [root]
            for base in pattern_roots:
                if not base.exists():
                    continue
                if base.is_file() and base.name == "CrashContext.runtime-xml":
                    candidates.append(base)
                    continue
                # Profile tree: */AppData/Local/CarlaUE4/Saved/Crashes/*/CrashContext.runtime-xml
                for path in base.glob(
                    "**/CarlaUE4/Saved/Crashes/*/CrashContext.runtime-xml"
                ):
                    candidates.append(path)
                for path in base.glob("*/CrashContext.runtime-xml"):
                    candidates.append(path)
                direct = base / "CrashContext.runtime-xml"
                if direct.is_file():
                    candidates.append(direct)
        except OSError:
            continue

    recent: list[Path] = []
    for path in candidates:
        try:
            if path.stat().st_mtime >= float(since_unix_s) - 2.0:
                recent.append(path)
        except OSError:
            continue
    if not recent:
        return None
    source = max(recent, key=lambda path: path.stat().st_mtime)
    try:
        tree = ET.parse(source)
        root_el = tree.getroot()
    except (OSError, ET.ParseError) as exc:
        return {
            "captured": False,
            "source_path": str(source),
            "capture_error": str(exc),
        }
    wanted = (
        "CrashGUID",
        "CrashType",
        "ErrorMessage",
        "SecondsSinceStart",
        "CommandLine",
        "ProcessId",
        "RHI.RHIName",
        "MemoryStats.bIsOOM",
        "ExecutableName",
        "BaseDir",
        "RootDir",
    )
    fields = {tag: _xml_find_text_recursive(root_el, tag) for tag in wanted}
    return {
        "captured": True,
        "source_path": str(source),
        "source_mtime_s": source.stat().st_mtime,
        "fields": fields,
        "CrashGUID": fields.get("CrashGUID"),
        "CrashType": fields.get("CrashType"),
        "ErrorMessage": fields.get("ErrorMessage"),
        "RHI": fields.get("RHI.RHIName"),
        "CommandLine": fields.get("CommandLine"),
        "SecondsSinceStart": fields.get("SecondsSinceStart"),
        "bIsOOM": fields.get("MemoryStats.bIsOOM"),
    }


def classify_startup_crash(crash_context: Mapping[str, Any] | None) -> str | None:
    """Return a startup error_code when CrashContext explains an early death."""

    if not crash_context or not crash_context.get("captured"):
        return None
    fields = crash_context.get("fields") if isinstance(crash_context.get("fields"), Mapping) else crash_context
    message = str(
        (fields or {}).get("ErrorMessage")
        or crash_context.get("ErrorMessage")
        or ""
    )
    if "Shader compilation failures are Fatal" in message:
        return CARLA_SHADER_PIPELINE_FATAL
    return None


def parse_resolution_from_command_line(arguments: str | None) -> tuple[int | None, int | None]:
    if not arguments:
        return None, None
    res_x = res_y = None
    for token in str(arguments).split():
        if token.startswith("-ResX="):
            try:
                res_x = int(token.split("=", 1)[1])
            except ValueError:
                pass
        elif token.startswith("-ResY="):
            try:
                res_y = int(token.split("=", 1)[1])
            except ValueError:
                pass
    return res_x, res_y


def query_windows_carla_command_line() -> str | None:
    """Best-effort read of the live CarlaUE4 command line via PowerShell."""

    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        return None
    command = (
        "$p = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Name -match '^CarlaUE4' } | Select-Object -First 1; "
        "if ($p) { $p.CommandLine }"
    )
    try:
        completed = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = (completed.stdout or "").strip()
    return line or None


def launch_params_match(
    command_line: str | None,
    *,
    rhi: str,
    render_offscreen: bool,
) -> bool | None:
    """Return True/False when cmdline is parseable, else None (unknown)."""

    if not command_line:
        return None
    effective = parse_rhi_from_command_line(command_line)
    if effective is None:
        return None
    return effective == normalize_rhi(rhi) and parse_offscreen_from_command_line(command_line) == bool(
        render_offscreen
    )


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
            [
                "pgrep",
                "-x",
                (
                    "CarlaUE4|CarlaUE4.sh|CarlaUE4-Linux|"
                    "CarlaUE4-Linux-Shipping"
                ),
            ],
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

    def _start_spec(
        self,
        *,
        rhi: str | None = None,
        render_offscreen: bool | None = None,
        map_name: str | None = None,
        launch_mode: str | None = None,
    ) -> dict[str, Any] | None:
        path = self.root / "safedrive_foundry" / "config" / "runtime" / "carla_start.toml"
        if not path.exists():
            return None
        try:
            from runtime.carla_engine_config import map_asset_path

            data = tomllib.loads(path.read_text(encoding="utf-8"))
            spec = dict(data.get("carla_start", {}))
            if spec.get("expected_version") != self.expected_version:
                return None
            raw_arguments = str(spec.get("arguments", "") or "")
            configured_rhi = spec.get("rhi")
            if configured_rhi is None or str(configured_rhi).strip() == "":
                configured_rhi = parse_rhi_from_command_line(raw_arguments) or "dx12"
            effective_rhi = normalize_rhi(rhi if rhi is not None else configured_rhi, default="dx12")
            if render_offscreen is not None:
                effective_offscreen = bool(render_offscreen)
            elif "render_offscreen" in spec:
                effective_offscreen = bool(spec.get("render_offscreen"))
            else:
                effective_offscreen = parse_offscreen_from_command_line(raw_arguments)

            mode = str(launch_mode or spec.get("launch_mode") or DEFAULT_LAUNCH_MODE).strip().lower()
            if mode not in VALID_LAUNCH_MODES:
                return None

            requested_map = (
                str(map_name).strip()
                if map_name is not None and str(map_name).strip()
                else str(spec.get("default_map") or "").strip() or parse_map_from_command_line(raw_arguments)
            )

            arguments = raw_arguments
            if requested_map:
                arguments = rewrite_arguments_map(arguments, requested_map) if arguments else (
                    build_carla_launch_arguments(
                        map_content=map_asset_path(requested_map),
                        rhi=effective_rhi,
                        render_offscreen=effective_offscreen,
                    )
                )
            arguments = rewrite_arguments_rhi_offscreen(
                arguments,
                rhi=effective_rhi,
                render_offscreen=effective_offscreen,
            )
            if mode == "explicit_arguments":
                if count_rhi_flags(arguments) != 1:
                    return None
                if count_offscreen_flags(arguments) != (1 if effective_offscreen else 0):
                    return None
            # default_engine mode keeps rewritten arguments only for evidence / explicit fallback.

            windows_executable = str(spec.get("windows_executable", "") or "").strip()
            working_directory = str(spec.get("windows_working_directory", "") or "").strip()
            if not working_directory and windows_executable:
                from runtime.carla_engine_config import windows_install_root_from_executable

                working_directory = windows_install_root_from_executable(windows_executable) or ""

            default_engine_ini = str(spec.get("default_engine_ini", "") or "").strip() or None

            spec["rhi"] = effective_rhi
            spec["render_offscreen"] = effective_offscreen
            spec["arguments"] = arguments
            spec["requested_rhi"] = normalize_rhi(rhi, default="dx12") if rhi is not None else effective_rhi
            spec["requested_map"] = requested_map
            spec["launch_mode"] = mode
            spec["windows_working_directory"] = working_directory
            if default_engine_ini:
                spec["default_engine_ini"] = default_engine_ini
            # In default_engine mode the process is started without ArgumentList.
            if mode == "default_engine":
                spec["effective_launch_arguments"] = ""
                spec["effective_rhi_source"] = "default_engine_config"
            else:
                spec["effective_launch_arguments"] = arguments
                spec["effective_rhi_source"] = "command_line"
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

    def _launch_result(self, spec: Mapping[str, Any]) -> LaunchResult:
        """Start CARLA once and return structured launch evidence (including PID)."""

        if self.start_process is not None:
            raw = self.start_process(spec)
            if isinstance(raw, LaunchResult):
                return raw
            if isinstance(raw, Mapping):
                return LaunchResult(
                    ok=bool(raw.get("ok", True)),
                    pid=int(raw["pid"]) if raw.get("pid") is not None else None,
                    executable=raw.get("executable") or spec.get("windows_executable"),  # type: ignore[arg-type]
                    working_directory=raw.get("working_directory") or spec.get("windows_working_directory"),  # type: ignore[arg-type]
                    launch_arguments=raw.get("launch_arguments")  # type: ignore[arg-type]
                    if "launch_arguments" in raw
                    else str(spec.get("effective_launch_arguments", "") or ""),
                    launch_mode=str(raw.get("launch_mode") or spec.get("launch_mode") or ""),
                    powershell_command=raw.get("powershell_command"),  # type: ignore[arg-type]
                    error=raw.get("error"),  # type: ignore[arg-type]
                    started_at=float(raw["started_at"]) if raw.get("started_at") is not None else time.time(),
                )
            return LaunchResult(ok=bool(raw), started_at=time.time(), launch_mode=str(spec.get("launch_mode") or ""))

        windows_executable, wsl_path = self._resolve_start_paths(spec)
        if wsl_path is not None and not wsl_path.exists():
            return LaunchResult(ok=False, error=f"wsl_path missing: {wsl_path}", started_at=time.time())

        working_directory = str(spec.get("windows_working_directory", "") or "").strip()
        launch_mode = str(spec.get("launch_mode") or DEFAULT_LAUNCH_MODE)
        if launch_mode == "default_engine":
            launch_arguments = ""
            arg_parts: list[str] = []
        else:
            launch_arguments = str(spec.get("effective_launch_arguments") or spec.get("arguments") or "")
            arg_parts = [p for p in launch_arguments.split() if p]

        started_at = time.time()
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
        if powershell is not None and windows_executable:
            if not working_directory:
                return LaunchResult(
                    ok=False,
                    executable=windows_executable,
                    working_directory=None,
                    launch_arguments=launch_arguments,
                    launch_mode=launch_mode,
                    error="windows_working_directory is required for Windows Start-Process",
                    started_at=started_at,
                )
            if not windows_path_exists(working_directory):
                return LaunchResult(
                    ok=False,
                    executable=windows_executable,
                    working_directory=working_directory,
                    launch_arguments=launch_arguments,
                    launch_mode=launch_mode,
                    error=f"working directory does not exist: {working_directory}",
                    started_at=started_at,
                )
            command = build_powershell_start_command(
                windows_executable=windows_executable,
                working_directory=working_directory,
                argument_parts=arg_parts,
            )
            try:
                completed = subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                return LaunchResult(
                    ok=False,
                    executable=windows_executable,
                    working_directory=working_directory,
                    launch_arguments=launch_arguments,
                    launch_mode=launch_mode,
                    powershell_command=command,
                    error=str(exc),
                    started_at=started_at,
                )
            pid: int | None = None
            for line in (completed.stdout or "").splitlines():
                token = line.strip()
                if token.isdigit():
                    pid = int(token)
                    break
            if completed.returncode != 0 and pid is None:
                return LaunchResult(
                    ok=False,
                    executable=windows_executable,
                    working_directory=working_directory,
                    launch_arguments=launch_arguments,
                    launch_mode=launch_mode,
                    powershell_command=command,
                    error=(completed.stderr or completed.stdout or "Start-Process failed").strip(),
                    started_at=started_at,
                )
            return LaunchResult(
                ok=True,
                pid=pid,
                executable=windows_executable,
                working_directory=working_directory,
                launch_arguments=launch_arguments,
                launch_mode=launch_mode,
                powershell_command=command,
                started_at=started_at,
            )

        # Optional native Linux/WSL CARLA binary (CarlaUE4.sh) when present.
        linux_launcher = str(spec.get("linux_executable", "") or "").strip()
        if not linux_launcher and wsl_path is not None:
            sibling = wsl_path.parent / "CarlaUE4.sh"
            if sibling.exists():
                linux_launcher = str(sibling)
        if linux_launcher and Path(linux_launcher).exists():
            try:
                args = [linux_launcher, *arg_parts] if arg_parts else [linux_launcher]
                proc = subprocess.Popen(
                    args,
                    cwd=str(Path(linux_launcher).parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return LaunchResult(
                    ok=True,
                    pid=proc.pid,
                    executable=linux_launcher,
                    working_directory=str(Path(linux_launcher).parent),
                    launch_arguments=launch_arguments,
                    launch_mode=launch_mode,
                    started_at=started_at,
                )
            except OSError as exc:
                return LaunchResult(ok=False, error=str(exc), started_at=started_at)
        return LaunchResult(ok=False, error="no launch path available", started_at=started_at)

    def _launch_known(self, spec: Mapping[str, Any]) -> bool:
        """Backward-compatible bool wrapper around :meth:`_launch_result`."""

        return self._launch_result(spec).ok

    def _attach_launch_evidence(
        self,
        report: ConnectionReport,
        *,
        spec: Mapping[str, Any] | None = None,
        launch: LaunchResult | None = None,
        requested_map: str | None = None,
        configured_default_map: str | None = None,
        crash_context: Mapping[str, Any] | None = None,
        process_exit_code: int | None = None,
        startup_wall_time: float | None = None,
    ) -> None:
        details = dict(report.details or {})
        if spec is not None:
            details.setdefault("launch_mode", spec.get("launch_mode"))
            details.setdefault("requested_rhi", spec.get("requested_rhi") or spec.get("rhi"))
            details.setdefault("effective_rhi", spec.get("rhi"))
            details.setdefault("effective_rhi_source", spec.get("effective_rhi_source"))
            details.setdefault("render_offscreen", bool(spec.get("render_offscreen")))
            details.setdefault("executable", spec.get("windows_executable"))
            details.setdefault("working_directory", spec.get("windows_working_directory"))
            details.setdefault(
                "launch_arguments",
                spec.get("effective_launch_arguments")
                if spec.get("launch_mode") == "default_engine"
                else spec.get("arguments"),
            )
            details.setdefault("requested_map", spec.get("requested_map") or requested_map)
        if requested_map is not None:
            details["requested_map"] = requested_map
        if configured_default_map is not None:
            details["configured_default_map"] = configured_default_map
        if report.map is not None:
            details["actual_map"] = report.map
        if launch is not None:
            details["pid"] = launch.pid
            details["executable"] = launch.executable or details.get("executable")
            details["working_directory"] = launch.working_directory or details.get("working_directory")
            details["launch_arguments"] = (
                launch.launch_arguments
                if launch.launch_arguments is not None
                else details.get("launch_arguments")
            )
            details["launch_mode"] = launch.launch_mode or details.get("launch_mode")
            details["powershell_command"] = launch.powershell_command
            details["launch"] = launch.to_dict()
        if process_exit_code is not None:
            details["process_exit_code"] = process_exit_code
        if startup_wall_time is not None:
            details["startup_wall_time"] = startup_wall_time
        if crash_context is not None:
            details["crash_context"] = dict(crash_context)
        report.details = details

    def ensure(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        poll_interval_seconds: float = 0.5,
        rhi: str | None = None,
        render_offscreen: bool | None = None,
        map_name: str | None = None,
        launch_mode: str | None = None,
        auto_pin_default_engine: bool = True,
    ) -> ConnectionReport:
        """Reuse a compatible instance or start CARLA once (map-aware, no mode cycling)."""

        from runtime.carla_engine_config import (
            maps_match,
            pin_default_engine_config,
            read_default_engine_config,
            resolve_default_engine_ini,
            verify_default_engine_matches,
            windows_install_root_from_executable,
        )

        wall_start = time.time()
        report = self.preflight(host=host, port=port)
        requested_rhi = normalize_rhi(rhi, default="dx12") if rhi is not None else None
        requested_map = str(map_name).strip() if map_name is not None and str(map_name).strip() else None

        if report.status == READY:
            report.details = dict(report.details or {})
            cmdline = query_windows_carla_command_line()
            report.details["server_command_line"] = cmdline
            report.details["actual_map"] = report.map
            if requested_map is not None:
                report.details["requested_map"] = requested_map
                if not maps_match(report.map, requested_map):
                    report.status = FAILED_FINAL
                    report.error_code = MAP_MISMATCH
                    report.error_message = (
                        f"running map {report.map!r} does not match requested {requested_map!r}; "
                        "cold-start required (no client.load_world)"
                    )
                    report.recovery_action = "require_cold_start"
                    report.recovery_result = "map_mismatch_existing_instance"
                    report.task_action = "BLOCKED"
                    return report
            if requested_rhi is not None or render_offscreen is not None:
                want_rhi = requested_rhi or parse_rhi_from_command_line(cmdline or "") or "dx12"
                want_off = (
                    bool(render_offscreen)
                    if render_offscreen is not None
                    else parse_offscreen_from_command_line(cmdline or "")
                )
                match = launch_params_match(cmdline, rhi=want_rhi, render_offscreen=want_off)
                report.details["requested_rhi"] = want_rhi
                report.details["effective_rhi"] = parse_rhi_from_command_line(cmdline or "")
                report.details["render_offscreen"] = want_off
                # Manual/default_engine processes may have no -dx12 on the command line.
                if match is False and parse_rhi_from_command_line(cmdline or "") is not None:
                    report.status = RETRYABLE_FAILURE
                    report.error_code = LAUNCH_PARAM_MISMATCH
                    report.error_message = (
                        "running CARLA RHI/offscreen does not match request; cold-start required"
                    )
                    report.recovery_action = "require_cold_start"
                    report.recovery_result = "launch_param_mismatch"
                    report.task_action = "IN_PROGRESS"
                    return report
                if parse_rhi_from_command_line(cmdline or "") is None and requested_rhi is not None:
                    report.details["effective_rhi"] = requested_rhi
                    report.details["effective_rhi_source"] = "default_engine_config"
            else:
                parsed = parse_rhi_from_command_line(cmdline or "")
                report.details["effective_rhi"] = parsed
                report.details["render_offscreen"] = parse_offscreen_from_command_line(cmdline or "")
                if parsed is None:
                    report.details["effective_rhi_source"] = "default_engine_config"
                else:
                    report.details["effective_rhi_source"] = "command_line"
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

        spec = self._start_spec(
            rhi=rhi,
            render_offscreen=render_offscreen,
            map_name=requested_map,
            launch_mode=launch_mode,
        )
        if spec is None:
            report.status = BLOCKED_EXTERNAL
            report.needs_user_action = True
            report.error_code = NEEDS_USER_ACTION
            report.error_message = "No verified CARLA 0.9.16 startup specification is available"
            report.task_action = "BLOCKED_EXTERNAL"
            return report

        effective_map = requested_map or spec.get("requested_map")
        effective_rhi = str(spec.get("rhi") or "dx12")
        mode = str(spec.get("launch_mode") or DEFAULT_LAUNCH_MODE)
        working_directory = str(spec.get("windows_working_directory") or "").strip()
        if not working_directory:
            report.status = BLOCKED_EXTERNAL
            report.needs_user_action = True
            report.error_code = INVALID_LAUNCH_CONFIG
            report.error_message = "windows_working_directory is not configured in carla_start.toml"
            report.task_action = "BLOCKED_EXTERNAL"
            self._attach_launch_evidence(report, spec=spec, requested_map=effective_map)
            return report
        if not windows_path_exists(working_directory):
            report.status = BLOCKED_EXTERNAL
            report.needs_user_action = True
            report.error_code = WORKING_DIRECTORY_MISSING
            report.error_message = f"CARLA working directory does not exist: {working_directory}"
            report.task_action = "BLOCKED_EXTERNAL"
            self._attach_launch_evidence(report, spec=spec, requested_map=effective_map)
            return report

        install_root = working_directory or windows_install_root_from_executable(
            str(spec.get("windows_executable") or "")
        )
        ini_override = spec.get("default_engine_ini")
        ini_path = Path(str(ini_override)) if ini_override else resolve_default_engine_ini(install_root)
        engine_cfg = read_default_engine_config(ini_path)
        configured_default_map = engine_cfg.configured_map_token

        if mode == "default_engine":
            mismatch = verify_default_engine_matches(
                engine_cfg,
                requested_map=str(effective_map) if effective_map else None,
                requested_rhi=effective_rhi,
            )
            if mismatch is not None:
                if auto_pin_default_engine:
                    pin_result = pin_default_engine_config(
                        requested_map=str(effective_map) if effective_map else None,
                        requested_rhi=effective_rhi,
                        path=ini_path,
                    )
                    report.details = dict(report.details or {})
                    report.details["default_engine_pin"] = pin_result
                    if not pin_result.get("ok"):
                        report.status = BLOCKED_EXTERNAL
                        report.needs_user_action = True
                        report.error_code = str(pin_result.get("error_code") or MAP_OR_RHI_CONFIG_MISMATCH)
                        report.error_message = str(pin_result.get("message") or mismatch.message)
                        report.task_action = "BLOCKED_EXTERNAL"
                        self._attach_launch_evidence(
                            report,
                            spec=spec,
                            requested_map=effective_map,
                            configured_default_map=configured_default_map,
                        )
                        return report
                    engine_cfg = read_default_engine_config(ini_path)
                    configured_default_map = engine_cfg.configured_map_token
                    mismatch = verify_default_engine_matches(
                        engine_cfg,
                        requested_map=str(effective_map) if effective_map else None,
                        requested_rhi=effective_rhi,
                    )
                if mismatch is not None:
                    report.status = BLOCKED_EXTERNAL
                    report.needs_user_action = True
                    report.error_code = MAP_OR_RHI_CONFIG_MISMATCH
                    report.error_message = mismatch.message
                    report.task_action = "BLOCKED_EXTERNAL"
                    report.details = dict(report.details or {})
                    report.details["config_mismatch"] = mismatch.to_dict()
                    self._attach_launch_evidence(
                        report,
                        spec=spec,
                        requested_map=effective_map,
                        configured_default_map=configured_default_map,
                    )
                    return report

        launch = self._launch_result(spec)
        if not launch.ok:
            report.status = BLOCKED_EXTERNAL
            report.needs_user_action = True
            err = launch.error or "Verified CARLA startup could not be invoked without external interaction"
            if "working directory" in err.lower():
                report.error_code = WORKING_DIRECTORY_MISSING
            else:
                report.error_code = NEEDS_USER_ACTION
            report.error_message = err
            report.task_action = "BLOCKED_EXTERNAL"
            report.recovery_action = "start_known_carla"
            report.recovery_result = "launch_failed"
            self._attach_launch_evidence(
                report,
                spec=spec,
                launch=launch,
                requested_map=effective_map,
                configured_default_map=configured_default_map,
                startup_wall_time=time.time() - wall_start,
            )
            return report

        deadline = time.monotonic() + max(0.0, startup_timeout_seconds)
        while time.monotonic() < deadline:
            self.sleeper(max(0.0, poll_interval_seconds))
            if launch.pid is not None:
                alive = query_windows_pid_alive(launch.pid)
                if alive is False:
                    crash = find_recent_crash_context(since_unix_s=launch.started_at or wall_start)
                    shader_code = classify_startup_crash(crash)
                    report.status = BLOCKED_EXTERNAL if shader_code else RETRYABLE_FAILURE
                    report.error_code = shader_code or CARLA_PROCESS_EXITED_EARLY
                    report.needs_user_action = bool(shader_code)
                    report.error_message = (
                        "CARLA process exited during startup with shader pipeline fatal"
                        if shader_code
                        else f"CARLA process pid={launch.pid} exited before RPC READY"
                    )
                    report.recovery_action = "start_known_carla"
                    report.recovery_result = "process_exited_early"
                    report.task_action = "BLOCKED_EXTERNAL" if shader_code else "IN_PROGRESS"
                    report.process_state = "NOT_RUNNING"
                    self._attach_launch_evidence(
                        report,
                        spec=spec,
                        launch=launch,
                        requested_map=effective_map,
                        configured_default_map=configured_default_map,
                        crash_context=crash,
                        process_exit_code=1,
                        startup_wall_time=time.time() - wall_start,
                    )
                    return report
            ready = self.preflight(host=host, port=port, retry_host=False)
            if ready.status == READY:
                ready.recovery_action = "start_known_carla"
                ready.recovery_result = "started_and_handshake_ready"
                self._attach_launch_evidence(
                    ready,
                    spec=spec,
                    launch=launch,
                    requested_map=effective_map,
                    configured_default_map=configured_default_map,
                    startup_wall_time=time.time() - wall_start,
                )
                if effective_map and not maps_match(ready.map, str(effective_map)):
                    ready.status = FAILED_FINAL
                    ready.error_code = MAP_MISMATCH
                    ready.error_message = (
                        f"READY but actual_map={ready.map!r} requested_map={effective_map!r}; "
                        "refusing client.load_world"
                    )
                    ready.recovery_result = "map_mismatch_after_start"
                    ready.task_action = "BLOCKED"
                    ready.details["actual_map"] = ready.map
                    ready.details["requested_map"] = effective_map
                    return ready
                return ready

        # Alive (or unknown) but RPC never became ready → true startup timeout.
        report.status = RETRYABLE_FAILURE
        report.error_code = STARTUP_TIMEOUT
        report.error_message = f"CARLA did not become ready within {startup_timeout_seconds:.1f}s"
        report.recovery_action = "start_known_carla"
        report.recovery_result = "timeout"
        report.task_action = "IN_PROGRESS"
        if launch.pid is not None:
            alive = query_windows_pid_alive(launch.pid)
            if alive is False:
                crash = find_recent_crash_context(since_unix_s=launch.started_at or wall_start)
                shader_code = classify_startup_crash(crash)
                report.status = BLOCKED_EXTERNAL if shader_code else RETRYABLE_FAILURE
                report.error_code = shader_code or CARLA_PROCESS_EXITED_EARLY
                report.needs_user_action = bool(shader_code)
                report.error_message = (
                    "CARLA process exited during startup with shader pipeline fatal"
                    if shader_code
                    else f"CARLA process pid={launch.pid} exited before RPC READY"
                )
                report.recovery_result = "process_exited_early"
                self._attach_launch_evidence(
                    report,
                    spec=spec,
                    launch=launch,
                    requested_map=effective_map,
                    configured_default_map=configured_default_map,
                    crash_context=crash,
                    process_exit_code=1,
                    startup_wall_time=time.time() - wall_start,
                )
                return report
        self._attach_launch_evidence(
            report,
            spec=spec,
            launch=launch,
            requested_map=effective_map,
            configured_default_map=configured_default_map,
            startup_wall_time=time.time() - wall_start,
        )
        return report


def exit_code(report: ConnectionReport) -> int:
    if report.status == READY:
        return EXIT_READY
    if report.status == RETRYABLE_FAILURE:
        return EXIT_RETRYABLE_FAILURE
    if report.status == BLOCKED_EXTERNAL:
        return EXIT_BLOCKED_EXTERNAL
    return EXIT_FAILED_FINAL
