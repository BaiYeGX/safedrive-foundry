"""G0 environment diagnostics and validation-report helpers."""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .sync_contract import (
    BLOCKED,
    FAIL,
    PASS,
    WARN,
    ContractViolation,
    SyncConfig,
    build_deterministic_trace,
    compare_traces,
    load_sync_config,
    run_contract_fault_injection,
    run_deterministic_smoke,
    validate_carla_settings,
    validate_tick_masters,
)


@dataclass(frozen=True)
class DoctorCheck:
    check_id: str
    status: str
    message: str
    code: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check_id,
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "details": _json_safe(dict(self.details)),
        }


@dataclass
class DoctorReport:
    root: Path
    checks: list[DoctorCheck]
    generated_at_utc: str
    invocation: Sequence[str] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        statuses = {check.status for check in self.checks}
        if FAIL in statuses:
            return FAIL
        if BLOCKED in statuses:
            return BLOCKED
        if WARN in statuses:
            return WARN
        return PASS

    def to_dict(self) -> dict[str, Any]:
        counts = {status: sum(check.status == status for check in self.checks) for status in (PASS, WARN, FAIL, BLOCKED)}
        return {
            "schema": "safedrive.g0.doctor.v1",
            "generated_at_utc": self.generated_at_utc,
            "project_root": str(self.root),
            "summary": {"status": self.status, "counts": counts},
            "invocation": list(self.invocation),
            "checks": [check.to_dict() for check in self.checks],
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _clean_output(value: str) -> str:
    return value.replace("\x00", "").replace("\r", "").strip()


def _run_command(command: Sequence[str], timeout_seconds: float = 8.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc), "error": "not_found"}
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": _clean_output(exc.stdout or ""),
            "stderr": _clean_output(exc.stderr or ""),
            "error": "timeout",
        }
    return {
        "returncode": completed.returncode,
        "stdout": _clean_output(completed.stdout),
        "stderr": _clean_output(completed.stderr),
    }


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError as exc:  # pragma: no cover - only Python < 3.11
        raise ContractViolation("toml_parser_missing", "Python 3.11+ is required") from exc
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _preferred_version(lock: Mapping[str, Any], section: str, key: str, default: str) -> str:
    value = lock.get(section, {})
    if isinstance(value, Mapping):
        return str(value.get(key, default))
    return default


def _version_tuple(value: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _probe_tcp(host: str, port: int, timeout_seconds: float) -> tuple[bool, str | None]:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True, None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def classify_carla_probe(
    *,
    socket_open: bool,
    api_available: bool,
    handshake_ok: bool,
    client_version: str | None = None,
    server_version: str | None = None,
    expected_version: str = "0.9.16",
    error: str | None = None,
) -> DoctorCheck:
    """Map a CARLA probe to a stable diagnostic code.

    The arguments are injectable so validation can prove negative paths without
    stopping a real server or pretending that a failed live probe passed.
    """

    if not socket_open:
        return DoctorCheck(
            "carla.rpc",
            FAIL,
            "CARLA RPC endpoint is unreachable; server may be stopped or the endpoint is wrong",
            "carla_not_started",
            {"error": error},
        )
    if not api_available:
        return DoctorCheck(
            "carla.handshake",
            FAIL,
            "RPC port is open but the CARLA Python API is unavailable; possible port conflict",
            "carla_client_unavailable_or_port_conflict",
            {"error": error},
        )
    if not handshake_ok:
        return DoctorCheck(
            "carla.handshake",
            FAIL,
            "RPC port is open but the CARLA handshake failed; possible port conflict or protocol error",
            "port_conflict_or_non_carla_service",
            {"error": error},
        )
    if (
        client_version != expected_version
        or server_version != expected_version
        or client_version != server_version
    ):
        return DoctorCheck(
            "carla.version",
            FAIL,
            "CARLA client/server version does not match the frozen expected version",
            "carla_version_mismatch",
            {
                "expected": expected_version,
                "client": client_version,
                "server": server_version,
            },
        )
    return DoctorCheck(
        "carla.handshake",
        PASS,
        "CARLA RPC, client/server version and world handshake passed",
        "carla_ready",
        {"client": client_version, "server": server_version},
    )


def classify_gpu_probe(visible: bool, *, details: Mapping[str, Any] | None = None) -> DoctorCheck:
    if visible:
        return DoctorCheck("gpu.visible", PASS, "NVIDIA GPU is visible to the diagnostic process", "gpu_visible", details or {})
    return DoctorCheck(
        "gpu.visible",
        FAIL,
        "NVIDIA GPU is not visible to the diagnostic process",
        "gpu_not_visible",
        details or {},
    )


def classify_disk_probe(free_gib: float, minimum_free_gib: float) -> DoctorCheck:
    if free_gib < minimum_free_gib:
        return DoctorCheck(
            "disk.free",
            FAIL,
            "free disk space is below the configured G0 safety floor",
            "low_disk_space",
            {"free_gib": round(free_gib, 3), "minimum_free_gib": minimum_free_gib},
        )
    return DoctorCheck(
        "disk.free",
        PASS,
        "free disk space is above the configured G0 safety floor",
        "disk_space_ok",
        {"free_gib": round(free_gib, 3), "minimum_free_gib": minimum_free_gib},
    )


def _load_carla(root: Path) -> tuple[Any | None, str | None, str | None]:
    candidates: list[Path | None] = [None]
    candidates.extend(
        [
            root / "tools" / "python-3.12.3-embed-amd64" / "Lib" / "site-packages",
            Path(sys.executable).parent / "Lib" / "site-packages",
        ]
    )
    errors: list[str] = []
    for candidate in candidates:
        if candidate is not None:
            if not candidate.exists():
                continue
            sys.path.insert(0, str(candidate))
        try:
            module = importlib.import_module("carla")
            return module, str(candidate) if candidate is not None else "sys.path", None
        except Exception as exc:  # CARLA may fail while loading its platform DLLs.
            errors.append(f"{candidate or 'sys.path'}: {type(exc).__name__}: {exc}")
    return None, None, "; ".join(errors)


def _check_carla(
    *,
    root: Path,
    host: str,
    port: int,
    expected_version: str,
    timeout_seconds: float,
) -> tuple[DoctorCheck, DoctorCheck, dict[str, Any]]:
    socket_open, socket_error = _probe_tcp(host, port, timeout_seconds)
    port_check = DoctorCheck(
        "carla.port",
        PASS if socket_open else FAIL,
        f"CARLA RPC TCP endpoint {host}:{port} is " + ("reachable" if socket_open else "unreachable"),
        "port_open" if socket_open else "carla_not_started",
        {"host": host, "port": port, "error": socket_error},
    )
    if not socket_open:
        return (
            port_check,
            classify_carla_probe(
                socket_open=False,
                api_available=False,
                handshake_ok=False,
                expected_version=expected_version,
                error=socket_error,
            ),
            {"socket_open": False, "socket_error": socket_error},
        )

    carla_module, source, import_error = _load_carla(root)
    if carla_module is None:
        handshake = classify_carla_probe(
            socket_open=True,
            api_available=False,
            handshake_ok=False,
            expected_version=expected_version,
            error=import_error,
        )
        return port_check, handshake, {"socket_open": True, "api_source": source, "import_error": import_error}

    client_version: str | None = None
    server_version: str | None = None
    handshake_error: str | None = None
    handshake_ok = False
    try:
        client = carla_module.Client(host, port)
        client.set_timeout(timeout_seconds)
        client_version = str(client.get_client_version())
        server_version = str(client.get_server_version())
        client.get_world()
        handshake_ok = True
    except Exception as exc:  # pragma: no cover - depends on external endpoint
        handshake_error = f"{type(exc).__name__}: {exc}"
    handshake = classify_carla_probe(
        socket_open=True,
        api_available=True,
        handshake_ok=handshake_ok,
        client_version=client_version,
        server_version=server_version,
        expected_version=expected_version,
        error=handshake_error,
    )
    details = {
        "socket_open": True,
        "api_source": source,
        "client_version": client_version,
        "server_version": server_version,
        "handshake_error": handshake_error,
    }
    return port_check, handshake, details


def _parse_wsl_distros(output: str) -> list[str]:
    distros: list[str] = []
    for raw_line in _clean_output(output).splitlines():
        line = raw_line.strip().lstrip("*").strip()
        if not line or line.lower().startswith(("windows subsystem", "name", "usage", "install")):
            continue
        if line not in distros:
            distros.append(line)
    return distros


def _running_in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        version = ""
    if "microsoft" in version or "wsl" in version:
        return True
    return Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()


def _check_wsl() -> tuple[DoctorCheck, str | None]:
    # Already inside WSL: do not require wsl.exe or nested distro lookup.
    if _running_in_wsl():
        distro = os.environ.get("WSL_DISTRO_NAME") or platform.uname().node or "wsl-local"
        return (
            DoctorCheck(
                "wsl.available",
                PASS,
                "diagnostic process is already running inside WSL",
                "wsl_in_guest",
                {"distro": distro, "mode": "in_guest"},
            ),
            distro,
        )
    version = _run_command(["wsl.exe", "--version"])
    if version.get("returncode") == 127:
        return DoctorCheck("wsl.available", BLOCKED, "wsl.exe is not available; WSL installation needs external action", "wsl_not_available", version), None
    listing = _run_command(["wsl.exe", "-l", "-q"])
    distros = _parse_wsl_distros(listing.get("stdout", ""))
    if listing.get("returncode") != 0 or not distros:
        return (
            DoctorCheck(
                "wsl.available",
                BLOCKED,
                "WSL is installed but no usable distribution is registered for this task",
                "wsl_distribution_unavailable",
                {"version": version, "listing": listing, "distros": distros},
            ),
            None,
        )
    return (
        DoctorCheck(
            "wsl.available",
            PASS,
            "WSL has at least one registered distribution",
            "wsl_ready",
            {"version": version, "distros": distros},
        ),
        distros[0],
    )


def _check_ros(distro: str | None, wsl_check: DoctorCheck) -> tuple[DoctorCheck, dict[str, Any]]:
    if wsl_check.status != PASS or not distro:
        return (
            DoctorCheck(
                "ros2.available",
                BLOCKED,
                "ROS 2 check is blocked because no usable WSL distribution is available",
                "ros2_blocked_by_wsl",
                {"wsl_status": wsl_check.status},
            ),
            {},
        )
    if _running_in_wsl():
        result = _run_command(
            [
                "bash",
                "-lc",
                "source /opt/ros/jazzy/setup.bash && command -v ros2 && ros2 --help",
            ]
        )
        if result.get("returncode") == 0 and "ros2" in result.get("stdout", ""):
            return DoctorCheck("ros2.available", PASS, "ROS 2 Jazzy command is available in current WSL", "ros2_ready_in_guest", result), result
        return DoctorCheck("ros2.available", FAIL, "ROS 2 Jazzy command is not available in current WSL", "ros2_unavailable", result), result
    result = _run_command(
        [
            "wsl.exe",
            "-d",
            distro,
            "--",
            "bash",
            "-lc",
            # ROS 2 Jazzy's CLI does not expose a ``--version`` option; use
            # the portable help command after resolving the executable.
            "source /opt/ros/jazzy/setup.bash && command -v ros2 && ros2 --help",
        ]
    )
    if result.get("returncode") == 0 and "ros2" in result.get("stdout", ""):
        return DoctorCheck("ros2.available", PASS, "ROS 2 Jazzy command is available in WSL", "ros2_ready", result), result
    return DoctorCheck("ros2.available", FAIL, "ROS 2 Jazzy command is not available in WSL", "ros2_unavailable", result), result


def _check_clock(distro: str | None, ros_check: DoctorCheck, clock_topic: str) -> DoctorCheck:
    if ros_check.status != PASS or not distro:
        return DoctorCheck(
            "ros.clock",
            BLOCKED,
            "live /clock observation is blocked because ROS 2 is unavailable",
            "clock_observation_blocked",
            {"ros_status": ros_check.status},
        )
    if _running_in_wsl():
        result = _run_command(
            [
                "bash",
                "-lc",
                "source /opt/ros/jazzy/setup.bash && ros2 topic list",
            ]
        )
    else:
        result = _run_command(
            [
                "wsl.exe",
                "-d",
                distro,
                "--",
                "bash",
                "-lc",
                "source /opt/ros/jazzy/setup.bash && ros2 topic list",
            ]
        )
    if result.get("returncode") == 0 and clock_topic in result.get("stdout", "").splitlines():
        return DoctorCheck("ros.clock", PASS, f"ROS clock topic {clock_topic} is discoverable", "clock_topic_visible", result)
    return DoctorCheck(
        "ros.clock",
        WARN,
        f"ROS 2 is available but {clock_topic} is not currently observable; start the sync driver",
        "clock_topic_not_observed",
        result,
    )


def _check_ancillary_ports(host: str, ports: Mapping[str, int], timeout_seconds: float) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for name, port in ports.items():
        if name == "rpc":
            continue
        open_, error = _probe_tcp(host, int(port), timeout_seconds)
        checks.append(
            DoctorCheck(
                f"carla.port.{name}",
                PASS if open_ else WARN,
                f"CARLA {name} endpoint {host}:{port} is " + ("reachable" if open_ else "not reachable"),
                "port_open" if open_ else "port_not_observed",
                {"host": host, "port": port, "error": error},
            )
        )
    return checks


def run_doctor(
    root: Path,
    *,
    carla_host: str | None = None,
    carla_port: int | None = None,
    expected_version: str | None = None,
    min_free_gib: float = 20.0,
    timeout_seconds: float = 3.0,
    invocation: Sequence[str] = (),
) -> DoctorReport:
    root = root.resolve()
    checks: list[DoctorCheck] = []
    config_path = root / "safedrive_foundry" / "config" / "carla_ros.toml"
    versions_path = root / "versions.lock"
    config_data: dict[str, Any] = {}
    lock_data: dict[str, Any] = {}
    sync_config = SyncConfig()

    if root.exists() and (root / "safedrive_foundry").is_dir():
        checks.append(DoctorCheck("paths.project", PASS, "project root and G0 skeleton are present", "project_path_ok", {"root": root}))
    else:
        checks.append(DoctorCheck("paths.project", FAIL, "project root or G0 skeleton is missing", "project_path_missing", {"root": root}))

    try:
        config_data = _read_toml(config_path)
        sync_config = SyncConfig.from_mapping(config_data)
        errors = sync_config.validate()
        if errors:
            checks.append(DoctorCheck("sync.config", FAIL, "synchronization configuration is invalid", "invalid_sync_config", {"errors": errors}))
        else:
            checks.append(DoctorCheck("sync.config", PASS, "fixed-step and frame contract configuration is valid", "sync_config_ok", sync_config.to_dict()))
    except (OSError, ValueError, ContractViolation) as exc:
        checks.append(DoctorCheck("sync.config", FAIL, "synchronization configuration cannot be loaded", "sync_config_unreadable", {"error": str(exc), "path": config_path}))

    try:
        lock_data = _read_toml(versions_path)
        expected_lock_version = _preferred_version(lock_data, "preferred", "carla", "0.9.16")
        checks.append(
            DoctorCheck(
                "versions.lock",
                PASS,
                "authoritative version lock is readable",
                "versions_lock_ok",
                {"carla": expected_lock_version, "path": versions_path},
            )
        )
    except (OSError, ValueError, ContractViolation) as exc:
        expected_lock_version = "0.9.16"
        checks.append(DoctorCheck("versions.lock", FAIL, "authoritative version lock cannot be read", "versions_lock_unreadable", {"error": str(exc), "path": versions_path}))

    expected_version = expected_version or os.environ.get("CARLA_EXPECTED_VERSION") or expected_lock_version
    carla_table = config_data.get("carla", {}) if isinstance(config_data.get("carla", {}), Mapping) else {}
    sync_table = config_data.get("sync", {}) if isinstance(config_data.get("sync", {}), Mapping) else {}
    carla_host = carla_host or os.environ.get("CARLA_DOCTOR_HOST") or os.environ.get("CARLA_HOST") or str(carla_table.get("host", "127.0.0.1"))
    carla_port = int(carla_port or os.environ.get("CARLA_PORT") or carla_table.get("port", 2000))
    ancillary_ports = {
        "rpc": carla_port,
        "streaming": int(carla_table.get("streaming_port", 2001)),
        "traffic_manager": int(carla_table.get("traffic_manager_port", 2002)),
    }

    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    expected_python = _preferred_version(lock_data, "preferred", "python", "3.12")
    expected_python_tuple = _version_tuple(expected_python)
    current_tuple = (sys.version_info.major, sys.version_info.minor)
    if expected_python_tuple is None or current_tuple != expected_python_tuple:
        checks.append(DoctorCheck("python.version", WARN, "Python major/minor differs from the preferred frozen interpreter", "python_version_mismatch", {"current": current, "expected": expected_python}))
    else:
        checks.append(DoctorCheck("python.version", PASS, "Python major/minor matches the preferred frozen interpreter", "python_version_ok", {"current": current, "expected": expected_python}))

    gpu = _run_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], timeout_seconds)
    gpu_visible = gpu.get("returncode") == 0 and bool(gpu.get("stdout"))
    checks.append(classify_gpu_probe(gpu_visible, details=gpu))

    wsl_check, distro = _check_wsl()
    checks.append(wsl_check)
    ros_check, ros_details = _check_ros(distro, wsl_check)
    checks.append(ros_check)

    default_carla = "/mnt/e/CARLA_0.9.16" if _running_in_wsl() else "E:/CARLA_0.9.16"
    carla_root = Path(os.environ.get("CARLA_ROOT", default_carla))
    if not carla_root.exists() and _running_in_wsl():
        # Accept Windows-style CARLA_ROOT by mapping to /mnt/<drive>/...
        env_root = os.environ.get("CARLA_ROOT", "E:/CARLA_0.9.16")
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", env_root.replace("\\", "/"))
        if match:
            mapped = Path("/mnt") / match.group(1).lower() / match.group(2)
            if mapped.exists():
                carla_root = mapped
    if carla_root.exists():
        checks.append(DoctorCheck("paths.carla", PASS, "configured CARLA installation path exists", "carla_path_ok", {"path": str(carla_root)}))
    else:
        checks.append(DoctorCheck("paths.carla", FAIL, "configured CARLA installation path does not exist", "carla_installation_missing", {"path": str(carla_root)}))

    try:
        usage = shutil.disk_usage(root)
        checks.append(classify_disk_probe(usage.free / (1024**3), min_free_gib))
    except OSError as exc:
        checks.append(DoctorCheck("disk.free", FAIL, "disk free-space check failed", "disk_probe_failed", {"error": str(exc)}))

    port_check, handshake_check, carla_details = _check_carla(
        root=root,
        host=carla_host,
        port=carla_port,
        expected_version=expected_version,
        timeout_seconds=timeout_seconds,
    )
    checks.extend((port_check, handshake_check))
    checks.extend(_check_ancillary_ports(carla_host, ancillary_ports, timeout_seconds))
    checks.append(_check_clock(distro, ros_check, str(sync_table.get("clock_topic", sync_config.clock_topic))))

    # This is a static ownership check; the live node also claims a process-local lease.
    checks.append(
        DoctorCheck(
            "sync.tick_master",
            PASS if not validate_tick_masters([sync_config.tick_master], sync_config.tick_master) else FAIL,
            "configuration declares exactly one tick master" if not validate_tick_masters([sync_config.tick_master], sync_config.tick_master) else "tick master ownership is ambiguous",
            "single_tick_master" if not validate_tick_masters([sync_config.tick_master], sync_config.tick_master) else "multiple_tick_masters",
            {"tick_master": sync_config.tick_master},
        )
    )

    return DoctorReport(
        root=root,
        checks=checks,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        invocation=invocation,
    )


def write_doctor_reports(report: DoctorReport, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SafeDrive G0-05 environment doctor",
        "",
        f"- Overall status: **{report.status}**",
        f"- Generated (UTC): `{report.generated_at_utc}`",
        f"- Project root: `{report.root}`",
        "",
        "| Check | Status | Code | Message |",
        "|---|---|---|---|",
    ]
    for check in report.checks:
        message = check.message.replace("|", "\\|")
        lines.append(f"| `{check.check_id}` | **{check.status}** | `{check.code or ''}` | {message} |")
    lines.extend(["", "## Details", ""])
    for check in report.checks:
        lines.extend(
            [
                f"### `{check.check_id}`",
                "",
                "```json",
                json.dumps(_json_safe(dict(check.details)), ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def print_doctor_report(report: DoctorReport) -> None:
    print(f"SafeDrive G0 doctor: {report.status}")
    for check in report.checks:
        suffix = f" [{check.code}]" if check.code else ""
        print(f"{check.status:8} {check.check_id}: {check.message}{suffix}")


def _validation_case(check_id: str, condition: bool, message: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": PASS if condition else FAIL,
        "message": message if condition else f"FAILED: {message}",
        "details": _json_safe(dict(details or {})),
    }


def run_g0_validation(root: Path, config: SyncConfig) -> dict[str, Any]:
    """Run offline repeatability, fault-injection and checkpoint validation."""

    checks: list[dict[str, Any]] = []
    config_errors = config.validate()
    checks.append(_validation_case("sync.config", not config_errors, "fixed-step configuration is legal", {"errors": config_errors, "config": config.to_dict()}))
    owner_errors = validate_tick_masters([config.tick_master], config.tick_master)
    checks.append(_validation_case("sync.tick_master", not owner_errors, "one configured tick master is declared", {"errors": owner_errors, "owner": config.tick_master}))

    first = build_deterministic_trace(seed=1234, steps=12, config=config)
    second = build_deterministic_trace(seed=1234, steps=12, config=config)
    repeat = compare_traces(first, second, tolerance_seconds=config.timestamp_tolerance_seconds)
    checks.append(_validation_case("determinism.same_seed", repeat["pass"], "same-seed traces have identical frames, event order and state hashes", repeat))

    fault_results = run_contract_fault_injection(config)
    expected_codes = {
        "duplicate_tick": "duplicate_tick",
        "missing_frame": "missing_frame",
        "stale_message": "stale_message",
        "multiple_tick_masters": "multiple_tick_masters",
    }
    for result in fault_results:
        observed = result["observed"]
        expected_code = expected_codes[result["id"]]
        checks.append(_validation_case(f"fault.{result['id']}", observed.get("code") == expected_code, f"injection detected as {expected_code}", observed))

    environment_faults = [
        (
            "environment.carla_not_started",
            classify_carla_probe(socket_open=False, api_available=False, handshake_ok=False),
            "carla_not_started",
        ),
        (
            "environment.port_conflict",
            classify_carla_probe(socket_open=True, api_available=True, handshake_ok=False, error="test listener"),
            "port_conflict_or_non_carla_service",
        ),
        (
            "environment.version_mismatch",
            classify_carla_probe(socket_open=True, api_available=True, handshake_ok=True, client_version="0.9.15", server_version="0.9.16"),
            "carla_version_mismatch",
        ),
        ("environment.gpu_not_visible", classify_gpu_probe(False), "gpu_not_visible"),
        ("environment.disk_low", classify_disk_probe(1.0, 20.0), "low_disk_space"),
    ]
    for check_id, observed, expected_code in environment_faults:
        checks.append(_validation_case(check_id, observed.code == expected_code, f"environment fault detected as {expected_code}", observed.to_dict()))

    settings = SimpleNamespace(
        synchronous_mode=True,
        fixed_delta_seconds=config.fixed_delta_seconds,
        substepping=config.substepping,
        max_substep_delta_time=config.max_substep_delta_time,
        max_substeps=config.max_substeps,
    )
    settings_errors = validate_carla_settings(settings, config)
    checks.append(_validation_case("carla.settings", not settings_errors, "CARLA fixed-step settings satisfy the contract", {"errors": settings_errors}))

    with tempfile.TemporaryDirectory(prefix="safedrive-g0-05-") as temporary:
        run_dir = Path(temporary)
        checkpoint = run_dir / "checkpoint.json"
        resumed_trace_path = run_dir / "resumed.json"
        interrupted_status, _ = run_deterministic_smoke(
            seed=77,
            steps=10,
            config=config,
            checkpoint_path=checkpoint,
            trace_path=resumed_trace_path,
            interrupt_after=3,
        )
        completed_status, resumed_trace = run_deterministic_smoke(
            seed=77,
            steps=10,
            config=config,
            checkpoint_path=checkpoint,
            trace_path=resumed_trace_path,
            resume=True,
        )
        expected_trace = build_deterministic_trace(seed=77, steps=10, config=config)
        resume_comparison = compare_traces(expected_trace, resumed_trace or {}, tolerance_seconds=config.timestamp_tolerance_seconds)
        checks.append(
            _validation_case(
                "recovery.checkpoint_resume",
                interrupted_status == "INTERRUPTED" and completed_status == "COMPLETED" and resume_comparison["pass"],
                "interrupted smoke resumes from an atomic checkpoint and matches a clean run",
                {"interrupted_status": interrupted_status, "completed_status": completed_status, "comparison": resume_comparison},
            )
        )

    status = PASS if all(item["status"] == PASS for item in checks) else FAIL
    return {
        "schema": "safedrive.g0.validation.v1",
        "status": status,
        "project_root": str(root.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "summary": {
            "total": len(checks),
            "passed": sum(item["status"] == PASS for item in checks),
            "failed": sum(item["status"] == FAIL for item in checks),
        },
    }


def write_validation_reports(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SafeDrive G0-05 offline validation",
        "",
        f"- Overall status: **{report['status']}**",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        "",
        "| Check | Status | Message |",
        "|---|---|---|",
    ]
    for item in report["checks"]:
        lines.append(f"| `{item['id']}` | **{item['status']}** | {item['message']} |")
    lines.extend(["", "## Evidence", "", "```json", json.dumps(_json_safe(report["summary"]), indent=2), "```", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "classify_carla_probe",
    "classify_disk_probe",
    "classify_gpu_probe",
    "print_doctor_report",
    "run_doctor",
    "run_g0_validation",
    "write_doctor_reports",
    "write_validation_reports",
]
