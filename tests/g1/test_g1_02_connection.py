from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from runtime.carla_connection import (  # noqa: E402
    BLOCKED_EXTERNAL,
    CLIENT_VERSION_MISMATCH,
    ConnectionResolver,
    EXIT_BLOCKED_EXTERNAL,
    EXIT_FAILED_FINAL,
    EXIT_READY,
    EXIT_RETRYABLE_FAILURE,
    HOST_RESOLUTION_FAILED,
    MAP_QUERY_FAILED,
    NEEDS_USER_ACTION,
    RETRYABLE_FAILURE,
    RPC_HANDSHAKE_FAILED,
    READY,
    SERVER_NOT_RUNNING,
    SERVER_VERSION_MISMATCH,
    STARTUP_TIMEOUT,
    TCP_UNREACHABLE,
    WORLD_SETTINGS_QUERY_FAILED,
    _parse_default_routes,
    _rank_host_candidates,
    exit_code,
    windows_path_to_wsl,
    wsl_path_to_windows,
)


class Settings:
    synchronous_mode = False
    fixed_delta_seconds = None
    substepping = True
    max_substep_delta_time = 0.01
    max_substeps = 10


class World:
    def __init__(self, map_name: str = "Carla/Maps/Town10HD_Opt") -> None:
        self.map_name = map_name
        self.settings = Settings()

    def get_map(self):
        if self.map_name == "ERROR":
            raise RuntimeError("map query failed")
        return type("Map", (), {"name": self.map_name})()

    def get_settings(self):
        if self.map_name == "SETTINGS_ERROR":
            raise RuntimeError("settings query failed")
        return self.settings


class Client:
    def __init__(self, client_version: str = "0.9.16", server_version: str = "0.9.16", world: World | None = None, error: Exception | None = None) -> None:
        self.client_version, self.server_version, self.world, self.error = client_version, server_version, world or World(), error
        self.timeout = None

    def set_timeout(self, value: float) -> None:
        self.timeout = value

    def get_client_version(self) -> str:
        if self.error:
            raise self.error
        return self.client_version

    def get_server_version(self) -> str:
        return self.server_version

    def get_world(self) -> World:
        return self.world


class G102ConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT

    def resolver(self, **kwargs) -> ConnectionResolver:
        return ConnectionResolver(self.root, process_query=lambda: "RUNNING", **kwargs)

    def test_host_and_port_precedence(self) -> None:
        with patch.dict(os.environ, {"CARLA_HOST": "env-host", "CARLA_PORT": "2222"}, clear=False):
            resolver = self.resolver(host_discoverer=lambda: "route-host")
            self.assertEqual(resolver.resolve_host().host, "env-host")
            self.assertEqual(resolver.resolve_host("explicit-host").host, "explicit-host")
            self.assertEqual(resolver.resolve_port(), 2222)
            self.assertEqual(resolver.resolve_port(3333), 3333)
        resolver = self.resolver(host_discoverer=lambda: "route-host")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolver.resolve_host().host, "route-host")
            self.assertEqual(resolver.resolve_host().source, "wsl_default_gateway")
            self.assertEqual(resolver.resolve_port(), 2000)

    def test_old_host_failure_re_discovers_once_only_when_address_changes(self) -> None:
        discovered = iter(["old-host", "new-host"])
        probes: list[str] = []

        def probe(host: str, port: int, timeout: float):
            probes.append(host)
            return (host == "new-host", "old host refused" if host == "old-host" else None)

        resolver = self.resolver(host_discoverer=lambda: next(discovered), tcp_probe=probe, client_factory=lambda h, p: Client())
        with patch.dict(os.environ, {}, clear=True):
            report = resolver.preflight()
        self.assertEqual(probes, ["old-host", "new-host"])
        self.assertEqual(report.previous_host, "old-host")
        self.assertEqual(report.host, "new-host")
        self.assertEqual(report.retry_count, 1)
        self.assertEqual(report.status, READY)

        calls = []
        resolver = self.resolver(host_discoverer=lambda: "same-host", tcp_probe=lambda h, p, t: (False, "no"))
        with patch.dict(os.environ, {}, clear=True):
            resolver.preflight()
        self.assertEqual(calls, [])  # same address is not probed a second time

    def test_tcp_server_and_rpc_error_classes(self) -> None:
        resolver = self.resolver(tcp_probe=lambda h, p, t: (False, "connection refused"))
        report = resolver.preflight(host="host")
        self.assertEqual(report.error_code, TCP_UNREACHABLE)
        self.assertEqual(exit_code(report), EXIT_RETRYABLE_FAILURE)

        resolver = ConnectionResolver(self.root, process_query=lambda: "NOT_RUNNING", tcp_probe=lambda h, p, t: (False, "refused"), host_discoverer=lambda: "host")
        report = resolver.preflight(host="host")
        self.assertEqual(report.error_code, SERVER_NOT_RUNNING)

        resolver = self.resolver(tcp_probe=lambda h, p, t: (True, None), client_factory=lambda h, p: Client(error=RuntimeError("RPC broke")))
        report = resolver.preflight(host="host")
        self.assertEqual(report.error_code, RPC_HANDSHAKE_FAILED)
        self.assertEqual(report.status, RETRYABLE_FAILURE)

    def test_version_map_and_settings_errors_are_not_ready(self) -> None:
        for factory, code in (
            (lambda h, p: Client(client_version="0.9.15"), CLIENT_VERSION_MISMATCH),
            (lambda h, p: Client(server_version="0.9.15"), SERVER_VERSION_MISMATCH),
            (lambda h, p: Client(world=World("ERROR")), MAP_QUERY_FAILED),
            (lambda h, p: Client(world=World("SETTINGS_ERROR")), WORLD_SETTINGS_QUERY_FAILED),
        ):
            report = self.resolver(tcp_probe=lambda h, p, t: (True, None), client_factory=factory).preflight(host="host")
            self.assertEqual(report.error_code, code)
            self.assertNotEqual(report.status, READY)
            self.assertEqual(exit_code(report), EXIT_FAILED_FINAL)

    def test_host_resolution_failure_is_structured(self) -> None:
        resolver = self.resolver(host_discoverer=lambda: (_ for _ in ()).throw(RuntimeError("no route")))
        with patch.dict(os.environ, {}, clear=True):
            report = resolver.preflight()
        self.assertEqual(report.error_code, HOST_RESOLUTION_FAILED)
        self.assertEqual(report.status, RETRYABLE_FAILURE)

    def test_tick_owner_is_observed_without_acquiring_lease(self) -> None:
        report = self.resolver(tcp_probe=lambda h, p, t: (True, None), client_factory=lambda h, p: Client()).status(host="host")
        self.assertIn("configured=", report.tick_owner or "")

    def test_ensure_reuses_ready_instance_without_starting_another(self) -> None:
        started = []
        resolver = self.resolver(tcp_probe=lambda h, p, t: (True, None), client_factory=lambda h, p: Client(), start_process=lambda spec: started.append(spec) or True)
        report = resolver.ensure(host="host")
        self.assertEqual(report.status, READY)
        self.assertEqual(report.recovery_action, "already_running")
        self.assertEqual(started, [])
        self.assertEqual(exit_code(report), EXIT_READY)

    def test_ensure_unknown_process_requires_user_and_never_starts(self) -> None:
        started = []
        resolver = ConnectionResolver(self.root, process_query=lambda: "UNKNOWN", tcp_probe=lambda h, p, t: (False, "refused"), start_process=lambda spec: started.append(spec) or True)
        report = resolver.ensure(host="host")
        self.assertEqual(report.status, BLOCKED_EXTERNAL)
        self.assertEqual(report.error_code, NEEDS_USER_ACTION)
        self.assertTrue(report.needs_user_action)
        self.assertEqual(started, [])
        self.assertEqual(exit_code(report), EXIT_BLOCKED_EXTERNAL)

    def test_ensure_known_start_waits_bounded_and_returns_timeout(self) -> None:
        started = []
        resolver = ConnectionResolver(
            self.root,
            process_query=lambda: "NOT_RUNNING",
            tcp_probe=lambda h, p, t: (False, "refused"),
            start_process=lambda spec: started.append(spec) or True,
            sleeper=lambda _: None,
        )
        # No actual CARLA startup is attempted in the unit test; the verified
        # startup spec is present, and the bounded wait is exercised.
        report = resolver.ensure(host="host", startup_timeout_seconds=0.0, poll_interval_seconds=0.0)
        self.assertEqual(started[0]["expected_version"], "0.9.16")
        self.assertEqual(report.error_code, STARTUP_TIMEOUT)
        self.assertEqual(report.status, RETRYABLE_FAILURE)
        self.assertEqual(exit_code(report), EXIT_RETRYABLE_FAILURE)

    def test_proxy_like_gateway_is_ranked_after_real_hosts(self) -> None:
        routes = _parse_default_routes(
            "default via 198.18.0.2 dev eth0 metric 1\n"
            "default via 172.30.80.1 dev eth1 metric 20\n"
        )
        ranked = _rank_host_candidates(routes)
        self.assertEqual([host for host, _ in ranked], ["172.30.80.1", "198.18.0.2"])
        self.assertEqual(ranked[0][1], "wsl_default_gateway")
        self.assertEqual(ranked[1][1], "wsl_default_gateway_proxy")

    def test_windows_wsl_path_round_trip(self) -> None:
        self.assertEqual(windows_path_to_wsl(r"E:\CARLA_0.9.16\CarlaUE4.exe"), Path("/mnt/e/CARLA_0.9.16/CarlaUE4.exe"))
        self.assertEqual(wsl_path_to_windows("/mnt/e/CARLA_0.9.16/CarlaUE4.exe"), r"E:\CARLA_0.9.16\CarlaUE4.exe")

    def test_start_paths_prefer_wsl_mount_for_existence(self) -> None:
        resolver = self.resolver()
        windows, wsl = resolver._resolve_start_paths(
            {
                "windows_executable": r"E:\CARLA_0.9.16\CarlaUE4.exe",
                "wsl_path": "/mnt/e/CARLA_0.9.16/CarlaUE4.exe",
            }
        )
        self.assertEqual(windows, r"E:\CARLA_0.9.16\CarlaUE4.exe")
        self.assertEqual(wsl, Path("/mnt/e/CARLA_0.9.16/CarlaUE4.exe"))


if __name__ == "__main__":
    unittest.main()
