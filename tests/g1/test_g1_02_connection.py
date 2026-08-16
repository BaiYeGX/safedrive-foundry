from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from runtime.carla_connection import (  # noqa: E402
    BLOCKED_EXTERNAL,
    CARLA_PROCESS_EXITED_EARLY,
    CARLA_SHADER_PIPELINE_FATAL,
    CLIENT_VERSION_MISMATCH,
    ConnectionResolver,
    EXIT_BLOCKED_EXTERNAL,
    EXIT_FAILED_FINAL,
    EXIT_READY,
    EXIT_RETRYABLE_FAILURE,
    FAILED_FINAL,
    HOST_RESOLUTION_FAILED,
    LAUNCH_PARAM_MISMATCH,
    LaunchResult,
    MAP_MISMATCH,
    MAP_OR_RHI_CONFIG_MISMATCH,
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
    WindowsCarlaProcess,
    _parse_default_routes,
    _rank_host_candidates,
    build_carla_launch_arguments,
    build_powershell_start_command,
    classify_startup_crash,
    count_offscreen_flags,
    count_rhi_flags,
    exit_code,
    find_recent_crash_context,
    launch_params_match,
    normalize_rhi,
    parse_offscreen_from_command_line,
    parse_rhi_from_command_line,
    rewrite_arguments_map,
    rewrite_arguments_rhi_offscreen,
    windows_path_to_wsl,
    wsl_path_to_windows,
)
from runtime.carla_engine_config import (  # noqa: E402
    pin_default_engine_config,
    read_default_engine_config,
    verify_default_engine_matches,
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
        patcher = patch("runtime.carla_connection.query_windows_carla_command_line", return_value=r"E:\CARLA_0.9.16\CarlaUE4.exe -dx12")
        self.addCleanup(patcher.stop)
        patcher.start()

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
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, default_map="Town03", rhi="dx12")
            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town03", rhi="dx12")
            started = []
            resolver = ConnectionResolver(
                root,
                process_query=lambda: "NOT_RUNNING",
                tcp_probe=lambda h, p, t: (False, "refused"),
                start_process=lambda spec: started.append(spec) or True,
                sleeper=lambda _: None,
            )
            # Keep all config reads/writes inside the temporary fixture.  The
            # bounded-wait test must never pin the user's real DefaultEngine.ini.
            with (
                patch("runtime.carla_connection.windows_path_exists", return_value=True),
                patch("runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini),
            ):
                report = resolver.ensure(
                    host="host",
                    startup_timeout_seconds=0.0,
                    poll_interval_seconds=0.0,
                )
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

    def test_launch_arguments_rhi_exclusive_and_offscreen_optional(self) -> None:
        dx11 = build_carla_launch_arguments(rhi="dx11", render_offscreen=False)
        dx12 = build_carla_launch_arguments(rhi="dx12", render_offscreen=True)
        self.assertEqual(count_rhi_flags(dx11), 1)
        self.assertEqual(count_rhi_flags(dx12), 1)
        self.assertIn("-dx11", dx11)
        self.assertNotIn("-dx12", dx11)
        self.assertIn("-dx12", dx12)
        self.assertNotIn("-dx11", dx12)
        self.assertEqual(count_offscreen_flags(dx11), 0)
        self.assertEqual(count_offscreen_flags(dx12), 1)
        rewritten = rewrite_arguments_rhi_offscreen(
            dx11 + " -dx12", rhi="dx11", render_offscreen=True
        )
        self.assertEqual(count_rhi_flags(rewritten), 1)
        self.assertEqual(parse_rhi_from_command_line(rewritten), "dx11")
        self.assertTrue(parse_offscreen_from_command_line(rewritten))

    def _write_start_toml(
        self,
        root: Path,
        *,
        launch_mode: str = "default_engine",
        default_map: str = "Town03",
        rhi: str = "dx12",
        arguments: str | None = None,
        working_directory: str = r"E:\CARLA_0.9.16",
        default_engine_ini: str | Path | None = None,
    ) -> Path:
        cfg = root / "safedrive_foundry" / "config" / "runtime"
        cfg.mkdir(parents=True, exist_ok=True)
        args = arguments or (
            f"/Game/Carla/Maps/{default_map} -windowed -ResX=640 -ResY=360 "
            f"-quality-level=Low -nosound -{rhi} -carla-rpc-port=2000"
        )
        ini_line = f'default_engine_ini = "{str(default_engine_ini)}"' if default_engine_ini else ""
        path = cfg / "carla_start.toml"
        lines = [
            "[carla_start]",
            'expected_version = "0.9.16"',
            r'windows_executable = "E:\\CARLA_0.9.16\\CarlaUE4.exe"',
            f'windows_working_directory = "{working_directory.replace(chr(92), chr(92)+chr(92))}"',
            'wsl_path = "/mnt/e/CARLA_0.9.16/CarlaUE4.exe"',
            f'launch_mode = "{launch_mode}"',
            f'default_map = "{default_map}"',
            f'rhi = "{rhi}"',
            "render_offscreen = false",
            f'arguments = "{args}"',
        ]
        if ini_line:
            lines.append(ini_line)
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_default_engine_ini(self, path: Path, *, map_name: str, rhi: str = "dx12") -> Path:
        asset = f"/Game/Carla/Maps/{map_name}.{map_name}"
        rhi_value = "DefaultGraphicsRHI_DX12" if rhi == "dx12" else "DefaultGraphicsRHI_DX11"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "[/Script/EngineSettings.GameMapsSettings]",
                    f"EditorStartupMap={asset}",
                    f"GameDefaultMap={asset}",
                    f"ServerDefaultMap={asset}",
                    f"TransitionMap={asset}",
                    "",
                    "[/Script/WindowsTargetPlatform.WindowsTargetSettings]",
                    f"DefaultGraphicsRHI={rhi_value}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def test_start_spec_normalizes_rhi_and_offscreen_from_toml_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(
                root,
                launch_mode="explicit_arguments",
                default_map="Town04",
                rhi="dx12",
                arguments=(
                    "/Game/Carla/Maps/Town04 -windowed -ResX=640 -ResY=360 "
                    "-quality-level=Low -nosound -dx11 -carla-rpc-port=2000"
                ),
            )
            # Force render_offscreen true in file.
            text = (root / "safedrive_foundry/config/runtime/carla_start.toml").read_text(encoding="utf-8")
            (root / "safedrive_foundry/config/runtime/carla_start.toml").write_text(
                text.replace("render_offscreen = false", "render_offscreen = true"),
                encoding="utf-8",
            )
            resolver = ConnectionResolver(root, process_query=lambda: "NOT_RUNNING")
            spec = resolver._start_spec()
            assert spec is not None
            self.assertEqual(spec["rhi"], "dx12")
            self.assertTrue(spec["render_offscreen"])
            self.assertEqual(count_rhi_flags(spec["arguments"]), 1)
            self.assertEqual(parse_rhi_from_command_line(spec["arguments"]), "dx12")
            self.assertEqual(count_offscreen_flags(spec["arguments"]), 1)
            override = resolver._start_spec(rhi="dx11", render_offscreen=False)
            assert override is not None
            self.assertEqual(override["rhi"], "dx11")
            self.assertFalse(override["render_offscreen"])
            self.assertEqual(parse_rhi_from_command_line(override["arguments"]), "dx11")
            self.assertEqual(count_offscreen_flags(override["arguments"]), 0)

    def test_ensure_rejects_ready_instance_with_rhi_mismatch(self) -> None:
        with patch(
            "runtime.carla_connection.query_windows_carla_command_line",
            return_value=r'E:\CARLA_0.9.16\CarlaUE4.exe /Game/Carla/Maps/Town04 -dx11 -carla-rpc-port=2000',
        ):
            resolver = self.resolver(
                tcp_probe=lambda h, p, t: (True, None),
                client_factory=lambda h, p: Client(),
            )
            report = resolver.ensure(host="host", rhi="dx12", render_offscreen=False)
        self.assertEqual(report.status, RETRYABLE_FAILURE)
        self.assertEqual(report.error_code, LAUNCH_PARAM_MISMATCH)
        self.assertEqual(report.details.get("requested_rhi"), "dx12")
        self.assertEqual(report.details.get("effective_rhi"), "dx11")

    def test_launch_params_match_helper(self) -> None:
        self.assertTrue(
            launch_params_match(
                "CarlaUE4.exe -dx12 -RenderOffScreen -carla-rpc-port=2000",
                rhi="dx12",
                render_offscreen=True,
            )
        )
        self.assertFalse(
            launch_params_match(
                "CarlaUE4.exe -dx11 -carla-rpc-port=2000",
                rhi="dx12",
                render_offscreen=False,
            )
        )
        self.assertIsNone(launch_params_match(None, rhi="dx11", render_offscreen=False))
        self.assertEqual(normalize_rhi("dx11"), "dx11")

    def test_powershell_command_includes_working_directory(self) -> None:
        cmd = build_powershell_start_command(
            windows_executable=r"E:\CARLA_0.9.16\CarlaUE4.exe",
            working_directory=r"E:\CARLA_0.9.16",
            argument_parts=None,
        )
        self.assertIn("-WorkingDirectory 'E:\\CARLA_0.9.16'", cmd)
        self.assertIn("-PassThru", cmd)
        self.assertIn("Start-Process", cmd)
        self.assertNotIn("-ArgumentList", cmd)

    def test_default_engine_mode_omits_argument_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, launch_mode="default_engine", default_map="Town03")
            captured: list[LaunchResult] = []

            def start(spec):
                result = LaunchResult(
                    ok=True,
                    pid=4242,
                    executable=spec.get("windows_executable"),
                    working_directory=spec.get("windows_working_directory"),
                    launch_arguments=spec.get("effective_launch_arguments", ""),
                    launch_mode=spec.get("launch_mode"),
                    powershell_command=build_powershell_start_command(
                        windows_executable=str(spec["windows_executable"]),
                        working_directory=str(spec["windows_working_directory"]),
                        argument_parts=(
                            str(spec.get("effective_launch_arguments") or "").split()
                            if spec.get("launch_mode") == "explicit_arguments"
                            else None
                        ),
                    ),
                    started_at=0.0,
                )
                captured.append(result)
                return result

            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town03", rhi="dx12")
            resolver = ConnectionResolver(
                root,
                process_query=lambda: "NOT_RUNNING",
                tcp_probe=lambda h, p, t: (False, "refused"),
                start_process=start,
                sleeper=lambda _: None,
            )
            with patch("runtime.carla_connection.windows_path_exists", return_value=True), patch(
                "runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini
            ), patch(
                "runtime.carla_connection.query_windows_pid_alive", return_value=True
            ):
                report = resolver.ensure(
                    host="host",
                    map_name="Town03",
                    rhi="dx12",
                    startup_timeout_seconds=0.0,
                    poll_interval_seconds=0.0,
                    auto_pin_default_engine=False,
                )
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0].launch_mode, "default_engine")
            self.assertEqual(captured[0].launch_arguments, "")
            self.assertNotIn("-ArgumentList", captured[0].powershell_command or "")
            self.assertIn("-WorkingDirectory", captured[0].powershell_command or "")
            self.assertEqual(report.error_code, STARTUP_TIMEOUT)

    def test_explicit_mode_splits_and_escapes_arguments(self) -> None:
        cmd = build_powershell_start_command(
            windows_executable=r"E:\CARLA_0.9.16\CarlaUE4.exe",
            working_directory=r"E:\CARLA_0.9.16",
            argument_parts=[
                "/Game/Carla/Maps/Town03",
                "-windowed",
                "-ResX=640",
                "-ResY=360",
                "-quality-level=Low",
                "-nosound",
                "-dx12",
                "-carla-rpc-port=2000",
            ],
        )
        self.assertIn("-ArgumentList '/Game/Carla/Maps/Town03','-windowed','-ResX=640'", cmd)
        self.assertIn("'-dx12'", cmd)
        self.assertIn("-WorkingDirectory 'E:\\CARLA_0.9.16'", cmd)
        rewritten = rewrite_arguments_map(
            "/Game/Carla/Maps/Town04 -windowed -dx12 -carla-rpc-port=2000",
            "Town03",
        )
        self.assertTrue(rewritten.startswith("/Game/Carla/Maps/Town03.Town03"))
        self.assertIn("-dx12", rewritten)

    def test_explicit_start_spec_uses_packaged_unreal_asset_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, launch_mode="explicit_arguments", default_map="Town12")
            resolver = ConnectionResolver(root, process_query=lambda: "NOT_RUNNING")
            spec = resolver._start_spec(
                map_name="Town05", rhi="dx12", launch_mode="explicit_arguments"
            )
            assert spec is not None
            self.assertTrue(
                str(spec["effective_launch_arguments"]).startswith(
                    "/Game/Carla/Maps/Town05.Town05 "
                )
            )

    def test_request_map_mismatch_rejects_without_auto_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, launch_mode="default_engine", default_map="Town04")
            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town04", rhi="dx12")
            started: list[Any] = []
            resolver = ConnectionResolver(
                root,
                process_query=lambda: "NOT_RUNNING",
                tcp_probe=lambda h, p, t: (False, "refused"),
                start_process=lambda spec: started.append(spec) or True,
            )
            with patch("runtime.carla_connection.windows_path_exists", return_value=True), patch(
                "runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini
            ):
                report = resolver.ensure(
                    host="host",
                    map_name="Town03",
                    rhi="dx12",
                    auto_pin_default_engine=False,
                    startup_timeout_seconds=1.0,
                )
            self.assertEqual(started, [])
            self.assertEqual(report.error_code, MAP_OR_RHI_CONFIG_MISMATCH)
            self.assertEqual(report.status, BLOCKED_EXTERNAL)
            self.assertTrue(report.needs_user_action)
            self.assertEqual(report.details.get("requested_map"), "Town03")

    def test_pid_early_exit_classifies_without_waiting_full_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, launch_mode="default_engine", default_map="Town03")
            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town03")
            sleeps: list[float] = []

            def start(spec):
                return LaunchResult(
                    ok=True,
                    pid=99,
                    executable=str(spec.get("windows_executable")),
                    working_directory=str(spec.get("windows_working_directory")),
                    launch_arguments="",
                    launch_mode="default_engine",
                    started_at=1000.0,
                )

            resolver = ConnectionResolver(
                root,
                process_query=lambda: "NOT_RUNNING",
                tcp_probe=lambda h, p, t: (False, "refused"),
                start_process=start,
                sleeper=lambda s: sleeps.append(s),
            )
            with patch("runtime.carla_connection.windows_path_exists", return_value=True), patch(
                "runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini
            ), patch(
                "runtime.carla_connection.query_windows_pid_alive", return_value=False
            ), patch(
                "runtime.carla_connection.find_recent_crash_context", return_value=None
            ):
                report = resolver.ensure(
                    host="host",
                    map_name="Town03",
                    rhi="dx12",
                    startup_timeout_seconds=180.0,
                    poll_interval_seconds=0.5,
                    auto_pin_default_engine=False,
                )
            self.assertEqual(report.error_code, CARLA_PROCESS_EXITED_EARLY)
            self.assertLess(len(sleeps), 5)
            self.assertEqual(report.details.get("pid"), 99)

    def test_shader_fatal_classified_from_crash_context(self) -> None:
        crash = {
            "captured": True,
            "fields": {
                "CrashGUID": "UE4CC-test",
                "CrashType": "Assert",
                "ErrorMessage": "Shader compilation failures are Fatal",
                "RHI.RHIName": "D3D12",
                "SecondsSinceStart": "0",
                "CommandLine": "CarlaUE4 /Game/Carla/Maps/Town04 -dx12",
                "MemoryStats.bIsOOM": "0",
            },
            "CrashGUID": "UE4CC-test",
            "ErrorMessage": "Shader compilation failures are Fatal",
        }
        self.assertEqual(classify_startup_crash(crash), CARLA_SHADER_PIPELINE_FATAL)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, launch_mode="default_engine", default_map="Town03")
            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town03")

            def start(spec):
                return LaunchResult(ok=True, pid=7, started_at=1.0, launch_mode="default_engine")

            resolver = ConnectionResolver(
                root,
                process_query=lambda: "NOT_RUNNING",
                tcp_probe=lambda h, p, t: (False, "refused"),
                start_process=start,
                sleeper=lambda _: None,
            )
            with patch("runtime.carla_connection.windows_path_exists", return_value=True), patch(
                "runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini
            ), patch(
                "runtime.carla_connection.query_windows_pid_alive", return_value=False
            ), patch(
                "runtime.carla_connection.find_recent_crash_context", return_value=crash
            ):
                report = resolver.ensure(
                    host="host",
                    map_name="Town03",
                    rhi="dx12",
                    startup_timeout_seconds=180.0,
                    poll_interval_seconds=0.0,
                    auto_pin_default_engine=False,
                )
            self.assertEqual(report.error_code, CARLA_SHADER_PIPELINE_FATAL)
            self.assertEqual(report.status, BLOCKED_EXTERNAL)
            self.assertTrue(report.needs_user_action)
            self.assertEqual(report.details["crash_context"]["CrashGUID"], "UE4CC-test")

    def test_alive_but_rpc_unready_is_startup_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, launch_mode="default_engine", default_map="Town03")
            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town03")

            def start(spec):
                return LaunchResult(ok=True, pid=55, started_at=1.0, launch_mode="default_engine")

            resolver = ConnectionResolver(
                root,
                process_query=lambda: "NOT_RUNNING",
                tcp_probe=lambda h, p, t: (False, "refused"),
                start_process=start,
                sleeper=lambda _: None,
            )
            with patch("runtime.carla_connection.windows_path_exists", return_value=True), patch(
                "runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini
            ), patch(
                "runtime.carla_connection.query_windows_pid_alive", return_value=True
            ):
                report = resolver.ensure(
                    host="host",
                    map_name="Town03",
                    rhi="dx12",
                    startup_timeout_seconds=0.0,
                    poll_interval_seconds=0.0,
                    auto_pin_default_engine=False,
                )
            self.assertEqual(report.error_code, STARTUP_TIMEOUT)
            self.assertEqual(report.status, RETRYABLE_FAILURE)
            self.assertEqual(report.details.get("pid"), 55)

    def test_ready_map_mismatch_returns_map_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_start_toml(root, launch_mode="default_engine", default_map="Town03")
            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town03")
            calls = {"n": 0}

            def start(spec):
                return LaunchResult(ok=True, pid=11, started_at=1.0, launch_mode="default_engine")

            def factory(h, p):
                return Client(world=World("Carla/Maps/Town04"))

            def probe(h, p, t):
                calls["n"] += 1
                # First preflight: not ready. After launch: TCP up.
                return (calls["n"] > 1, None if calls["n"] > 1 else "refused")

            resolver = ConnectionResolver(
                root,
                process_query=lambda: "NOT_RUNNING" if calls["n"] <= 1 else "RUNNING",
                tcp_probe=probe,
                client_factory=factory,
                start_process=start,
                sleeper=lambda _: None,
            )
            with patch("runtime.carla_connection.windows_path_exists", return_value=True), patch(
                "runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini
            ), patch(
                "runtime.carla_connection.query_windows_pid_alive", return_value=True
            ):
                report = resolver.ensure(
                    host="host",
                    map_name="Town03",
                    rhi="dx12",
                    startup_timeout_seconds=5.0,
                    poll_interval_seconds=0.0,
                    auto_pin_default_engine=False,
                )
            self.assertEqual(report.error_code, MAP_MISMATCH)
            self.assertEqual(report.status, FAILED_FINAL)
            self.assertEqual(report.details.get("requested_map"), "Town03")
            self.assertIn("Town04", str(report.details.get("actual_map") or report.map))

    def test_already_running_compatible_does_not_start_second(self) -> None:
        started: list[Any] = []
        resolver = self.resolver(
            tcp_probe=lambda h, p, t: (True, None),
            client_factory=lambda h, p: Client(world=World("Carla/Maps/Town03")),
            start_process=lambda spec: started.append(spec) or True,
        )
        with patch(
            "runtime.carla_connection.query_windows_carla_command_line",
            return_value=r"E:\CARLA_0.9.16\CarlaUE4.exe",
        ):
            report = resolver.ensure(host="host", map_name="Town03", rhi="dx12")
        self.assertEqual(report.status, READY)
        self.assertEqual(report.recovery_action, "already_running")
        self.assertEqual(started, [])
        self.assertEqual(report.details.get("requested_map"), "Town03")
        self.assertEqual(report.details.get("effective_rhi_source"), "default_engine_config")

    def test_pin_default_engine_is_noop_when_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ini = self._write_default_engine_ini(Path(tmp) / "DefaultEngine.ini", map_name="Town03", rhi="dx12")
            before = ini.read_text(encoding="utf-8")
            result = pin_default_engine_config(
                requested_map="Town03",
                requested_rhi="dx12",
                path=ini,
            )
            self.assertTrue(result["ok"])
            self.assertFalse(result["written"])
            self.assertEqual(ini.read_text(encoding="utf-8"), before)

    def test_pin_default_engine_atomic_write_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ini = self._write_default_engine_ini(Path(tmp) / "DefaultEngine.ini", map_name="Town04", rhi="dx11")
            result = pin_default_engine_config(
                requested_map="Town03",
                requested_rhi="dx12",
                path=ini,
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["written"])
            self.assertTrue(result.get("backup_path"))
            self.assertTrue(Path(str(result["backup_path"])).is_file())
            after = read_default_engine_config(ini)
            self.assertIsNone(
                verify_default_engine_matches(after, requested_map="Town03", requested_rhi="dx12")
            )
            self.assertIn("Town03.Town03", after.game_default_map or "")
            self.assertEqual(after.configured_rhi, "dx12")

    def test_restart_refuses_synchronous_world_and_multiple_processes(self) -> None:
        process = WindowsCarlaProcess(7, r"E:\CARLA_0.9.16\CarlaUE4.exe", "CarlaUE4.exe")
        world = World("Carla/Maps/Town03")
        world.settings.synchronous_mode = True
        closed: list[int] = []
        resolver = self.resolver(
            tcp_probe=lambda h, p, t: (True, None),
            client_factory=lambda h, p: Client(world=world),
            windows_processes=lambda: (process,),
            request_process_close=lambda pid: closed.append(pid) or True,
        )
        report = resolver.restart(host="host", map_name="Town01")
        self.assertEqual(report.status, BLOCKED_EXTERNAL)
        self.assertEqual(closed, [])

        world.settings.synchronous_mode = False
        resolver = self.resolver(
            tcp_probe=lambda h, p, t: (True, None),
            client_factory=lambda h, p: Client(world=world),
            windows_processes=lambda: (
                process,
                WindowsCarlaProcess(8, process.executable_path, process.command_line),
            ),
            request_process_close=lambda pid: closed.append(pid) or True,
        )
        report = resolver.restart(host="host", map_name="Town01")
        self.assertEqual(report.error_code, NEEDS_USER_ACTION)
        self.assertEqual(closed, [])

    def test_restart_graceful_close_pin_ensure_and_single_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ini = self._write_default_engine_ini(root / "DefaultEngine.ini", map_name="Town03")
            self._write_start_toml(root, default_map="Town03", default_engine_ini=ini)
            state = {"running": True, "map": "Town03", "starts": 0, "closes": 0, "pid": 77}

            def close(pid: int) -> bool:
                self.assertEqual(pid, 77)
                state["closes"] += 1
                state["running"] = False
                return True

            def start(spec: Any) -> LaunchResult:
                state["starts"] += 1
                state["running"] = True
                state["map"] = "Town01"
                state["pid"] = 88
                return LaunchResult(ok=True, pid=88, launch_mode="default_engine", started_at=1.0)

            resolver = ConnectionResolver(
                root,
                process_query=lambda: "RUNNING" if state["running"] else "NOT_RUNNING",
                tcp_probe=lambda h, p, t: (bool(state["running"]), None),
                client_factory=lambda h, p: Client(world=World(f"Carla/Maps/{state['map']}")),
                start_process=start,
                windows_processes=lambda: (
                    WindowsCarlaProcess(
                        state["pid"], r"E:\CARLA_0.9.16\CarlaUE4.exe", r"E:\CARLA_0.9.16\CarlaUE4.exe"
                    ),
                ) if state["running"] else (),
                request_process_close=close,
                sleeper=lambda _: None,
            )
            with patch("runtime.carla_connection.windows_path_exists", return_value=True), patch(
                "runtime.carla_engine_config.resolve_default_engine_ini", return_value=ini
            ), patch("runtime.carla_connection.query_windows_carla_command_line", return_value=r"E:\CARLA_0.9.16\CarlaUE4.exe"), patch(
                "runtime.carla_connection.query_windows_pid_alive", return_value=True
            ):
                report = resolver.restart(
                    host="host",
                    map_name="Town01",
                    rhi="dx12",
                    shutdown_timeout_seconds=1.0,
                    startup_timeout_seconds=1.0,
                    poll_interval_seconds=0.0,
                )
            self.assertEqual(report.status, READY)
            self.assertEqual(report.map, "Carla/Maps/Town01")
            self.assertEqual(state["closes"], 1)
            self.assertEqual(state["starts"], 1)
            self.assertEqual(report.recovery_action, "safe_cold_restart")
            self.assertEqual(read_default_engine_config(ini).configured_map_token, "Town01")

    def test_find_recent_crash_context_parses_shader_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            crash_dir = base / "User" / "AppData/Local/CarlaUE4/Saved/Crashes/UE4CC-shader"
            crash_dir.mkdir(parents=True)
            xml = crash_dir / "CrashContext.runtime-xml"
            xml.write_text(
                "<FGenericCrashContext><RuntimeProperties>"
                "<CrashGUID>UE4CC-shader</CrashGUID>"
                "<CrashType>Assert</CrashType>"
                "<ErrorMessage>Shader compilation failures are Fatal</ErrorMessage>"
                "<RHI.RHIName>D3D12</RHI.RHIName>"
                "<SecondsSinceStart>0</SecondsSinceStart>"
                "<CommandLine>CarlaUE4 /Game/Carla/Maps/Town04 -dx12</CommandLine>"
                "<MemoryStats.bIsOOM>0</MemoryStats.bIsOOM>"
                "</RuntimeProperties></FGenericCrashContext>",
                encoding="utf-8",
            )
            found = find_recent_crash_context(since_unix_s=0.0, crash_root=base)
            assert found is not None
            self.assertTrue(found["captured"])
            self.assertEqual(found["CrashGUID"], "UE4CC-shader")
            self.assertEqual(classify_startup_crash(found), CARLA_SHADER_PIPELINE_FATAL)


if __name__ == "__main__":
    unittest.main()
