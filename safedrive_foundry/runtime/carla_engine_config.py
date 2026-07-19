"""Shared DefaultEngine.ini map/RHI helpers for CARLA cold-start.

Used by ``sdf sim ensure`` and live runners so map pinning is not duplicated.
Only the documented map/RHI keys are touched; other Unreal settings are left alone.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Nested large-map packages under Content/Carla/Maps/<Town>/<Town>.umap
NESTED_MAP_PACKAGE: frozenset[str] = frozenset({"Town11", "Town12", "Town13", "Town15"})

_MAP_KEY_NAMES = (
    "EditorStartupMap",
    "GameDefaultMap",
    "ServerDefaultMap",
    "TransitionMap",
)
_RHI_KEY = "DefaultGraphicsRHI"
_RHI_INI_VALUES = {
    "dx11": "DefaultGraphicsRHI_DX11",
    "dx12": "DefaultGraphicsRHI_DX12",
}
_RHI_INI_TO_TOKEN = {
    "DefaultGraphicsRHI_DX11": "dx11",
    "DefaultGraphicsRHI_DX12": "dx12",
    "DefaultGraphicsRHI_Vulkan": "vulkan",
    "DefaultGraphicsRHI_OpenGL": "opengl",
}
_VALID_RHI = frozenset({"dx11", "dx12"})
_RHI_ALIASES = {
    "dx11": "dx11",
    "d3d11": "dx11",
    "11": "dx11",
    "dx12": "dx12",
    "d3d12": "dx12",
    "12": "dx12",
}


def _normalize_rhi(value: str | None, *, default: str = "dx12") -> str:
    if value is None or str(value).strip() == "":
        return default
    token = str(value).strip().lower().lstrip("-")
    token = _RHI_ALIASES.get(token, token)
    if token not in _VALID_RHI:
        raise ValueError(f"unsupported RHI {value!r}; expected one of {sorted(_VALID_RHI)}")
    return token


def _windows_path_to_wsl(path: str) -> Path | None:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path.strip())
    if not match:
        return None
    drive, rest = match.group(1).lower(), match.group(2).replace("\\", "/")
    return Path("/mnt") / drive / rest


def _wsl_path_to_windows(path: str | Path) -> str | None:
    text = str(path).replace("\\", "/")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if not match:
        return None
    drive, rest = match.group(1).upper(), match.group(2).replace("/", "\\")
    return f"{drive}:\\{rest}"


def default_engine_ini_candidates(windows_install_root: str | None = None) -> list[Path]:
    """Return candidate paths for DefaultEngine.ini (WSL mounts preferred)."""

    candidates: list[Path] = []
    roots: list[str] = []
    if windows_install_root:
        roots.append(windows_install_root)
    roots.append(r"E:\CARLA_0.9.16")
    for root in roots:
        wsl = _windows_path_to_wsl(root)
        if wsl is not None:
            candidates.append(wsl / "CarlaUE4" / "Config" / "DefaultEngine.ini")
        candidates.append(Path(root) / "CarlaUE4" / "Config" / "DefaultEngine.ini")
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def resolve_default_engine_ini(windows_install_root: str | None = None) -> Path | None:
    for path in default_engine_ini_candidates(windows_install_root):
        if path.is_file():
            return path
    return None


def map_content_path(map_name: str) -> str:
    """Unreal content path without asset suffix: /Game/Carla/Maps/Town03."""

    token = str(map_name).strip()
    if not token:
        raise ValueError("map_name must be non-empty")
    if token.startswith("/Game/"):
        return token.split(".", 1)[0]
    if token in NESTED_MAP_PACKAGE:
        return f"/Game/Carla/Maps/{token}/{token}"
    return f"/Game/Carla/Maps/{token}"


def map_asset_path(map_name: str) -> str:
    """Full DefaultEngine map asset id: /Game/Carla/Maps/Town03.Town03."""

    token = str(map_name).strip()
    if "." in token and token.startswith("/Game/"):
        return token
    content = map_content_path(token)
    leaf = token.split("/")[-1]
    return f"{content}.{leaf}"


def normalize_map_token(value: str | None) -> str | None:
    """Reduce DefaultEngine or world map strings to a comparable town token."""

    if value is None:
        return None
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    leaf = text.split("/")[-1]
    base = leaf.split(".", 1)[0] if "." in leaf else leaf
    return base.strip() or None


def maps_match(actual: str | None, requested: str | None) -> bool:
    """True when the live world map is compatible with the requested town token."""

    if not requested:
        return True
    if not actual:
        return False
    req = normalize_map_token(requested) or str(requested)
    act = str(actual)
    act_token = normalize_map_token(act) or act
    if req == act_token:
        return True
    if req in act or act.endswith(req):
        return True
    return False


def rhi_to_ini_value(rhi: str) -> str:
    token = _normalize_rhi(rhi, default="dx12")
    return _RHI_INI_VALUES[token]


def rhi_from_ini_value(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in _RHI_INI_TO_TOKEN:
        return _RHI_INI_TO_TOKEN[text]
    lowered = text.lower()
    if "dx12" in lowered or "d3d12" in lowered:
        return "dx12"
    if "dx11" in lowered or "d3d11" in lowered:
        return "dx11"
    return None


@dataclass(frozen=True)
class DefaultEngineConfig:
    path: Path | None
    game_default_map: str | None = None
    server_default_map: str | None = None
    editor_startup_map: str | None = None
    transition_map: str | None = None
    default_graphics_rhi: str | None = None
    raw_keys: dict[str, str] = field(default_factory=dict)

    @property
    def configured_map_token(self) -> str | None:
        return normalize_map_token(self.game_default_map or self.server_default_map)

    @property
    def configured_rhi(self) -> str | None:
        return rhi_from_ini_value(self.default_graphics_rhi)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "game_default_map": self.game_default_map,
            "server_default_map": self.server_default_map,
            "editor_startup_map": self.editor_startup_map,
            "transition_map": self.transition_map,
            "default_graphics_rhi": self.default_graphics_rhi,
            "configured_map_token": self.configured_map_token,
            "configured_rhi": self.configured_rhi,
        }


def _parse_ini_key(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}=(.*)$", text)
    if not match:
        return None
    return match.group(1).strip() or None


def read_default_engine_config(path: Path | None = None) -> DefaultEngineConfig:
    """Parse the map/RHI keys from DefaultEngine.ini (read-only)."""

    resolved = path or resolve_default_engine_ini()
    if resolved is None or not resolved.is_file():
        return DefaultEngineConfig(path=resolved)
    text = resolved.read_text(encoding="utf-8", errors="replace")
    keys = {name: _parse_ini_key(text, name) for name in (*_MAP_KEY_NAMES, _RHI_KEY)}
    return DefaultEngineConfig(
        path=resolved,
        game_default_map=keys.get("GameDefaultMap"),
        server_default_map=keys.get("ServerDefaultMap"),
        editor_startup_map=keys.get("EditorStartupMap"),
        transition_map=keys.get("TransitionMap"),
        default_graphics_rhi=keys.get(_RHI_KEY),
        raw_keys={k: v for k, v in keys.items() if v is not None},
    )


@dataclass(frozen=True)
class ConfigMismatch:
    error_code: str
    message: str
    expected: Mapping[str, Any]
    actual: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "expected": dict(self.expected),
            "actual": dict(self.actual),
        }


def verify_default_engine_matches(
    config: DefaultEngineConfig,
    *,
    requested_map: str | None,
    requested_rhi: str | None,
) -> ConfigMismatch | None:
    """Return a mismatch description when DefaultEngine.ini disagrees with the request."""

    expected: dict[str, Any] = {}
    actual: dict[str, Any] = {
        "game_default_map": config.game_default_map,
        "server_default_map": config.server_default_map,
        "default_graphics_rhi": config.default_graphics_rhi,
    }
    if requested_map:
        asset = map_asset_path(requested_map)
        expected["map_asset"] = asset
        expected["map_token"] = normalize_map_token(requested_map)
        game_ok = maps_match(config.game_default_map, requested_map)
        server_ok = maps_match(config.server_default_map, requested_map)
        if not (game_ok and server_ok):
            return ConfigMismatch(
                error_code="MAP_OR_RHI_CONFIG_MISMATCH",
                message=(
                    f"DefaultEngine map mismatch: requested={requested_map} "
                    f"GameDefaultMap={config.game_default_map} "
                    f"ServerDefaultMap={config.server_default_map}; "
                    f"set both to {asset}"
                ),
                expected=expected,
                actual=actual,
            )
    if requested_rhi:
        want = _normalize_rhi(requested_rhi, default="dx12")
        expected["rhi"] = want
        expected["default_graphics_rhi"] = rhi_to_ini_value(want)
        have = config.configured_rhi
        if have != want:
            return ConfigMismatch(
                error_code="MAP_OR_RHI_CONFIG_MISMATCH",
                message=(
                    f"DefaultEngine RHI mismatch: requested={want} "
                    f"DefaultGraphicsRHI={config.default_graphics_rhi}; "
                    f"set {_RHI_KEY}={rhi_to_ini_value(want)}"
                ),
                expected=expected,
                actual=actual,
            )
    return None


def _replace_or_append_key(text: str, key: str, value: str) -> tuple[str, bool]:
    """Return (new_text, changed)."""

    pattern = re.compile(rf"(?m)^({re.escape(key)})=.*$")
    replacement = f"{key}={value}"
    if pattern.search(text):
        current = _parse_ini_key(text, key)
        if current == value:
            return text, False
        new_text = pattern.sub(replacement, text, count=1)
        return new_text, new_text != text
    section = "[/Script/EngineSettings.GameMapsSettings]"
    if key != _RHI_KEY and section in text:
        idx = text.index(section) + len(section)
        return text[:idx] + "\n" + replacement + text[idx:], True
    rhi_section = "[/Script/WindowsTargetPlatform.WindowsTargetSettings]"
    if key == _RHI_KEY and rhi_section in text:
        idx = text.index(rhi_section) + len(rhi_section)
        return text[:idx] + "\n" + replacement + text[idx:], True
    return text.rstrip() + "\n" + replacement + "\n", True


def pin_default_engine_config(
    *,
    requested_map: str | None,
    requested_rhi: str | None,
    path: Path | None = None,
    create_backup: bool = True,
) -> dict[str, Any]:
    """Atomically pin map/RHI keys. No-op when already matching.

    Returns a structured result with backup path and whether a write occurred.
    """

    resolved = path or resolve_default_engine_ini()
    if resolved is None or not resolved.is_file():
        return {
            "ok": False,
            "error_code": "NEEDS_USER_ACTION",
            "message": (
                "DefaultEngine.ini not found; set GameDefaultMap/ServerDefaultMap and "
                f"{_RHI_KEY} manually under CarlaUE4/Config/"
            ),
            "path": None,
            "written": False,
        }

    before = read_default_engine_config(resolved)
    mismatch = verify_default_engine_matches(
        before, requested_map=requested_map, requested_rhi=requested_rhi
    )
    if mismatch is None:
        return {
            "ok": True,
            "path": str(resolved),
            "written": False,
            "backup_path": None,
            "before": before.to_dict(),
            "after": before.to_dict(),
            "message": "DefaultEngine.ini already matches request; no rewrite",
        }

    text = resolved.read_text(encoding="utf-8", errors="replace")
    changed = False
    if requested_map:
        asset = map_asset_path(requested_map)
        for key in _MAP_KEY_NAMES:
            text, key_changed = _replace_or_append_key(text, key, asset)
            changed = changed or key_changed
    if requested_rhi:
        rhi_value = rhi_to_ini_value(requested_rhi)
        text, key_changed = _replace_or_append_key(text, _RHI_KEY, rhi_value)
        changed = changed or key_changed

    if not changed:
        return {
            "ok": False,
            "error_code": "MAP_OR_RHI_CONFIG_MISMATCH",
            "message": "DefaultEngine.ini parse/replace produced no changes",
            "path": str(resolved),
            "written": False,
            "before": before.to_dict(),
        }

    backup_path: str | None = None
    if create_backup:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = resolved.with_name(f"{resolved.name}.safedrive-backup-{stamp}")
        shutil.copy2(resolved, backup)
        backup_path = str(backup)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".DefaultEngine.ini.",
        suffix=".tmp",
        dir=str(resolved.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, resolved)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    after = read_default_engine_config(resolved)
    still = verify_default_engine_matches(
        after, requested_map=requested_map, requested_rhi=requested_rhi
    )
    if still is not None:
        return {
            "ok": False,
            "error_code": still.error_code,
            "message": f"post-write verification failed: {still.message}",
            "path": str(resolved),
            "written": True,
            "backup_path": backup_path,
            "before": before.to_dict(),
            "after": after.to_dict(),
            "mismatch": still.to_dict(),
        }
    return {
        "ok": True,
        "path": str(resolved),
        "written": True,
        "backup_path": backup_path,
        "before": before.to_dict(),
        "after": after.to_dict(),
        "message": "DefaultEngine.ini pinned and re-verified",
    }


def windows_install_root_from_executable(windows_executable: str | None) -> str | None:
    """Derive E:\\CARLA_0.9.16 from E:\\CARLA_0.9.16\\CarlaUE4.exe."""

    if not windows_executable:
        return None
    text = str(windows_executable).strip().rstrip("\\/")
    if not text:
        return None
    parent = str(Path(text).parent)
    if parent.startswith("/mnt/"):
        return _wsl_path_to_windows(parent)
    return parent
