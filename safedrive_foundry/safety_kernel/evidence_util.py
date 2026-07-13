"""Evidence pack helpers: host resources, hashes, git fingerprint, write gate.

Formal evidence is only written when SDF_WRITE_G2_EVIDENCE=1 so ordinary unit
tests do not silently overwrite production evidence packs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
from pathlib import Path
from typing import Any, Mapping


def write_evidence_enabled() -> bool:
    return os.environ.get("SDF_WRITE_G2_EVIDENCE", "").strip() in {"1", "true", "TRUE", "yes"}


def host_resource_snapshot() -> dict[str, Any]:
    """Process-local resources (offline regression). Not CARLA/GPU live profile."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux ru_maxrss is KiB; macOS is bytes — report both raw and best-effort MiB.
    peak_rss = float(usage.ru_maxrss)
    peak_rss_mib = peak_rss / 1024.0 if platform.system() == "Linux" else peak_rss / (1024.0 * 1024.0)
    return {
        "hostname": platform.node() or "unknown",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "process_peak_rss_raw": peak_rss,
        "process_peak_rss_mib_approx": round(peak_rss_mib, 3),
        "process_user_time_s": float(usage.ru_utime),
        "process_system_time_s": float(usage.ru_stime),
        "CARLA_quality_rendering_mode": "offline_n/a",
        "Windows_CARLA_VRAM": "offline_n/a",
        "WSL_CUDA_allocated_reserved": "offline_n/a",
        "whole_GPU_peak": "offline_n/a",
        "model_precision_quantization": "n/a_safety_kernel_cpu",
        "OOM_thermal_disconnect_recovery": "none_observed",
        "disk_artifact_note": "json_evidence_only",
        "profile": "offline_cpu_regression",
        "workload_profile": "regression",
    }


def git_fingerprint(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.check_output(
                args,
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            return out.strip()
        except Exception:
            return "unavailable"

    head = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--short"])
    dirty = bool(status) if status != "unavailable" else None
    return {
        "git_head": head,
        "git_branch": branch,
        "git_dirty": dirty,
        "git_status_short": status[:2000] if status != "unavailable" else status,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_evidence_pack(
    evidence_dir: Path,
    *,
    task: str,
    schema: str,
    summary: Mapping[str, Any],
    readme_md: str,
    config_hash: str,
    contracts_schema_hash: str,
    command: str,
    repo_root: Path,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write summary/manifest/README when write gate is open. Returns manifest dict."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(dict(summary), indent=2, sort_keys=True) + "\n"
    summary_path = evidence_dir / "summary.json"
    summary_path.write_text(summary_text, encoding="utf-8")
    artifact_hash = sha256_text(summary_text)
    fp = git_fingerprint(repo_root)
    man: dict[str, Any] = {
        "task": task,
        "schema": schema,
        "artifacts": ["summary.json", "README.md", "manifest.json"],
        "artifact_hash_summary_sha256": artifact_hash,
        "config_hash": config_hash,
        "contracts_schema_hash": contracts_schema_hash,
        "command": command,
        **fp,
    }
    if extra_manifest:
        man.update(dict(extra_manifest))
    man_text = json.dumps(man, indent=2, sort_keys=True) + "\n"
    (evidence_dir / "manifest.json").write_text(man_text, encoding="utf-8")
    (evidence_dir / "README.md").write_text(readme_md, encoding="utf-8")
    return man
