"""The ``sdf`` G0 command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .doctor import (
    BLOCKED,
    FAIL,
    PASS,
    WARN,
    print_doctor_report,
    run_doctor,
    run_g0_validation,
    write_doctor_reports,
    write_validation_reports,
)
from .sync_contract import (
    ContractViolation,
    SyncConfig,
    compare_traces,
    load_sync_config,
    run_carla_trace,
    run_deterministic_smoke,
)


def _source_root() -> Path:
    """Find the repository root for both the source shim and installed entrypoint."""

    candidates: list[Path] = []
    configured = os.environ.get("SAFEDRIVE_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend([Path.cwd(), Path(__file__).resolve().parents[4]])
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "versions.lock").is_file() and (candidate / "safedrive_foundry").is_dir():
            return candidate
    return Path.cwd().resolve()


def _root_from_args(args: argparse.Namespace) -> Path:
    value = getattr(args, "root", None)
    return Path(value).resolve() if value else _source_root()


def _resolve_path(root: Path, value: str | None, default: Path) -> Path:
    candidate = Path(value) if value else default
    return candidate if candidate.is_absolute() else root / candidate


def _load_config(root: Path) -> SyncConfig:
    return load_sync_config(root / "safedrive_foundry" / "config" / "carla_ros.toml")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_doctor_paths(root: Path) -> tuple[Path, Path]:
    base = root / "docs" / "environment" / "evidence" / "g0-05"
    return base / "doctor.json", base / "doctor.md"


def _default_validation_paths(root: Path) -> tuple[Path, Path]:
    base = root / "docs" / "environment" / "evidence" / "g0-05"
    return base / "validation.json", base / "validation.md"


def _config_host_port(root: Path) -> tuple[str, int]:
    try:
        import tomllib

        data = tomllib.loads((root / "safedrive_foundry" / "config" / "carla_ros.toml").read_text(encoding="utf-8"))
        carla = data.get("carla", {})
        return str(carla.get("host", "127.0.0.1")), int(carla.get("port", 2000))
    except (OSError, ValueError, TypeError):
        return "127.0.0.1", 2000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdf",
        description="SafeDrive Foundry G0 environment, synchronization and evidence commands.",
    )
    parser.add_argument("--root", help="project root (defaults to the current SafeDrive workspace)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="diagnose G0 environment and write JSON/Markdown reports")
    doctor.add_argument("--root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    doctor.add_argument("--json-out", help="JSON report path")
    doctor.add_argument("--markdown-out", help="Markdown report path")
    doctor.add_argument("--carla-host", help="CARLA RPC host override")
    doctor.add_argument("--carla-port", type=int, help="CARLA RPC port override")
    doctor.add_argument("--expected-version", help="CARLA client/server version override")
    doctor.add_argument("--min-free-gib", type=float, default=20.0, help="minimum free disk space")
    doctor.add_argument("--timeout", type=float, default=3.0, help="probe timeout in seconds")

    smoke = subparsers.add_parser("sync-smoke", help="run or resume deterministic or live CARLA synchronization smoke")
    smoke.add_argument("--root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    smoke.add_argument("--seed", type=int, default=1234)
    smoke.add_argument("--steps", type=int, default=12)
    smoke.add_argument("--run-id", default=None)
    smoke.add_argument("--run-dir", help="directory for checkpoint and trace")
    smoke.add_argument("--checkpoint", help="checkpoint path")
    smoke.add_argument("--trace", help="trace path")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--interrupt-after", type=int, help="write a checkpoint and stop after N new frames")
    smoke.add_argument("--carla", action="store_true", help="run a live CARLA tick smoke instead of offline simulation")
    smoke.add_argument("--carla-host", help="live CARLA host")
    smoke.add_argument("--carla-port", type=int, help="live CARLA port")
    smoke.add_argument("--timeout", type=float, default=5.0)

    compare = subparsers.add_parser("compare", help="compare two synchronization traces")
    compare.add_argument("trace_a")
    compare.add_argument("trace_b")
    compare.add_argument("--tolerance", type=float, default=1e-6)

    validate = subparsers.add_parser("validate-g0", help="run offline G0-05 contract and recovery validation")
    validate.add_argument("--root", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    validate.add_argument("--json-out", help="JSON validation report path")
    validate.add_argument("--markdown-out", help="Markdown validation report path")
    return parser


def _cmd_doctor(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    json_default, markdown_default = _default_doctor_paths(root)
    report = run_doctor(
        root,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        expected_version=args.expected_version,
        min_free_gib=args.min_free_gib,
        timeout_seconds=args.timeout,
        invocation=sys.argv,
    )
    json_path = _resolve_path(root, args.json_out, json_default)
    markdown_path = _resolve_path(root, args.markdown_out, markdown_default)
    write_doctor_reports(report, json_path, markdown_path)
    print_doctor_report(report)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return {PASS: 0, WARN: 0, FAIL: 1, BLOCKED: 2}[report.status]


def _cmd_sync_smoke(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    try:
        config = _load_config(root)
    except ContractViolation as exc:
        print(f"FAIL sync config [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1

    if args.carla:
        default_host, default_port = _config_host_port(root)
        host = args.carla_host or os.environ.get("CARLA_HOST") or default_host
        port = args.carla_port or int(os.environ.get("CARLA_PORT", default_port))
        try:
            trace = run_carla_trace(
                host=host,
                port=port,
                timeout_seconds=args.timeout,
                steps=args.steps,
                config=config,
            )
        except (ContractViolation, OSError, RuntimeError, ValueError) as exc:
            code = exc.code if isinstance(exc, ContractViolation) else type(exc).__name__
            message = exc.message if isinstance(exc, ContractViolation) else str(exc)
            print(f"FAIL live CARLA sync [{code}]: {message}", file=sys.stderr)
            return 1
        run_id = args.run_id or f"carla-{args.seed}"
        default_dir = root / "docs" / "environment" / "evidence" / "g0-05" / "smoke" / run_id
        run_dir = _resolve_path(root, args.run_dir, default_dir)
        trace_path = _resolve_path(root, args.trace, run_dir / "trace.json")
        _write_json(trace_path, trace)
        print(f"COMPLETED live CARLA sync trace: {trace_path}")
        print(f"frames={trace['frames']}")
        return 0

    run_id = args.run_id or f"seed-{args.seed}"
    default_dir = root / "docs" / "environment" / "evidence" / "g0-05" / "smoke" / run_id
    run_dir = _resolve_path(root, args.run_dir, default_dir)
    checkpoint_path = _resolve_path(root, args.checkpoint, run_dir / "checkpoint.json")
    trace_path = _resolve_path(root, args.trace, run_dir / "trace.json")
    try:
        status, trace = run_deterministic_smoke(
            seed=args.seed,
            steps=args.steps,
            config=config,
            checkpoint_path=checkpoint_path,
            trace_path=trace_path,
            resume=args.resume,
            interrupt_after=args.interrupt_after,
        )
    except (ContractViolation, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, ContractViolation) else type(exc).__name__
        message = exc.message if isinstance(exc, ContractViolation) else str(exc)
        print(f"FAIL deterministic sync smoke [{code}]: {message}", file=sys.stderr)
        return 1
    if status == "INTERRUPTED":
        print(f"INTERRUPTED checkpoint: {checkpoint_path}")
        return 75
    assert trace is not None
    print(f"COMPLETED deterministic sync trace: {trace_path}")
    print(f"frames={trace['frames']}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        first = _load_json(Path(args.trace_a))
        second = _load_json(Path(args.trace_b))
        result = compare_traces(first, second, tolerance_seconds=args.tolerance)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FAIL trace comparison: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    root = _root_from_args(args)
    try:
        config = _load_config(root)
    except ContractViolation as exc:
        print(f"FAIL validation config [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    report = run_g0_validation(root, config)
    json_default, markdown_default = _default_validation_paths(root)
    json_path = _resolve_path(root, args.json_out, json_default)
    markdown_path = _resolve_path(root, args.markdown_out, markdown_default)
    write_validation_reports(report, json_path, markdown_path)
    print(f"G0-05 validation: {report['status']} ({report['summary']['passed']}/{report['summary']['total']} passed)")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    return 0 if report["status"] == PASS else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "sync-smoke":
        return _cmd_sync_smoke(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "validate-g0":
        return _cmd_validate(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
