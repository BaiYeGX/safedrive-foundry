#!/usr/bin/env python3
"""Validate the SafeDrive Foundry task catalog and its roadmap dependencies.

The checker is intentionally stdlib-only.  It reads the repository as it is on
disk, treats only tasks/G0..G8 as active task directories, and excludes any
path containing an ``archive`` component from the active catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TASK_COUNT = 48
PHASES = tuple(f"G{i}" for i in range(9))
TASK_ID_RE = re.compile(r"(?<![A-Za-z0-9])G[0-8]-\d{2}(?!\d)")
FILENAME_RE = re.compile(r"^(G[0-8]-\d{2})_[^/]+\.md$")
TITLE_RE = re.compile(r"^\s*#\s+(G[0-8]-\d{2})\b", re.MULTILINE)
STATUS_RE = re.compile(
    r"(?:\*\*)?状态(?:\*\*)?\s*[：:]\s*`?\s*([A-Z][A-Z_]*)"
)
DEPENDENCY_RE = re.compile(
    r"^\s*(?:-\s*)?(?:\*\*)?(?:依赖|Dependencies?)"
    r"(?:\*\*)?\s*[：:]\s*(.*?)\s*$",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"\b(?P<start>G[0-8]-(?P<start_num>\d{2}))\s*"
    r"(?:～|~|–|—|至|-)\s*"
    r"(?:(?P<end_phase>G[0-8])-)?(?P<end_num>\d{2})(?!\d)"
)
STARTUP_SECTION_RE = re.compile(
    r"^## 启动读取清单\s*$\n(?P<body>.*?)(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
DOC_PATH_RE = re.compile(r"(?<![A-Za-z0-9_./-])(docs/[A-Za-z0-9_./-]+\.md)")

TASK_STATES = {
    "PENDING",
    "CURRENT",
    "IN_PROGRESS",
    "PAUSED",
    "BLOCKED",
    "BLOCKED_EXTERNAL",
    "DECISION_REQUIRED",
    "VALIDATING",
    "COMPLETED",
    "COMPLETED_WITH_LIMITS",
    "FAILED_FINAL",
}


def _id_sort_key(task_id: str) -> tuple[int, int]:
    phase, number = task_id.split("-")
    return int(phase[1:]), int(number)


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if details:
        result["details"] = details
    return result


def _is_archived(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return "archive" in relative.parts


def _expand_dependency_text(value: str) -> list[str]:
    """Normalize IDs and compact same-phase ranges from a dependency cell."""

    value = value.strip()
    if not value or re.fullmatch(r"(?:无|none|n/a|-)\s*", value, re.IGNORECASE):
        return []

    dependencies: set[str] = set()
    ranges: list[tuple[int, int]] = []
    for match in RANGE_RE.finditer(value):
        start = match.group("start")
        start_phase = int(start[1])
        start_num = int(match.group("start_num"))
        end_phase_text = match.group("end_phase")
        end_phase = start_phase if end_phase_text is None else int(end_phase_text[1])
        end_num = int(match.group("end_num"))
        if start_phase == end_phase and start_num <= end_num:
            ranges.append((match.start(), match.end()))
            for number in range(start_num, end_num + 1):
                dependencies.add(f"G{start_phase}-{number:02d}")

    remainder = RANGE_RE.sub(" ", value)
    dependencies.update(TASK_ID_RE.findall(remainder))
    return sorted(dependencies, key=_id_sort_key)


def _read_first_dependency(text: str) -> tuple[str | None, str | None]:
    for line in text.splitlines():
        match = DEPENDENCY_RE.match(line)
        if match:
            return match.group(1), None
    return None, "missing dependency field"


def _read_status(text: str) -> str | None:
    match = STATUS_RE.search(text)
    return match.group(1) if match else None


def _active_task_files(root: Path) -> list[Path]:
    tasks_root = root / "tasks"
    if not tasks_root.exists():
        return []
    return sorted(
        path
        for path in tasks_root.rglob("*.md")
        if not _is_archived(path, root)
    )


def _parse_roadmap(root: Path) -> tuple[dict[str, list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    roadmap_path = root / "ROADMAP.md"
    errors: list[dict[str, Any]] = []
    entries: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    if not roadmap_path.exists():
        errors.append(_error("missing_roadmap", "ROADMAP.md does not exist"))
        return {}, rows, errors

    for line_number, line in enumerate(roadmap_path.read_text(encoding="utf-8").splitlines(), 1):
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        task_id = cells[0].strip("` ")
        if not re.fullmatch(r"G[0-8]-\d{2}", task_id):
            continue
        dependencies = _expand_dependency_text(cells[2])
        row = {
            "id": task_id,
            "dependencies": dependencies,
            "line": line_number,
            "title": cells[1],
        }
        rows.append(row)
        entries[task_id].append(dependencies)

    for task_id, dependency_rows in sorted(entries.items(), key=lambda item: _id_sort_key(item[0])):
        if len(dependency_rows) > 1:
            errors.append(
                _error(
                    "duplicate_roadmap_id",
                    f"ROADMAP contains {task_id} more than once",
                    task_id=task_id,
                )
            )
    return {task_id: deps[0] for task_id, deps in entries.items()}, rows, errors


def _find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(node: str) -> None:
        if node in active_set:
            start = active.index(node)
            cycle = active[start:] + [node]
            if cycle not in cycles:
                cycles.append(cycle)
            return
        if node in visited:
            return
        active.append(node)
        active_set.add(node)
        for dependency in graph.get(node, []):
            if dependency in graph:
                visit(dependency)
        active.pop()
        active_set.remove(node)
        visited.add(node)

    for node in sorted(graph, key=_id_sort_key):
        visit(node)
    return cycles


def _check_progress(root: Path, task_statuses: dict[str, str]) -> list[dict[str, Any]]:
    progress_path = root / "PROGRESS.md"
    if not progress_path.exists():
        return []
    text = progress_path.read_text(encoding="utf-8")
    current_match = re.search(r"\|\s*当前任务\s*\|([^|]+)\|", text)
    status_match = re.search(r"\|\s*当前状态\s*\|([^|]+)\|", text)
    if not current_match or not status_match:
        return []
    current_cell = current_match.group(1).replace("*", "").replace("`", "").strip()
    if re.match(r"^(?:无|none|n/a)\b", current_cell, re.IGNORECASE):
        return []
    current_ids = TASK_ID_RE.findall(current_cell)
    current_status_match = re.search(r"[A-Z][A-Z_]+", status_match.group(1))
    if not current_ids or not current_status_match:
        return []
    task_id = current_ids[0]
    expected = current_status_match.group(0)
    actual = task_statuses.get(task_id)
    if actual is None:
        return [_error("progress_current_task_missing", f"PROGRESS points to missing task {task_id}", task_id=task_id)]
    if actual != expected:
        return [
            _error(
                "progress_status_mismatch",
                f"PROGRESS status for {task_id} is {expected}, task file is {actual}",
                task_id=task_id,
                progress_status=expected,
                task_status=actual,
            )
        ]
    return []


def run_check(root: str | Path, expected_count: int = EXPECTED_TASK_COUNT) -> dict[str, Any]:
    """Return a JSON-serializable catalog report for ``root``."""

    root_path = Path(root).resolve()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    roadmap, roadmap_rows, roadmap_errors = _parse_roadmap(root_path)
    errors.extend(roadmap_errors)

    files = _active_task_files(root_path)
    start_task_path = root_path / "START_TASK.md"
    start_task_text = start_task_path.read_text(encoding="utf-8") if start_task_path.exists() else ""
    if not start_task_path.exists():
        errors.append(_error("missing_start_task", "START_TASK.md does not exist"))
    files_by_id: dict[str, list[Path]] = defaultdict(list)
    parsed_tasks: list[dict[str, Any]] = []
    task_statuses: dict[str, str] = {}

    for path in files:
        relative = path.relative_to(root_path)
        filename_match = FILENAME_RE.fullmatch(path.name)
        filename_id = filename_match.group(1) if filename_match else None
        text = path.read_text(encoding="utf-8")
        title_match = TITLE_RE.search(text)
        body_id = title_match.group(1) if title_match else None
        status = _read_status(text)
        raw_dependencies, dependency_error = _read_first_dependency(text)
        dependencies = _expand_dependency_text(raw_dependencies or "")

        task = {
            "path": relative.as_posix(),
            "filename_id": filename_id,
            "body_id": body_id,
            "status": status,
            "dependencies": dependencies,
        }
        parsed_tasks.append(task)

        if relative.parts[1] in {f"G{i}" for i in range(2, 9)}:
            startup_match = STARTUP_SECTION_RE.search(text)
            if startup_match is None:
                errors.append(
                    _error(
                        "missing_startup_reading_list",
                        f"{relative.as_posix()} has no 启动读取清单",
                        path=relative.as_posix(),
                    )
                )
            else:
                doc_paths = sorted(set(DOC_PATH_RE.findall(startup_match.group("body"))))
                if not doc_paths:
                    errors.append(
                        _error(
                            "empty_startup_document_route",
                            f"{relative.as_posix()} startup list names no docs/*.md document",
                            path=relative.as_posix(),
                        )
                    )
                for doc_path in doc_paths:
                    if not (root_path / doc_path).is_file():
                        errors.append(
                            _error(
                                "missing_startup_document",
                                f"{relative.as_posix()} routes missing document {doc_path}",
                                path=relative.as_posix(),
                                document=doc_path,
                            )
                        )
        if start_task_text and relative.as_posix() not in start_task_text:
            errors.append(
                _error(
                    "missing_start_route",
                    f"START_TASK.md does not route {relative.as_posix()}",
                    path=relative.as_posix(),
                )
            )

        if filename_id is None:
            errors.append(
                _error(
                    "invalid_filename",
                    f"active task file has invalid filename: {relative.as_posix()}",
                    path=relative.as_posix(),
                )
            )
        else:
            files_by_id[filename_id].append(path)
            expected_phase = filename_id.split("-")[0]
            actual_parent = relative.parent.as_posix()
            if actual_parent != f"tasks/{expected_phase}":
                errors.append(
                    _error(
                        "wrong_directory",
                        f"{relative.as_posix()} is not directly under tasks/{expected_phase}/",
                        path=relative.as_posix(),
                        expected_directory=f"tasks/{expected_phase}",
                    )
                )
            if body_id != filename_id:
                errors.append(
                    _error(
                        "body_id_mismatch",
                        f"{relative.as_posix()} filename ID {filename_id} differs from body ID {body_id}",
                        path=relative.as_posix(),
                        filename_id=filename_id,
                        body_id=body_id,
                    )
                )

        if body_id is None:
            errors.append(
                _error(
                    "missing_body_id",
                    f"{relative.as_posix()} has no task ID in its first heading",
                    path=relative.as_posix(),
                )
            )
        if status is None:
            errors.append(
                _error("missing_status", f"{relative.as_posix()} has no task status", path=relative.as_posix())
            )
        elif status not in TASK_STATES:
            errors.append(
                _error(
                    "invalid_status",
                    f"{relative.as_posix()} uses illegal task status {status}",
                    path=relative.as_posix(),
                    status=status,
                )
            )
        if dependency_error:
            errors.append(
                _error("missing_dependency_field", f"{relative.as_posix()} has no direct dependency field", path=relative.as_posix())
            )

        if filename_id and status:
            task_statuses[filename_id] = status

    for task_id, paths in sorted(files_by_id.items(), key=lambda item: _id_sort_key(item[0])):
        if len(paths) <= 1:
            continue
        hashes = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
        duplicate_kind = "identical" if len(hashes) == 1 else "content_conflict"
        severity = "DECISION_REQUIRED" if duplicate_kind == "content_conflict" else "ERROR"
        errors.append(
            _error(
                "duplicate_task_file",
                f"active task ID {task_id} has {len(paths)} files ({duplicate_kind}; {severity})",
                task_id=task_id,
                paths=[path.relative_to(root_path).as_posix() for path in paths],
                duplicate_kind=duplicate_kind,
                severity=severity,
            )
        )

    roadmap_ids = set(roadmap)
    active_ids = set(files_by_id)
    if len(roadmap_rows) != expected_count:
        errors.append(
            _error(
                "roadmap_task_count",
                f"ROADMAP contains {len(roadmap_rows)} task rows; expected {expected_count}",
                actual=len(roadmap_rows),
                expected=expected_count,
            )
        )
    if len(files) != expected_count:
        errors.append(
            _error(
                "active_task_file_count",
                f"active task directories contain {len(files)} markdown files; expected {expected_count}",
                actual=len(files),
                expected=expected_count,
            )
        )

    missing = sorted(roadmap_ids - active_ids, key=_id_sort_key)
    unexpected = sorted(active_ids - roadmap_ids, key=_id_sort_key)
    if missing:
        errors.append(_error("missing_task_file", "ROADMAP IDs without an active task file", task_ids=missing))
    if unexpected:
        errors.append(_error("unexpected_task_file", "active task IDs absent from ROADMAP", task_ids=unexpected))

    graph: dict[str, list[str]] = {}
    unique_task_records = {task_id: paths[0] for task_id, paths in files_by_id.items() if len(paths) == 1}
    parsed_by_path = {root_path / task["path"]: task for task in parsed_tasks}
    for task_id, path in unique_task_records.items():
        task = parsed_by_path[path]
        graph[task_id] = task["dependencies"]
        roadmap_deps = roadmap.get(task_id)
        if roadmap_deps is None:
            continue
        if set(task["dependencies"]) != set(roadmap_deps):
            errors.append(
                _error(
                    "dependency_mismatch",
                    f"{task_id} task dependency field differs from ROADMAP",
                    task_id=task_id,
                    task_dependencies=task["dependencies"],
                    roadmap_dependencies=roadmap_deps,
                )
            )

    known_ids = roadmap_ids | active_ids
    for task_id, dependencies in graph.items():
        for dependency in dependencies:
            if dependency not in known_ids:
                errors.append(
                    _error(
                        "missing_dependency",
                        f"{task_id} depends on missing task {dependency}",
                        task_id=task_id,
                        dependency=dependency,
                    )
                )
            if dependency == task_id:
                errors.append(_error("self_dependency", f"{task_id} depends on itself", task_id=task_id))
            elif re.fullmatch(r"G[0-8]-\d{2}", dependency):
                task_phase, task_number = _id_sort_key(task_id)
                dependency_phase, dependency_number = _id_sort_key(dependency)
                if dependency_phase > task_phase:
                    errors.append(
                        _error(
                            "future_dependency",
                            f"{task_id} depends on future-phase task {dependency}",
                            task_id=task_id,
                            dependency=dependency,
                        )
                    )
                elif dependency_phase == task_phase and dependency_number >= task_number:
                    errors.append(
                        _error(
                            "future_dependency",
                            f"{task_id} depends on same-phase future task {dependency}",
                            task_id=task_id,
                            dependency=dependency,
                        )
                    )

    for cycle in _find_cycles(graph):
        errors.append(_error("dependency_cycle", "dependency cycle detected", cycle=cycle))

    errors.extend(_check_progress(root_path, task_statuses))

    phase_counts = {phase: 0 for phase in PHASES}
    for task_id in roadmap_ids:
        phase_counts[task_id.split("-")[0]] += 1
    report = {
        "ok": not errors,
        "root": str(root_path),
        "expected_task_count": expected_count,
        "summary": {
            "roadmap_rows": len(roadmap_rows),
            "roadmap_unique_ids": len(roadmap_ids),
            "active_task_files": len(files),
            "active_unique_ids": len(active_ids),
            "phase_counts": phase_counts,
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "roadmap": {
            "ids": sorted(roadmap_ids, key=_id_sort_key),
            "rows": roadmap_rows,
        },
        "tasks": sorted(parsed_tasks, key=lambda task: (_id_sort_key(task["filename_id"]) if task["filename_id"] else (99, 99), task["path"])),
        "errors": errors,
        "warnings": warnings,
    }
    return report


def _print_human(report: dict[str, Any]) -> None:
    summary = report["summary"]
    state = "PASS" if report["ok"] else "FAIL"
    print(
        f"TASK_CATALOG {state}: roadmap={summary['roadmap_rows']} "
        f"active_files={summary['active_task_files']} unique_ids={summary['active_unique_ids']} "
        f"errors={summary['error_count']}"
    )
    phases = " ".join(f"{phase}={count}" for phase, count in summary["phase_counts"].items())
    print(f"PHASE_COUNTS {phases}")
    for error in report["errors"]:
        print(f"[{error['code']}] {error['message']}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (defaults to the directory containing this script's project)",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = run_check(args.root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
