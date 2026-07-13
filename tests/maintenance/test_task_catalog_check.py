from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.maintenance.task_catalog_check import (  # noqa: E402
    EXPECTED_TASK_COUNT,
    main,
    run_check,
)


PHASE_COUNTS = {"G0": 6, "G1": 9, "G2": 5, "G3": 5, "G4": 5, "G5": 5, "G6": 5, "G7": 3, "G8": 5}


def task_ids() -> list[str]:
    return [
        f"{phase}-{number:02d}"
        for phase, count in PHASE_COUNTS.items()
        for number in range(1, count + 1)
    ]


def dependencies_for(ids: list[str]) -> dict[str, list[str]]:
    return {task_id: ([] if index == 0 else [ids[index - 1]]) for index, task_id in enumerate(ids)}


def write_task(root: Path, task_id: str, dependencies: list[str], *, body_id: str | None = None, status: str = "PENDING", suffix: str = "TASK") -> Path:
    phase = task_id.split("-")[0]
    path = root / "tasks" / phase / f"{task_id}_{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    dependency_text = "、".join(dependencies) if dependencies else "无"
    title_id = body_id or task_id
    path.write_text(
        f"# {title_id}: test task\n\n"
        f"**状态**：{status}\n"
        f"**依赖**：{dependency_text}\n\n"
        + ("## 启动读取清单\n\n1. docs/test.md\n\n" if int(phase[1:]) >= 2 else "")
        + "## 目标\n\nA test task.\n\n"
        "## 断点记录\n\nPENDING; resume from the last check.\n",
        encoding="utf-8",
    )
    return path


def write_roadmap(root: Path, ids: list[str], dependencies: dict[str, list[str]]) -> None:
    lines = ["# Test roadmap", "", "| ID | 能力包 | 直接依赖 |", "|---|---|---|"]
    for task_id in ids:
        dep_text = "、".join(dependencies[task_id]) if dependencies[task_id] else "无"
        lines.append(f"| {task_id} | test | {dep_text} |")
    (root / "ROADMAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_start_task(root: Path, ids: list[str]) -> None:
    routes = [
        f"tasks/{task_id.split('-')[0]}/{task_id}_TASK.md"
        for task_id in ids
    ]
    (root / "START_TASK.md").write_text("\n".join(routes) + "\n", encoding="utf-8")


class TaskCatalogCheckTests(unittest.TestCase):
    def make_project(self) -> tuple[Path, list[str], dict[str, list[str]]]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        (root / "docs").mkdir()
        (root / "docs" / "test.md").write_text("# Test reference\n", encoding="utf-8")
        ids = task_ids()
        dependencies = dependencies_for(ids)
        for task_id in ids:
            write_task(root, task_id, dependencies[task_id])
        write_roadmap(root, ids, dependencies)
        write_start_task(root, ids)
        return root, ids, dependencies

    def test_legal_directory(self) -> None:
        root, _, _ = self.make_project()
        report = run_check(root)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["summary"]["roadmap_rows"], EXPECTED_TASK_COUNT)
        self.assertEqual(report["summary"]["active_task_files"], EXPECTED_TASK_COUNT)

    def test_missing_task(self) -> None:
        root, ids, _ = self.make_project()
        (root / "tasks" / ids[-1].split("-")[0] / f"{ids[-1]}_TASK.md").unlink()
        report = run_check(root)
        self.assertIn("missing_task_file", {error["code"] for error in report["errors"]})

    def test_duplicate_task(self) -> None:
        root, _, dependencies = self.make_project()
        original = root / "tasks" / "G1" / "G1-01_TASK.md"
        write_task(root, "G1-01", dependencies["G1-01"], suffix="DUPLICATE")
        report = run_check(root)
        duplicate = next(error for error in report["errors"] if error["code"] == "duplicate_task_file")
        self.assertEqual(duplicate["details"]["duplicate_kind"], "identical")
        self.assertTrue(original.exists())

    def test_body_id_mismatch(self) -> None:
        root, _, _ = self.make_project()
        path = root / "tasks" / "G1" / "G1-01_TASK.md"
        path.write_text(path.read_text(encoding="utf-8").replace("# G1-01:", "# G1-02:"), encoding="utf-8")
        report = run_check(root)
        self.assertIn("body_id_mismatch", {error["code"] for error in report["errors"]})

    def test_illegal_status(self) -> None:
        root, _, _ = self.make_project()
        path = root / "tasks" / "G1" / "G1-01_TASK.md"
        path.write_text(path.read_text(encoding="utf-8").replace("**状态**：PENDING", "**状态**：BROKEN"), encoding="utf-8")
        report = run_check(root)
        self.assertIn("invalid_status", {error["code"] for error in report["errors"]})

    def test_missing_dependency(self) -> None:
        root, _, _ = self.make_project()
        path = root / "tasks" / "G1" / "G1-01_TASK.md"
        path.write_text(path.read_text(encoding="utf-8").replace("G0-06", "G0-99"), encoding="utf-8")
        report = run_check(root)
        self.assertIn("missing_dependency", {error["code"] for error in report["errors"]})

    def test_self_dependency(self) -> None:
        root, _, _ = self.make_project()
        path = root / "tasks" / "G1" / "G1-01_TASK.md"
        path.write_text(path.read_text(encoding="utf-8").replace("G0-06", "G1-01"), encoding="utf-8")
        report = run_check(root)
        self.assertIn("self_dependency", {error["code"] for error in report["errors"]})

    def test_dependency_cycle(self) -> None:
        root, _, _ = self.make_project()
        first = root / "tasks" / "G0" / "G0-01_TASK.md"
        first.write_text(first.read_text(encoding="utf-8").replace("**依赖**：无", "**依赖**：G0-02"), encoding="utf-8")
        report = run_check(root)
        self.assertIn("dependency_cycle", {error["code"] for error in report["errors"]})

    def test_archive_excluded(self) -> None:
        root, _, dependencies = self.make_project()
        write_task(root / "tasks" / "archive" / "task_duplicates", "G1-01", dependencies["G1-01"], suffix="ARCHIVED")
        report = run_check(root)
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["summary"]["active_task_files"], EXPECTED_TASK_COUNT)

    def test_json_output(self) -> None:
        root, _, _ = self.make_project()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["--root", str(root), "--json"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["summary"]["roadmap_rows"], EXPECTED_TASK_COUNT)

    def test_progress_with_no_current_task_ignores_historical_ids(self) -> None:
        root, _, _ = self.make_project()
        (root / "PROGRESS.md").write_text(
            "| 当前任务 | **无**（G1-01～G1-09 已关闭） |\n"
            "| 当前状态 | **COMPLETED_WITH_LIMITS** |\n",
            encoding="utf-8",
        )
        report = run_check(root)
        self.assertTrue(report["ok"], report["errors"])

    def test_missing_startup_reading_list(self) -> None:
        root, _, _ = self.make_project()
        path = root / "tasks" / "G2" / "G2-01_TASK.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace("## 启动读取清单\n\n1. docs/test.md\n\n", ""),
            encoding="utf-8",
        )
        report = run_check(root)
        self.assertIn("missing_startup_reading_list", {error["code"] for error in report["errors"]})

    def test_missing_start_route(self) -> None:
        root, _, _ = self.make_project()
        path = root / "START_TASK.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace("tasks/G8/G8-05_TASK.md\n", ""),
            encoding="utf-8",
        )
        report = run_check(root)
        self.assertIn("missing_start_route", {error["code"] for error in report["errors"]})

    def test_missing_startup_document(self) -> None:
        root, _, _ = self.make_project()
        path = root / "tasks" / "G2" / "G2-01_TASK.md"
        path.write_text(
            path.read_text(encoding="utf-8").replace("docs/test.md", "docs/missing.md"),
            encoding="utf-8",
        )
        report = run_check(root)
        self.assertIn("missing_startup_document", {error["code"] for error in report["errors"]})


if __name__ == "__main__":
    unittest.main()
