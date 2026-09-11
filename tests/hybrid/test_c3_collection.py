"""Fail-closed tests for the bounded C3 supplement collector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.h6_cora_collect_sft as collector


def _args(root: Path) -> SimpleNamespace:
    release_root = root / "release"
    release_root.mkdir()
    (release_root / "release-index.json").write_text(
        json.dumps({"samples": [{"pair_id": f"old-{i}"} for i in range(3)]}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        run_id="c3-repair-fixture",
        output_root=root / "generated",
        release_root=release_root,
        plan=None,
        manifest=None,
    )


class C3CollectionTests(unittest.TestCase):
    def test_plan_is_physical_root_and_attempt_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            plan_path = collector._plan(args)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["status"], "PRE_REGISTERED")
            self.assertEqual(plan["limits"]["max_roots"], 48)
            self.assertEqual(plan["limits"]["max_attempts"], 96)
            self.assertEqual(plan["limits"]["train_roots"], 32)
            self.assertEqual(plan["limits"]["validation_roots"], 16)
            self.assertEqual(len(plan["rows"]), 48)
            self.assertEqual(len({row["root_id"] for row in plan["rows"]}), 48)
            self.assertEqual(sum(len(row["attempt_ids"]) for row in plan["rows"]), 96)
            self.assertEqual(plan["old_identity"]["root_count"], 3)

    def test_external_carla_failure_writes_only_explicit_failed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            collector._plan(args)
            resolver_report = SimpleNamespace(
                status="RETRYABLE_FAILURE",
                error_code="SERVER_NOT_RUNNING",
                error_message="fixture external crash",
            )
            fake_resolver = SimpleNamespace(preflight=lambda: resolver_report)
            with patch.object(collector, "ConnectionResolver", return_value=fake_resolver):
                output = collector._collect(args)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "CARLA_BLOCKED_EXTERNAL")
            self.assertEqual(payload["accepted_count"], 0)
            self.assertEqual(len(payload["rows"]), 48)
            self.assertTrue(all(row["accepted"] is False for row in payload["rows"]))
            self.assertTrue(all(row["attempts_started"] == 0 for row in payload["rows"]))
            self.assertTrue(all(row["status"] == "NOT_STARTED_CARLA_BLOCKED" for row in payload["rows"]))
            self.assertFalse(list(output.parent.glob("**/*.png")))

    def test_ready_server_is_not_allowed_to_bypass_shared_runtime_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            collector._plan(args)
            resolver_report = SimpleNamespace(status=collector.READY)
            fake_resolver = SimpleNamespace(preflight=lambda: resolver_report)
            with patch.object(collector, "ConnectionResolver", return_value=fake_resolver):
                output = collector._collect(args)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "CARLA_READY_COLLECTION_NOT_ARMED")
            self.assertEqual(payload["accepted_count"], 0)
            self.assertTrue(all(row["status"] == "NOT_STARTED_LIVE_CAPTURE_GUARD" for row in payload["rows"]))


if __name__ == "__main__":
    unittest.main()
