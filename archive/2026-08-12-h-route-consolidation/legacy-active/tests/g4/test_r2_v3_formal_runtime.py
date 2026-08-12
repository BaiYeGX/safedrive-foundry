from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from driving_vla.evaluation.k2_v3_artifact import (
    K2AnchorArtifactV3,
)
from driving_vla.evaluation.r2_world_ready_v3 import (
    build_core_blind_manifest_v3,
)
from driving_vla.model.checkpoint_contract import (
    validate_checkpoint_for_use,
    write_checkpoint_manifest,
)
from driving_vla.model.k2_v3_codec import (
    build_k2_v3_bundle,
)
from driving_vla.model.k2_v3_guard import (
    attach_k2_v3_guard,
)
from driving_vla.model.k2_v3_types import (
    AlternativeKind,
)
from driving_vla.model.navigation_contract import (
    build_route_context,
)
from driving_vla.runtime.k2_execution import (
    select_k2_semantic_v3,
)
from scripts.r2_v3_aggregate_formal import main as aggregate_main


class R2V3FormalRuntimeTest(unittest.TestCase):
    def test_v3_anchor_roundtrip_and_cold_force_selection(self) -> None:
        route = tuple((float(index), 0.0) for index in range(40))
        context = build_route_context(route)
        bundle = attach_k2_v3_guard(
            build_k2_v3_bundle(
                native_path_xy=route,
                route_context=context,
                ego_v=4.0,
                base_speed_mps=5.0,
                alternative_kind=AlternativeKind.TEMPORAL_YIELD,
                alternative_available=True,
                alternative_reason="registered_conflict",
                temporal_target_speed_mps=0.0,
                backbone_forward_id="forward-1",
            )
        )
        self.assertEqual(bundle.guard_status, "OK")
        artifact = K2AnchorArtifactV3(
            pair_id="pair",
            scenario_id="scenario",
            seed_id="seed_a",
            anchor_run_id="anchor",
            anchor_carla_frame=7,
            anchor_simulation_time_s=1.5,
            requested_initial_state_hash="a" * 64,
            measured_initial_state_hash="b" * 64,
            observation_fingerprint={"front_rgb_sha256": "c" * 64},
            model_checkpoint_hash="d" * 64,
            executor_config_hash="e" * 32,
            bundle=bundle,
        )
        restored = K2AnchorArtifactV3.from_json_bytes(
            artifact.to_json_bytes()
        )
        self.assertEqual(
            restored.artifact_content_hash(),
            artifact.artifact_content_hash(),
        )
        selected = select_k2_semantic_v3(
            restored.bundle,
            mode="force",
            force_index=1,
        )
        self.assertEqual(selected.candidate_index, 1)

    def test_pending_checkpoint_allows_only_registered_blind_use(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "k2_v3_formal.pt"
            checkpoint.write_bytes(b"frozen-head")
            write_checkpoint_manifest(
                root / "CHECKPOINT_STATUS.json",
                checkpoint_path=checkpoint,
                status="R2V3_FROZEN_PENDING_BLIND",
                allowed_uses=["r2v3_blind_audit"],
                forbidden_uses=["r2v3_formal", "world_campaign"],
            )
            result = validate_checkpoint_for_use(
                checkpoint,
                "r2v3_blind_audit",
            )
        self.assertTrue(result["ok"])

    def test_core_aggregator_requires_pair_and_long_evidence(self) -> None:
        manifest = build_core_blind_manifest_v3()
        checkpoint_hash = "a" * 64
        pair_rows = []
        long_rows = []
        for index, case in enumerate(manifest["cases"]):
            pair_rows.append(
                {
                    **case,
                    "status": "COMPLETED",
                    "checkpoint_sha256": checkpoint_hash,
                    "comparable": index >= 2,
                    "decisive": 2 <= index < 8,
                    "winner": (
                        (index - 2) % 2
                        if 2 <= index < 8
                        else None
                    ),
                    "pair_label": "SYMMETRIC_SAFE",
                    "candidate1_available": bool(
                        case["dual_candidate_expected"]
                    ),
                    "safe_candidate_exists": True,
                    "both_bad": False,
                    "guard_mpc_failure": False,
                }
            )
            long_rows.append(
                {
                    **case,
                    "checkpoint_sha256": checkpoint_hash,
                    "completed": True,
                    "collision": False,
                    "offroad": False,
                    "wrong_exit": False,
                    "route_completion": {"completed": True},
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "core.json"
            pair_path = root / "pairs.json"
            long_path = root / "long.json"
            output = root / "report.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            pair_path.write_text(
                json.dumps(
                    {
                        "phase": "core_blind",
                        "source_manifest_hash": manifest["manifest_hash"],
                        "rows": pair_rows,
                    }
                ),
                encoding="utf-8",
            )
            long_path.write_text(
                json.dumps(
                    {
                        "phase": "core_blind",
                        "source_manifest_hash": manifest["manifest_hash"],
                        "rows": long_rows,
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                sys,
                "argv",
                [
                    "r2_v3_aggregate_formal.py",
                    "--manifest",
                    str(manifest_path),
                    "--pair-report",
                    str(pair_path),
                    "--long-report",
                    str(long_path),
                    "--out",
                    str(output),
                ],
            ):
                with redirect_stdout(io.StringIO()):
                    exit_code = aggregate_main()
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["records"]), 12)


if __name__ == "__main__":
    unittest.main()
