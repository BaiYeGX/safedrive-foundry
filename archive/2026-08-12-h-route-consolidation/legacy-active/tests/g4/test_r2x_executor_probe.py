"""R2-X1: V2 artifact + offline executor control divergence (contract_probe)."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))
sys.path.insert(0, str(ROOT))

from driving_vla.evaluation.k2_spatial_artifact import (  # noqa: E402
    artifact_from_bundle_v2,
    bundle_from_artifact_v2,
    make_dummy_observation_fingerprint,
)
from driving_vla.evaluation.paired_contract import (  # noqa: E402
    K2_ANCHOR_SCHEMA_V2,
    K2AnchorArtifactV2,
    content_hash,
)
from driving_vla.model.k2_spatial_builder import (  # noqa: E402
    build_spatial_k2_bundle_from_residuals,
    synthetic_diverse_residuals,
)
from driving_vla.model.k2_spatial_guard import attach_spatial_guard  # noqa: E402
from driving_vla.runtime.k2_execution import select_k2_spatial  # noqa: E402


def _guarded_diverse():
    native = tuple((float(i) * 1.2, 0.0) for i in range(20))
    nom, alt = synthetic_diverse_residuals(20, lateral_sign=1.0, lineage="contract_probe")
    alt["raw_d"] = [min(2.5, 0.4 * i) for i in range(20)]
    b = build_spatial_k2_bundle_from_residuals(
        native_path_xy=native,
        ego_xy=native[0],
        ego_v=5.0,
        base_speed_mps=6.0,
        residual_nominal=nom,
        residual_defensive=alt,
        observation_identity={"t": "probe"},
        backbone_forward_id="fwd-probe",
        defensive_available=True,
    )
    b = replace(b, set_diagnostics={"eligible_for_diversity": True})
    return attach_spatial_guard(b, require_diversity_if_eligible=True)


class V2ArtifactTest(unittest.TestCase):
    def test_roundtrip_preserves_path_hashes(self) -> None:
        g = _guarded_diverse()
        self.assertEqual(g.guard_status, "OK", msg=g.guard_reasons)
        fp = make_dummy_observation_fingerprint(
            k2_bundle_hash=content_hash({"n": g.native_path_hash}, nibble=16)
        )
        art = artifact_from_bundle_v2(
            g,
            pair_id="p1",
            scenario_id="s1",
            seed_id="seed_a",
            anchor_run_id="a1",
            anchor_carla_frame=0,
            anchor_simulation_time_s=0.0,
            requested_initial_state_hash="0" * 64,
            measured_initial_state_hash="0" * 64,
            observation_fingerprint=fp,
            model_checkpoint_hash="m",
            executor_config_hash="e",
            evidence_lineage="contract_probe",
        )
        self.assertEqual(art.schema_version, K2_ANCHOR_SCHEMA_V2)
        raw = art.to_json_bytes()
        art2 = K2AnchorArtifactV2.from_json_bytes(raw)
        cold = bundle_from_artifact_v2(art2)
        cold = replace(
            cold,
            guard_status=art2.guard_status,
            guard_reasons=art2.guard_reasons,
        )
        s0 = select_k2_spatial(cold, mode="force", force_index=0)
        s1 = select_k2_spatial(cold, mode="force", force_index=1)
        self.assertNotEqual(
            s0.execution_spec.spatial_path_hash, s1.execution_spec.spatial_path_hash
        )
        self.assertEqual(art2.evidence_lineage, "contract_probe")

    def test_contract_probe_lineage_not_spatial_mode_head(self) -> None:
        nom, alt = synthetic_diverse_residuals(8)
        self.assertEqual(nom["head_lineage"], "contract_probe")


class ExecutorProbeScriptTest(unittest.TestCase):
    def test_probe_script_pass(self) -> None:
        import subprocess

        script = ROOT / "scripts" / "r2x_executor_probe.py"
        out_dir = ROOT / "docs" / "runtime-evidence" / "r2x-executor-probe"
        proc = subprocess.run(
            [sys.executable, str(script), "--out", str(out_dir), "--n-ticks", "30"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
        report_path = out_dir / "executor_probe_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["evidence_lineage"], "contract_probe")
        sp = report["spatial_contract_probe"]
        self.assertTrue(sp["proposal_paths_differ"])
        self.assertTrue(sp["committed_paths_differ"])
        self.assertTrue(sp["control_seq_differ"] or sp["first_div_tick"] is not None)


if __name__ == "__main__":
    unittest.main()
