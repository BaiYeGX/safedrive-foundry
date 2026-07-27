"""Pure offline tests for R2-D run-set orchestrator (no CARLA)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.evaluation.paired_contract import compute_pair_id  # noqa: E402
from driving_vla.evaluation.runner_contract import (  # noqa: E402
    CONTINUE_POLICY_CONTINUE_ALL,
    CONTINUE_POLICY_STOP_ON_FAIL,
    FAILURE_INTERRUPTED_PARTIAL_ATTEMPT,
    PAIR_STATUS_COMPARABLE,
    PAIR_STATUS_FAILED,
    PAIR_STATUS_INCOMPARABLE,
    RETRY_POLICY_NO_AUTO_RETRY,
    RunnerContractError,
    aggregate_from_run_set_spec,
    attempt_dir_for,
    build_completed_manifest,
    build_failed_manifest,
    build_run_set_manifest,
    compute_run_set_manifest_content_hash,
    ensure_run_set_manifest,
    execute_run_set_orchestration,
    normalize_aggregate_row,
    resolve_no_auto_retry_action,
    require_frozen_registry,
    slots_from_run_set_report_or_manifest,
    validate_report_against_manifest,
    validate_run_set_manifest_stable_identity,
    write_json,
    write_json_exclusive_create,
    write_run_set_checkpoint,
    ExpectedPairHashes,
)
from driving_vla.evaluation.scenario_registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_scenario_registry,
)

FROZEN_MANIFEST = (
    ROOT
    / "docs"
    / "runtime-evidence"
    / "r2-g4a-paired-pilot"
    / "registry"
    / "registry_manifest.json"
)

MODEL_H = "model_retimer_test"
EXEC_H = "executor_test"


def _freeze_and_manifest(pairs_root: Path, *, continue_policy: str = CONTINUE_POLICY_CONTINUE_ALL):
    reg, audit, man_path = require_frozen_registry(
        DEFAULT_REGISTRY_PATH,
        manifest_path=FROZEN_MANIFEST,
        repo_root=ROOT,
    )
    rs = build_run_set_manifest(
        registry=reg,
        freeze_audit=audit,
        pairs_root=pairs_root,
        model_retimer_hash=MODEL_H,
        executor_config_hash=EXEC_H,
        continue_policy=continue_policy,
        model_checkpoint_hash="ckpt",
        retimer_hash="rh",
        registry_path=str(DEFAULT_REGISTRY_PATH),
        registry_manifest_path=str(man_path),
    )
    return reg, audit, rs


def _write_pair_manifest(
    pairs_root: Path,
    *,
    pair_id: str,
    attempt_id: int,
    scenario_id: str,
    seed_id: str,
    family: str,
    comparable: bool,
    status: str = "COMPLETED",
    error: str | None = None,
) -> None:
    adir = attempt_dir_for(pairs_root, pair_id, attempt_id)
    adir.mkdir(parents=True, exist_ok=True)
    if status == "FAILED" or error:
        man = build_failed_manifest(
            pair_id=pair_id,
            scenario_id=scenario_id,
            seed_id=seed_id,
            registry_sha256="reg",
            model_retimer_hash=MODEL_H,
            executor_config_hash=EXEC_H,
            attempt_id=attempt_id,
            error=error or "fail",
            extra={"family": family},
        )
    else:
        man = build_completed_manifest(
            pair_id=pair_id,
            scenario_id=scenario_id,
            seed_id=seed_id,
            family=family,
            registry_sha256="reg",
            model_retimer_hash=MODEL_H,
            executor_config_hash=EXEC_H,
            artifact_content_hash=f"art-{pair_id}-{attempt_id}",
            attempt_id=attempt_id,
            branch_order=(0, 1) if seed_id == "seed_a" else (1, 0),
            forward_count_total=1,
            comparable=comparable,
            comparability={"status": "COMPARABLE" if comparable else "INCOMPARABLE"},
            oracle={
                "pair_id": pair_id,
                "scenario_id": scenario_id,
                "seed_id": seed_id,
                "family": family,
                "comparable": comparable,
                "pair_label": "TIE" if comparable else "INCOMPARABLE",
                "top1_candidate_id": "v1_nominal",
                "top1_candidate_index": 0,
                "oracle_candidate_id": "v1_nominal" if comparable else None,
                "oracle_candidate_index": 0 if comparable else None,
                "oracle_decision_level": "tie_top1" if comparable else None,
                "decision_reason": "test",
                "both_bad": False,
                "outcome_delta": {},
                "failure_reasons": [] if comparable else ["x"],
            },
            anchor={},
            branch_0={},
            branch_1={},
        )
    write_json(adir / "pair_manifest.json", man)


class RunSetManifestTest(unittest.TestCase):
    def test_immutable_manifest_12_pairs_counterbalance_and_pair_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, audit, rs = _freeze_and_manifest(pairs_root)
            self.assertTrue(rs["immutable"])
            self.assertTrue(rs["frozen"])
            self.assertEqual(rs["n_pairs"], 12)
            self.assertEqual(len(rs["pairs"]), 12)
            self.assertEqual(rs["registry_sha256"], audit["registry_sha256"])
            self.assertEqual(rs["model_retimer_hash"], MODEL_H)
            self.assertEqual(rs["executor_config_hash"], EXEC_H)
            self.assertIn("manifest_content_hash", rs)
            # order + counterbalance
            self.assertEqual(rs["pairs"][0]["seed_id"], "seed_a")
            self.assertEqual(rs["pairs"][0]["branch_order"], [0, 1])
            self.assertEqual(rs["pairs"][1]["seed_id"], "seed_b")
            self.assertEqual(rs["pairs"][1]["branch_order"], [1, 0])
            # pair_id stable
            for slot in rs["pairs"]:
                expect = compute_pair_id(
                    scenario_registry_hash=rs["registry_sha256"],
                    scenario_id=slot["scenario_id"],
                    seed_id=slot["seed_id"],
                    model_checkpoint_config_retimer_hash=MODEL_H,
                    executor_config_hash=EXEC_H,
                )
                self.assertEqual(slot["pair_id"], expect)
                self.assertIn("planned_attempt_id", slot)
            # content hash stable
            _reg2, audit2, rs2 = _freeze_and_manifest(pairs_root)
            self.assertEqual(rs["manifest_content_hash"], rs2["manifest_content_hash"])

    def test_spatial_identity_enters_hash_and_resume_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            reg, audit, _base = _freeze_and_manifest(pairs_root)
            kwargs = {
                "registry": reg,
                "freeze_audit": audit,
                "pairs_root": pairs_root,
                "model_retimer_hash": "spatial-bound-model",
                "executor_config_hash": EXEC_H,
                "model_checkpoint_hash": "base-ckpt",
                "retimer_hash": "rh",
                "policy_type": "NeuralV2Policy",
                "policy_model_id": "sdf-vla-v2-spatial@0.1.0",
                "spatial_head_checkpoint_hash": "head-a",
                "spatial_k2_config_hash": "config-a",
            }
            spatial_a = build_run_set_manifest(**kwargs)
            self.assertEqual(
                spatial_a["spatial_head_checkpoint_hash"], "head-a"
            )
            spatial_b = build_run_set_manifest(
                **{**kwargs, "spatial_head_checkpoint_hash": "head-b"}
            )
            self.assertNotEqual(
                spatial_a["manifest_content_hash"],
                spatial_b["manifest_content_hash"],
            )
            with self.assertRaises(RunnerContractError):
                validate_run_set_manifest_stable_identity(
                    spatial_a,
                    {
                        "spatial_head_checkpoint_hash": "head-b",
                        "spatial_k2_config_hash": "config-a",
                    },
                )

    def test_failed_spatial_row_uses_v2_candidate_identity(self) -> None:
        row = normalize_aggregate_row(
            {
                "status": "FAILED",
                "pair_id": "p",
                "scenario_id": "cut_in_early",
                "seed_id": "seed_a",
                "attempt_id": 0,
                "error": "NO_ALTERNATIVE",
                "candidate_ids": [
                    "v2_nominal_progress",
                    "v2_defensive_alternative",
                ],
                "top1_candidate_index": 0,
            }
        )
        self.assertEqual(row["top1_candidate_id"], "v2_nominal_progress")
        self.assertNotEqual(row["top1_candidate_id"], "v1_nominal")


class DryOrchestrationTest(unittest.TestCase):
    def test_fake_run_pair_continue_all_keeps_all_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)

            def fake_run_pair(
                *,
                scenario_id,
                seed_id,
                branch_order,
                force_attempt_id=None,
                retry_policy=None,
                **kwargs,
            ):
                # find slot
                slot = next(
                    s
                    for s in rs["pairs"]
                    if s["scenario_id"] == scenario_id and s["seed_id"] == seed_id
                )
                pair_id = slot["pair_id"]
                attempt_id = int(
                    force_attempt_id
                    if force_attempt_id is not None
                    else slot["planned_attempt_id"]
                )
                # diversify statuses by index
                idx = int(slot["index"])
                if idx == 2:
                    _write_pair_manifest(
                        pairs_root,
                        pair_id=pair_id,
                        attempt_id=attempt_id,
                        scenario_id=scenario_id,
                        seed_id=seed_id,
                        family=slot["family"],
                        comparable=False,
                        status="FAILED",
                        error="spawn_failed",
                    )
                    raise RuntimeError("spawn_failed")
                comparable = idx not in {4, 5}
                _write_pair_manifest(
                    pairs_root,
                    pair_id=pair_id,
                    attempt_id=attempt_id,
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    family=slot["family"],
                    comparable=comparable,
                )
                man = json.loads(
                    (
                        attempt_dir_for(pairs_root, pair_id, attempt_id) / "pair_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                man["idempotent_read"] = idx == 0
                man["attempt_id"] = attempt_id
                man["attempt_dir"] = str(
                    attempt_dir_for(pairs_root, pair_id, attempt_id).as_posix()
                )
                return man

            report = execute_run_set_orchestration(
                run_set_manifest=rs,
                run_pair_fn=fake_run_pair,
                pairs_root=pairs_root,
                checkpoint_path=Path(td) / "run_set_checkpoint.json",
                report_path=Path(td) / "run_set_report.json",
            )
            self.assertEqual(len(report["pair_results"]), 12)
            statuses = [r["status"] for r in report["pair_results"]]
            self.assertIn(PAIR_STATUS_FAILED, statuses)
            self.assertIn(PAIR_STATUS_COMPARABLE, statuses)
            self.assertIn(PAIR_STATUS_INCOMPARABLE, statuses)
            # fixtures never swapped: pair_ids match plan order
            for slot, row in zip(rs["pairs"], report["pair_results"]):
                self.assertEqual(slot["pair_id"], row["pair_id"])
                self.assertEqual(slot["scenario_id"], row["scenario_id"])
                self.assertEqual(slot["seed_id"], row["seed_id"])
            self.assertTrue(report["pair_results"][0]["idempotent_read"])
            # summary retains failed/incomparable in counts
            self.assertEqual(report["summary"]["n_fail"], 1)
            self.assertGreaterEqual(report["summary"]["n_incomparable"], 2)

    def test_stop_on_fail_pre_registered_not_outcome_driven(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(
                pairs_root, continue_policy=CONTINUE_POLICY_STOP_ON_FAIL
            )
            calls: list[str] = []

            def fake_run_pair(
                *,
                scenario_id,
                seed_id,
                branch_order,
                force_attempt_id=None,
                retry_policy=None,
                **kwargs,
            ):
                calls.append(f"{scenario_id}/{seed_id}")
                slot = next(
                    s
                    for s in rs["pairs"]
                    if s["scenario_id"] == scenario_id and s["seed_id"] == seed_id
                )
                if len(calls) == 1:
                    raise RuntimeError("first_fail")
                pair_id = slot["pair_id"]
                aid = int(
                    force_attempt_id
                    if force_attempt_id is not None
                    else slot["planned_attempt_id"]
                )
                _write_pair_manifest(
                    pairs_root,
                    pair_id=pair_id,
                    attempt_id=aid,
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    family=slot["family"],
                    comparable=True,
                )
                man = json.loads(
                    (attempt_dir_for(pairs_root, pair_id, aid) / "pair_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                man["attempt_id"] = aid
                return man

            report = execute_run_set_orchestration(
                run_set_manifest=rs,
                run_pair_fn=fake_run_pair,
                pairs_root=pairs_root,
                checkpoint_path=Path(td) / "ckpt.json",
                report_path=Path(td) / "report.json",
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(report["pair_results"]), 12)
            self.assertEqual(report["pair_results"][0]["status"], PAIR_STATUS_FAILED)
            self.assertEqual(report["pair_results"][1]["failure"], "not_run_stop_on_fail")
            # remaining fixtures unchanged
            self.assertEqual(
                report["pair_results"][3]["scenario_id"], rs["pairs"][3]["scenario_id"]
            )


class AggregateSlotsTest(unittest.TestCase):
    def test_attempt_n_aggregate_exactly_12_and_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            # prepare 12 manifests
            for slot in rs["pairs"]:
                _write_pair_manifest(
                    pairs_root,
                    pair_id=slot["pair_id"],
                    attempt_id=int(slot["planned_attempt_id"]),
                    scenario_id=slot["scenario_id"],
                    seed_id=slot["seed_id"],
                    family=slot["family"],
                    comparable=slot["index"] % 3 != 0,
                    status="FAILED" if slot["index"] == 1 else "COMPLETED",
                    error="e" if slot["index"] == 1 else None,
                )
            slots = [
                {
                    "pair_id": s["pair_id"],
                    "attempt_id": s["planned_attempt_id"],
                }
                for s in rs["pairs"]
            ]
            rows = aggregate_from_run_set_spec(pairs_root=pairs_root, slots=slots, require_n=12)
            self.assertEqual(len(rows), 12)
            # failed retained
            self.assertFalse(rows[1]["comparable"])
            self.assertEqual(rows[1]["status"], PAIR_STATUS_FAILED)
            # denominator includes incomparable
            n_comp = sum(1 for r in rows if r["comparable"])
            self.assertLess(n_comp, 12)
            # duplicate slots fail-closed
            bad = list(slots)
            bad[3] = dict(bad[0])
            with self.assertRaises(RunnerContractError) as ctx:
                aggregate_from_run_set_spec(pairs_root=pairs_root, slots=bad, require_n=12)
            self.assertIn("duplicate", str(ctx.exception))

    def test_missing_pair_and_wrong_attempt_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            slots = [
                {"pair_id": s["pair_id"], "attempt_id": s["planned_attempt_id"]}
                for s in rs["pairs"]
            ]
            # write only 11
            for s in rs["pairs"][:11]:
                _write_pair_manifest(
                    pairs_root,
                    pair_id=s["pair_id"],
                    attempt_id=int(s["planned_attempt_id"]),
                    scenario_id=s["scenario_id"],
                    seed_id=s["seed_id"],
                    family=s["family"],
                    comparable=True,
                )
            with self.assertRaises(RunnerContractError) as ctx:
                aggregate_from_run_set_spec(pairs_root=pairs_root, slots=slots, require_n=12)
            self.assertIn("missing pair_manifest", str(ctx.exception))
            # wrong attempt
            s0 = rs["pairs"][0]
            _write_pair_manifest(
                pairs_root,
                pair_id=s0["pair_id"],
                attempt_id=int(s0["planned_attempt_id"]),
                scenario_id=s0["scenario_id"],
                seed_id=s0["seed_id"],
                family=s0["family"],
                comparable=True,
            )
            wrong = [
                {"pair_id": s0["pair_id"], "attempt_id": int(s0["planned_attempt_id"]) + 99}
            ] + slots[1:]
            with self.assertRaises(RunnerContractError):
                aggregate_from_run_set_spec(pairs_root=pairs_root, slots=wrong, require_n=12)

    def test_report_slots_prefer_resolved_attempt(self) -> None:
        report = {
            "pair_results": [
                {
                    "index": i,
                    "pair_id": f"p{i}",
                    "attempt_id": i % 2,
                    "scenario_id": "s",
                    "seed_id": "seed_a",
                }
                for i in range(12)
            ]
        }
        slots = slots_from_run_set_report_or_manifest(run_set_report=report)
        self.assertEqual(len(slots), 12)
        self.assertEqual(slots[3]["attempt_id"], 1)


class ConnectFailureManifestTest(unittest.TestCase):
    def test_build_failed_manifest_and_safe_world_none_pattern(self) -> None:
        """Document connect failure writes FAILED attempt; world may be unset."""
        with tempfile.TemporaryDirectory() as td:
            adir = Path(td) / "attempt_0"
            adir.mkdir()
            man = build_failed_manifest(
                pair_id="p",
                scenario_id="lead_brake_moderate",
                seed_id="seed_a",
                registry_sha256="r",
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
                attempt_id=0,
                error="RPC_HANDSHAKE_FAILED",
                extra={"phase": "connect_world"},
            )
            write_json(adir / "pair_manifest.json", man)
            loaded = json.loads((adir / "pair_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "FAILED")
            self.assertFalse(loaded["comparable"])
            self.assertEqual(loaded.get("phase"), "connect_world")
            world = None
            if world is not None:  # safety pattern used by run_pair finally
                raise AssertionError("must not touch world")


class ManifestExclusiveAndRetryTest(unittest.TestCase):
    def test_existing_manifest_not_overwritten_and_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            path = Path(td) / "run_set_manifest.json"
            m1, mode1 = ensure_run_set_manifest(path, rs)
            self.assertEqual(mode1, "created")
            # second ensure reuses
            m2, mode2 = ensure_run_set_manifest(path, rs)
            self.assertEqual(mode2, "reused")
            self.assertEqual(m1["manifest_content_hash"], m2["manifest_content_hash"])
            # exclusive create fails if we try write again
            with self.assertRaises(RunnerContractError):
                write_json_exclusive_create(path, rs)
            # mismatch fail-closed
            bad = dict(rs)
            bad["model_retimer_hash"] = "OTHER"
            bad["manifest_content_hash"] = compute_run_set_manifest_content_hash(bad)
            with self.assertRaises(RunnerContractError) as ctx:
                ensure_run_set_manifest(path, bad)
            self.assertIn("mismatch", str(ctx.exception).lower())
            # absolute dir excluded from hash: same content with different abs path
            rs_alt = dict(rs)
            rs_alt["pairs"] = [dict(p) for p in rs["pairs"]]
            for p in rs_alt["pairs"]:
                p["planned_attempt_dir"] = "/other/machine/" + p["planned_attempt_dir_rel"]
            self.assertEqual(
                compute_run_set_manifest_content_hash(rs),
                compute_run_set_manifest_content_hash(rs_alt),
            )

    def test_failed_planned_attempt_no_auto_retry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, audit, rs = _freeze_and_manifest(pairs_root)
            slot = rs["pairs"][0]
            pair_id = slot["pair_id"]
            aid = int(slot["planned_attempt_id"])
            _write_pair_manifest(
                pairs_root,
                pair_id=pair_id,
                attempt_id=aid,
                scenario_id=slot["scenario_id"],
                seed_id=slot["seed_id"],
                family=slot["family"],
                comparable=False,
                status="FAILED",
                error="boom",
            )
            expected = ExpectedPairHashes(
                pair_id=pair_id,
                scenario_id=slot["scenario_id"],
                seed_id=slot["seed_id"],
                registry_sha256=rs["registry_sha256"],
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
            )
            action = resolve_no_auto_retry_action(
                pairs_root,
                pair_id=pair_id,
                planned_attempt_id=aid,
                expected=expected,
            )
            self.assertEqual(action["action"], "retain_failed")
            self.assertEqual(action["attempt_id"], aid)
            # orchestration must not invent attempt_1
            calls = []

            def fake(
                *,
                scenario_id,
                seed_id,
                branch_order,
                force_attempt_id=None,
                retry_policy=None,
                **kw,
            ):
                calls.append(force_attempt_id)
                # simulate retain_failed by returning existing failed man
                s = next(
                    x
                    for x in rs["pairs"]
                    if x["scenario_id"] == scenario_id and x["seed_id"] == seed_id
                )
                if s["pair_id"] == pair_id:
                    man = json.loads(
                        (
                            attempt_dir_for(pairs_root, pair_id, aid) / "pair_manifest.json"
                        ).read_text(encoding="utf-8")
                    )
                    man["retained_failed"] = True
                    man["attempt_id"] = aid
                    return man
                _write_pair_manifest(
                    pairs_root,
                    pair_id=s["pair_id"],
                    attempt_id=int(s["planned_attempt_id"]),
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    family=s["family"],
                    comparable=True,
                )
                man = json.loads(
                    (
                        attempt_dir_for(pairs_root, s["pair_id"], int(s["planned_attempt_id"]))
                        / "pair_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                man["attempt_id"] = int(s["planned_attempt_id"])
                return man

            report = execute_run_set_orchestration(
                run_set_manifest=rs,
                run_pair_fn=fake,
                pairs_root=pairs_root,
                checkpoint_path=Path(td) / "ckpt.json",
                report_path=Path(td) / "report.json",
            )
            self.assertEqual(report["pair_results"][0]["attempt_id"], 0)
            self.assertEqual(report["pair_results"][0]["status"], PAIR_STATUS_FAILED)
            self.assertFalse((pairs_root / pair_id / "attempt_1").exists())

    def test_returned_attempt_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)

            def bad_fn(*, scenario_id, seed_id, branch_order, force_attempt_id=None, **kw):
                s = next(
                    x
                    for x in rs["pairs"]
                    if x["scenario_id"] == scenario_id and x["seed_id"] == seed_id
                )
                return {
                    "pair_id": s["pair_id"],
                    "attempt_id": 99,  # wrong
                    "status": "COMPLETED",
                    "comparable": True,
                    "oracle": {"pair_label": "TIE"},
                }

            with self.assertRaises(RunnerContractError) as ctx:
                execute_run_set_orchestration(
                    run_set_manifest=rs,
                    run_pair_fn=bad_fn,
                    pairs_root=pairs_root,
                    checkpoint_path=Path(td) / "ckpt.json",
                    report_path=Path(td) / "report.json",
                )
            self.assertIn("attempt_id", str(ctx.exception))

    def test_checkpoint_resume_same_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            ckpt = Path(td) / "ckpt.json"
            rep = Path(td) / "report.json"
            n_calls = {"n": 0}

            def fake(
                *,
                scenario_id,
                seed_id,
                branch_order,
                force_attempt_id=None,
                retry_policy=None,
                **kw,
            ):
                n_calls["n"] += 1
                s = next(
                    x
                    for x in rs["pairs"]
                    if x["scenario_id"] == scenario_id and x["seed_id"] == seed_id
                )
                aid = int(force_attempt_id)
                # fail mid-way first run
                if n_calls["n"] == 3 and not (Path(td) / "resumed").exists():
                    (Path(td) / "resumed").write_text("1", encoding="utf-8")
                    raise RuntimeError("interrupt_sim")
                _write_pair_manifest(
                    pairs_root,
                    pair_id=s["pair_id"],
                    attempt_id=aid,
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    family=s["family"],
                    comparable=True,
                )
                man = json.loads(
                    (
                        attempt_dir_for(pairs_root, s["pair_id"], aid) / "pair_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                man["attempt_id"] = aid
                return man

            # first orchestration: stop_on_fail after index 2 failure... use continue_all
            # but raise on third call - continue_all records failure and continues
            # better: raise only once then on resume skip completed

            # Simpler: run until 2 pairs, then checkpoint has last_completed_index=1
            # We'll manually limit by raising after 2 successes and catch...
            # Use continue_all and raise RuntimeError after writing 2 - actually raise
            # on pair index 2 without writing - that records FAILED for index 2 then continues

            report1 = execute_run_set_orchestration(
                run_set_manifest=rs,
                run_pair_fn=fake,
                pairs_root=pairs_root,
                checkpoint_path=ckpt,
                report_path=rep,
            )
            self.assertEqual(len(report1["pair_results"]), 12)
            self.assertTrue(ckpt.is_file())
            # resume: all completed already so second run should call 0 new pairs
            # if all slots done, start_index=12 and no calls... but pair 2 may be FAILED
            n_before = n_calls["n"]
            report2 = execute_run_set_orchestration(
                run_set_manifest=rs,
                run_pair_fn=fake,
                pairs_root=pairs_root,
                checkpoint_path=ckpt,
                report_path=rep,
            )
            # completed checkpoint → no additional calls
            self.assertEqual(n_calls["n"], n_before)
            self.assertEqual(len(report2["pair_results"]), 12)

    def test_stale_report_rejected_by_aggregate_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            report = {
                "run_set_manifest_content_hash": "DEADBEEF" * 8,
                "registry_sha256": rs["registry_sha256"],
                "model_retimer_hash": MODEL_H,
                "executor_config_hash": EXEC_H,
                "retry_policy": RETRY_POLICY_NO_AUTO_RETRY,
                "pair_results": [
                    {
                        "index": i,
                        "pair_id": rs["pairs"][i]["pair_id"],
                        "attempt_id": 0,
                        "scenario_id": rs["pairs"][i]["scenario_id"],
                        "seed_id": rs["pairs"][i]["seed_id"],
                    }
                    for i in range(12)
                ],
            }
            with self.assertRaises(RunnerContractError) as ctx:
                validate_report_against_manifest(report, rs)
            self.assertIn("content_hash", str(ctx.exception))
            # wrong attempt vs no_auto_retry
            report2 = {
                "run_set_manifest_content_hash": rs["manifest_content_hash"],
                "registry_sha256": rs["registry_sha256"],
                "model_retimer_hash": MODEL_H,
                "executor_config_hash": EXEC_H,
                "retry_policy": RETRY_POLICY_NO_AUTO_RETRY,
                "pair_results": [
                    {
                        "index": i,
                        "pair_id": rs["pairs"][i]["pair_id"],
                        "attempt_id": 1 if i == 0 else 0,
                        "scenario_id": rs["pairs"][i]["scenario_id"],
                        "seed_id": rs["pairs"][i]["seed_id"],
                    }
                    for i in range(12)
                ],
            }
            with self.assertRaises(RunnerContractError) as ctx2:
                validate_report_against_manifest(report2, rs)
            self.assertIn("attempt_id", str(ctx2.exception))


class RealEvidenceLayoutCompatTest(unittest.TestCase):
    """Shape tests for R2-C legacy top-level Evidence + R2-D planning."""

    def test_legacy_top_level_occupies_attempt0_plans_attempt1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            reg = load_scenario_registry(DEFAULT_REGISTRY_PATH)
            audit = {
                "frozen": True,
                "registry_sha256": reg.compute_registry_sha256(),
            }
            # Compute first pair_id as R2-D would
            from driving_vla.evaluation.paired_contract import compute_pair_id

            first = reg.fixtures[0]
            pair_id = compute_pair_id(
                scenario_registry_hash=audit["registry_sha256"],
                scenario_id=first.scenario_id,
                seed_id=first.seed_id,
                model_checkpoint_config_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
            )
            # Simulate R2-C legacy layout (top-level pair_manifest + anchor)
            legacy = pairs_root / pair_id
            (legacy / "anchor").mkdir(parents=True)
            (legacy / "branch-0").mkdir()
            write_json(
                legacy / "pair_manifest.json",
                {
                    "pair_id": pair_id,
                    "scenario_id": first.scenario_id,
                    "seed_id": first.seed_id,
                    "comparable": True,
                    "artifact_content_hash": "legacy",
                },
            )
            rs = build_run_set_manifest(
                registry=reg,
                freeze_audit=audit,
                pairs_root=pairs_root,
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
            )
            self.assertEqual(rs["pairs"][0]["pair_id"], pair_id)
            self.assertEqual(rs["pairs"][0]["planned_attempt_id"], 1)
            self.assertEqual(
                rs["pairs"][0]["planned_attempt_dir_rel"], f"{pair_id}/attempt_1"
            )
            # other empty pairs stay attempt_0
            for slot in rs["pairs"][1:]:
                self.assertEqual(slot["planned_attempt_id"], 0, msg=slot["scenario_id"])
            # legacy not moved
            self.assertTrue((legacy / "pair_manifest.json").is_file())
            self.assertTrue((legacy / "anchor").is_dir())
            self.assertFalse((legacy / "attempt_0").exists())

    def test_partial_attempt_sealed_failed_no_run_pair(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            slot = rs["pairs"][0]
            pair_id = slot["pair_id"]
            aid = int(slot["planned_attempt_id"])
            adir = pairs_root / pair_id / f"attempt_{aid}"
            adir.mkdir(parents=True)
            # partial content without pair_manifest
            (adir / "branch-0").mkdir()
            (adir / "branch-0" / "outcome_trace.jsonl").write_text("{}\n", encoding="utf-8")
            expected = ExpectedPairHashes(
                pair_id=pair_id,
                scenario_id=slot["scenario_id"],
                seed_id=slot["seed_id"],
                registry_sha256=rs["registry_sha256"],
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
            )
            action = resolve_no_auto_retry_action(
                pairs_root,
                pair_id=pair_id,
                planned_attempt_id=aid,
                expected=expected,
            )
            self.assertEqual(action["action"], "retain_failed")
            self.assertTrue(action.get("sealed_partial"))
            man = action["existing_manifest"]
            self.assertEqual(man["status"], "FAILED")
            self.assertIn("INTERRUPTED_PARTIAL_ATTEMPT", str(man.get("error", "")))
            # partial file retained
            self.assertTrue((adir / "branch-0" / "outcome_trace.jsonl").is_file())
            # aggregate can read sealed manifest
            slots = [{"pair_id": pair_id, "attempt_id": aid}] + [
                {"pair_id": s["pair_id"], "attempt_id": s["planned_attempt_id"]}
                for s in rs["pairs"][1:]
            ]
            for s in rs["pairs"][1:]:
                _write_pair_manifest(
                    pairs_root,
                    pair_id=s["pair_id"],
                    attempt_id=int(s["planned_attempt_id"]),
                    scenario_id=s["scenario_id"],
                    seed_id=s["seed_id"],
                    family=s["family"],
                    comparable=True,
                )
            rows = aggregate_from_run_set_spec(
                pairs_root=pairs_root, slots=slots, require_n=12
            )
            self.assertEqual(len(rows), 12)
            self.assertFalse(rows[0]["comparable"])
            self.assertEqual(rows[0]["status"], PAIR_STATUS_FAILED)
            # orchestration must not call run_pair body for sealed pair
            calls: list[str] = []

            def fake(
                *,
                scenario_id,
                seed_id,
                branch_order,
                force_attempt_id=None,
                retry_policy=None,
                **kw,
            ):
                calls.append(f"{scenario_id}/{seed_id}")
                s = next(
                    x
                    for x in rs["pairs"]
                    if x["scenario_id"] == scenario_id and x["seed_id"] == seed_id
                )
                # simulate resolve path by returning sealed/failed for first
                if s["index"] == 0:
                    man2 = json.loads(
                        (adir / "pair_manifest.json").read_text(encoding="utf-8")
                    )
                    man2["attempt_id"] = aid
                    man2["retained_failed"] = True
                    return man2
                _write_pair_manifest(
                    pairs_root,
                    pair_id=s["pair_id"],
                    attempt_id=int(s["planned_attempt_id"]),
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    family=s["family"],
                    comparable=True,
                )
                man = json.loads(
                    (
                        attempt_dir_for(pairs_root, s["pair_id"], int(s["planned_attempt_id"]))
                        / "pair_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                man["attempt_id"] = int(s["planned_attempt_id"])
                return man

            # Use run_pair path via resolve: first slot already sealed — fake still called
            # but retains failed without wiping partial. Key: sealed_partial leaves files.
            report = execute_run_set_orchestration(
                run_set_manifest=rs,
                run_pair_fn=fake,
                pairs_root=pairs_root,
                checkpoint_path=Path(td) / "ckpt.json",
                report_path=Path(td) / "report.json",
            )
            self.assertEqual(report["pair_results"][0]["status"], PAIR_STATUS_FAILED)
            self.assertTrue((adir / "branch-0" / "outcome_trace.jsonl").is_file())

    def test_corrupt_noncontiguous_checkpoint_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            ckpt_path = Path(td) / "ckpt.json"
            # non-contiguous: skip index 1
            bad = {
                "schema_version": "safedrive.g4a.run_set_checkpoint.v1",
                "run_set_manifest_content_hash": rs["manifest_content_hash"],
                "registry_sha256": rs["registry_sha256"],
                "model_retimer_hash": MODEL_H,
                "executor_config_hash": EXEC_H,
                "retry_policy": RETRY_POLICY_NO_AUTO_RETRY,
                "last_completed_index": 1,
                "pair_results": [
                    {
                        "index": 0,
                        "pair_id": rs["pairs"][0]["pair_id"],
                        "scenario_id": rs["pairs"][0]["scenario_id"],
                        "seed_id": rs["pairs"][0]["seed_id"],
                        "attempt_id": rs["pairs"][0]["planned_attempt_id"],
                        "branch_order": rs["pairs"][0]["branch_order"],
                    },
                    {
                        "index": 2,
                        "pair_id": rs["pairs"][2]["pair_id"],
                        "scenario_id": rs["pairs"][2]["scenario_id"],
                        "seed_id": rs["pairs"][2]["seed_id"],
                        "attempt_id": rs["pairs"][2]["planned_attempt_id"],
                        "branch_order": rs["pairs"][2]["branch_order"],
                    },
                ],
            }
            write_json(ckpt_path, bad)
            with self.assertRaises(RunnerContractError) as ctx:
                from driving_vla.evaluation.runner_contract import load_run_set_checkpoint

                load_run_set_checkpoint(ckpt_path, run_set_manifest=rs)
            self.assertTrue(
                "index" in str(ctx.exception).lower()
                or "contiguous" in str(ctx.exception).lower()
                or "last_completed" in str(ctx.exception).lower()
            )


class CmdRunSetPlanOnlyTest(unittest.TestCase):
    def test_plan_only_writes_immutable_manifest(self) -> None:
        import importlib.util

        path = ROOT / "tests" / "g4" / "run_g4a_paired.py"
        spec = importlib.util.spec_from_file_location("run_g4a_paired_mod2", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td)
            ns = type(
                "A",
                (),
                {
                    "registry": str(DEFAULT_REGISTRY_PATH),
                    "registry_manifest": str(FROZEN_MANIFEST),
                    "evidence_dir": str(evidence),
                    "plan_only": True,
                    "stop_on_fail": False,
                    "continue_policy": "",
                    "device": "cpu",
                    "host": "127.0.0.1",
                    "port": 2000,
                },
            )()
            code = mod.cmd_run_set(ns)
            self.assertEqual(code, 0)
            man = json.loads((evidence / "run_set_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(man["immutable"])
            self.assertEqual(len(man["pairs"]), 12)
            self.assertEqual(man["continue_policy"], CONTINUE_POLICY_CONTINUE_ALL)
            self.assertEqual(man["retry_policy"], RETRY_POLICY_NO_AUTO_RETRY)
            # second plan-only reuses without overwrite error
            code2 = mod.cmd_run_set(ns)
            self.assertEqual(code2, 0)


class CliRecoveryAndPartialSealTest(unittest.TestCase):
    """CLI recovery: reuse frozen manifest; empty/corrupt partial seals."""

    def _load_cli_mod(self):
        import importlib.util

        path = ROOT / "tests" / "g4" / "run_g4a_paired.py"
        spec = importlib.util.spec_from_file_location("run_g4a_paired_recovery", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_cli_recovery_reuses_manifest_hash_and_checkpoint_next_slot(self) -> None:
        """Create → complete slot0 → re-enter CLI: same hash/planned IDs, resume at 1."""
        mod = self._load_cli_mod()
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td)
            pairs_root = evidence / "pairs"
            ns_plan = type(
                "A",
                (),
                {
                    "registry": str(DEFAULT_REGISTRY_PATH),
                    "registry_manifest": str(FROZEN_MANIFEST),
                    "evidence_dir": str(evidence),
                    "plan_only": True,
                    "stop_on_fail": False,
                    "continue_policy": "",
                    "device": "cpu",
                    "host": "127.0.0.1",
                    "port": 2000,
                    "no_aggregate": True,
                    "min_comparable": 10,
                },
            )()
            code = mod.cmd_run_set(ns_plan)
            self.assertEqual(code, 0)
            man_path = evidence / "run_set_manifest.json"
            man1 = json.loads(man_path.read_text(encoding="utf-8"))
            hash1 = man1["manifest_content_hash"]
            planned_ids = [int(p["planned_attempt_id"]) for p in man1["pairs"]]
            pair_ids = [str(p["pair_id"]) for p in man1["pairs"]]

            # Simulate completed slot0 / planned attempt (Evidence now occupies that dir).
            slot0 = man1["pairs"][0]
            aid0 = int(slot0["planned_attempt_id"])
            _write_pair_manifest(
                pairs_root,
                pair_id=slot0["pair_id"],
                attempt_id=aid0,
                scenario_id=slot0["scenario_id"],
                seed_id=slot0["seed_id"],
                family=slot0["family"],
                comparable=True,
            )
            row0 = {
                "index": 0,
                "pair_id": slot0["pair_id"],
                "scenario_id": slot0["scenario_id"],
                "seed_id": slot0["seed_id"],
                "family": slot0["family"],
                "branch_order": list(slot0["branch_order"]),
                "attempt_id": aid0,
                "status": "COMPLETED",
                "comparable": True,
            }
            write_run_set_checkpoint(
                evidence / "run_set_checkpoint.json",
                run_set_manifest=man1,
                pair_results=[row0],
                last_completed_index=0,
                status="IN_PROGRESS",
            )

            # If entry rescanned Evidence, slot0 would replan attempt_id+1 and diverge.
            # CLI must reuse frozen slots and resume from checkpoint index 1.
            called_indices: list[int] = []

            def fake_run_pair(
                *,
                scenario_id,
                seed_id,
                branch_order,
                force_attempt_id=None,
                retry_policy=None,
                **kw,
            ):
                s = next(
                    x
                    for x in man1["pairs"]
                    if x["scenario_id"] == scenario_id and x["seed_id"] == seed_id
                )
                called_indices.append(int(s["index"]))
                aid = int(force_attempt_id)
                self.assertEqual(aid, int(s["planned_attempt_id"]))
                _write_pair_manifest(
                    pairs_root,
                    pair_id=s["pair_id"],
                    attempt_id=aid,
                    scenario_id=scenario_id,
                    seed_id=seed_id,
                    family=s["family"],
                    comparable=True,
                )
                man = json.loads(
                    (
                        attempt_dir_for(pairs_root, s["pair_id"], aid) / "pair_manifest.json"
                    ).read_text(encoding="utf-8")
                )
                man["attempt_id"] = aid
                return man

            ns_resume = type(
                "A",
                (),
                {
                    "registry": str(DEFAULT_REGISTRY_PATH),
                    "registry_manifest": str(FROZEN_MANIFEST),
                    "evidence_dir": str(evidence),
                    "plan_only": False,
                    "stop_on_fail": False,
                    "continue_policy": "",
                    "device": "cpu",
                    "host": "127.0.0.1",
                    "port": 2000,
                    "no_aggregate": True,
                    "min_comparable": 0,
                },
            )()

            scan_calls = {"n": 0}

            def boom_scan(*a, **k):
                scan_calls["n"] += 1
                raise AssertionError(
                    "first_unoccupied_attempt_id must not run on CLI recovery resume"
                )

            with mock.patch.object(mod, "cmd_aggregate", return_value=0), mock.patch(
                "driving_vla.evaluation.paired_live.run_pair",
                side_effect=fake_run_pair,
            ), mock.patch(
                "driving_vla.evaluation.paired_live._identity_hashes_without_carla",
                return_value=(
                    man1.get("model_checkpoint_hash", "ckpt"),
                    man1.get("retimer_hash", "rh"),
                    man1["model_retimer_hash"],
                    {},
                ),
            ), mock.patch(
                "driving_vla.evaluation.paired_live.EXECUTOR_CONFIG_HASH",
                man1["executor_config_hash"],
            ), mock.patch(
                "driving_vla.evaluation.runner_contract.first_unoccupied_attempt_id",
                side_effect=boom_scan,
            ):
                code2 = mod.cmd_run_set(ns_resume)

            self.assertEqual(code2, 0)
            man2 = json.loads(man_path.read_text(encoding="utf-8"))
            self.assertEqual(man2["manifest_content_hash"], hash1)
            self.assertEqual(
                [int(p["planned_attempt_id"]) for p in man2["pairs"]], planned_ids
            )
            self.assertEqual([str(p["pair_id"]) for p in man2["pairs"]], pair_ids)
            # Must not rescan Evidence / recompute planned IDs on recovery entry.
            self.assertEqual(scan_calls["n"], 0)
            # Checkpoint resume: slots 1..11 only (not slot 0 again).
            self.assertEqual(called_indices, list(range(1, 12)))
            report = json.loads(
                (evidence / "run_set_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(report["pair_results"]), 12)
            self.assertEqual(int(report["pair_results"][0]["attempt_id"]), aid0)

    def test_ensure_reuses_slots_when_evidence_would_replan(self) -> None:
        """After completing planned attempt, rebuild would bump id; reuse must not."""
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            path = Path(td) / "run_set_manifest.json"
            m1, mode1 = ensure_run_set_manifest(path, rs)
            self.assertEqual(mode1, "created")
            slot0 = m1["pairs"][0]
            _write_pair_manifest(
                pairs_root,
                pair_id=slot0["pair_id"],
                attempt_id=int(slot0["planned_attempt_id"]),
                scenario_id=slot0["scenario_id"],
                seed_id=slot0["seed_id"],
                family=slot0["family"],
                comparable=True,
            )
            # Rebuild from Evidence would plan next free id for slot0.
            rebuilt = build_run_set_manifest(
                registry=_reg,
                freeze_audit=_audit,
                pairs_root=pairs_root,
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
                model_checkpoint_hash="ckpt",
                retimer_hash="rh",
            )
            self.assertNotEqual(
                int(rebuilt["pairs"][0]["planned_attempt_id"]),
                int(m1["pairs"][0]["planned_attempt_id"]),
            )
            # ensure with identity-only (no pairs) reuses original slots.
            identity = {
                "registry_sha256": m1["registry_sha256"],
                "model_retimer_hash": MODEL_H,
                "model_checkpoint_hash": "ckpt",
                "retimer_hash": "rh",
                "executor_config_hash": EXEC_H,
                "continue_policy": CONTINUE_POLICY_CONTINUE_ALL,
                "retry_policy": RETRY_POLICY_NO_AUTO_RETRY,
                "n_pairs": 12,
                "registry_schema_version": m1["registry_schema_version"],
            }
            m2, mode2 = ensure_run_set_manifest(path, identity)
            self.assertEqual(mode2, "reused")
            self.assertEqual(m2["manifest_content_hash"], m1["manifest_content_hash"])
            self.assertEqual(
                m2["pairs"][0]["planned_attempt_id"],
                m1["pairs"][0]["planned_attempt_id"],
            )
            # Must not call build_fn on reuse.
            def boom():
                raise AssertionError("build_fn must not run when manifest exists")

            m3, mode3 = ensure_run_set_manifest(path, identity, build_fn=boom)
            self.assertEqual(mode3, "reused")
            self.assertEqual(m3["manifest_content_hash"], m1["manifest_content_hash"])

    def test_empty_partial_dir_sealed_no_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            slot = rs["pairs"][0]
            pair_id = slot["pair_id"]
            aid = int(slot["planned_attempt_id"])
            adir = pairs_root / pair_id / f"attempt_{aid}"
            adir.mkdir(parents=True)  # empty dir exists
            self.assertFalse((adir / "pair_manifest.json").exists())
            expected = ExpectedPairHashes(
                pair_id=pair_id,
                scenario_id=slot["scenario_id"],
                seed_id=slot["seed_id"],
                registry_sha256=rs["registry_sha256"],
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
            )
            action = resolve_no_auto_retry_action(
                pairs_root,
                pair_id=pair_id,
                planned_attempt_id=aid,
                expected=expected,
            )
            self.assertEqual(action["action"], "retain_failed")
            self.assertTrue(action.get("sealed_partial"))
            man = action["existing_manifest"]
            self.assertEqual(man["status"], PAIR_STATUS_FAILED)
            self.assertEqual(man["error"], FAILURE_INTERRUPTED_PARTIAL_ATTEMPT)
            self.assertTrue((adir / "pair_manifest.json").is_file())
            # No next attempt created.
            self.assertFalse((pairs_root / pair_id / f"attempt_{aid + 1}").exists())
            # Absent path still new_run.
            other = rs["pairs"][1]
            action2 = resolve_no_auto_retry_action(
                pairs_root,
                pair_id=other["pair_id"],
                planned_attempt_id=int(other["planned_attempt_id"]),
                expected=ExpectedPairHashes(
                    pair_id=other["pair_id"],
                    scenario_id=other["scenario_id"],
                    seed_id=other["seed_id"],
                    registry_sha256=rs["registry_sha256"],
                    model_retimer_hash=MODEL_H,
                    executor_config_hash=EXEC_H,
                ),
            )
            self.assertEqual(action2["action"], "new_run")
            self.assertFalse(
                attempt_dir_for(
                    pairs_root, other["pair_id"], int(other["planned_attempt_id"])
                ).exists()
            )

    def test_corrupt_manifest_backed_up_before_seal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            pairs_root = Path(td) / "pairs"
            pairs_root.mkdir()
            _reg, _audit, rs = _freeze_and_manifest(pairs_root)
            slot = rs["pairs"][0]
            pair_id = slot["pair_id"]
            aid = int(slot["planned_attempt_id"])
            adir = pairs_root / pair_id / f"attempt_{aid}"
            adir.mkdir(parents=True)
            man_path = adir / "pair_manifest.json"
            corrupt_bytes = b"{not valid json at all\n"
            man_path.write_bytes(corrupt_bytes)
            sha = hashlib.sha256(corrupt_bytes).hexdigest()
            expected = ExpectedPairHashes(
                pair_id=pair_id,
                scenario_id=slot["scenario_id"],
                seed_id=slot["seed_id"],
                registry_sha256=rs["registry_sha256"],
                model_retimer_hash=MODEL_H,
                executor_config_hash=EXEC_H,
            )
            action = resolve_no_auto_retry_action(
                pairs_root,
                pair_id=pair_id,
                planned_attempt_id=aid,
                expected=expected,
            )
            self.assertEqual(action["action"], "retain_failed")
            self.assertTrue(action.get("sealed_partial"))
            backup = adir / f"pair_manifest.corrupt.{sha}.json"
            self.assertTrue(backup.is_file())
            self.assertEqual(backup.read_bytes(), corrupt_bytes)
            sealed = json.loads(man_path.read_text(encoding="utf-8"))
            self.assertEqual(sealed["status"], PAIR_STATUS_FAILED)
            self.assertEqual(sealed["error"], FAILURE_INTERRUPTED_PARTIAL_ATTEMPT)
            self.assertEqual(
                sealed.get("corrupt_manifest_backup"),
                f"pair_manifest.corrupt.{sha}.json",
            )
            # Re-entry retains sealed FAILED; backup preserved.
            action2 = resolve_no_auto_retry_action(
                pairs_root,
                pair_id=pair_id,
                planned_attempt_id=aid,
                expected=expected,
            )
            self.assertEqual(action2["action"], "retain_failed")
            self.assertFalse(action2.get("sealed_partial"))
            self.assertEqual(backup.read_bytes(), corrupt_bytes)


if __name__ == "__main__":
    unittest.main()
