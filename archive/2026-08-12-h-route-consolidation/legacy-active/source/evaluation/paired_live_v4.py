"""Live paired evaluation for the R2 K2 V4 ordered-token head.

The fixture/session mechanics deliberately reuse the already-audited V3
runner.  The policy is V4 and its same-forward raw token dump is copied into
the anchor evidence.  Branches are cold rebuilt from the frozen artifact and
may collect actor-future sidecars, but never call the VLA again.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from driving_vla.evaluation.comparability import evaluate_pair_comparability
from driving_vla.evaluation.fixture_runtime import (
    FixtureError,
    connect_world,
    restore_async,
)
from driving_vla.evaluation.paired_live import (
    EXECUTOR_CONFIG_HASH,
    _file_sha256,
    run_branch,
)
from driving_vla.evaluation.paired_live_v3 import run_anchor_v3
from driving_vla.evaluation.oracle_v2 import evaluate_pair_oracle_v2
from driving_vla.evaluation.paired_contract import compute_pair_id, content_hash
from driving_vla.evaluation.scenario_registry import ScenarioSeedFixture
from driving_vla.model.k2_v3_types import load_k2_v3_config
from driving_vla.model.neural_policy import NeuralV4Policy
from driving_vla.model.simlingo_runtime import SimLingoNeuralRuntime

REPORT_SCHEMA = "safedrive.r2_v4.live_pair.v1"
FINAL_NAMESPACE = "r3_final_head_formal"


def _safe(metrics: Any) -> bool:
    return bool(
        int(metrics.collision_episode_count) == 0
        and float(metrics.offroad_fraction) < 0.02
        and bool(metrics.completed_primary_horizon)
    )


def build_v4_model_identity(checkpoint: Path | str) -> dict[str, str]:
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError(f"K2 V4 checkpoint missing: {path}")
    checkpoint_hash = _file_sha256(path)
    model_hash = content_hash(
        {
            "policy": NeuralV4Policy.model_id,
            "checkpoint_sha256": checkpoint_hash,
            "k2_v3_config_hash": load_k2_v3_config().config_hash(),
            "schema": "safedrive.k2.semantic_head.v4",
        },
        nibble=64,
    )
    return {
        "checkpoint_sha256": checkpoint_hash,
        "k2_v3_config_hash": load_k2_v3_config().config_hash(),
        "model_retimer_hash": model_hash,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pair_v4(
    *,
    registry: Any,
    fixture: ScenarioSeedFixture,
    checkpoint: Path | str,
    evidence_dir: Path,
    host: str = "",
    port: int = 2000,
    device: str = "cuda",
    shared_policy: Any | None = None,
    branch_order: tuple[int, int] = (0, 1),
    namespace: str = FINAL_NAMESPACE,
    collect_actor_future: bool = True,
    checkpoint_use: str = "r2v4_formal",
    source_manifest_hash: str = "",
    repeat_group: str = "",
    aa_noise_identity: str = "",
    carla_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute exactly one V4 pair and freeze all restartable evidence."""
    if str(namespace) in {
        "r2v4_blind_audit",
        "r3_r2_blind_holdout",
        "r3_final_head_formal",
    } and not bool(
        collect_actor_future
    ):
        raise ValueError(
            f"{namespace} requires actor-future sidecar on every paired run"
        )
    if not str(host).strip():
        raise ValueError("V4 paired execution requires a host from READY preflight")
    evidence_dir = Path(evidence_dir)
    if evidence_dir.exists():
        raise FileExistsError(f"refusing existing V4 pair dir: {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    identity = build_v4_model_identity(checkpoint)
    registry_hash = str(registry.registry_sha256 or registry.compute_registry_sha256())
    pair_id = compute_pair_id(
        scenario_registry_hash=registry_hash,
        scenario_id=fixture.scenario_id,
        seed_id=fixture.seed_id,
        model_checkpoint_config_retimer_hash=identity["model_retimer_hash"],
        executor_config_hash=EXECUTOR_CONFIG_HASH,
    )
    client = None
    world = None
    forward_counter = {"n": 0}
    try:
        client, world = connect_world(
            host=host,
            port=port,
            map_name=fixture.map_name,
            sim_dt_s=float(fixture.sim_dt_s),
            sync=True,
            timeout_s=90.0,
            retries=3,
        )
        if shared_policy is None:
            runtime = SimLingoNeuralRuntime(device=device)
            load = runtime.load()
            if not load.ok:
                raise RuntimeError(f"SimLingo load failed: {load.error}")
            policy = NeuralV4Policy(
                runtime=runtime,
                semantic_head_checkpoint=str(checkpoint),
                keep_on_gpu=True,
                lazy=False,
                device=device,
                checkpoint_use=checkpoint_use,
            )
        else:
            policy = shared_policy
        policy.ensure_loaded()
        artifact, anchor = run_anchor_v3(
            client=client,
            world=world,
            fixture=fixture,
            policy=policy,
            pair_id=pair_id,
            model_checkpoint_hash=identity["checkpoint_sha256"],
            registry_hash=registry_hash,
            evidence_dir=evidence_dir,
            forward_counter=forward_counter,
            include_scenario_family=False,
        )
        # Preserve the exact same-forward token dump beside the artifact.  A
        # missing dump is a hard contract failure for V4, never a silent V3
        # fallback.
        token_meta = dict(getattr(policy, "last_token_metadata", {}) or {})
        token_source = Path(str(token_meta.get("raw_tensor_path") or ""))
        if not token_source.is_file():
            raise FixtureError("V4_TOKEN_DUMP_MISSING")
        token_target = evidence_dir / "anchor" / "driving_tokens_v4.npy"
        token_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(token_source, token_target)
        token_meta["raw_tensor_path"] = str(token_target)
        _write_json(evidence_dir / "anchor" / "driving_tokens_v4.json", token_meta)
        trace = {
            "schema_version": "safedrive.r2_v4.formal_trace.v1",
            "namespace": str(namespace),
            "source_manifest_hash": str(source_manifest_hash),
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "raw_head_output_hash": str(
                artifact.bundle.observation_identity.get("raw_head_output_hash") or ""
            ),
            "raw_head_output": {
                "candidate_id": artifact.candidates[1].candidate_id,
                "alternative_kind": artifact.candidates[1].alternative_kind.value,
                "target_lane_side": artifact.candidates[1].target_lane_side.value,
                "available": bool(artifact.candidates[1].available),
                "availability_reason": artifact.candidates[1].availability_reason,
                "probability": float(artifact.candidates[1].probability),
                "metadata": dict(artifact.candidates[1].metadata),
            },
            "semantic_rescue_count": int(
                artifact.bundle.observation_identity.get("semantic_rescue_count", 0)
            ),
            "scenario_family_runtime_use": int(
                artifact.bundle.observation_identity.get("scenario_family_runtime_use", 0)
            ),
            "guard_status": artifact.bundle.guard_status,
            "guard_reasons": list(artifact.bundle.guard_reasons),
            "artifact_content_hash": artifact.artifact_content_hash(),
        }
        _write_json(evidence_dir / "anchor" / "formal_trace_v4.json", trace)

        valid = dict(artifact.bundle.guard_metrics.get("candidate_valid") or {})
        available = {
            index
            for index, candidate in enumerate(artifact.candidates)
            if bool(candidate.available) and bool(valid.get(candidate.candidate_id, False))
        }
        if 0 not in available:
            raise FixtureError("V4 nominal candidate unavailable at anchor")
        results: dict[int, Any] = {}
        for index in branch_order:
            if index not in available:
                continue
            results[index] = run_branch(
                client=client,
                world=world,
                fixture=fixture,
                artifact=artifact,
                force_index=index,
                pair_id=pair_id,
                registry_hash=registry_hash,
                model_retimer_hash=identity["model_retimer_hash"],
                evidence_dir=evidence_dir,
                forward_counter=forward_counter,
                policy=policy,
                collect_actor_future=bool(collect_actor_future),
                include_scenario_family=False,
            )

        comparable = False
        comparability: dict[str, Any] = {
            "status": "SINGLETON_NO_RANKING",
            "comparable": False,
            "reasons": ["NO_ALTERNATIVE"],
            "failure_codes": [],
        }
        oracle = None
        if 1 in results:
            comparison = evaluate_pair_comparability(
                anchor=artifact,
                branch0=results[0].report,
                branch1=results[1].report,
                expected_registry_hash=registry_hash,
                expected_scenario_id=fixture.scenario_id,
                expected_seed_id=fixture.seed_id,
                expected_model_retimer_hash=identity["model_retimer_hash"],
                expected_executor_config_hash=EXECUTOR_CONFIG_HASH,
            )
            comparability = comparison.to_dict()
            comparable = bool(comparison.comparable)
            oracle = evaluate_pair_oracle_v2(
                pair_id=pair_id,
                scenario_id=fixture.scenario_id,
                seed_id=fixture.seed_id,
                family=fixture.family,
                comparable=comparable,
                top1_index=artifact.top1_index,
                metrics0=results[0].metrics if comparable else None,
                metrics1=results[1].metrics if comparable else None,
                incomparable_reasons=comparison.reasons,
                candidate_ids=(artifact.candidates[0].candidate_id, artifact.candidates[1].candidate_id),
            )
        safe = {index: _safe(result.metrics) for index, result in results.items()}
        decisive = bool(
            comparable
            and oracle is not None
            and oracle.oracle_decision_level not in {None, "tie_top1"}
        )
        condition_variant = (
            fixture.scenario_id.split("__", 1)[1]
            if "__" in fixture.scenario_id
            else "base"
        )
        repeat_group = str(repeat_group or fixture.scenario_id.split("__", 1)[0])
        aa_noise_identity = str(
            aa_noise_identity
            or content_hash(
                {
                    "namespace": "r3_aa_noise_probe",
                    "repeat_group": repeat_group,
                    "candidate_id": "v3_nominal_progress",
                },
                nibble=64,
            )
        ).lower()
        if not repeat_group or len(aa_noise_identity) != 64 or any(
            char not in "0123456789abcdef" for char in aa_noise_identity
        ):
            raise FixtureError("R3_REPEAT_AA_BINDING_INVALID")
        actor_controller_kind = (
            "reactive" if "reactive" in condition_variant.lower() else "fixed"
        )
        branch_fatal = any(
            int(summary.get("collision_episodes", 0)) > 0
            or float((summary.get("metrics") or {}).get("offroad_fraction", 0.0)) >= 0.02
            or not bool((summary.get("metrics") or {}).get("completed_primary_horizon", False))
            for summary in (result.summary for result in results.values())
        )
        strict_success = bool(
            comparable
            and not branch_fatal
            and safe.get(0, False)
            and safe.get(1, False)
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "namespace": str(namespace),
            "source_manifest_hash": str(source_manifest_hash),
            "pair_id": pair_id,
            "scenario_id": fixture.scenario_id,
            "seed_id": fixture.seed_id,
            "map_name": fixture.map_name,
            "family": fixture.family,
            "condition_variant": condition_variant,
            "repeat_group": repeat_group,
            "aa_noise_identity": aa_noise_identity,
            "actor_controller_kind": actor_controller_kind,
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "registry_hash": registry_hash,
            "artifact_content_hash": artifact.artifact_content_hash(),
            "bundle_hash": artifact.bundle.bundle_hash,
            "forward_count_total": int(forward_counter["n"]),
            "candidate1_available": 1 in available,
            "comparable": comparable,
            "decisive": decisive,
            "fatal": branch_fatal,
            "strict_success": strict_success,
            "winner": None if oracle is None else oracle.oracle_candidate_index,
            "pair_label": "NO_ALTERNATIVE" if oracle is None else oracle.pair_label,
            "both_bad": bool(oracle.both_bad) if oracle is not None else not safe.get(0, False),
            "safe_candidate_exists": any(safe.values()),
            "guard_mpc_failure": bool(
                not comparable
                and 1 in available
                and "MPC_DEADLINE_UNRELIABLE" in comparability.get("failure_codes", ())
            ),
            "comparability": comparability,
            "oracle": None if oracle is None else oracle.to_dict(),
            "anchor": anchor,
            "branches": {str(index): result.summary for index, result in results.items()},
            "actor_future_sidecar": bool(collect_actor_future),
            "semantic_rescue_count": int(trace["semantic_rescue_count"]),
            "scenario_family_runtime_use": int(trace["scenario_family_runtime_use"]),
            "carla_runtime": dict(carla_runtime or {}),
        }
        _write_json(evidence_dir / "pair_report.json", report)
        # Compatibility files make ActionBranchDatasetV1 construction
        # restartable without asking the collector to reinterpret a result.
        _write_json(
            evidence_dir / "pair_manifest.json",
            {
                "status": "COMPLETED",
                "namespace": str(namespace),
                "source_manifest_hash": str(source_manifest_hash),
                "r2_checkpoint_sha256": identity["checkpoint_sha256"],
                "pair_id": pair_id,
                "scenario_id": fixture.scenario_id,
                "seed_id": fixture.seed_id,
                "map_name": fixture.map_name,
                "family": fixture.family,
                "condition_variant": condition_variant,
                "repeat_group": repeat_group,
                "aa_noise_identity": aa_noise_identity,
                "actor_controller_kind": actor_controller_kind,
                "artifact_content_hash": artifact.artifact_content_hash(),
                "carla_runtime": dict(carla_runtime or {}),
            },
        )
        _write_json(evidence_dir / "pair_comparability.json", comparability)
        _write_json(
            evidence_dir / "pair_oracle.json",
            {} if oracle is None else oracle.to_dict(),
        )
        return report
    except Exception as exc:
        _write_json(
            evidence_dir / "failure.json",
            {"schema_version": REPORT_SCHEMA, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        if world is not None:
            restore_async(world)


def run_aa_repeat_v4(
    *,
    registry: Any,
    fixture: ScenarioSeedFixture,
    checkpoint: Path | str,
    evidence_dir: Path,
    host: str = "",
    port: int = 2000,
    device: str = "cuda",
    shared_policy: Any | None = None,
    namespace: str = "r3_aa_noise_probe",
    source_manifest_hash: str = "",
    checkpoint_use: str = "r3_final_head_formal",
    repeat_group: str = "",
    aa_noise_identity: str = "",
) -> dict[str, Any]:
    """Cold-rebuild the same candidate twice to estimate simulator A/A noise."""
    if not str(host).strip():
        raise ValueError("V4 A/A repeat requires a host from READY preflight")
    evidence_dir = Path(evidence_dir)
    if evidence_dir.exists():
        raise FileExistsError(f"refusing existing V4 A/A dir: {evidence_dir}")
    evidence_dir.mkdir(parents=True, exist_ok=False)
    identity = build_v4_model_identity(checkpoint)
    registry_hash = str(registry.registry_sha256 or registry.compute_registry_sha256())
    pair_id = compute_pair_id(
        scenario_registry_hash=registry_hash,
        scenario_id=fixture.scenario_id,
        seed_id=fixture.seed_id,
        model_checkpoint_config_retimer_hash=identity["model_retimer_hash"],
        executor_config_hash=EXECUTOR_CONFIG_HASH,
    )
    client = None
    world = None
    forward_counter = {"n": 0}
    try:
        client, world = connect_world(
            host=host,
            port=port,
            map_name=fixture.map_name,
            sim_dt_s=float(fixture.sim_dt_s),
            sync=True,
            timeout_s=90.0,
            retries=3,
        )
        if shared_policy is None:
            runtime = SimLingoNeuralRuntime(device=device)
            load = runtime.load()
            if not load.ok:
                raise RuntimeError(f"SimLingo load failed: {load.error}")
            policy = NeuralV4Policy(
                runtime=runtime,
                semantic_head_checkpoint=str(checkpoint),
                keep_on_gpu=True,
                lazy=False,
                device=device,
                checkpoint_use=checkpoint_use,
            )
        else:
            policy = shared_policy
        policy.ensure_loaded()
        artifact, _anchor = run_anchor_v3(
            client=client,
            world=world,
            fixture=fixture,
            policy=policy,
            pair_id=pair_id,
            model_checkpoint_hash=identity["checkpoint_sha256"],
            registry_hash=registry_hash,
            evidence_dir=evidence_dir,
            forward_counter=forward_counter,
            include_scenario_family=False,
        )
        token_meta = dict(getattr(policy, "last_token_metadata", {}) or {})
        token_source = Path(str(token_meta.get("raw_tensor_path") or ""))
        if not token_source.is_file():
            raise FixtureError("V4_TOKEN_DUMP_MISSING")
        token_target = evidence_dir / "anchor" / "driving_tokens_v4.npy"
        token_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(token_source, token_target)
        token_meta["raw_tensor_path"] = str(token_target)
        _write_json(evidence_dir / "anchor" / "driving_tokens_v4.json", token_meta)
        repeats: dict[str, Any] = {}
        for label in ("repeat_a", "repeat_b"):
            repeat_dir = evidence_dir / label
            result = run_branch(
                client=client,
                world=world,
                fixture=fixture,
                artifact=artifact,
                force_index=0,
                pair_id=pair_id,
                registry_hash=registry_hash,
                model_retimer_hash=identity["model_retimer_hash"],
                evidence_dir=repeat_dir,
                forward_counter=forward_counter,
                policy=policy,
                collect_actor_future=True,
                include_scenario_family=False,
            )
            repeats[label] = result.summary
        repeat_group = str(repeat_group or fixture.scenario_id.split("__", 1)[0])
        aa_noise_identity = str(
            aa_noise_identity
            or content_hash(
                {
                    "namespace": "r3_aa_noise_probe",
                    "repeat_group": repeat_group,
                    "candidate_id": "v3_nominal_progress",
                },
                nibble=64,
            )
        ).lower()
        if not repeat_group or len(aa_noise_identity) != 64 or any(
            char not in "0123456789abcdef" for char in aa_noise_identity
        ):
            raise FixtureError("R3_REPEAT_AA_BINDING_INVALID")
        report = {
            "schema_version": "safedrive.r3.aa_noise_repeat.v1",
            "namespace": namespace,
            "source_manifest_hash": source_manifest_hash,
            "pair_id": pair_id,
            "scenario_id": fixture.scenario_id,
            "seed_id": fixture.seed_id,
            "repeat_group": repeat_group,
            "aa_noise_identity": aa_noise_identity,
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "artifact_content_hash": artifact.artifact_content_hash(),
            "forward_count_total": int(forward_counter["n"]),
            "repeats": repeats,
            "actor_future_paths": {
                label: str(
                    evidence_dir / label / "branch-0" / "oracle" / "actor_future_trace.jsonl"
                )
                for label in repeats
            },
        }
        _write_json(evidence_dir / "aa_report.json", report)
        return report
    except Exception as exc:
        _write_json(evidence_dir / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if world is not None:
            restore_async(world)


__all__ = [
    "FINAL_NAMESPACE",
    "REPORT_SCHEMA",
    "build_v4_model_identity",
    "run_aa_repeat_v4",
    "run_pair_v4",
]
