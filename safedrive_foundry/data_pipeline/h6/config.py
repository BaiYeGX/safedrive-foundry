"""Configuration for the post-H5 VLA-primary experiment."""

from __future__ import annotations

from data_pipeline.h3.contracts import stable_sha256


H6_VLA90_CONFIG = {
    "schema_version": "safedrive.h6.vla90.v1",
    "arms": ["off", "on"],
    "decision_ticks": 50,
    "target_actual_vla_coverage": 0.90,
    "max_unsafe_rate_delta": 0.01,
    "progress_bootstrap_lower_95": 0.0,
    "scorer_deadline_ms": 50.0,
    "max_switches_per_30s": 2.0,
    "ping_pong_window_s": 0.5,
    "data_isolation": {
        "training_seeds": [89, 97],
        "formal_acceptance_seeds": [101, 103],
        "training_acceptance_disjoint": True,
    },
    "pre_roll": {
        "preserve_scenario_ticks": True,
        "target_speed_mps": 4.0,
        "minimum_ready_speed_mps": 2.5,
        "maximum_extra_ticks": 80,
        "route_follow": True,
        "maximum_normalized_steer": 0.35,
        "spectator_follow_ego": True,
        "spectator_follow_hz": 20.0,
        "dynamic_traffic_light_timing": True,
    },
    "world": {
        "requires_expert_and_vla_pair_score": True,
        "requires_vla_score_gte_expert": True,
        "minimum_vla_trust": 0.50,
    },
    "guard": {
        "verdicts": ["PASS", "REVIEW", "REJECT"],
        "world_eligible": ["PASS", "REVIEW"],
    },
    "safety": {
        "preferred_vla_repair_first": True,
        "same_tick_expert_fallback": True,
        "both_fail_minimal_risk": True,
    },
}
H6_VLA90_CONFIG_SHA256 = stable_sha256(H6_VLA90_CONFIG)

# H6 v1 consumed seed 101 and must remain byte-for-byte compatible.  VLA75 is
# a new acceptance contract with explicitly pre-registered, mutually
# disjoint formal lineages.  The first lineage intentionally starts at 103:
# the old v1 metadata mentioned 103, but no real seed-103 H6 run exists (that
# fact is checked by readiness before a formal run is allowed).
H6_VLA75_FORMAL_LINEAGES: dict[str, tuple[int, int]] = {
    "a": (103, 107),
    "b": (109, 113),
    "c": (127, 131),
}


def _validate_lineage_id(lineage_id: str) -> str:
    value = str(lineage_id).strip().lower()
    if value not in H6_VLA75_FORMAL_LINEAGES:
        raise ValueError(
            f"unknown_h6_vla75_formal_lineage:{lineage_id};"
            f"expected={sorted(H6_VLA75_FORMAL_LINEAGES)}"
        )
    return value


def build_h6_vla75_config(lineage_id: str) -> dict:
    """Build the immutable v2 contract for one pre-registered lineage.

    A fresh mapping is returned on every call so callers cannot mutate the
    hash input shared by another run.  Thresholds deliberately distinguish
    the raw World 90% preference gate from the applied-control 75% VLA gate;
    outcome attribution in the development data remains a separate 90% rule.
    """

    lineage = _validate_lineage_id(lineage_id)
    pilot_seed, full_second_seed = H6_VLA75_FORMAL_LINEAGES[lineage]
    return {
        "schema_version": "safedrive.h6.vla75.v2",
        "contract": "vla75-v2",
        "lineage_id": lineage,
        "arms": ["off", "on"],
        "decision_ticks": 50,
        # Flat aliases make the contract easy to inspect in manifests while
        # the nested ``targets``/``runtime`` blocks remain canonical for
        # structured consumers.
        "target_world_vla_preference": 0.90,
        "target_actual_vla_coverage": 0.75,
        "max_classic_mrm_share": 0.25,
        "max_unsafe_delta": 0.01,
        "forbid_target_only_unsafe": True,
        "scorer_deadline_ms": 50.0,
        "max_switches_per_30s": 2,
        "ping_pong_window_ticks": 10,
        "targets": {
            "world_vla_preference": 0.90,
            "actual_vla_coverage": 0.75,
            "classic_mrm_share": 0.25,
            "unsafe_delta": 0.01,
            "progress_bootstrap_lower_95": 0.0,
        },
        "world": {
            "raw_pair_required": True,
            "raw_score_vla_gte_expert": True,
            "raw_preference_utility_vla_gte_expert": True,
            "minimum_vla_trust": 0.50,
            "maximum_vla_risk": 0.20,
            "source_blind_features": True,
            "schema_version": "safedrive.world.vla75.pair_exec.v1",
        },
        "runtime": {
            "scorer_deadline_ms": 50.0,
            "max_world_incremental_gpu_gib": 1.5,
            # Frozen local admission budget from docs/RESOURCES.md.  This is
            # an Evidence requirement, not an assumption that a run passed.
            "max_whole_gpu_peak_gib": 14.5,
            "max_switches_per_30s": 2,
            "ping_pong_window_ticks": 10,
            "emergency_switch_margin": 1.5,
            "ema_alpha_grid": [0.25, 0.50, 0.75],
            "hold_ticks_grid": [6, 10, 14],
            "hysteresis_grid": [0.05, 0.10, 0.20],
        },
        "safety": {
            "preferred_vla_repair_first": True,
            "same_tick_expert_fallback": True,
            "both_fail_minimal_risk": True,
            "final_revalidation_required": True,
            "applied_requires_track_approved": True,
        },
        "provenance": {
            "raw_and_stabilized_separate": True,
            "applied_bound_after_control": True,
            "require_run_lock": True,
            "require_spectator_follow": True,
            "require_candidate_config_model_feature_worktree_hashes": True,
        },
        "data_isolation": {
            "development_seeds": [89, 97],
            "training_seed": 89,
            "calibration_seed": 97,
            "consumed_seed_101": True,
            "formal_pilot_seed": pilot_seed,
            "formal_full_seeds": [pilot_seed, full_second_seed],
            "training_outcome_attribution_minimum": 0.90,
            "training_acceptance_disjoint": True,
        },
        "matrix": {
            "maps": ["Town01", "Town03", "Town05"],
            "pilot_families": [
                "free_flow",
                "emergency_lead_brake",
                "aggressive_cut_in",
                "red_light_dilemma",
            ],
            "full_family_count": 9,
            "weathers": ["ClearNoon", "CloudyNoon"],
            "pilot_pairs": 12,
            "full_pairs": 108,
        },
    }


def h6_vla75_config_sha256(lineage_id: str) -> str:
    """Return the stable hash of a lineage-specific v2 configuration."""

    return stable_sha256(build_h6_vla75_config(lineage_id))


# Convenience aliases are for tooling that needs a concrete default while the
# public builder/hash functions remain the authoritative API.  They represent
# lineage A only and must not be used to claim B or C evidence.
H6_VLA75_CONFIG = build_h6_vla75_config("a")
H6_VLA75_CONFIG_SHA256 = h6_vla75_config_sha256("a")


__all__ = [
    "H6_VLA90_CONFIG",
    "H6_VLA90_CONFIG_SHA256",
    "H6_VLA75_FORMAL_LINEAGES",
    "H6_VLA75_CONFIG",
    "H6_VLA75_CONFIG_SHA256",
    "build_h6_vla75_config",
    "h6_vla75_config_sha256",
]
