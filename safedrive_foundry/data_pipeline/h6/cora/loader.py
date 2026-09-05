"""Fail-closed C3-facing loader for verified C2 roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from data_pipeline.h2.store import file_sha256

from .config import CORA_C2_CONFIG, CORA_C2_CONFIG_SHA256
from .feature import validate_cora_feature_view
from .matrix import CORA_MATRIX_SHA256
from .outcomes import read_public_label
from .store import CoraDataStore


TRAIN_ALLOWED_SPLITS = frozenset({"train", "validation"})
ALL_DATA_SPLITS = frozenset(
    {"coverage_pilot", "train", "validation", "calibration", "locked_development"}
)


def load_cora_roots(
    dataset_root: Path | str,
    *,
    splits: Sequence[str] = ("train",),
    purpose: str = "training",
    require_complete_pair: bool = True,
    allow_pilot: bool = False,
) -> tuple[dict[str, Any], ...]:
    path = Path(dataset_root)
    wanted = frozenset(str(item) for item in splits)
    if not wanted or not wanted.issubset(ALL_DATA_SPLITS):
        raise ValueError(f"cora_loader_splits:{sorted(wanted)}")
    if purpose == "training" and not wanted.issubset(TRAIN_ALLOWED_SPLITS):
        raise ValueError(f"cora_loader_training_split_forbidden:{sorted(wanted - TRAIN_ALLOWED_SPLITS)}")
    if purpose not in {"training", "calibration", "evaluation", "audit"}:
        raise ValueError(f"cora_loader_purpose:{purpose}")
    if "coverage_pilot" in wanted and not allow_pilot:
        raise ValueError("cora_loader_pilot_requires_explicit_opt_in")
    if purpose == "calibration" and wanted != {"calibration"}:
        raise ValueError(f"cora_loader_calibration_scope:{sorted(wanted)}")
    if purpose == "evaluation" and not wanted.issubset({"validation", "locked_development"}):
        raise ValueError(f"cora_loader_evaluation_scope:{sorted(wanted)}")
    if (path / "repair-protocol.json").is_file():
        from .repair import merged_roots, read_json
        if purpose != "audit":
            gate_path = path / "final-delivery.json"
            if not gate_path.is_file():
                raise ValueError("cora_repair_quality_gate_missing")
            gate = read_json(gate_path)
            if gate.get("status") != "GATE_PASSED" or gate.get("passed") is not True:
                raise ValueError("cora_repair_quality_gate_failed")
        return tuple(record for record in merged_roots(path) if record["split"] in wanted
            and not record.get("diagnostic")
            and (not require_complete_pair or record.get("nominal_pair_outcome_mask")))
    store = CoraDataStore(path.parent, path.name)
    if path.name != str(CORA_C2_CONFIG["dataset_id"]):
        raise ValueError("cora_loader_dataset_id")
    manifest_ok, failures = store.verify_manifest()
    if not manifest_ok:
        raise ValueError(f"cora_loader_manifest:{list(failures)}")
    output = []
    for record in store.iter_roots():
        root_id = str(record.get("root_id", ""))
        if not store.has_valid_root(root_id):
            raise ValueError(f"cora_loader_root_artifacts:{root_id}")
        if record.get("matrix_sha256") != CORA_MATRIX_SHA256:
            raise ValueError("cora_loader_matrix_hash")
        if record.get("config_sha256") != CORA_C2_CONFIG_SHA256:
            raise ValueError("cora_loader_config_hash")
        split = str(record.get("split", ""))
        if split == "reserved_formal":
            raise ValueError("cora_loader_formal_materialization")
        if split not in wanted:
            continue
        if require_complete_pair and not bool(record.get("nominal_pair_outcome_mask", False)):
            continue
        feature_paths = record.get("feature_paths", {})
        feature_hashes = record.get("feature_sha256", {})
        feature_views: dict[str, Any] = {}
        for key, relative in feature_paths.items():
            feature_path = path / str(relative)
            if not feature_path.is_file() or file_sha256(feature_path) != feature_hashes.get(key):
                raise ValueError(f"cora_loader_feature_hash:{record.get('root_id')}:{key}")
            feature = json.loads(feature_path.read_text(encoding="utf-8"))
            validate_cora_feature_view(feature)
            feature_views[str(key)] = feature
        branches = tuple(record.get("branches", ()))
        public_heads: dict[str, Any] = {}
        public_label_paths: dict[str, str] = {}
        for branch in branches:
            paths = branch.get("artifact_paths", {})
            hashes = branch.get("artifact_sha256", {})
            if "label" not in paths or "label" not in hashes:
                raise ValueError(f"cora_loader_label_missing:{root_id}:{branch.get('proposal_id')}")
            label_path = path / str(paths["label"])
            if not label_path.is_file() or file_sha256(label_path) != hashes["label"]:
                raise ValueError(f"cora_loader_label_hash:{root_id}:{branch.get('proposal_id')}")
            proposal_digest = str(branch.get("proposal_sha256", ""))
            public = read_public_label(store, root_id, proposal_digest)
            if (
                public.get("proposal_id") != branch.get("proposal_id")
                or public.get("source_label_path") != paths["label"]
                or public.get("source_label_sha256") != hashes["label"]
            ):
                raise ValueError(
                    f"cora_loader_public_label_binding:{root_id}:{branch.get('proposal_id')}"
                )
            public_heads[str(branch.get("proposal_id"))] = public["heads"]
            public_label_paths[str(branch.get("proposal_id"))] = str(
                store.public_label_path(root_id, proposal_digest).relative_to(store.root)
            )
        prepared = dict(record)
        prepared["root_cluster_id"] = root_id
        prepared["feature_views"] = feature_views
        prepared["per_head_targets"] = {
            proposal_id: {
                str(name): {
                    "value": head.get("value"),
                    "mask": bool(head.get("valid", False)),
                    "unit": str(head.get("unit", "")),
                    "derivation_version": str(head.get("derivation_version", "")),
                }
                for name, head in heads.items()
                if isinstance(head, dict)
            }
            for proposal_id, heads in public_heads.items()
        }
        prepared["public_label_paths"] = public_label_paths
        prepared["intervention_edges"] = [
            edge for edge in record.get("edges", ()) if edge.get("edge_kind") == "intervention_base"
        ]
        output.append(prepared)
    if not output:
        raise ValueError("cora_loader_no_rows")
    return tuple(output)


__all__ = ["ALL_DATA_SPLITS", "TRAIN_ALLOWED_SPLITS", "load_cora_roots"]
