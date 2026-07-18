"""Leakage auditor: cross-split, wrong-frame, regression write, near-dup (G3-01)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable

from data_pipeline.vla.schema import SampleRecord, content_hash, payload_hash
from data_pipeline.vla.split import SplitAssigner, SplitName


class LeakageError(ValueError):
    """Raised when a sample would introduce train/eval leakage or corrupt frozen sets."""


def near_dup_fingerprint(sample: SampleRecord) -> str:
    """Coarse fingerprint for near-duplicate detection (ignores frame identity).

    Uses image URI + route + quantized ego so nearby / near-identical observations
    collide even when full payload hashes differ.
    """
    pi = sample.layers.policy_input or {}
    ego = pi.get("ego_state") or {}
    try:
        x = float(ego.get("x", 0.0))
        y = float(ego.get("y", 0.0))
        yaw = float(ego.get("yaw", 0.0))
        v = float(ego.get("v", 0.0))
    except (TypeError, ValueError):
        x = y = yaw = v = 0.0
    material = {
        "town": sample.identity.town,
        "route_id": sample.identity.route_id,
        "uri": pi.get("front_rgb_uri"),
        "route": pi.get("route"),
        "ego_q": {
            "x": int(round(x * 2)),
            "y": int(round(y * 2)),
            "yaw": int(round(yaw / 0.087)),
            "v": int(round(v * 2)),
        },
    }
    raw = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class LeakageAuditor:
    assigner: SplitAssigner
    near_dup_hash_prefix: int = 16
    # Full content hashes already committed per split (resume / exact re-write).
    committed: dict[SplitName, set[str]] = field(default_factory=dict)
    # Identity keys seen (run|frame|scenario|attempt) → split
    identity_index: dict[str, SplitName] = field(default_factory=dict)
    # Payload hashes (identity excluded) → split for cross-split / near-dup detection.
    payload_to_split: dict[str, SplitName] = field(default_factory=dict)
    # Prefix of near_dup_fingerprint → split (uses near_dup_hash_prefix).
    near_dup_prefix_index: dict[str, SplitName] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in SplitName:
            self.committed.setdefault(name, set())

    def _near_prefix(self, sample: SampleRecord) -> str:
        n = max(4, int(self.near_dup_hash_prefix))
        return near_dup_fingerprint(sample)[:n]

    def _ensure_hashes(self, sample: SampleRecord) -> tuple[str, str]:
        ph = payload_hash(sample)
        h = sample.content_hash or content_hash(sample)
        sample.content_hash = h
        return h, ph

    def check_admit(self, sample: SampleRecord, *, target_split: SplitName | None = None) -> SplitName:
        """Validate sample can be admitted; return assigned split. Raises LeakageError."""
        split = target_split or self.assigner.assign_sample(sample)
        key = sample.identity.key()
        h, ph = self._ensure_hashes(sample)
        near_pfx = self._near_prefix(sample)

        # 1) Wrong-frame / identity reuse in a different split
        if key in self.identity_index and self.identity_index[key] != split:
            raise LeakageError(
                f"cross_split_identity: {key} already in {self.identity_index[key].value}, "
                f"refusing {split.value}"
            )

        # 2) Same payload already in another split (cross-split leak)
        if ph in self.payload_to_split and self.payload_to_split[ph] != split:
            raise LeakageError(
                f"cross_split_content: payload {ph[:12]}… already in "
                f"{self.payload_to_split[ph].value}, refusing {split.value}"
            )

        # 3) Exact payload duplicate anywhere
        if ph in self.payload_to_split:
            raise LeakageError(f"near_duplicate: payload hash already in {self.payload_to_split[ph].value}")

        # 3b) Near-duplicate via coarse fingerprint prefix (uses near_dup_hash_prefix)
        if near_pfx in self.near_dup_prefix_index:
            raise LeakageError(
                f"near_duplicate: fingerprint prefix {near_pfx} already in "
                f"{self.near_dup_prefix_index[near_pfx].value} "
                f"(near_dup_hash_prefix={self.near_dup_hash_prefix})"
            )

        # 4) Exact content rewrite into same split
        if h in self.committed.get(split, set()):
            raise LeakageError(f"near_duplicate: content hash already in {split.value}")

        # 5) Regression frozen fields must not enter TRAIN
        if split == SplitName.TRAIN and sample.layers.regression_frozen:
            raise LeakageError(
                "regression_write: regression_frozen fields must not enter TRAIN split"
            )

        # 6) Forbid promoting regression-pinned identity into train/val/test
        assigned_natural = self.assigner.assign_sample(sample)
        if assigned_natural == SplitName.REGRESSION and split != SplitName.REGRESSION:
            raise LeakageError(
                f"regression_escape: identity is regression-pinned but target={split.value}"
            )

        return split

    def admit(self, sample: SampleRecord, *, target_split: SplitName | None = None) -> SplitName:
        split = self.check_admit(sample, target_split=target_split)
        h, ph = self._ensure_hashes(sample)
        self.committed[split].add(h)
        self.identity_index[sample.identity.key()] = split
        self.payload_to_split[ph] = split
        self.near_dup_prefix_index[self._near_prefix(sample)] = split
        return split

    def bulk_index(self, samples: Iterable[SampleRecord]) -> None:
        for s in samples:
            self.admit(s)
