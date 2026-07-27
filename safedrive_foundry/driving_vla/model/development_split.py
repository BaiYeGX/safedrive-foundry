"""Deterministic episode-group split for post-v4 Spatial K2 development.

The old v4 exam has already been observed and may be retired into development
data, but it must never be presented as a new blind exam.  This helper keeps
episodes intact and selects one validation episode per scenario family.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

REQUIRED_DEVELOPMENT_FAMILIES = (
    "left_cut_in",
    "right_cut_in",
    "lead_brake",
    "crossing",
    "empty",
)


def _tie_hash(seed: int, episode_id: str) -> str:
    return hashlib.sha256(f"{seed}|{episode_id}".encode("utf-8")).hexdigest()


def assign_balanced_episode_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 29,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        episode = str(row.get("episode_id") or "")
        if not episode:
            raise ValueError("development split requires episode_id")
        by_episode[episode].append(row)
    family_episodes: dict[str, list[str]] = defaultdict(list)
    for episode, episode_rows in by_episode.items():
        families = {str(row.get("scenario_family") or "") for row in episode_rows}
        if len(families) != 1:
            raise ValueError(f"episode {episode} spans families {sorted(families)}")
        family_episodes[next(iter(families))].append(episode)

    missing = [
        family
        for family in REQUIRED_DEVELOPMENT_FAMILIES
        if len(family_episodes.get(family, ())) < 2
    ]
    if missing:
        raise ValueError(
            "need at least two episodes per development family: "
            + ", ".join(missing)
        )

    val_episodes: set[str] = set()
    for family in REQUIRED_DEVELOPMENT_FAMILIES:
        candidates = family_episodes[family]
        # Prefer a larger/more eligible episode, then use a seeded stable tie.
        ranked = sorted(
            candidates,
            key=lambda episode: (
                -sum(
                    bool(row.get("alternative_available", False))
                    for row in by_episode[episode]
                ),
                -len(by_episode[episode]),
                _tie_hash(seed, episode),
            ),
        )
        val_episodes.add(ranked[0])

    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        was_exam = bool(row.get("is_r2_exam_fixture", False))
        row["was_r2_v4_exam_fixture"] = was_exam
        row["is_r2_exam_fixture"] = False
        row["retired_exam_source"] = was_exam
        row["split_id"] = (
            "val" if str(row["episode_id"]) in val_episodes else "train"
        )
        row["dataset_version"] = "v7-development-retired-v4"
        out.append(row)

    split_counts = Counter(str(row["split_id"]) for row in out)
    train_episodes = {
        str(row["episode_id"]) for row in out if row["split_id"] == "train"
    }
    val_episode_set = {
        str(row["episode_id"]) for row in out if row["split_id"] == "val"
    }
    if train_episodes & val_episode_set:
        raise ValueError("episode leakage between train and val")
    manifest = {
        "seed": int(seed),
        "policy": "one_episode_per_required_family_to_val",
        "required_families": list(REQUIRED_DEVELOPMENT_FAMILIES),
        "val_episodes": sorted(val_episodes),
        "n_train_episodes": len(train_episodes),
        "n_val_episodes": len(val_episode_set),
        "split_counts": dict(split_counts),
        "episode_leakage": False,
        "old_v4_exam_retired_to_development": True,
        "new_blind_exam_required": True,
    }
    return out, manifest
