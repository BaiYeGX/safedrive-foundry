#!/usr/bin/env python3
"""X5A: collect same-forward SimLingo driving features for ~32 anchors.

Modes (mutually exclusive intent):
  --synthetic-tensors   Offline unit path only. NEVER counts as X5A real PASS.
  --fixture-rgb-dir     Real SimLingo forward on frozen RGB (.npy/.png) + labels.
  --from-r2-pilot       Real SimLingo on readonly R2 paired-pilot anchor RGBs
                        (probe-only; NOT training data).
  --carla               Live CARLA RGB collect (requires preflight READY).

Fail-closed contract:
  - Without an explicit mode that yields real or synthetic rows, exit non-zero.
  - Non-synthetic modes MUST NOT silently fall back to synthetic tensors.
  - Each anchor stores mean64 + full_pool + raw FP16 path + full lineage hashes.

Output: docs/runtime-evidence/r2x-feature-probe/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "safedrive_foundry"))

from driving_vla.model.driving_feature import (  # noqa: E402
    extract_driving_feature_bundle,
    feature_vector_hash,
)

OUT = ROOT / "docs/runtime-evidence/r2x-feature-probe"
R2_PILOT = ROOT / "docs/runtime-evidence/r2-g4a-paired-pilot/pairs"

# Probe family coverage targets (labels for linear probe; not a train registry)
PROBE_FAMILY_ORDER = (
    "left_cut_in",
    "right_cut_in",
    "lead_brake",
    "obstruction",
    "crossing",
    "empty",
)


def _stable_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _sha_array(arr: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(arr).tobytes())


def _write_anchor(out: Path, sample: dict) -> None:
    aid = sample["anchor_id"]
    d = out / "anchors" / aid
    d.mkdir(parents=True, exist_ok=True)
    (d / "feature.json").write_text(
        json.dumps(sample, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _conflict_side_from_family(fam: str) -> str:
    f = fam.lower()
    if "left" in f:
        return "left"
    if "right" in f:
        return "right"
    if "empty" in f or "clear" in f:
        return "empty"
    if "lead" in f or "brake" in f:
        return "center"
    if "cross" in f or "merge" in f:
        return "center"
    if "obstruct" in f:
        return "center"
    return "unknown"


def _family_from_scenario_id(scenario_id: str) -> str:
    s = str(scenario_id or "").lower()
    if "cut" in s and ("left" in s or "l_" in s or s.endswith("_l")):
        return "left_cut_in"
    if "cut" in s and ("right" in s or "r_" in s or s.endswith("_r")):
        return "right_cut_in"
    if "cut" in s:
        return "left_cut_in" if "late" in s else "crossing"
    if "lead" in s or "brake" in s:
        return "lead_brake"
    if "obstruct" in s or "block" in s:
        return "obstruction"
    if "cross" in s or "merge" in s or "yield" in s:
        return "crossing"
    if "empty" in s or "clear" in s or "open" in s:
        return "empty"
    return s[:24] or "unknown"


def collect_synthetic(n: int = 32, seed: int = 0) -> list[dict]:
    """Synthetic adaptor tensors with L/R signal in early channels (probe sanity)."""
    rng = np.random.RandomState(seed)
    families = (
        ["left_cut_in"] * 7
        + ["right_cut_in"] * 7
        + ["lead_brake"] * 5
        + ["obstruction"] * 4
        + ["crossing"] * 4
        + ["empty"] * 5
    )
    rows: list[dict] = []
    for i in range(min(n, len(families))):
        fam = families[i]
        t, c = 8, 96
        arr = rng.randn(1, t, c).astype(np.float32) * 0.2
        if "left" in fam:
            arr[:, :, :16] += 2.5
            side = "left"
        elif "right" in fam:
            arr[:, :, 16:32] += 2.5
            side = "right"
        elif "empty" in fam:
            side = "empty"
        else:
            arr[:, :, 40:48] += 1.0
            side = "center"
        aid = f"syn_{_stable_id(f'{fam}_{i}')}"
        raw_path = OUT / "anchors" / aid / "raw_tokens_fp16.npy"
        bundle = extract_driving_feature_bundle(
            arr, require=True, raw_tensor_path=str(raw_path)
        )
        sample = {
            "anchor_id": aid,
            "scenario_family": fam,
            "conflict_side": side,
            "observation_hash": _stable_id(f"obs_{fam}_{i}"),
            "image_hash": _stable_id(f"img_{fam}_{i}"),
            "backbone_forward_id": f"syn-fwd-{i}",
            "simlingo_mode": "synthetic_tensor",
            "is_real_simlingo_feature": False,
            "driving_feature_ok": True,
            "mean64": list(bundle.mean64),
            "mean64_hash": bundle.mean64_hash,
            "driving_feature": list(bundle.mean64),
            "driving_feature_hash": bundle.mean64_hash,
            "full_pool": list(bundle.full_pool),
            "full_pool_hash": bundle.full_pool_hash,
            "raw_shape": list(bundle.raw_shape),
            "raw_dtype": bundle.raw_dtype,
            "raw_content_hash": bundle.raw_content_hash,
            "driving_feature_raw_hash": bundle.raw_content_hash,
            "raw_tensor_path": bundle.raw_tensor_path,
            "source_mean64": bundle.source_mean64,
            "source_full_pool": bundle.source_full_pool,
            "adaptor_name": bundle.adaptor_name,
            "note": "synthetic for offline probe pipeline only; not X5A real PASS",
        }
        rows.append(sample)
    return rows


def _discover_r2_pilot_rgb(max_n: int) -> list[dict[str, Any]]:
    """Readonly discovery of frozen R2 anchor RGBs (probe only, not train)."""
    found: list[dict[str, Any]] = []
    if not R2_PILOT.is_dir():
        return found
    for pair_dir in sorted(R2_PILOT.iterdir()):
        if not pair_dir.is_dir():
            continue
        # prefer attempt_*/anchor then top-level anchor
        candidates = sorted(pair_dir.glob("attempt_*/anchor/anchor_front_rgb.npy"))
        if not candidates:
            top = pair_dir / "anchor" / "anchor_front_rgb.npy"
            if top.is_file():
                candidates = [top]
        for rgb_path in candidates:
            cfg_path = rgb_path.parent / "run_config.json"
            scenario_id = "unknown"
            seed_id = "unknown"
            pair_id = pair_dir.name
            if cfg_path.is_file():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    scenario_id = str(cfg.get("scenario_id") or scenario_id)
                    seed_id = str(cfg.get("seed_id") or seed_id)
                    pair_id = str(cfg.get("pair_id") or pair_id)
                except Exception:  # noqa: BLE001
                    pass
            found.append(
                {
                    "rgb_path": rgb_path,
                    "scenario_id": scenario_id,
                    "seed_id": seed_id,
                    "pair_id": pair_id,
                    "scenario_family": _family_from_scenario_id(scenario_id),
                }
            )
            if len(found) >= max_n:
                return found
    return found


def _discover_fixture_dir(d: Path, max_n: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if not d.is_dir():
        return found
    for p in sorted(d.rglob("*.npy")):
        # skip raw token dumps if nested under probe out
        if "raw_tokens" in p.name:
            continue
        # label file optional: same stem .json with scenario_family
        meta_path = p.with_suffix(".json")
        fam = "unknown"
        side = "unknown"
        scenario_id = p.stem
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                fam = str(meta.get("scenario_family") or fam)
                side = str(meta.get("conflict_side") or side)
                scenario_id = str(meta.get("scenario_id") or scenario_id)
            except Exception:  # noqa: BLE001
                pass
        if fam == "unknown":
            fam = _family_from_scenario_id(scenario_id)
        if side == "unknown":
            side = _conflict_side_from_family(fam)
        found.append(
            {
                "rgb_path": p,
                "scenario_id": scenario_id,
                "seed_id": "fixture",
                "pair_id": p.stem,
                "scenario_family": fam,
                "conflict_side": side,
            }
        )
        if len(found) >= max_n:
            break
    return found


def _load_rgb(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"bad_rgb_shape:{arr.shape}")
    return arr


def collect_real_simlingo(
    entries: list[dict[str, Any]],
    *,
    out: Path,
    source_tag: str,
    keep_on_gpu: bool = True,
    stability_repeat: bool = True,
) -> list[dict]:
    """One SimLingo forward per entry; dump raw/full_pool/mean64; optional 2nd fwd."""
    from driving_vla.adapter.policy_adapter import ObservationBundle
    from driving_vla.model.neural_policy import NeuralV0Policy

    pol = NeuralV0Policy(lazy=True, keep_on_gpu=keep_on_gpu)
    pol.ensure_loaded()
    ckpt_hash = ""
    try:
        rt = pol.runtime
        if rt is not None:
            ckpt = getattr(rt, "checkpoint_path", None) or getattr(rt, "weights_path", None)
            if ckpt and Path(str(ckpt)).is_file():
                h = hashlib.sha256(Path(str(ckpt)).read_bytes()).hexdigest()[:16]
                ckpt_hash = h
    except Exception:  # noqa: BLE001
        ckpt_hash = "unknown"

    rows: list[dict] = []
    for i, ent in enumerate(entries):
        rgb_path = Path(ent["rgb_path"])
        image = _load_rgb(rgb_path)
        fam = str(ent.get("scenario_family") or "unknown")
        side = str(ent.get("conflict_side") or _conflict_side_from_family(fam))
        scenario_id = str(ent.get("scenario_id") or fam)
        seed_id = str(ent.get("seed_id") or "na")
        pair_id = str(ent.get("pair_id") or f"entry_{i}")
        aid = f"real_{_stable_id(f'{source_tag}_{pair_id}_{scenario_id}_{seed_id}_{i}')}"
        adir = out / "anchors" / aid
        adir.mkdir(parents=True, exist_ok=True)
        raw_path = adir / "raw_tokens_fp16.npy"
        np.save(str(adir / "front_rgb.npy"), image)

        img_hash = _sha_array(image)
        obs = ObservationBundle(
            run_id=f"x5a-collect-{aid}",
            frame_id=f"frame-{i}",
            scenario_id=scenario_id,
            simulation_time_s=0.0,
            wall_time_s=time.time(),
            carla_frame=0,
            ego_x=0.0,
            ego_y=0.0,
            ego_yaw=0.0,
            ego_v=5.0,
            route_xy=tuple((float(x), 0.0) for x in range(0, 40, 2)),
            front_rgb=image,
            meta={
                "official_contract": True,
                "image_layout": "bgr",
                "target_ego_1": (10.0, 0.0),
                "target_ego_2": (20.0, 0.0),
                "command_text": None,
                "prompt_mode": "target_point",
            },
        )
        t0 = time.perf_counter()
        native = pol.predict_native(obs)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        if not native.driving_feature_ok:
            raise RuntimeError(
                f"driving_feature_not_ok anchor={aid} err={native.driving_feature_error}"
            )
        if not native.driving_feature:
            raise RuntimeError(f"driving_feature_empty anchor={aid}")

        # Prefer full same-forward raw dump if runtime still holds bundle
        bundle_obj = None
        try:
            rt = pol.runtime
            if rt is not None and getattr(rt, "model", None) is not None:
                bundle_obj = getattr(rt.model, "_sdf_driving_feature_bundle", None)
        except Exception:  # noqa: BLE001
            bundle_obj = None

        mean64 = list(native.driving_feature)
        full_pool = list(native.driving_feature_full_pool or [])
        raw_shape = list(native.driving_feature_raw_shape or [])
        raw_dtype = str(native.driving_feature_raw_dtype or "")
        raw_hash = str(native.driving_feature_raw_hash or "")
        mean64_hash = str(native.driving_feature_hash or feature_vector_hash(mean64))
        full_hash = str(
            native.driving_feature_full_pool_hash or feature_vector_hash(full_pool)
        )
        raw_note = "from_native_fields"
        raw_tokens = None
        try:
            rt = pol.runtime
            if rt is not None and getattr(rt, "model", None) is not None:
                raw_tokens = getattr(rt.model, "_sdf_driving_raw_tokens", None)
        except Exception:  # noqa: BLE001
            raw_tokens = None
        if raw_tokens is not None:
            from driving_vla.model.driving_feature import dump_raw_tokens_fp16

            dumped_hash, dumped_shape, dumped_dtype = dump_raw_tokens_fp16(
                raw_tokens, raw_path
            )
            raw_hash = dumped_hash
            raw_shape = list(dumped_shape)
            raw_dtype = dumped_dtype
            raw_note = "raw_tokens_fp16"
        elif full_pool:
            np.save(
                str(raw_path),
                np.asarray(full_pool, dtype=np.float16).reshape(1, 1, -1),
            )
            raw_note = "full_pool_fp16_proxy"

        stability = {}
        if stability_repeat:
            native2 = pol.predict_native(obs)
            h2 = str(native2.driving_feature_hash or "")
            stability = {
                "repeat_mean64_hash": h2,
                "repeat_match": h2 == mean64_hash and bool(h2),
                "repeat_raw_hash": str(native2.driving_feature_raw_hash or ""),
            }
            if not stability["repeat_match"]:
                raise RuntimeError(
                    f"feature_unstable anchor={aid} h1={mean64_hash} h2={h2}"
                )

        path_xy = list(native.path_map_xy or ())
        path_hash = feature_vector_hash(
            [float(x) for p in path_xy for x in (p[0], p[1])]
        ) if path_xy else ""
        speed = list(native.speed_mps or ())
        speed_hash = feature_vector_hash(speed) if speed else ""

        sample = {
            "anchor_id": aid,
            "scenario_family": fam,
            "conflict_side": side,
            "scenario_id": scenario_id,
            "seed_id": seed_id,
            "pair_id": pair_id,
            "source_rgb_path": str(rgb_path.as_posix()),
            "observation_hash": img_hash,
            "image_hash": img_hash,
            "backbone_forward_id": f"v0-{mean64_hash}-{raw_hash}",
            "simlingo_mode": source_tag,
            "is_real_simlingo_feature": True,
            "model_checkpoint_hash": ckpt_hash,
            "driving_feature_ok": True,
            "mean64": mean64,
            "mean64_hash": mean64_hash,
            "driving_feature": mean64,
            "driving_feature_hash": mean64_hash,
            "full_pool": full_pool,
            "full_pool_hash": full_hash,
            "raw_shape": raw_shape,
            "raw_dtype": raw_dtype,
            "raw_content_hash": raw_hash,
            "driving_feature_raw_hash": raw_hash,
            "raw_tensor_path": str(raw_path.as_posix()),
            "raw_dump_note": raw_note,
            "source_mean64": str(native.driving_feature_source or "simlingo_driving_mean64_v1"),
            "source_full_pool": "simlingo_driving_full_pool_v1",
            "adaptor_name": "driving",
            "native_path_hash": path_hash,
            "native_speed_hash": speed_hash,
            "native_path_len": len(path_xy),
            "latency_ms": lat_ms,
            "peak_vram_mb": float(native.peak_vram_mb),
            "stability": stability,
            "note": (
                "real SimLingo same-forward feature; "
                "R2 pilot RGB is probe-only and excluded from head train sets"
            ),
        }
        rows.append(sample)
        print(
            f"[collect] {i+1}/{len(entries)} {aid} fam={fam} "
            f"mean64={mean64_hash} lat_ms={lat_ms:.0f}",
            flush=True,
        )
    return rows


def _preflight_status() -> str:
    import subprocess

    pre = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sdf.py"), "sim", "preflight", "--json"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    try:
        payload = json.loads(pre.stdout or "{}")
        return str(
            payload.get("status")
            or payload.get("preflight_status")
            or payload.get("state")
            or "UNKNOWN"
        )
    except Exception:  # noqa: BLE001
        return f"parse_failed_rc={pre.returncode}"


def _ego_lateral_lon(ego_tf: Any, npc_tf: Any) -> tuple[float, float]:
    import math

    eyaw = math.radians(float(ego_tf.rotation.yaw))
    dx = float(npc_tf.location.x) - float(ego_tf.location.x)
    dy = float(npc_tf.location.y) - float(ego_tf.location.y)
    lat = -math.sin(eyaw) * dx + math.cos(eyaw) * dy
    lon = math.cos(eyaw) * dx + math.sin(eyaw) * dy
    return lat, lon


def _side_from_lat(lat: float, *, thr: float = 0.8) -> str:
    if lat > thr:
        return "left"
    if lat < -thr:
        return "right"
    return "center"


def _family_from_fixture(scenario_id: str, family: str, side: str) -> str:
    fam = str(family or "").lower()
    sid = str(scenario_id or "").lower()
    if side == "empty":
        return "empty"
    if "cut" in fam or "cut" in sid:
        if side == "right":
            return "right_cut_in"
        return "left_cut_in"
    if "lead" in fam or "brake" in fam or "lead" in sid:
        return "lead_brake"
    if "cross" in fam or "cross" in sid or "merge" in sid:
        return "crossing"
    if "obstruct" in fam or "block" in sid:
        return "obstruction"
    if side == "empty":
        return "empty"
    return fam or sid or "unknown"


def _mirror_npc_lateral(fixture: Any) -> Any:
    """Mirror first NPC across ego longitudinal axis → right-side cut-in variant."""
    from dataclasses import replace
    import math

    if not fixture.actors:
        return fixture
    ego = fixture.ego
    npc = fixture.actors[0]
    yaw = math.radians(float(ego.transform.yaw_deg))
    # vector ego→npc
    dx = float(npc.transform.x) - float(ego.transform.x)
    dy = float(npc.transform.y) - float(ego.transform.y)
    lat = -math.sin(yaw) * dx + math.cos(yaw) * dy
    lon = math.cos(yaw) * dx + math.sin(yaw) * dy
    # flip lateral
    lat_m = -lat
    # back to map
    fx, fy = math.cos(yaw), math.sin(yaw)
    lx, ly = -math.sin(yaw), math.cos(yaw)
    nx = float(ego.transform.x) + lon * fx + lat_m * lx
    ny = float(ego.transform.y) + lon * fy + lat_m * ly
    new_tf = replace(npc.transform, x=nx, y=ny)
    # flip lateral velocity component similarly
    vx, vy = float(npc.initial_velocity.vx), float(npc.initial_velocity.vy)
    v_lon = math.cos(yaw) * vx + math.sin(yaw) * vy
    v_lat = -math.sin(yaw) * vx + math.cos(yaw) * vy
    v_lat = -v_lat
    nvx = v_lon * fx + v_lat * lx
    nvy = v_lon * fy + v_lat * ly
    new_vel = replace(npc.initial_velocity, vx=nvx, vy=nvy)
    new_npc = replace(npc, transform=new_tf, initial_velocity=new_vel, name="cutter_right")
    rest = tuple(fixture.actors[1:])
    return replace(
        fixture,
        scenario_id=f"{fixture.scenario_id}_right",
        actors=(new_npc, *rest),
        notes=(fixture.notes or "") + " | mirrored_right_for_x5a_probe",
    )


def _empty_fixture(fixture: Any) -> Any:
    from dataclasses import replace

    return replace(
        fixture,
        scenario_id=f"{fixture.scenario_id}_empty",
        actors=(),
        family="empty",
        notes=(fixture.notes or "") + " | ego_only_empty_probe",
    )


def collect_carla_live(
    n: int,
    out: Path,
    *,
    stability_repeat: bool = True,
    samples_per_episode: int = 3,
) -> list[dict]:
    """Live CARLA: registry fixtures + right-mirror + empty variants → SimLingo features."""
    import math
    from dataclasses import replace

    from driving_vla.adapter.policy_adapter import ObservationBundle
    from driving_vla.evaluation.fixture_runtime import (
        cleanup_session,
        connect_world,
        open_fixture_session,
        step_fixture,
    )
    from driving_vla.evaluation.paired_live import (
        _attach_sensors,
        _destroy_sensors,
        _ego_pose,
        _nav_targets_ego,
        _route_xy,
        _wait_camera,
    )
    from driving_vla.evaluation.scenario_registry import load_scenario_registry
    from driving_vla.model.neural_policy import NeuralV0Policy
    from driving_vla.model.simlingo_runtime import SIMLINGO_CAMERA_XYZ

    status = _preflight_status()
    if status != "READY":
        raise RuntimeError(f"CARLA_NOT_READY status={status}")

    reg_path = ROOT / "safedrive_foundry" / "config" / "g4a" / "scenario_registry_v1.toml"
    registry = load_scenario_registry(reg_path)

    # Diversity-first worklist so --n 32 still covers left/right/empty/lead/cross
    work: list[tuple[Any, str]] = []  # (fixture, variant_tag)
    cut_ins = [f for f in registry.fixtures if "cut" in f.scenario_id]
    leads = [f for f in registry.fixtures if "lead" in f.scenario_id]
    crosses = [f for f in registry.fixtures if "cross" in f.scenario_id]
    # empty first (3)
    for f in (leads[:1] + cut_ins[:1] + crosses[:1]):
        work.append((_empty_fixture(f), "empty"))
    # right-mirrored cut-ins (up to 4)
    for f in cut_ins[:4]:
        work.append((_mirror_npc_lateral(f), "right_mirror"))
    # left cut-ins base
    for f in cut_ins:
        work.append((f, "base"))
    # lead / cross
    for f in leads + crosses:
        work.append((f, "base"))
    # any remaining fixtures
    seen_ids = {(getattr(f, "scenario_id", ""), getattr(f, "seed_id", ""), v) for f, v in work}
    for f in registry.fixtures:
        key = (f.scenario_id, f.seed_id, "base")
        if key not in seen_ids:
            work.append((f, "base"))

    client, world = connect_world(host="127.0.0.1", port=2000, map_name="Town03", sync=True)
    pol = NeuralV0Policy(lazy=True, keep_on_gpu=True)
    pol.ensure_loaded()
    ckpt_hash = ""
    try:
        rt = pol.runtime
        if rt is not None:
            ckpt = getattr(rt, "checkpoint_path", None) or getattr(rt, "weights_path", None)
            if ckpt and Path(str(ckpt)).is_file():
                ckpt_hash = hashlib.sha256(Path(str(ckpt)).read_bytes()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        ckpt_hash = "unknown"

    rows: list[dict] = []
    tick_offsets = [0, 8, 16][: max(1, int(samples_per_episode))]

    try:
        for fix, variant in work:
            if len(rows) >= n:
                break
            session = None
            sb = None
            try:
                session = open_fixture_session(client, world, fix, settle_ticks=6)
                ego = next(s.actor for s in session.spawned if s.role == "ego")
                sb = _attach_sensors(world, ego)
                # gather multi-tick samples
                for ti, off in enumerate(tick_offsets):
                    if len(rows) >= n:
                        break
                    if off > 0:
                        step_fixture(session, n_ticks=off if ti == 0 else max(1, off - tick_offsets[ti - 1]), sim_dt_s=float(fix.sim_dt_s))
                    image = _wait_camera(sb, world, session, min_frames=2, max_ticks=40)
                    pose = _ego_pose(ego)
                    # live conflict side from NPC geometry
                    side = "empty"
                    lat_v = 0.0
                    lon_v = 0.0
                    npcs = [s for s in session.spawned if s.role != "ego"]
                    if not npcs:
                        side = "empty"
                    else:
                        ego_tf = ego.get_transform()
                        best = None
                        for sp in npcs:
                            la, lo = _ego_lateral_lon(ego_tf, sp.actor.get_transform())
                            if best is None or abs(lo) < abs(best[1]):
                                best = (la, lo)
                        lat_v, lon_v = best if best else (0.0, 0.0)
                        side = _side_from_lat(lat_v)
                    fam = _family_from_fixture(fix.scenario_id, fix.family, side)
                    if variant == "empty":
                        fam = "empty"
                        side = "empty"
                    elif variant == "right_mirror" and side == "center":
                        # still label intended right for probe
                        side = "right"
                        fam = "right_cut_in"

                    route_xy = _route_xy(fix)
                    tp1, tp2 = _nav_targets_ego(
                        route_xy, ego_x=pose.x, ego_y=pose.y, ego_yaw=pose.yaw
                    )
                    episode_id = f"{fix.scenario_id}__{fix.seed_id}__{variant}"
                    aid = f"live_{_stable_id(f'{episode_id}_t{ti}_{len(rows)}')}"
                    adir = out / "anchors" / aid
                    adir.mkdir(parents=True, exist_ok=True)
                    raw_path = adir / "raw_tokens_fp16.npy"
                    np.save(str(adir / "front_rgb.npy"), image)
                    img_hash = _sha_array(image)

                    obs = ObservationBundle(
                        run_id=f"x5a-live-{aid}",
                        frame_id=f"t{ti}",
                        scenario_id=fix.scenario_id,
                        simulation_time_s=float(world.get_snapshot().timestamp.elapsed_seconds),
                        wall_time_s=time.time(),
                        carla_frame=int(world.get_snapshot().frame),
                        ego_x=float(pose.x),
                        ego_y=float(pose.y),
                        ego_yaw=float(pose.yaw),
                        ego_v=float(pose.speed_mps),
                        route_xy=route_xy,
                        front_rgb=image,
                        meta={
                            "official_contract": True,
                            "image_layout": "bgr",
                            "target_ego_1": tp1,
                            "target_ego_2": tp2,
                            "command_text": None,
                            "prompt_mode": "target_point",
                            "camera_mount_xyz": list(SIMLINGO_CAMERA_XYZ),
                        },
                    )
                    t0 = time.perf_counter()
                    native = pol.predict_native(obs)
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    if not native.driving_feature_ok or not native.driving_feature:
                        raise RuntimeError(
                            f"feature_fail {aid} ok={native.driving_feature_ok} "
                            f"err={native.driving_feature_error}"
                        )

                    mean64 = list(native.driving_feature)
                    full_pool = list(native.driving_feature_full_pool or [])
                    mean64_hash = str(native.driving_feature_hash or feature_vector_hash(mean64))
                    full_hash = str(
                        native.driving_feature_full_pool_hash or feature_vector_hash(full_pool)
                    )
                    raw_hash = str(native.driving_feature_raw_hash or "")
                    raw_shape = list(native.driving_feature_raw_shape or [])
                    raw_dtype = str(native.driving_feature_raw_dtype or "")
                    raw_note = "from_native"
                    try:
                        rt = pol.runtime
                        raw_tokens = (
                            getattr(rt.model, "_sdf_driving_raw_tokens", None)
                            if rt is not None and getattr(rt, "model", None) is not None
                            else None
                        )
                    except Exception:  # noqa: BLE001
                        raw_tokens = None
                    if raw_tokens is not None:
                        from driving_vla.model.driving_feature import dump_raw_tokens_fp16

                        raw_hash, shape, raw_dtype = dump_raw_tokens_fp16(raw_tokens, raw_path)
                        raw_shape = list(shape)
                        raw_note = "raw_tokens_fp16"
                    elif full_pool:
                        np.save(
                            str(raw_path),
                            np.asarray(full_pool, dtype=np.float16).reshape(1, 1, -1),
                        )
                        raw_note = "full_pool_fp16_proxy"

                    stability: dict[str, Any] = {}
                    if stability_repeat:
                        n2 = pol.predict_native(obs)
                        h2 = str(n2.driving_feature_hash or "")
                        stability = {
                            "repeat_mean64_hash": h2,
                            "repeat_match": h2 == mean64_hash and bool(h2),
                        }
                        if not stability["repeat_match"]:
                            raise RuntimeError(
                                f"feature_unstable {aid} h1={mean64_hash} h2={h2}"
                            )

                    path_xy = [(float(p[0]), float(p[1])) for p in (native.path_map_xy or ())]
                    path_hash = (
                        feature_vector_hash([float(x) for p in path_xy for x in (p[0], p[1])])
                        if path_xy
                        else ""
                    )
                    speed_mps = [float(v) for v in (native.speed_mps or ())]
                    sample = {
                        "anchor_id": aid,
                        "scenario_family": fam,
                        "conflict_side": side,
                        "scenario_id": fix.scenario_id,
                        "seed_id": fix.seed_id,
                        "episode_id": episode_id,
                        "pair_id": episode_id,
                        "variant": variant,
                        "tick_index": ti,
                        "actor_lat_m": lat_v,
                        "actor_lon_m": lon_v,
                        "ego_v": float(pose.speed_mps),
                        "base_speed_mps": float(speed_mps[0]) if speed_mps else float(pose.speed_mps),
                        "native_path_xy": [[float(x), float(y)] for x, y in path_xy],
                        "native_speed_mps": speed_mps,
                        "observation_hash": img_hash,
                        "image_hash": img_hash,
                        "backbone_forward_id": f"v0-{mean64_hash}-{raw_hash}",
                        "simlingo_mode": "carla_live",
                        "is_real_simlingo_feature": True,
                        "model_checkpoint_hash": ckpt_hash,
                        "driving_feature_ok": True,
                        "mean64": mean64,
                        "mean64_hash": mean64_hash,
                        "driving_feature": mean64,
                        "driving_feature_hash": mean64_hash,
                        "full_pool": full_pool,
                        "full_pool_hash": full_hash,
                        "raw_shape": raw_shape,
                        "raw_dtype": raw_dtype,
                        "raw_content_hash": raw_hash,
                        "driving_feature_raw_hash": raw_hash,
                        "raw_tensor_path": str(raw_path.as_posix()),
                        "raw_dump_note": raw_note,
                        "source_mean64": str(
                            native.driving_feature_source or "simlingo_driving_mean64_v1"
                        ),
                        "source_full_pool": "simlingo_driving_full_pool_v1",
                        "adaptor_name": "driving",
                        "native_path_hash": path_hash,
                        "native_path_len": len(path_xy),
                        "latency_ms": lat_ms,
                        "peak_vram_mb": float(native.peak_vram_mb),
                        "stability": stability,
                        "note": "carla_live X5A probe; not automatic train inclusion",
                    }
                    rows.append(sample)
                    _write_anchor(out, sample)
                    print(
                        f"[live] {len(rows)}/{n} {aid} fam={fam} side={side} "
                        f"var={variant} mean64={mean64_hash} lat_ms={lat_ms:.0f}",
                        flush=True,
                    )
            finally:
                if sb is not None:
                    _destroy_sensors(sb)
                if session is not None:
                    cleanup_session(session, soft=True)
    finally:
        try:
            # leave server running; only clear actors
            from driving_vla.evaluation.fixture_runtime import purge_episode_actors

            purge_episode_actors(world, client=client)
        except Exception:  # noqa: BLE001
            pass

    if len(rows) < 8:
        raise RuntimeError(f"live_collect_too_few n={len(rows)}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="X5A feature collect (fail-closed)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument(
        "--synthetic-tensors",
        action="store_true",
        help="offline synthetic only; never counts as X5A real PASS",
    )
    ap.add_argument(
        "--from-r2-pilot",
        action="store_true",
        help="real SimLingo on frozen R2 pilot anchor RGB (probe only)",
    )
    ap.add_argument(
        "--fixture-rgb-dir",
        type=str,
        default="",
        help="directory of *.npy RGB (+ optional *.json labels)",
    )
    ap.add_argument(
        "--carla",
        action="store_true",
        help="live CARLA collect (preflight READY required)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-stability-repeat", action="store_true")
    ap.add_argument(
        "--samples-per-episode",
        type=int,
        default=3,
        help="CARLA live: anchors sampled per fixture episode",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Fresh collect: remove prior anchors so probe never mixes modes
    anchors_root = out / "anchors"
    if anchors_root.is_dir():
        import shutil

        shutil.rmtree(anchors_root)
    anchors_root.mkdir(parents=True, exist_ok=True)

    modes = sum(
        [
            bool(args.synthetic_tensors),
            bool(args.from_r2_pilot),
            bool(args.fixture_rgb_dir),
            bool(args.carla),
        ]
    )
    if modes == 0:
        print(
            "ERROR: specify exactly one of --synthetic-tensors | --from-r2-pilot | "
            "--fixture-rgb-dir DIR | --carla. Silent synthetic fallback is forbidden.",
            file=sys.stderr,
        )
        return 2
    if modes > 1:
        print("ERROR: specify only one collect mode", file=sys.stderr)
        return 2

    try:
        if args.synthetic_tensors:
            rows = collect_synthetic(n=args.n, seed=args.seed)
            mode = "synthetic_tensor"
            is_real = False
        elif args.from_r2_pilot:
            entries = _discover_r2_pilot_rgb(args.n)
            if not entries:
                raise RuntimeError("no R2 pilot anchor_front_rgb.npy found")
            rows = collect_real_simlingo(
                entries,
                out=out,
                source_tag="r2_pilot_frozen_rgb",
                stability_repeat=not args.no_stability_repeat,
            )
            mode = "r2_pilot_frozen_rgb"
            is_real = True
        elif args.fixture_rgb_dir:
            entries = _discover_fixture_dir(Path(args.fixture_rgb_dir), args.n)
            if not entries:
                raise RuntimeError(f"no RGB npy under {args.fixture_rgb_dir}")
            rows = collect_real_simlingo(
                entries,
                out=out,
                source_tag="fixture_rgb",
                stability_repeat=not args.no_stability_repeat,
            )
            mode = "fixture_rgb"
            is_real = True
        else:
            rows = collect_carla_live(
                args.n,
                out,
                stability_repeat=not args.no_stability_repeat,
                samples_per_episode=int(args.samples_per_episode),
            )
            mode = "carla_live"
            is_real = True
    except Exception as exc:  # noqa: BLE001
        err = {
            "schema_version": "safedrive.r2x.feature_collect.v1",
            "status": "FAILED",
            "error": f"{type(exc).__name__}:{exc}",
            "is_real_simlingo_feature": False,
        }
        (out / "collect_manifest.json").write_text(
            json.dumps(err, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(err, indent=2), flush=True)
        return 1

    # clear previous anchors only when writing new set into default tree?
    # write each
    for r in rows:
        _write_anchor(out, r)

    families = sorted({r["scenario_family"] for r in rows})
    manifest = {
        "schema_version": "safedrive.r2x.feature_collect.v1",
        "status": "OK",
        "n": len(rows),
        "out": str(out.as_posix()),
        "mode": mode,
        "is_real_simlingo_feature": is_real,
        "x5a_real_pass_eligible": bool(is_real and len(rows) >= 8),
        "anchor_ids": [r["anchor_id"] for r in rows],
        "families": families,
        "note": (
            "synthetic never satisfies X5A real gate; "
            "r2 pilot RGB is real feature but must not enter head train splits"
        ),
    }
    (out / "collect_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
