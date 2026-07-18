# Pure VLA CARLA Demo — Status Brief for External Consult

**Date:** 2026-07-19  
**Project:** SafeDrive Foundry (CARLA 0.9.16 + ROS2 SIL; Windows CARLA + WSL2 Ubuntu client)  
**Hardware:** RTX 4080 16GB (single GPU, no iGPU), Intel i5-13600KF  
**Purpose of this doc:** Hand to ChatGPT / another expert to diagnose why pure VLA closed-loop driving fails (spins at intersections → D3D crash).

> **2026-07-18 update:** Sections 2–9 describe the failed historical pipeline and are
> retained as diagnostic evidence. The active implementation is now
> `run_g3_vla_mpc_minimal.py` → `run_g3_vla_mpc_stable.py`; see Section 11 and
> `G3_VLA_MPC_STABLE_RUNBOOK.md`. No new live pass is claimed yet.

---

## 1. User requirements (what they want)

### Must

1. **Pure VLA controls both steering and speed** in CARLA closed loop.  
   - VLA must produce **path + speed** (not only speed).  
   - Controller tracks VLA trajectory (direction + speed).  
2. **Do NOT use CARLA map centerline / waypoints as the driving reference** for control.  
   - Reason given: path should look like something transferable later (not “cheat” with HD map geometry as the plan).  
   - Map may still be used for: spawn, optional *navigation-style* target point in the VLA text prompt (like a coarse GPS goal), not as MPC/pure-pursuit path.  
3. **Watchable demo**: ~1 minute realtime, smoother than ~5 FPS slideshow.  
4. **Single machine**: one 4080 shared by Unreal Engine (CARLA) and PyTorch (VLA).  
5. **Map at CARLA process launch**, not mid-session `client.load_world()` — mid-session map switch causes **fatal** (shader / D3D device lost).  
6. **Not claiming G3 stage VERIFIED** for this visual demo; honest measurement only.

### Explicitly rejected by user

| Approach | Why rejected |
|----------|----------------|
| Geometry = CARLA lane centerline, VLA only sets speed | “Feels smooth” but not pure VLA; user forbids this for their goal |
| `set_transform` hard snap / lateral yank to road | Feels “dragged back”; not acceptable |
| Mid-session `load_world` to change town | Causes fatal; map must be in UE startup args |

### Acceptable intermediate (historical)

- “Smooth” demos that used **map centerline + VLA speed** were subjectively good (~30 FPS feel, clean turns) but **do not meet pure-VLA requirement**.

---

## 2. What we implemented (current demo stack)

**Main script:** `tests/g3/run_g3_vla_smooth_b_demo.py`  
**Evidence dir:** `docs/architecture/evidence/g3-05/visual_demo_b/`

### Pipeline (current pure-VLA mode)

```text
RGB camera (attach to ego)
    → async SimLingo VLA (CUDA, keep-on-GPU)
    → TrajectoryArray (map frame path + speeds)
    → temporal smooth + self-straighten (NO map centerline)
    → pure-pursuit (steer) + speed PID (throttle/brake)
    → carla.VehicleControl + world.tick (sync ~20 Hz, realtime paced)
```

### Model / runtime

| Item | Value |
|------|--------|
| Model | SimLingo neural V0 (InternVL2-1B vision + PEFT LoRA) |
| Checkpoint | project `models/simlingo/...` + InternVL2-1B local weights |
| Device | Force CUDA; fail if weights not on GPU |
| VRAM | ~1.9 GB resident, ~2.2 GB peak during forward |
| Infer latency | ~110–180 ms P50 (after warmup) |
| Async period | ~0.45–0.7 s between forwards (leave GPU time for UE) |
| Preprocess | Crop + InternVL dynamic_preprocess 448; tensors moved to CUDA bf16 |

### Control (current)

- **Not** dense MPC on reshaped path (reshape was suspected of distorting VLA geometry).  
- **Pure pursuit** on smoothed VLA polyline + **speed PID**.  
- Steer rate limit / LPF to reduce chatter.  
- Cap speed ~12–15 m/s in 1‑min tests.  
- **No** Safety Kernel / QP / RATO on this demo path.

### Boot order (anti D3D)

1. Connect CARLA **before** loading CUDA model.  
2. Spawn vehicle → settle ticks.  
3. Attach small RGB camera.  
4. **Then** load SimLingo on GPU.  
5. Map check: `current_map` must match `--map` or exit `MAP_MISMATCH` (no live `load_world`).

### CARLA start config

`safedrive_foundry/config/runtime/carla_start.toml`:

```text
arguments = "/Game/Carla/Maps/Town03 -windowed -ResX=800 -ResY=600 -quality-level=Low -nosound -dx11 -carla-rpc-port=2000"
```

Map must be in **startup ArgumentList**. `sdf sim ensure` reuses an already-running instance and **will not** change map if CARLA is already up on another town.

---

## 3. Observed failures (user + logs)

### A. Pure VLA driving quality

- User: after first turn / at junctions, car **spins like a headless fly** (乱转圈).  
- User: straight segments **wobble left-right** (扭扭车), sometimes on double yellow.  
- Log pattern (latest pure-pursuit run):  
  - Long stretches labeled `SELF_STRAIGHT` with low `lat_rms`.  
  - Then `CURVED` with `head_ddeg` ±25–37°, `lat_rms` ~2.8–3.3 m, **steer saturates ~0.5–0.9**.  
  - Positions loop in a small area near an intersection (e.g. around x∈[-45,-35], y∈[-72,-65]).  
  - Speed often low (~1–4 m/s) then bursts; VLA `v_mean` unstable.

### B. D3D / CARLA fatal

- User dialog:  
  `Unreal Engine is exiting due to D3D device being lost (0x887A020 INTERNAL_ERROR)`.  
- Client side:  
  `time-out of 60000ms while waiting for the simulator` on `world.tick` / actor destroy.  
- **Not** primarily “VRAM full”: Task Manager often shows free VRAM; crash correlates with **UE + CUDA on same 4080**, spawn/camera/load spikes, or long stalls during wild control.  
- Project already documents: **do not mid-session `load_world`** (shader fatal risk). Mid-session map switch was attempted once → 60s timeout and dead sim.

### C. Map mismatch

- Running instance often **Town10HD_Opt** while demo/config requested **Town03**.  
- `sdf sim ensure` with process already RUNNING keeps old map.  
- Fix requires: kill CarlaUE4 → launch with map in args → then demo `--map` check.

### D. Metrics from a “completed” 60 s-class run (older pipeline)

Example summary (not the junction-spin run): ~58 s sim, **only ~45 m** distance, VRAM peak ~2.2 GB, realtime ~20 FPS at 20 Hz sync. Low distance indicates poor progress / crawling, not healthy cruising.

---

## 4. What was tried (and outcomes)

| Experiment | Result |
|------------|--------|
| Map centerline geometry + VLA speed only | **Smooth**, good turns, high wall FPS; **rejected** (not pure VLA) |
| VLA path projected into map corridor | Less vanish; still map-dependent; user rejected |
| Soft corridor + hard `set_transform` stick | Felt “dragged”; rejected |
| Pure VLA + dense MPC + reshape | Wobbly / erratic |
| Pure VLA + temporal smooth + self-straighten + PP/PID | Still **spins at intersection**, then **D3D timeout** |
| CUDA memory fraction 0.35–0.5 | VRAM looked empty; later removed; model fully on CUDA now |
| Mid-session `load_world` | Timeout / fatal |
| Keep model on GPU after vehicle spawn | Correct boot order; model ~1.9–2.2 GB on CUDA |

---

## 5. Architecture constraints (project)

- Windows: CARLA 0.9.16 UE4.  
- WSL2 Ubuntu: Python client, SimLingo, demo scripts.  
- Same physical GPU for **D3D (UE)** and **CUDA (PyTorch)**.  
- SafeDrive Foundry product path is normally: VLA → Safety Kernel → MPC (G2/G3); this **visual pure-VLA demo bypasses Safety** on purpose for watchability.  
- G0 frozen; not part of this demo.

---

## 6. Hypotheses to validate (for GPT / expert)

Please treat these as open questions:

1. **Model capability / domain gap**  
   Is SimLingo V0 + this camera/prompt/target-point setup simply not closed-loop capable at junctions without map or stronger training?

2. **Closed-loop error accumulation**  
   Async 0.45–0.7 s updates + wrong path at intersection → pure pursuit locks onto a circling polyline → high steer forever.

3. **Self-straighten bug at junctions**  
   `lat_rms` / mean-heading logic may **mislabeled curved junction as straight** or **over-straighten**, producing inconsistent goals (log shows rapid STRAIGHT↔CURVED flips).

4. **Target point / language prompt**  
   Sparse nav polyline as target may send the model into wrong turn or U-turn intent at intersections.

5. **GPU dual-stack instability**  
   High steer thrash + continuous ticks + VLA forward → UE frame stall → Windows TDR → D3D device lost (even with free VRAM).

6. **Controller**  
   Pure pursuit on short/noisy VLA horizon may be unstable at low speed / high curvature; needs longer horizon, curvature-aware speed, or stop-and-replan.

7. **Camera**  
   320×180 may be too weak; 704×396 better but more GPU load with UE.

---

## 7. Questions for GPT (copy-paste)

1. Given single RTX 4080 shared by CARLA UE (D3D) and SimLingo (CUDA), what is a robust closed-loop architecture for **pure neural trajectory + speed** without HD-map centerline tracking?  

2. At urban intersections, VLA path shows large `head_ddeg` and `lat_rms`, steer saturates, vehicle circles, then `world.tick` times out / D3D device lost. Is the primary failure **policy**, **tracking**, or **GPU contention**? How to separate them experimentally?  

3. What post-processing on VLA trajectories (no map) is standard to stop junction spinning while still “pure VLA”? (e.g. confidence, progress check, freeze last good path, min radius, stop if path self-intersects)  

4. Is async inference every 0.5–0.7 s fundamentally too slow for junction closed loop with pure pursuit?  

5. How should **target waypoint** in SimLingo-style prompts be set for open-loop-to-closed-loop transfer without leaking map centerline into control?  

6. Practical Windows TDR / D3D device lost mitigations when UE and PyTorch share one GPU during aggressive vehicle motion.

---

## 8. Key file paths

| Path | Role |
|------|------|
| `tests/g3/run_g3_vla_smooth_b_demo.py` | Main pure-VLA visual demo |
| `safedrive_foundry/driving_vla/model/simlingo_runtime.py` | Load/forward, keep-on-GPU |
| `safedrive_foundry/driving_vla/model/neural_policy.py` | V0 policy, ego→map path |
| `safedrive_foundry/config/runtime/carla_start.toml` | CARLA launch + **startup map** |
| `docs/architecture/evidence/g3-05/visual_demo_b/` | Demo JSON summaries |
| `docs/architecture/G1_DEVELOPMENT_AND_ACCEPTANCE_REVIEW.md` | Notes: avoid live `load_world` |

---

## 9. One-paragraph summary

We need **pure VLA closed-loop control of throttle and steering in CARLA** (no HD-map centerline as the driven path), on a **single 4080** shared with UE. GPU residency and boot order are largely fixed (~2 GB CUDA). Driving quality is still poor: **at intersections the vehicle spins with saturated steering while VLA reports high curvature / high lateral RMS**, then **CARLA dies with D3D device lost / RPC timeout**. Map must be selected **at UE process start**, not via mid-session `load_world`. User rejects map-centerline “assisted” smoothness even though it worked visually. Looking for root-cause ranking and a pure-VLA-compatible fix path that can drive normally in CARLA first.

---

## 10. Causal re-analysis (accepted) — NOT “SimLingo is useless” yet

**Conclusion so far:** circling / wobble **cannot** be attributed solely to SimLingo. The glue code could manufacture circling even with a decent model. Observed behavior was **not** a clean “raw VLA → MPC” experiment.

### Confirmed glue bugs

| # | Bug | Location | Effect |
|---|-----|----------|--------|
| 1 | Fixed nav target `route_xy[10]` (~15–20 m from **spawn only**) | `neural_policy._route_target` | After car passes point, **ego-frame target is behind** → model prompted to U-turn / circle |
| 2 | Fake second target `target1 + (5,0)` | `simlingo_runtime.build_driving_input` | Wrong turn cue at junctions |
| 3 | 65% blend with previous **world** points, no time align | `smooth_vla_trajectory` | Pulls path **backward** toward stale rear points |
| 4 | “Lookahead” = farthest point from ego | `pure_vla_control` | On bent/self-intersecting path picks wrong tip → steer sat |
| 5 | Real MPC often not in the loop | demo used PP helper, not `ControlLoop` | Logs were not “VLA→MPC” |
| 6 | No per-frame dump of **raw** VLA path | demos | Cannot score model vs glue |

Causal chain:

```text
stale / behind VLA nav targets
→ VLA raw path (unknown quality)
→ unaligned temporal blend
→ self-straighten reshape
→ farthest-point pursuit (or non-MPC)
→ circle / wobble / D3D thrash
```

### Minimal fix experiment (implemented)

**Script:** `tests/g3/run_g3_vla_mpc_minimal.py`

Designed to stop after proving:

1. Advancing targets at ~**15 m** and ~**30 m** ahead of **current** ego (not fixed index).  
2. Both targets passed into SimLingo (`target_point2_xy` real, not +5 m fake).  
3. **Raw** VLA path used for control (no prev blend / self-straighten).  
4. `ControlLoop.set_trajectory` + `step`; count `mode == "mpc"`.  
5. Steer rad → CARLA `[-1,1]`.  
6. Debug draw: green raw VLA, yellow MPC ref, cyan ego trail.  
7. Logs: target behind?, path forward ratio, self-intersect, MPC mode, steer sat.  
8. Straight-ish spawn, **3–4 m/s**, ~20–25 s proof, no junction tuning.

**Policy fix (global):** `NeuralV0Policy._route_targets()` advances along route; `forward_numpy(..., target_point2_xy=...)`.

**How to interpret next run:**

| Green raw VLA path | Vehicle | Blame |
|--------------------|---------|--------|
| Always forward, stable | Still circles | Tracking / MPC / coord |
| Jumps left-right each frame | — | Model / camera / prompt |
| Targets log `behind=True` | Circles | Nav target bug still present |

**Do not claim “SimLingo cannot drive” until green raw path is logged with targets always ahead and real MPC tracking.**

---

## 11. Implemented replacement pipeline — awaiting Windows live verification

The causal analysis above led to a replacement rather than further tuning of the old
pure-pursuit/smoothing stack:

1. The camera mount, FOV and 2:1 input are aligned with the SimLingo calibration.
2. `NeuralV0Policy.predict_native()` exposes the original 20-point spatial path and
   speed head; the formal T10 canonicalizer remains backward compatible but is not
   used as a fake 20 Hz control path.
3. `VLAPathManager` arc-length-aligns overlapping predictions, commits a speed-aware
   near prefix, blends only the future tail, and rejects backward, self-intersecting,
   implausibly curved or discontinuous paths. Rejection holds the last committed
   VLA path; it does not substitute a map centerline.
4. `ConstrainedVLAMPC` uses a 2-second linear-bicycle QP with steering-angle,
   steering-rate and steering-acceleration constraints plus curvature speed limiting.
5. Inference is serialized with CARLA ticks on the shared RTX 4080 and synchronized
   before rendering resumes. Debug path drawing occurs only on VLA updates.
6. Passing now requires all of CTE RMS, steering saturation, sign-flip rate,
   MPC-use fraction, distance/displacement, accepted VLA paths and actual progress;
   “did not circle” alone cannot pass.

The 2026-07-19 speed/endurance revision additionally removes the erroneous
`max(vla_speed, 0.85 * cap)` floor. `--v-ref 15` is now only a cap. A dedicated
speed planner applies multiplicative calibration, immediately honors VLA braking,
and slew-limits only acceleration. The MPC limits speed by sustained curvature,
remaining VLA path braking distance and path freshness; an invalid geometry update
may lower speed but may not increase it. A rolling 600m coarse route replaces the
old fixed 160m route, and behind/exhausted route targets are rejected instead of
being replaced by a fake straight target.

The runner now records actual/target speed, path age, collisions, lane invasions,
off-road fraction, route progress, inference latency and peak VRAM. Five-minute
runs checkpoint every 60s. Town11/12/13 receive CARLA Large Map hero/tile-streaming
setup; unstable Large Maps are opt-in and excluded from random runs.

Offline result: all **64 tests** under `tests/g3` pass, including straight and R=30m
arc closed-loop simulations, semantic speed braking, short/stale path braking and
route exhaustion behavior. These changes are not yet live-revalidated after the
Windows driver restart. Follow the exact 20s → 60s → 300s sequence in
`G3_VLA_MPC_STABLE_RUNBOOK.md`; do not claim G3 VERIFIED from offline tests.
