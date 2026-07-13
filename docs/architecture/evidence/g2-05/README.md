# G2-05 Fault Matrix & Safety Stage Evidence (Offline)

Offline CPU fault injection and mode comparison. **Not** live CARLA short-loop VERIFIED.

## Semantics

- State hard faults (stale obs, privilege, non-finite ego/actors) **lock** the tick.
- Soft-stale / OOD / overconfident gates apply to **learning** sources only.
- Solver timeout never executes a timed-out solution.
- Matrix includes low_attachment / actuator_saturation / solver_timeout / model_timeout.

## Reproduce

```text
SDF_WRITE_G2_EVIDENCE=1 PYTHONPATH=safedrive_foundry python3 -m unittest tests.g2.test_g2_05_latency_evidence -v
```

## Limits

- No live CARLA short closed loop
- C3 claim remains offline MEASURED, not live VERIFIED
