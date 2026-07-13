# G2-04 Arbitration / Shadow / Fallback Evidence

Offline CPU regression for deterministic arbitration pipeline.

## Reproduce

```text
SDF_WRITE_G2_EVIDENCE=1 PYTHONPATH=safedrive_foundry python3 -m unittest tests.g2.test_g2_04_latency_evidence -v
```

## Limits

- Not live CARLA 50Hz VERIFIED
- Shadow is compare-only (no control / tick ownership)
- Fresh SafetyKernel per scenario (no mode pollution)
