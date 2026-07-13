# G2-01 Safety Kernel Evidence

Offline CPU regression for contracts / Validator / state machine.

## Reproduce

```text
SDF_WRITE_G2_EVIDENCE=1 PYTHONPATH=safedrive_foundry python3 -m unittest tests.g2.test_g2_01_latency_evidence -v
```

## Limits

- Not live CARLA 50Hz VERIFIED
