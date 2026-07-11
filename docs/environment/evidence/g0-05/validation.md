# SafeDrive G0-05 offline validation

- Overall status: **PASS**
- Generated (UTC): `2026-07-11T18:12:04.475810+00:00`

| Check | Status | Message |
|---|---|---|
| `sync.config` | **PASS** | fixed-step configuration is legal |
| `sync.tick_master` | **PASS** | one configured tick master is declared |
| `determinism.same_seed` | **PASS** | same-seed traces have identical frames, event order and state hashes |
| `fault.duplicate_tick` | **PASS** | injection detected as duplicate_tick |
| `fault.missing_frame` | **PASS** | injection detected as missing_frame |
| `fault.stale_message` | **PASS** | injection detected as stale_message |
| `fault.multiple_tick_masters` | **PASS** | injection detected as multiple_tick_masters |
| `environment.carla_not_started` | **PASS** | environment fault detected as carla_not_started |
| `environment.port_conflict` | **PASS** | environment fault detected as port_conflict_or_non_carla_service |
| `environment.version_mismatch` | **PASS** | environment fault detected as carla_version_mismatch |
| `environment.gpu_not_visible` | **PASS** | environment fault detected as gpu_not_visible |
| `environment.disk_low` | **PASS** | environment fault detected as low_disk_space |
| `carla.settings` | **PASS** | CARLA fixed-step settings satisfy the contract |
| `recovery.checkpoint_resume` | **PASS** | interrupted smoke resumes from an atomic checkpoint and matches a clean run |

## Evidence

```json
{
  "total": 14,
  "passed": 14,
  "failed": 0
}
```
