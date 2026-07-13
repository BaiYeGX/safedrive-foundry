# G1-03 evidence

## Real CARLA OpenDRIVE

- Source: CARLA 0.9.16 `Content/Carla/Maps/OpenDrive/{Town01,Town03,Town10HD}.xodr`
- Copies: `carla-opendrive/` and `safedrive_foundry/classic_stack/map/fixtures/carla/`
- Hashes: `carla-opendrive/manifest.json` / `fixtures/carla/manifest.json`
- Graph cache samples: `map-cache-carla/`
- Summary: `summary.json`

## Notes

- Synthetic fixture regression remains under `fixtures/Town0*.xodr` for fixed-node unit tests.
- Continuous `load_world` map switching is avoided after a Fatal Error on this host.
