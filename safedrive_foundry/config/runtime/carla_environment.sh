#!/usr/bin/env bash
# Compatibility shell adapter only.
# Prefer:  python3 scripts/sdf.py sim preflight|status|ensure
# Business code must use runtime.carla_connection.ConnectionResolver, not a
# copied IP. This script only exports CARLA_* for interactive shells.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source this file: source safedrive_foundry/config/runtime/carla_environment.sh" >&2
  exit 2
fi

SDF_RUNTIME_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SDF_WINDOWS_HOST="$(PYTHONPATH="$SDF_RUNTIME_ROOT/safedrive_foundry${PYTHONPATH:+:$PYTHONPATH}" python3 - "$SDF_RUNTIME_ROOT" <<'PY'
import sys
from pathlib import Path
from runtime.carla_connection import ConnectionResolver

root = Path(sys.argv[1])
print(ConnectionResolver(root).resolve_host(force_dynamic=True).host)
PY
)" || {
  echo "Cannot resolve the dynamic CARLA host; CARLA_HOST was not set." >&2
  return 1
}

export CARLA_HOST="$SDF_WINDOWS_HOST"
export CARLA_PORT="${CARLA_PORT:-2000}"
export CARLA_EXPECTED_VERSION="${CARLA_EXPECTED_VERSION:-0.9.16}"
export CARLA_TIMEOUT_SECONDS="${CARLA_TIMEOUT_SECONDS:-10.0}"

unset SDF_RUNTIME_ROOT
unset SDF_WINDOWS_HOST
