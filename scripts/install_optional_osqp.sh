#!/usr/bin/env bash
# Optional local OSQP install for G2 longitudinal QP (tools/ is gitignored).
# Does not modify system Python; append-only site path is discovered by safety_kernel.repair.qp_solver.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${ROOT}/tools/wsl_site_packages"
mkdir -p "${TARGET}"
python3 -m pip install --target "${TARGET}" "osqp>=0.6.2,<2"
python3 - <<PY
import sys
sys.path.append("${TARGET}")
import osqp
print(f"osqp {osqp.__version__} installed at ${TARGET}")
PY
