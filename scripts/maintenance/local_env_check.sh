#!/usr/bin/env bash
# SafeDrive local env check (run inside WSL Ubuntu-24.04)
# Prefer project venv for torch/carla VLA checks.
set +e
echo "=== WSL local_env_check ==="
echo "date: $(date -Iseconds)"
echo "whoami: $(whoami)"

SDF_VENV="${SDF_VENV:-/home/sdf/.venvs/sdf}"
SDF_PYTHON="${SDF_PYTHON:-${SDF_VENV}/bin/python}"

echo "--- uname ---"
uname -a

echo "--- nvidia-smi ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WSL nvidia-smi: NOT_FOUND"
fi

echo "--- system python3 (must NOT be used for VLA) ---"
command -v python3
python3 --version
python3 - <<'PY'
try:
    import torch
    print("system torch", torch.__version__)
except Exception as e:
    print("system torch:", type(e).__name__, e, "(expected under PEP 668)")
PY

echo "--- project venv: ${SDF_VENV} ---"
if [ -x "${SDF_PYTHON}" ]; then
  echo "python: ${SDF_PYTHON}"
  "${SDF_PYTHON}" --version
  "${SDF_PYTHON}" - <<'PY'
import sys
print("executable", sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
        print("cuda_sum_check", int(torch.arange(1024, device="cuda").sum().item()))
except Exception as e:
    print("torch:", type(e).__name__, e)
try:
    import carla
    print("carla ok")
except Exception as e:
    print("carla:", type(e).__name__, e)
PY
else
  echo "MISS ${SDF_PYTHON}"
  echo "Run: bash scripts/maintenance/install_torch_cu126.sh"
fi

echo "--- paths ---"
for p in \
  "/mnt/e/autonomous driving" \
  "/mnt/e/autonomous driving/simlingo-main" \
  "/mnt/e/autonomous driving/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt" \
  "/mnt/e/autonomous driving/models/InternVL2-1B/model.safetensors" \
  "/mnt/e/CARLA_0.9.16/CarlaUE4.exe" \
  "/mnt/e/autonomous driving/scripts/sdf.py" \
  "/home/sdf/.venvs/sdf/bin/python"
do
  if [ -e "$p" ]; then
    echo "OK   $p"
  else
    echo "MISS $p"
  fi
done

echo "--- ros2 ---"
if [ -f /opt/ros/jazzy/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  echo "ROS Jazzy setup: OK"
  command -v ros2 && echo "ros2: available"
else
  echo "ROS Jazzy setup: MISSING"
fi

cd "/mnt/e/autonomous driving" || exit 1

# Prefer project venv for doctor/sim when available
if [ -x "${SDF_PYTHON}" ]; then
  export PATH="${SDF_VENV}/bin:${PATH}"
fi

echo "--- sdf doctor ---"
python3 scripts/sdf.py doctor
echo "doctor_exit=$?"

echo "--- sdf sim status ---"
python3 scripts/sdf.py sim status
echo "sim_status_exit=$?"
echo "NOTE: if CARLA process is NOT_RUNNING, RPC_HANDSHAKE_FAILED is expected."

echo "--- sdf sim preflight ---"
python3 scripts/sdf.py sim preflight
echo "preflight_exit=$?"

echo "--- task catalog ---"
PYTHONPATH=. python3 scripts/maintenance/task_catalog_check.py
echo "catalog_exit=$?"

echo "=== local_env_check done ==="
echo "VLA rule: always use ${SDF_VENV} (not system python3, not carla_ros venv)."
