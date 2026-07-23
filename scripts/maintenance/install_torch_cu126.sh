#!/usr/bin/env bash
# Install PyTorch into the project WSL venv (PEP 668 safe).
# Preferred: /home/sdf/.venvs/sdf  (G0 baseline)
# Fallback:  $HOME/.venvs/sdf
set -euo pipefail

PREFERRED="/home/sdf/.venvs/sdf"
FALLBACK="${HOME}/.venvs/sdf"

if [ -x "${PREFERRED}/bin/python" ]; then
  VENV="${PREFERRED}"
elif [ -x "${FALLBACK}/bin/python" ]; then
  VENV="${FALLBACK}"
else
  VENV="${PREFERRED}"
  if [ ! -d "$(dirname "${VENV}")" ]; then
    # Prefer /home/sdf if that user/home exists
    if [ -d /home/sdf ]; then
      mkdir -p /home/sdf/.venvs
      VENV="/home/sdf/.venvs/sdf"
    else
      mkdir -p "${HOME}/.venvs"
      VENV="${HOME}/.venvs/sdf"
    fi
  fi
  echo "Creating venv: ${VENV}"
  python3 -m venv "${VENV}"
fi

PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
echo "Using: ${PY}"
"${PY}" --version

# Match docs/RESOURCES.md when possible
TORCH_SPEC="${TORCH_SPEC:-torch==2.12.1}"
echo "Installing ${TORCH_SPEC} (cu126) ..."
"${PIP}" install -U pip
"${PIP}" install "${TORCH_SPEC}" --index-url https://download.pytorch.org/whl/cu126

echo "Verify:"
"${PY}" - <<'PY'
import torch
print("executable", __import__("sys").executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    s = torch.arange(1024, device="cuda").sum().item()
    print("cuda_sum_check", s)
else:
    print("device", None)
PY

echo "Activate later with:"
echo "  source ${VENV}/bin/activate"
echo "install_torch_cu126 done"
