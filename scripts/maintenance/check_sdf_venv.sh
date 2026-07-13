#!/usr/bin/env bash
set +e
echo "whoami: $(whoami)"
echo "home: $HOME"
ls -la "$HOME/.venvs" 2>/dev/null || echo "no $HOME/.venvs"
ls -la /home/sdf/.venvs 2>/dev/null || echo "no /home/sdf/.venvs"

for v in "$HOME/.venvs/sdf" /home/sdf/.venvs/sdf "$HOME/.venvs/carla_ros" /home/sdf/.venvs/carla_ros; do
  echo "--- try $v ---"
  if [ -x "$v/bin/python" ]; then
    "$v/bin/python" - <<'PY'
import sys
print("python", sys.executable, sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch:", type(e).__name__, e)
try:
    import carla
    print("carla ok")
except Exception as e:
    print("carla:", type(e).__name__, e)
PY
  else
    echo "missing $v/bin/python"
  fi
done
