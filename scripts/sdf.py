"""Source-tree shim for the installed ``sdf`` entry point."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SOURCE = ROOT / "safedrive_foundry" / "ros_ws" / "src" / "safedrive_carla_bridge"
sys.path.insert(0, str(PACKAGE_SOURCE))

from safedrive_carla_bridge.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
