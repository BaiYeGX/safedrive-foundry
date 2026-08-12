"""R2 / G4A offline evaluation contracts.

This package is deliberately separate from runtime control namespaces.
Oracle helpers must never be imported by PathManager, SpeedPlanner, or MPC.
"""

from __future__ import annotations

# Do not re-export oracle helpers into a flat control-facing API.
# Import submodules explicitly: paired_contract, scenario_registry,
# comparability, outcome_metrics, oracle.

__all__: list[str] = []
