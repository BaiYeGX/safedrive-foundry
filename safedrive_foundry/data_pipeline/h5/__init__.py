"""H5 closed-loop readiness package.

This package contains the World-on/off selector glue that must be used before
entering closed-loop evaluation: risk-gated scoring and candidate hysteresis.
"""
from .runtime import H5WorldRouter

__all__ = ["H5WorldRouter"]
