"""H4 locked evaluation package.

This package is intentionally separate from H3 training/evaluation.  It reads
test labels only through the locked-evaluation entry point, never from H3
training code.
"""
from .contracts import H4_CONFIG, H4_CONFIG_SHA256

__all__ = ["H4_CONFIG", "H4_CONFIG_SHA256"]
