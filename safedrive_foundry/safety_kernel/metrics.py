"""Latency and coverage metrics helpers for Safety Kernel evidence."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass(frozen=True)
class LatencyReport:
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    deadline_ms: float
    deadline_miss_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "deadline_ms": self.deadline_ms,
            "deadline_miss_count": self.deadline_miss_count,
        }


def build_latency_report(
    samples_ms: Sequence[float],
    *,
    deadline_ms: float,
    deadline_miss_count: int | None = None,
) -> LatencyReport:
    misses = deadline_miss_count
    if misses is None:
        misses = sum(1 for v in samples_ms if v > deadline_ms)
    if not samples_ms:
        return LatencyReport(
            count=0,
            p50_ms=float("nan"),
            p95_ms=float("nan"),
            p99_ms=float("nan"),
            max_ms=float("nan"),
            mean_ms=float("nan"),
            deadline_ms=deadline_ms,
            deadline_miss_count=0,
        )
    return LatencyReport(
        count=len(samples_ms),
        p50_ms=percentile(samples_ms, 0.50),
        p95_ms=percentile(samples_ms, 0.95),
        p99_ms=percentile(samples_ms, 0.99),
        max_ms=max(samples_ms),
        mean_ms=statistics.fmean(samples_ms),
        deadline_ms=deadline_ms,
        deadline_miss_count=int(misses),
    )
