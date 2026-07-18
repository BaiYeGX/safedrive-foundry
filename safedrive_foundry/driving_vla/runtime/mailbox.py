"""Async-safe last-good candidate mailbox (control loop never joins GPU)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from safety_kernel.contracts.types import PolicyCandidateSet


@dataclass
class MailboxEntry:
    candidate_set: PolicyCandidateSet
    published_wall_s: float
    latency_s: float
    ok: bool = True
    reason: str = ""


class CandidateMailbox:
    """Thread-safe single-slot mailbox for ~5Hz VLA → 50Hz control."""

    def __init__(self, *, soft_stale_s: float = 0.20) -> None:
        self.soft_stale_s = soft_stale_s
        self._lock = threading.Lock()
        self._entry: Optional[MailboxEntry] = None
        self._vla_ok = True
        self._last_error = ""

    def publish(self, cset: PolicyCandidateSet, *, latency_s: float) -> None:
        with self._lock:
            self._entry = MailboxEntry(
                candidate_set=cset,
                published_wall_s=time.time(),
                latency_s=latency_s,
                ok=True,
            )
            self._vla_ok = True
            self._last_error = ""

    def mark_degraded(self, reason: str) -> None:
        with self._lock:
            self._vla_ok = False
            self._last_error = reason
            if self._entry is not None:
                self._entry = MailboxEntry(
                    candidate_set=self._entry.candidate_set,
                    published_wall_s=self._entry.published_wall_s,
                    latency_s=self._entry.latency_s,
                    ok=False,
                    reason=reason,
                )

    def latest(self, *, now_wall_s: float | None = None) -> Optional[MailboxEntry]:
        now = time.time() if now_wall_s is None else now_wall_s
        with self._lock:
            if self._entry is None:
                return None
            age = now - self._entry.published_wall_s
            if age > self.soft_stale_s:
                return MailboxEntry(
                    candidate_set=self._entry.candidate_set,
                    published_wall_s=self._entry.published_wall_s,
                    latency_s=self._entry.latency_s,
                    ok=False,
                    reason="soft_stale",
                )
            return self._entry

    def vla_ok(self) -> bool:
        with self._lock:
            return self._vla_ok

    def last_error(self) -> str:
        with self._lock:
            return self._last_error
