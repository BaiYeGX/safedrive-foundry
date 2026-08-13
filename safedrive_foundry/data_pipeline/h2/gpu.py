"""Device-level GPU memory sampling for H2 runtime evidence."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence


class GPUSamplingError(RuntimeError):
    """Raised when device-level memory admission cannot be measured."""


@dataclass(frozen=True)
class GPUSample:
    monotonic_s: float
    elapsed_s: float
    index: int
    uuid: str
    name: str
    total_mib: float
    used_mib: float

    @property
    def used_gib(self) -> float:
        return self.used_mib / 1024.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["used_gib"] = self.used_gib
        return payload


def _parse_query(stdout: str, *, monotonic_s: float, elapsed_s: float) -> tuple[GPUSample, ...]:
    samples: list[GPUSample] = []
    for index, row in enumerate(csv.reader(line for line in stdout.splitlines() if line.strip())):
        if len(row) < 5:
            continue
        try:
            samples.append(
                GPUSample(
                    monotonic_s=monotonic_s,
                    elapsed_s=elapsed_s,
                    index=int(row[0].strip()),
                    uuid=row[1].strip(),
                    name=row[2].strip(),
                    total_mib=float(row[3].strip()),
                    used_mib=float(row[4].strip()),
                )
            )
        except (TypeError, ValueError):
            continue
    return tuple(samples)


class GPUMemorySampler:
    """Poll nvidia-smi every 100 ms and retain an auditable sample timeline."""

    def __init__(self, *, interval_s: float = 0.1, gpu_index: int = 0) -> None:
        if interval_s <= 0.0:
            raise ValueError("GPU sample interval must be positive")
        self.interval_s = float(interval_s)
        self.gpu_index = int(gpu_index)
        self._started_s: float | None = None
        self._samples: list[GPUSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None

    @property
    def command(self) -> tuple[str, ...]:
        return (
            "nvidia-smi", "--id", str(self.gpu_index),
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        )

    def _query(self) -> tuple[GPUSample, ...]:
        now = time.monotonic()
        started = self._started_s if self._started_s is not None else now
        try:
            completed = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                check=False,
                timeout=max(0.5, self.interval_s * 4.0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._error = f"nvidia_smi:{type(exc).__name__}:{exc}"
            return ()
        if completed.returncode != 0:
            self._error = f"nvidia_smi_exit:{completed.returncode}:{(completed.stderr or '').strip()}"
            return ()
        parsed = _parse_query(completed.stdout, monotonic_s=now, elapsed_s=now - started)
        if not parsed:
            self._error = "nvidia_smi_empty_or_unparseable"
        return parsed

    def _record(self, values: Sequence[GPUSample]) -> None:
        if not values:
            return
        with self._lock:
            self._samples.extend(values)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._record(self._query())
            self._stop.wait(self.interval_s)

    def start(self) -> "GPUMemorySampler":
        if self._thread is not None:
            raise RuntimeError("GPU sampler already started")
        self._started_s = time.monotonic()
        initial = self._query()
        if not initial:
            raise GPUSamplingError(self._error or "GPU device memory query failed")
        self._record(initial)
        self._thread = threading.Thread(target=self._run, name="h2-gpu-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 8.0))
        self._record(self._query())
        with self._lock:
            samples = tuple(self._samples)
        peak = max(samples, key=lambda item: item.used_mib) if samples else None
        return {
            "interval_s": self.interval_s,
            "gpu_index": self.gpu_index,
            "command": list(self.command),
            "sample_count": len(samples),
            "error": self._error,
            "samples": [sample.to_dict() for sample in samples],
            "peak_used_gib": 0.0 if peak is None else peak.used_gib,
            "peak_used_mib": 0.0 if peak is None else peak.used_mib,
            "device": None if peak is None else {
                "index": peak.index,
                "uuid": peak.uuid,
                "name": peak.name,
                "total_mib": peak.total_mib,
            },
            "torch": self._torch_snapshot(),
        }

    def peak_used_gib(self) -> float:
        with self._lock:
            return max((sample.used_gib for sample in self._samples), default=0.0)

    @staticmethod
    def _torch_snapshot() -> dict[str, Any]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}
            return {
                "available": True,
                "allocated_mib": float(torch.cuda.memory_allocated(0)) / 1024**2,
                "reserved_mib": float(torch.cuda.memory_reserved(0)) / 1024**2,
                "peak_allocated_mib": float(torch.cuda.max_memory_allocated(0)) / 1024**2,
                "peak_reserved_mib": float(torch.cuda.max_memory_reserved(0)) / 1024**2,
            }
        except Exception as exc:  # pragma: no cover - host runtime dependent
            return {"available": False, "error": f"torch_snapshot:{type(exc).__name__}:{exc}"}


__all__ = ["GPUMemorySampler", "GPUSample", "GPUSamplingError"]
