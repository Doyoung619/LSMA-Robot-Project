"""Bounded nonblocking producer; disk I/O and rolling metrics have one worker."""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
import uuid
import numpy as np

from .metrics import DEFAULT_CONFIG, evaluate_samples


SCHEMA_VERSION = 1
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runs"


class TelemetryError(RuntimeError):
    """Logging cannot keep up or the journal cannot be written."""


def serial(value):
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serial(v) for v in value]
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    """Readers see either the old or the new complete document."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, allow_nan=False,
                                    indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class RolloutRecorder:
    def __init__(self, root=None, *, profile, analysis=None, metadata=None,
                 metric_config=None, metric_interval=1., window_seconds=120.,
                 queue_size=8192, metrics=None):
        if metric_interval <= 0 or window_seconds <= 0 or queue_size < 1:
            raise ValueError("Positive metric interval, window and queue size required")
        self.profile, self.analysis = profile, analysis
        self.config = {**DEFAULT_CONFIG, **(metric_config or {})}
        self.metric_interval, self.window_seconds = metric_interval, window_seconds
        self.metrics = dict(metrics or {})
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ_") + uuid.uuid4().hex[:10]
        self.run_dir = Path(root or DEFAULT_ROOT).resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        profile_data = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
        self.nj = len(profile_data["joint_names"])
        self.manifest = {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "robot": serial(profile_data),
            "profile_fingerprint": profile.fingerprint() if hasattr(profile, "fingerprint") else None,
            "audio": serial(analysis.to_dict() if hasattr(analysis, "to_dict") else analysis),
            "metadata": serial(metadata or {}), "metric_configuration": self.config,
            "rolling_window_seconds": window_seconds,
            "sample_timing": "music_time uses source host-monotonic receipt mapped to music clock; observed_monotonic is polling time; transport delay remains included",
            "synchronization_scope": "estimated beat reference and host receipt timing; not acoustic ground truth",
            "files": {name: name + ".jsonl" for name in ("samples", "events", "metrics")},
        }
        write_json(self.run_dir / "manifest.json", self.manifest)
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._closed = False
        self._lock = threading.Lock()
        self._sequence = 0
        # A frozen or backward source clock must not grow metric memory forever.
        self._max_buffer_samples = max(128, int(window_seconds / profile_data.get("command_period", .025) * 8) + 1)
        self._buffers = defaultdict(lambda: deque(maxlen=self._max_buffer_samples))
        self._clipped_buffers = Counter()
        self._last_metric = {}
        self._counts = Counter()
        self._thread = threading.Thread(target=self._work, name="rita-telemetry", daemon=True)
        self._thread.start()

    def _submit(self, stream, payload):
        # Validation/serialization precedes acceptance; the caller's mutable
        # arrays and adapter dictionaries cannot change already queued data.
        data = serial(payload)
        json.dumps(data, allow_nan=False)
        with self._lock:
            if self._closed: raise TelemetryError("Recorder is closed")
            if self._error: raise TelemetryError(self._error)
            data.update(sequence=self._sequence, logged_monotonic=time.monotonic())
            self._sequence += 1
            try: self._queue.put_nowait((stream, data))
            except queue.Full:
                self._error = "Telemetry queue full; refusing silent sample loss"
                raise TelemetryError(self._error)

    def record_sample(self, source, music_time, q, dq=None, *, feedback_monotonic=None,
                      observed_monotonic=None, fresh=True, metadata=None):
        if not isinstance(source, str) or not source.strip(): raise ValueError("A sample source is required")
        q = np.asarray(q, dtype=float)
        if q.shape != (self.nj,) or not np.isfinite(q).all():
            raise ValueError("Sample q must match ordered robot joints and be finite")
        if not np.isfinite(music_time): raise ValueError("Finite music time required")
        if dq is not None:
            dq = np.asarray(dq, dtype=float)
            if dq.shape != q.shape or not np.isfinite(dq).all(): raise ValueError("Invalid sample dq")
        self._submit("samples", {"source": source, "music_time": float(music_time),
            "q": q, "dq": dq, "fresh": bool(fresh), "feedback_monotonic": feedback_monotonic,
            "observed_monotonic": observed_monotonic, "metadata": metadata or {}})

    def record_event(self, kind, music_time=None, **payload):
        if not isinstance(kind, str) or not kind: raise ValueError("Event kind required")
        self._submit("events", {"kind": kind, "music_time": music_time, **payload})

    def record_segment(self, segment, stage, **payload):
        self.record_event("segment_" + stage, music_time=segment.start,
                          segment=asdict(segment), **payload)

    def update_manifest(self, **fields):
        self._submit("manifest", fields)

    def _snapshot(self, source, samples, *, provisional):
        result = evaluate_samples(samples, self.profile, self.analysis, self.config)
        extensions = {}
        for name, metric in self.metrics.items():
            try:
                value = serial(metric(samples, self.profile, self.analysis))
                json.dumps(value, allow_nan=False)
                extensions[name] = {"status": "ok", "value": value}
            except Exception as exc:
                extensions[name] = {"status": "error", "value": None, "error": str(exc)}
        if provisional:
            result.setdefault("quality", {})["rolling_count_limit"] = self._max_buffer_samples
            result["quality"]["rolling_count_clipped_samples"] = self._clipped_buffers[source]
        return {"source": source, "provisional": provisional,
                "window_start_seconds": samples[0]["music_time"] if samples else None,
                "window_end_seconds": samples[-1]["music_time"] if samples else None,
                **result, "extensions": extensions}

    def _work(self):
        files = {}
        try:
            files = {name: (self.run_dir / (name + ".jsonl")).open("a", encoding="utf-8")
                     for name in ("samples", "events", "metrics")}
            last_flush = time.monotonic()
            latest = {}
            while True:
                try: item = self._queue.get(timeout=.2)
                except queue.Empty:
                    for file in files.values(): file.flush()
                    continue
                if item is None: break
                stream, data = item
                if stream == "manifest":
                    self.manifest.update(data)
                    write_json(self.run_dir / "manifest.json", self.manifest)
                    continue
                files[stream].write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
                if stream == "samples":
                    source, t = data["source"], data["music_time"]
                    self._counts[source] += 1
                    buf = self._buffers[source]
                    if len(buf) == buf.maxlen: self._clipped_buffers[source] += 1
                    buf.append(data)
                    while buf and buf[0]["music_time"] < t - self.window_seconds: buf.popleft()
                    if t - self._last_metric.get(source, -float("inf")) >= self.metric_interval:
                        snapshot = self._snapshot(source, list(buf), provisional=True)
                        files["metrics"].write(json.dumps(serial(snapshot), ensure_ascii=False, allow_nan=False) + "\n")
                        latest[source] = snapshot
                        write_json(self.run_dir / "live.json", {"run_id": self.run_id,
                                   "provisional": True, "channels": latest})
                        self._last_metric[source] = t
                        for file in files.values(): file.flush()
                if time.monotonic() - last_flush >= .5:
                    for file in files.values(): file.flush()
                    last_flush = time.monotonic()
        except Exception as exc:
            self._error = f"Telemetry worker failed: {type(exc).__name__}: {exc}"
        finally:
            for file in files.values(): file.close()

    def close(self, status="completed", error=None, **metadata):
        with self._lock:
            if self._closed: return getattr(self, "summary", None)
            self._closed = True
        # Called after motion is stopped. Drain the worker before final replay.
        while self._thread.is_alive():
            try: self._queue.put(None, timeout=.2); break
            except queue.Full: continue
        self._thread.join()
        from .reader import iter_records
        grouped = defaultdict(list)
        for sample in iter_records(self.run_dir / "samples.jsonl"):
            grouped[sample["source"]].append(sample)
        channels = {source: self._snapshot(source, samples, provisional=False)
                    for source, samples in grouped.items()}
        self.summary = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
            "status": "failed" if self._error else status, "error": error or self._error,
            "telemetry_error": self._error, "sample_counts": dict(self._counts),
            "channels": channels, "metadata": serial(metadata),
            "ended_utc": datetime.now(timezone.utc).isoformat()}
        write_json(self.run_dir / "summary.json", self.summary)
        self.manifest.update(status=self.summary["status"], ended_utc=self.summary["ended_utc"],
                             error=self.summary["error"])
        write_json(self.run_dir / "manifest.json", self.manifest)
        if self._error: raise TelemetryError(self._error)
        return self.summary

    def __enter__(self): return self

    def __exit__(self, kind, value, traceback):
        self.close(status="failed" if kind else "completed", error=str(value) if value else None)
