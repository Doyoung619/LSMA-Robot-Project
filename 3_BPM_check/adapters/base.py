"""Shared contract; no neural or scientific dependencies at import time."""
import math


class BPMEstimator:
    name = "base"

    def __init__(self, device="cpu", threads=1):
        self.device = device
        self.threads = threads
        self.metadata = {}

    def load(self):
        raise NotImplementedError

    def predict(self, audio_path):
        raise NotImplementedError


def result(method, bpm=None, beat_times=None, confidence=None, status="success", error=None):
    if status == "success" and (bpm is None or not math.isfinite(float(bpm)) or float(bpm) <= 0):
        status, error, bpm = "error", "No finite positive BPM", None
    return {"method": method, "bpm": None if bpm is None else float(bpm),
            "beat_times": None if beat_times is None else [float(x) for x in beat_times],
            "confidence": None if confidence is None else float(confidence),
            "status": status, "error": error}


def bpm_from_beats(beats):
    """Median of finite, strictly positive consecutive intervals; no octave correction."""
    values = [float(x) for x in beats]
    intervals = [b-a for a, b in zip(values, values[1:])
                 if math.isfinite(a) and math.isfinite(b) and b > a]
    if not intervals:
        raise ValueError("Insufficient valid consecutive beat intervals")
    import statistics
    return 60.0 / statistics.median(intervals)
