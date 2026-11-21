"""Lazy model registry: classical use never imports a neural framework."""
import importlib

METHODS = {
    "OUR_EXISTING": ("existing", "ExistingEstimator"),
    "OUR_V2": ("classical_v2", "ClassicalV2Estimator"),
    "OUR_V2_FAST": ("classical_v2_fast", "ClassicalV2FastEstimator"),
    "DeepRhythm": ("deeprhythm", "DeepRhythmEstimator"),
    "TempoCNN": ("tempocnn", "TempoCNNEstimator"),
    "BeatThis": ("beat_this", "BeatThisEstimator"),
    "BeatNet": ("beatnet", "BeatNetEstimator"),
}


def get_estimator(name, device="cpu", threads=1):
    module, cls = METHODS[name]
    return getattr(importlib.import_module(f"adapters.{module}"), cls)(device=device, threads=threads)
