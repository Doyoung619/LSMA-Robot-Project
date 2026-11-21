"""Lazy model registry: classical use never imports a neural framework."""
import importlib

METHODS = {
    "OUR_EXISTING": ("existing", "ExistingEstimator"),
    "DeepRhythm": ("deeprhythm", "DeepRhythmEstimator"),
    "TempoCNN": ("tempocnn", "TempoCNNEstimator"),
    "BeatThis": ("beat_this", "BeatThisEstimator"),
    "BeatNet": ("beatnet", "BeatNetEstimator"),
}


def get_estimator(name, device="cpu", threads=1):
    module, cls = METHODS[name]
    return getattr(importlib.import_module(f"adapters.{module}"), cls)(device=device, threads=threads)
