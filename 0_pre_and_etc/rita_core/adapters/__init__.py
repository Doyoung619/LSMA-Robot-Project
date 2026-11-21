"""Vendor libraries are imported only when their backend connects."""
from .base import RobotAdapter, AdapterError, CapabilityError, StaleFeedback, DispatchDeadlineMiss
from .mock import MockAdapter


def make_adapter(backend, profile, **kwargs):
    if backend == "mock":
        return MockAdapter(profile, **kwargs)
    if backend == "misty":
        from .misty import MistyAdapter
        return MistyAdapter(profile, **kwargs)
    if backend == "kinova":
        from .kinova import KinovaAdapter
        return KinovaAdapter(profile, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")
