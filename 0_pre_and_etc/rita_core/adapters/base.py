"""Adapter errors and contracts shared without importing vendor SDKs."""
from abc import ABC, abstractmethod
import numpy as np


class AdapterError(RuntimeError):
    pass


class StaleFeedback(AdapterError):
    pass


class DispatchDeadlineMiss(AdapterError):
    pass


class CapabilityError(AdapterError):
    pass


def terminal_hold_start(segment):
    """Start of an EXACT constant C2 suffix, not a sampled low-speed guess."""
    index = len(segment.controls)
    final_q = segment.controls[-1, -1]
    while index > 0 and np.max(np.abs(segment.controls[index-1]-final_q)) <= 1e-12:
        index -= 1
    return float(segment.knots[index])


def require_motion_permission(profile, execute):
    if not execute:
        raise PermissionError("Hardware motion requires explicit --execute")
    if not profile.local_verified:
        raise PermissionError("Hardware motion requires a verified, gitignored local profile")
    if not profile.workspace.get("verified", False):
        raise PermissionError("Local workspace has not been reviewed; example joint boxes are not hardware certification")


def require_segment(segment, profile, stop_to_stop=False):
    if not segment.validation.get("valid", False):
        raise AdapterError("Refusing an unvalidated reference segment")
    for t in (segment.start, segment.end):
        q, dq, ddq = segment.at(t)
        if len(q) != len(profile.joint_names):
            raise AdapterError("Segment joint count differs from ordered robot profile")
        if not np.all(np.isfinite(np.r_[q, dq, ddq])):
            raise AdapterError("Nonfinite segment boundary")
        if stop_to_stop and (np.max(np.abs(dq)) > 1e-7 or np.max(np.abs(ddq)) > 1e-7):
            raise CapabilityError("This hardware adapter supports stop-to-stop chunks only; nonzero boundary flow cannot be queued safely")
    declared_names = segment.metadata.get("joint_names")
    if declared_names is not None and tuple(declared_names) != tuple(profile.joint_names):
        raise AdapterError("Segment joint_names order differs from profile")
    if segment.metadata.get("units", "rad") != "rad":
        raise AdapterError("Segment units must be rad")


class RobotAdapter(ABC):
    @abstractmethod
    def connect(self): ...
    @abstractmethod
    def get_capabilities(self): ...
    @abstractmethod
    def read_state(self): ...
    @abstractmethod
    def execute_segment(self, segment, scheduled_monotonic=None, now_music=None): ...
    @abstractmethod
    def stop(self): ...
    @abstractmethod
    def close(self): ...
