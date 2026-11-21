from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json
import numpy as np


def array(value, name, n=None):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)) or (n is not None and len(a) != n):
        raise ValueError(f"{name}: expected finite vector of length {n}, got {a.shape}")
    return a


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if hasattr(value, "to_dict"): return value.to_dict()
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    return value


@dataclass
class BoundaryState:
    t: float
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray

    def __post_init__(self):
        self.q = array(self.q, "boundary.q")
        self.dq = array(self.dq, "boundary.dq", len(self.q))
        self.ddq = array(self.ddq, "boundary.ddq", len(self.q))
        if not np.isfinite(self.t): raise ValueError("boundary time must be finite seconds")


@dataclass
class AudioAnalysis:
    source: str
    audio_hash: str
    sample_rate: int
    duration: float
    beats: np.ndarray
    onset_times: np.ndarray
    onset_strength: np.ndarray
    energy: np.ndarray
    accent: np.ndarray
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("beats", "onset_times", "onset_strength", "energy", "accent"):
            setattr(self, name, array(getattr(self, name), name))
        if len(self.beats) < 2 or np.any(np.diff(self.beats) <= 0):
            raise ValueError("beat timestamps must be strictly increasing seconds, at least two")
        if len(self.onset_times) != len(self.onset_strength) or len(self.energy) != len(self.onset_times):
            raise ValueError("onset/energy lengths differ")
        if len(self.accent) != len(self.beats): raise ValueError("one accent per analyzed beat is required")
        if self.sample_rate <= 0 or self.duration <= 0: raise ValueError("invalid audio rate/duration")

    @property
    def bpm(self): return float(60/np.median(np.diff(self.beats)))

    @property
    def beat_timestamps(self): return self.beats

    def beat_time(self, index):
        if index < -1e-8 or index > len(self.beats)-1+1e-8:
            raise ValueError(f"beat index {index} outside analyzed grid 0..{len(self.beats)-1}")
        return float(np.interp(index, np.arange(len(self.beats)), self.beats))

    def to_dict(self): return serial(asdict(self))


@dataclass
class RobotProfile:
    name: str
    model: str
    joint_names: tuple
    neutral: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    vmax: np.ndarray
    amax: np.ndarray
    units: str = "rad"
    command_mode: str = "mock"
    command_period: float = .025
    feedback_freshness: float = .25
    latency_seconds: float = 0.
    dispatch_lead_seconds: float = .1
    latency_mode: str = "dispatch"
    workspace: dict = field(default_factory=dict)
    local_verified: bool = False
    endpoint: dict = field(default_factory=dict)
    min_excursion: float = .03
    max_stationary_seconds: float = .35
    speed_threshold: float = .005

    def __post_init__(self):
        self.joint_names = tuple(self.joint_names)
        n = len(self.joint_names)
        if not n or len(set(self.joint_names)) != n: raise ValueError("joint names must be ordered and unique")
        if self.units != "rad": raise ValueError("internal profile units must explicitly be rad; conversion belongs in adapter")
        for name in ("neutral", "lower", "upper", "vmax", "amax"):
            setattr(self, name, array(getattr(self, name), name, n))
        if np.any(self.lower >= self.upper) or np.any(self.neutral < self.lower) or np.any(self.neutral > self.upper):
            raise ValueError("neutral and joint position bounds are inconsistent")
        if np.any(self.vmax <= 0) or np.any(self.amax <= 0): raise ValueError("positive velocity/acceleration limits required")
        if min(self.command_period, self.feedback_freshness, self.min_excursion, self.max_stationary_seconds, self.speed_threshold) <= 0:
            raise ValueError("positive command/feedback/activity settings required")
        if self.latency_seconds < 0 or self.dispatch_lead_seconds < 0: raise ValueError("latency must be nonnegative")
        if self.latency_mode not in ("dispatch", "timestamp"): raise ValueError("unknown latency mode")

    @property
    def nj(self): return len(self.joint_names)
    @property
    def dof(self): return self.nj
    def to_dict(self): return serial(asdict(self))
    def fingerprint(self): return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass
class RobotState:
    t: float
    q: np.ndarray
    dq: np.ndarray
    joint_names: tuple
    units: str = "rad"
    fresh: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.joint_names = tuple(self.joint_names)
        self.q = array(self.q, "feedback.q", len(self.joint_names))
        self.dq = array(self.dq, "feedback.dq", len(self.joint_names))
        if self.units != "rad": raise ValueError("adapter feedback must already be converted to rad")
        if not np.isfinite(self.t): raise ValueError("feedback needs finite monotonic timestamp")


@dataclass
class DanceSegment:
    start: float
    duration: float
    knots: np.ndarray
    controls: np.ndarray
    accent_times: np.ndarray = field(default_factory=lambda: np.array([]))
    accent_strengths: np.ndarray = field(default_factory=lambda: np.array([]))
    rhythm_id: str = ""
    metrical_level: int = 0
    metadata: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)

    def __post_init__(self):
        self.knots = array(self.knots, "knots")
        self.controls = np.asarray(self.controls, dtype=float)
        if self.controls.ndim != 3 or self.controls.shape[1] != 6 or self.controls.shape[0] != len(self.knots)-1:
            raise ValueError("controls must have shape (pieces,6,joints) for quintic Bezier")
        if not np.all(np.isfinite(self.controls)) or np.any(np.diff(self.knots) <= 0): raise ValueError("invalid controls/knots")
        if abs(self.start-self.knots[0]) > 1e-7 or abs(self.start+self.duration-self.knots[-1]) > 1e-7:
            raise ValueError("segment and knot timestamps disagree")
        self.accent_times = array(self.accent_times, "accent_times")
        self.accent_strengths = array(self.accent_strengths, "accent_strengths")
        if len(self.accent_times) and not len(self.accent_strengths): self.accent_strengths = np.ones(len(self.accent_times))
        if len(self.accent_strengths) != len(self.accent_times): raise ValueError("accent time/strength lengths disagree")
        if len(self.accent_times) and (self.accent_times.min() < self.start-1e-8 or self.accent_times.max() > self.end+1e-8):
            raise ValueError("accent timestamp outside segment")
        if self.metrical_level not in (0, 1, 2, 3): raise ValueError("metrical level must be 0..3")

    @property
    def end(self): return self.start+self.duration

    def at(self, t):
        from .trajectory import evaluate
        return evaluate(self, t)

    def start_state(self):
        return BoundaryState(self.start, *self.at(self.start))

    def end_state(self):
        return BoundaryState(self.end, *self.at(self.end))


@dataclass
class PlannerState:
    current: BoundaryState
    reserved: BoundaryState | None = None
    queue: list = field(default_factory=list)
    visit_history: list = field(default_factory=list)
    recent_descriptors: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    fallback: str | None = None
