"""Explicit radian profiles. Hardware examples are NOT certified workspaces."""
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
from .contracts import RobotProfile

PROFILE_DIR = Path(__file__).with_name("profiles")


def default_profile(backend="mock"):
    if backend not in ("mock", "misty", "kinova"):
        raise ValueError(f"Unknown backend: {backend}")
    return load_profile(PROFILE_DIR / f"{backend}.example.json", backend=backend)


def mock_profile(**changes):
    profile = default_profile("mock")
    for key, value in changes.items():
        if not hasattr(profile, key):
            raise ValueError(f"Unknown RobotProfile field: {key}")
        setattr(profile, key, value)
    return profile


def load_profile(path=None, backend="mock"):
    if path is None:
        return default_profile(backend)
    path = Path(path)
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML profiles require optional PyYAML; JSON needs no extra package") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Robot profile must be an object")
    unknown = set(data) - {f.name for f in fields(RobotProfile)}
    if unknown:
        raise ValueError(f"Unknown profile fields: {sorted(unknown)}")
    required = {"name", "model", "joint_names", "neutral", "lower", "upper", "vmax", "amax", "units", "command_mode", "command_period", "feedback_freshness", "workspace", "endpoint", "local_verified"}
    missing = required-set(data)
    if missing:
        raise ValueError(f"Missing profile fields: {sorted(missing)}")
    if data["units"] != "rad":
        raise ValueError("Internal profile units must explicitly be rad; convert source limits deliberately")
    names = tuple(data["joint_names"])
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be a nonempty unique ordered list")
    data["joint_names"] = names
    for key in ("neutral", "lower", "upper", "vmax", "amax"):
        data[key] = np.asarray(data[key], dtype=float)
        if data[key].shape != (len(names),) or not np.all(np.isfinite(data[key])):
            raise ValueError(f"{key}: expected {len(names)} finite radian values in joint_names order")
    if np.any(data["lower"] >= data["upper"]) or np.any(data["neutral"] < data["lower"]) or np.any(data["neutral"] > data["upper"]):
        raise ValueError("Neutral pose must be inside strictly ordered bounds")
    if np.any(data["vmax"] <= 0) or np.any(data["amax"] <= 0):
        raise ValueError("Velocity and acceleration limits must be positive")
    for key in ("command_period", "feedback_freshness"):
        if not np.isfinite(data[key]) or data[key] <= 0:
            raise ValueError(f"{key} must be positive seconds")
    expected = {"mock": "mock", "misty": "misty_pose", "kinova": "follow_joint_trajectory"}[backend]
    if data["command_mode"] != expected:
        raise ValueError(f"Backend {backend} requires command_mode={expected}")
    if backend != "mock" and data["local_verified"] and not path.name.endswith((".local.json", ".local.yaml", ".local.yml")):
        raise ValueError("Verified hardware settings must be stored in a gitignored *.local.json/yaml profile")
    return RobotProfile(**data)
