"""Offline sampling of pinned, unmodified official Reachy pure move functions.

No ReachyMini object, hardware SDK, motion command, learned weight, random
selector, added transition interpolation or post-hoc scale selection is used.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import csv
from functools import lru_cache
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

from setup_external import SOURCE, ROOT, config, verify_source

METHOD_NATIVE = "Reachy primitive scheduler (native)"
METHOD_ADAPTED = "Reachy primitive scheduler + documented adaptation"
_NAMESPACE = "_rita_reachy_pinned_6808a976"


def _preview_defaults(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "DEFAULT_SKIPPED_MOVES":
            defaults["excluded_moves"] = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and node.name == "Config":
            for field in node.body:
                if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name) and field.target.id in ("beats_per_sequence", "start_move", "amplitude_scale", "control_ts"):
                    defaults[field.target.id] = ast.literal_eval(field.value)
    return defaults


@lru_cache(maxsize=1)
def _loaded():
    source = verify_source()
    settings = config()["scheduler"]
    preview = _preview_defaults(SOURCE / "examples" / "dance_demo.py")
    for key in ("excluded_moves", "beats_per_sequence", "start_move", "amplitude_scale"):
        if preview[key] != settings[key]:
            raise RuntimeError(f"Recorded scheduler differs from pinned official preview: {key}")
    # __init__.py imports DanceMove and the hardware SDK. Only the actual pure
    # upstream modules are needed: a private namespace gives their relative
    # imports a package context, without executing that optional wrapper.
    package = types.ModuleType(_NAMESPACE)
    package.__path__ = [str(SOURCE / "src" / "reachy_mini_dances_library")]
    package.__package__ = _NAMESPACE
    sys.modules[_NAMESPACE] = package
    dance = importlib.import_module(_NAMESPACE + ".collection.dance")
    moves = dance.AVAILABLE_MOVES
    order = [name for name in moves if name not in set(settings["excluded_moves"])]
    if not order or settings["start_move"] not in order:
        raise RuntimeError("Official preview start/order cannot be reproduced")
    start = order.index(settings["start_move"])
    order = order[start:]+order[:start]
    source.update({"available_move_count": len(moves), "available_move_ids": list(moves),
                   "preview_move_count": len(order), "preview_order": order,
                   "preview_cycle_beats": len(order)*settings["beats_per_sequence"],
                   "sdk_initialization_bypassed": True,
                   "loader_scope": "Unmodified official rhythmic_motion.py and collection/dance.py; private namespace bypasses package __init__ hardware-wrapper import only",
                   "original_control_period_seconds": preview["control_ts"]})
    return moves, order, source


def load_upstream():
    """Initialize and verify locally cached official code, outside timed loops."""
    return deepcopy(_loaded()[2])


def sample_native(times_seconds, beat_phase):
    """Sample the analytical counterpart of upstream's default cyclic preview.

    `beat_phase` is global cumulative musical beats: phase zero is the estimated
    first source beat. Crop calls do not reset scheduling or motion phase.
    Before that origin the native offset is neutral. Nonuniform beat timestamps
    can be converted to cumulative phase by the caller without a BPM constant.
    """
    times = np.asarray(times_seconds, dtype=float)
    phase = np.asarray(beat_phase, dtype=float)
    if times.ndim != 1 or phase.shape != times.shape or not len(times):
        raise ValueError("times_seconds and beat_phase must be matching nonempty vectors")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(phase)):
        raise ValueError("Sampling times and beat phases must be finite")
    if np.any(np.diff(times) <= 0) or np.any(np.diff(phase) < -1e-9):
        raise ValueError("Time must strictly increase and beat phase must not reverse")
    moves, order, source = _loaded()
    settings = config()["scheduler"]
    span = settings["beats_per_sequence"]
    params = {name: deepcopy(moves[name][1]) for name in order}
    xyz = np.zeros((len(times), 3)); rpy = xyz.copy(); antennas = np.zeros((len(times), 2))
    ids = []
    for i, p in enumerate(phase):
        if p < 0:
            ids.append("__pre_beat_neutral__")
            continue
        name = order[int(np.floor(p/span)) % len(order)]
        offsets = moves[name][0](float(p), **params[name])
        xyz[i] = np.asarray(offsets.position_offset, dtype=float)
        rpy[i] = np.asarray(offsets.orientation_offset, dtype=float)
        antennas[i] = np.asarray(offsets.antennas_offset, dtype=float)
        ids.append(name)
    if not all(np.all(np.isfinite(x)) for x in (xyz, rpy, antennas)):
        raise ValueError("An official Reachy primitive produced NaN/Inf")
    return {"position_m": xyz, "orientation_rad": rpy, "antenna_rad": antennas,
            "primitive_ids": ids, "source": deepcopy(source),
            "scheduler": {**settings, "cycle_beats": span*len(order),
                          "cycle_phase_periodicity_caveat": "Move selection repeats every cycle_beats; global motion phase is not reset, so a primitive's own frequency may require multiple selection cycles before all output channels repeat."}}


def adapt_misty(native, profile, *, anchor_first=True):
    """Fixed named Euler-coordinate projection, without smoothing or clipping."""
    mapping = config()["adaptation"]
    expected = tuple(mapping["target_order"])
    actual = tuple(profile.joint_names)
    if actual not in (expected, expected+("left_arm", "right_arm")):
        raise ValueError(f"Misty adaptation expects ordered joints {expected} optionally followed by left_arm/right_arm; got {actual}")
    if profile.units != "rad":
        raise ValueError("Misty reference profile must use explicit radians")
    rpy = np.asarray(native["orientation_rad"], dtype=float)
    if rpy.ndim != 2 or rpy.shape[1] != 3 or not np.all(np.isfinite(rpy)):
        raise ValueError("Reachy orientation must be finite (samples,3) roll/pitch/yaw radians")
    neutral = np.asarray(profile.neutral, dtype=float)
    if neutral.shape != (len(actual),):
        raise ValueError("Misty neutral and joint order dimensions disagree")
    if len(rpy)==0:
        raise ValueError("Misty adaptation needs at least one native sample")
    mapped=rpy[:, mapping["orientation_indices"]]*np.asarray(mapping["orientation_signs"])*mapping["orientation_scale"]
    anchor=mapped[0].copy() if anchor_first else np.zeros(3)
    q = np.tile(neutral, (len(rpy), 1))
    q[:, :3] += mapped-anchor
    return {"q": q, "mapping": {**deepcopy(mapping),"anchor_first":bool(anchor_first),"subtracted_first_euler_offset_rad":anchor.tolist()}}


def native_smoke(output=None, bpm=117.02546482191966, seconds=14., dt=.01):
    """Save a small native trajectory using the executed song's shared estimate."""
    output = Path(output or ROOT / "results" / "reachy_native_smoke")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter(); metadata = load_upstream(); init_seconds = time.perf_counter()-start
    times = np.arange(0, seconds, dt)
    phase = times*bpm/60
    start = time.perf_counter(); trajectory = sample_native(times, phase); elapsed = time.perf_counter()-start
    np.savez_compressed(output / "native_trajectory.npz", times_seconds=times, beat_phase=phase,
                        position_m=trajectory["position_m"], orientation_rad=trajectory["orientation_rad"],
                        antenna_rad=trajectory["antenna_rad"], primitive_ids=np.asarray(trajectory["primitive_ids"]))
    with (output / "native_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "beat_phase", "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "antenna_left_rad", "antenna_right_rad", "primitive_id"])
        for i, t in enumerate(times):
            writer.writerow([t,phase[i],*trajectory["position_m"][i],*trajectory["orientation_rad"][i],*trajectory["antenna_rad"][i],trajectory["primitive_ids"][i]])
    report = {"method":METHOD_NATIVE,"source":metadata,"scheduler":trajectory["scheduler"],
              "duration_seconds":seconds,"sample_interval_seconds":dt,"sample_count":len(times),
              "bpm":bpm,"phase_origin":"local beat-zero smoke; full benchmark supplies the shared song timeline",
              "initialization_seconds":init_seconds,"motion_generation_seconds":elapsed,
              "RTF_generation":elapsed/seconds,"primitive_ids_observed":list(dict.fromkeys(trajectory["primitive_ids"])),
              "hardware_executed":False,"simulation_state_available":False,"output_kind":"native head-pose/antenna reference offsets"}
    (output / "smoke_status.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Safely sample native Reachy pure functions, without hardware")
    parser.add_argument("--output");parser.add_argument("--bpm",type=float,default=117.02546482191966)
    parser.add_argument("--seconds",type=float,default=14.)
    args=parser.parse_args()
    print(json.dumps(native_smoke(args.output,args.bpm,args.seconds),indent=2))
