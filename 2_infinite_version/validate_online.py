"""Accelerated virtual-time infinite-planner checks; no audio or hardware I/O."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "0_pre_and_etc"))
from rita_core.contracts import AudioAnalysis, BoundaryState, RobotProfile
from rita_core.trajectory import validate_segment
from rita_infinite import InfinitePlanner, InfinitePlanningError


def git_identity():
    head = subprocess.run(["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(ROOT.parent), "status", "--porcelain"],
                            capture_output=True, text=True)
    return {"sha": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()), "git_available": status.returncode == 0}


def run(seconds, bpm, speed, acceleration, seed, budget, stop_to_stop=False):
    period = 60/bpm
    beats = np.arange(int((seconds+150)/period)+1)*period+.17
    a = AudioAnalysis("synthetic-click", "synthetic-grid", 44100, float(beats[-1]+period),
                      beats, beats, np.ones(len(beats)), np.ones(len(beats)),
                      np.resize([1, .5, .8, .5], len(beats)), {"fixture": True})
    p = RobotProfile("infinite-validation", "mock", tuple(f"j{i}" for i in range(6)),
                     np.zeros(6), np.full(6, -.8), np.full(6, .8),
                     np.full(6, speed), np.full(6, acceleration))
    provenance = {"git": git_identity(), "audio_hash": None,
                  "fixture_grid_hash": hashlib.sha256(beats.tobytes()).hexdigest(),
                  "source": "synthetic analyzed beat grid; no audio file used",
                  "profile": p.to_dict(), "analysis_settings": a.settings}
    planner = InfinitePlanner(a, p, seed, {"planning_budget_seconds": budget, "stop_to_stop": stop_to_stop})
    state = BoundaryState(a.beat_time(0), p.neutral.copy(), np.zeros(6), np.zeros(6))
    beat, previous, rows, failures = 0, None, [], []
    start = time.perf_counter()
    while state.t-a.beat_time(0) < seconds:
        try:
            seg = planner.next_segment(beat, state)
        except InfinitePlanningError as error:
            failures.append({"reason": error.reason, "at_seconds": state.t-a.beat_time(0),
                             "diagnostic": error.diagnostic})
            break
        report = validate_segment(seg, p, previous)
        if not report["valid"]:
            raise AssertionError(report)
        rows.append({"start": seg.start, "duration": seg.duration, "level": seg.metrical_level,
                     "rhythm": seg.rhythm_id, "novelty": seg.metadata.get("novelty_distance"),
                     "relaxations": seg.metadata.get("novelty_relaxation_steps", 0),
                     "fallback": seg.metadata["fallback"], "feature_cell": seg.metadata.get("feature_cell"),
                     "planning_seconds": seg.metadata["planning_seconds"],
                     "max_stationary_seconds": report["joined_stationary_seconds"],
                     "excursion": report["min_excursion"]})
        previous, state, beat = seg, seg.end_state(), seg.metadata["end_beat"]
        if len(rows) % 25 == 0:
            print(f"bpm={bpm:g} speed={speed:g}: {len(rows)} phrases / {state.t-a.beat_time(0):.1f}s virtual", flush=True)
    stats = planner.stats()
    latency = np.array(stats.pop("planning_seconds"))
    return {"seed": seed, "bpm": bpm, "speed_limit": speed, "acceleration_limit": acceleration,
            "backend": "mock-reference-only", "stop_to_stop": stop_to_stop,
            "provenance": provenance,
            "requested_virtual_seconds": seconds, "completed_virtual_seconds": state.t-a.beat_time(0),
            "wall_seconds": time.perf_counter()-start, "segments": len(rows), "failures": failures,
            "planning_median_seconds": float(np.median(latency)) if len(latency) else None,
            "planning_p95_seconds": float(np.percentile(latency, 95)) if len(latency) else None,
            "planning_max_seconds": float(latency.max()) if len(latency) else None,
            "max_joined_stationary_seconds": max((x["max_stationary_seconds"] for x in rows), default=None),
            "minimum_excursion": min((x["excursion"] for x in rows), default=None),
            "stats": stats, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=610)
    parser.add_argument("--bpm", type=float, default=120)
    parser.add_argument("--speed", type=float, default=1.2)
    parser.add_argument("--acceleration", type=float, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget", type=float, default=1.5)
    parser.add_argument("--stop-to-stop", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "validation_summary.json")
    args = parser.parse_args()
    result = run(args.seconds, args.bpm, args.speed, args.acceleration, args.seed, args.budget, args.stop_to_stop)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in ("rows", "stats")}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
