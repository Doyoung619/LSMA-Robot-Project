"""Export a finite/infinite planner reference for the Kortex velocity runner (rita.py).

Run in the planner environment (Python 3.11, `pip install -e .`):

    python 0_pre_and_etc/export_reference.py --mode finite --seconds 60

The Kortex API environment (Python 3.9, protobuf 3.5) cannot import the planners,
so the two halves are split by a file: this script plans the whole performance
headlessly through the normal `run_session` contracts (mock backend, certified
segments), then samples it to degrees at 100 Hz. `rita.py run --reference FILE`
executes it with the existing watchdog, joint guards and tracking-error stop.

The Kinova example profile's joint box, speed and acceleration limits are kept,
but the box is re-centred on the same interior posture the legacy demo uses,
computed from the latest `rita.py inspect` snapshot. Limits are additionally
capped by the firmware limits recorded in that snapshot. The infinite planner's
motion is therefore generated ahead of playback here, not during playback.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from rita_core.audio import analyze_audio, find_audio  # noqa: E402
from rita_core.profiles import load_profile, PROFILE_DIR  # noqa: E402
from rita_core.runtime import create_planner, run_session  # noqa: E402
from rita_robot.motion import RobotModel, smoother  # noqa: E402

MODEL = ROOT/"data/robot/GEN3-6DOF_VISION_URDF_ARM_V01.urdf"
DT = .01


class Recorder:
    """Planner wrapper that keeps every segment run_session executes."""
    def __init__(self, planner):
        self.planner, self.segments = planner, []

    def next_segment(self, start_beat, state):
        seg = self.planner.next_segment(start_beat, state)
        self.segments.append(seg)
        return seg

    def terminal_suffix(self, beat, boundary):
        finish = getattr(self.planner, "terminal_suffix", None)
        seg = finish(beat, boundary) if finish is not None else None
        if seg is not None: self.segments.append(seg)
        return seg

    def __getattr__(self, name):
        return getattr(self.planner, name)


def recentred_profile(snapshot, model, box_scale=1.):
    profile = load_profile(PROFILE_DIR/"kinova.example.json", backend="kinova")
    below, above = box_scale*(profile.lower-profile.neutral), box_scale*(profile.upper-profile.neutral)
    half_deg = np.degrees(np.maximum(-below, above))
    initial = np.asarray(snapshot["q"], dtype=float)
    # Legacy rule (motion.RobotModel.center) with the larger planner box:
    # keep the whole box at least 12 degrees inside the official URDF limits.
    center_deg = np.clip(initial, model.lower+half_deg+12, model.upper-half_deg-12)
    neutral = np.radians(center_deg)
    profile.neutral, profile.lower, profile.upper = neutral, neutral+below, neutral+above
    profile.vmax = np.minimum(profile.vmax, np.radians(snapshot["speed_limits"]))
    profile.amax = np.minimum(profile.amax, np.radians(snapshot["accel_limits"]))
    profile.name = "gen3-l53-recentred-for-kortex-velocity-runner"
    return profile, center_deg


def sample(segments, neutral, first_start):
    end = segments[-1].end
    # Grid reaches past the final knot; samples beyond it hold the exact end state.
    times = np.arange(int(np.ceil(end/DT))+1)*DT
    q = np.tile(neutral, (len(times), 1)); dq = np.zeros_like(q); ddq = np.zeros_like(q)
    for previous, seg in zip(segments, segments[1:]):
        if abs(seg.start-previous.end) > 1e-6:
            raise RuntimeError("Recorded segments are not contiguous")
    for i, seg in enumerate(segments):
        last = i == len(segments)-1
        mask = (times >= seg.start-1e-9) & ((times <= seg.end+1e-9) | last)
        if mask.any():
            q[mask], dq[mask], ddq[mask] = seg.at(np.clip(times[mask], seg.start, seg.end))
    if first_start > 0 and np.max(np.abs(q[0]-neutral)) > 1e-6:
        raise RuntimeError("Reference does not begin at the neutral pose")
    return times, q, dq, ddq


def append_return(times, q, v, a, center, vmax, amax):
    """Quintic rest-to-rest return to the centre so settle() never jumps."""
    delta = center-q[-1]
    if np.max(np.abs(v[-1])) > 1e-3 or np.max(np.abs(a[-1])) > 1e-2:
        raise RuntimeError("Planner reference does not end at rest")
    if np.max(np.abs(delta)) < 1e-6:
        return times, q, v, a
    duration = max(1.5, float(np.max(1.875*np.abs(delta)/(.5*vmax))),
                   float(np.max(np.sqrt(5.78*np.abs(delta)/(.5*amax)))))
    t = np.arange(1, int(np.ceil(duration/DT))+1)*DT
    u = np.minimum(t/duration, 1.)
    s = smoother(u)
    ds = 30*u**2*(1-u)**2/duration
    dds = 60*u*(1-u)*(1-2*u)/duration**2
    return (np.concatenate([times, times[-1]+t]), np.vstack([q, q[-1]+s[:, None]*delta]),
            np.vstack([v, ds[:, None]*delta]), np.vstack([a, dds[:, None]*delta]))


# Gripper tool point in the URDF flange frame, fitted (<=1.5 mm residual) to the
# Kortex tool_pose logged during the first reference run on this robot. rita.py's
# Controller stops when this point moves > TOOL_LIMIT from where it was before
# the run, or drops below TOOL_MIN_Z; exports keep a margin under both.
TOOL_OFFSET = np.array([.005, .0092, -.1205])
TOOL_LIMIT, TOOL_MARGIN, TOOL_MIN_Z = .22, .03, .25


class EnvelopeError(RuntimeError):
    pass


def tool_points(model, q):
    _, flange = model.fk(np.atleast_2d(q))
    return flange[:, :3, 3]+np.einsum("nij,j->ni", flange[:, :3, :3], TOOL_OFFSET)


def legacy_checks(model, q, v, a, center, snapshot):
    """Legacy create_plan envelope, firmware caps, and rita.py's tool envelope."""
    if np.any(q < model.lower+8) or np.any(q > model.upper-8):
        raise EnvelopeError("Reference violates the 8 degree joint position margin")
    if np.any(np.max(np.abs(v), axis=0) > np.asarray(snapshot["speed_limits"])+1e-6):
        raise EnvelopeError("Reference exceeds firmware joint speed limits")
    points, _ = model.fk(q[::5])
    center_points, _ = model.fk(center)
    excursion = float(np.max(np.linalg.norm(points-center_points, axis=-1)))
    if excursion > .20: raise EnvelopeError("Reference exceeds 20 cm arm workspace envelope")
    if np.min(points[:, 2:, 2]) < .12: raise EnvelopeError("Reference approaches base/table plane")
    if np.min(np.linalg.norm(points[:, 4:6]-points[:, 1:2], axis=-1)) < .15:
        raise EnvelopeError("Wrist approaches shoulder")
    # The controller measures from the pose BEFORE prepare(), so include the
    # joint-space move from the snapshot pose to the centre as well.
    initial = np.asarray(snapshot["q"], dtype=float)
    path = np.vstack([initial+u*(center-initial) for u in np.linspace(0, 1, 50)]+[q[::5]])
    tool = tool_points(model, path)
    tool_distance = float(np.max(np.linalg.norm(tool-tool_points(model, initial)[0], axis=1)))
    if tool_distance > TOOL_LIMIT-TOOL_MARGIN:
        raise EnvelopeError(f"Tool would move {tool_distance*100:.1f} cm from its pre-run pose "
                            f"(rita.py stops at {TOOL_LIMIT*100:.0f} cm; export allows {100*(TOOL_LIMIT-TOOL_MARGIN):.0f})")
    if np.min(tool[:, 2]) < TOOL_MIN_Z+.02:
        raise EnvelopeError("Tool would come within 2 cm of rita.py's minimum tool height")
    return excursion, tool_distance


def plan_once(args, snapshot, model, analysis, box_scale, config):
    profile, center_deg = recentred_profile(snapshot, model, box_scale)
    recorder = Recorder(create_planner(args.mode, analysis, profile, args.seed, config))
    out_dir = ROOT/"outputs"/f"reference_{args.mode}"
    report = run_session(analysis, profile, mode=args.mode, backend="mock", seed=args.seed,
                         seconds=args.seconds, headless=True, output=out_dir, config=config,
                         planner=recorder)
    run = report["run"]
    if not run["completed"] or run["error"] or report["feasibility"]["invalid_segments"]:
        raise SystemExit(f"Planning failed: {run['error']} (invalid segments: {report['feasibility']['invalid_segments']})")
    segments = recorder.segments
    times, q, dq, ddq = sample(segments, profile.neutral, segments[0].start)
    q_deg, v_deg, a_deg = np.degrees(q), np.degrees(dq), np.degrees(ddq)
    times, q_deg, v_deg, a_deg = append_return(times, q_deg, v_deg, a_deg, center_deg,
                                               np.degrees(profile.vmax), np.degrees(profile.amax))
    if times[-1] > analysis.duration-.5:
        raise SystemExit("Reference plus return is longer than the music; choose fewer --seconds")
    excursion, tool_distance = legacy_checks(model, q_deg, v_deg, a_deg, center_deg, snapshot)
    return dict(segments=segments, times=times, q=q_deg, v=v_deg, a=a_deg, center=center_deg,
                excursion=excursion, tool_distance=tool_distance, out_dir=out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=["finite", "infinite"], required=True)
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--audio", default=None, help="Defaults to the bundled Billie Jean.mp3")
    p.add_argument("--snapshot", default=str(ROOT/"data/robot_snapshot.json"),
                   help="Written by `rita.py inspect`; run it first with the robot in its start pose")
    p.add_argument("--config", default=None, help="Planner config JSON override")
    p.add_argument("--output", default=None)
    p.add_argument("--preset", choices=["max", "planner"], default="max",
                   help="max: full library amplitude (finite) / k=3 only (infinite), and the largest "
                        "joint box (up to x1.75 finite, x3 infinite) whose motion stays inside every "
                        "rita.py envelope from the CURRENT robot pose. planner: original example settings")
    p.add_argument("--no-variety", action="store_true",
                   help="max preset only: keep one slow metrical level (k=3) instead of mixing short/long")
    p.add_argument("--box-scale", type=float, default=None,
                   help="Fixed scale of the Kinova example joint box (about +/-11.5 deg); disables the search")
    p.add_argument("--amplitude-fraction", type=float, default=None,
                   help="finite only: fraction of the box the motion library uses (planner default 0.18)")
    args = p.parse_args()

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    model = RobotModel(MODEL)
    # Variety (default): alternate metrical levels; faster levels use a smaller motion so
    # they stay inside the acceleration caps (k=0 never passes at ~117 BPM with them).
    #   finite:   k=2 (move per 4 beats) at 25% of the box, k=3 (per 8 beats) at 70%
    #   infinite: k=1/2/3 (per 2/4/8 beats), amplitude shrinks per candidate as needed
    variety = {"finite": {"level_mix": True, "level_amplitude": {"2": 0.25, "3": 0.7}},
               "infinite": {"level_mix": True, "metrical_levels": [1, 2, 3],
                            "amplitude_scales": [1, .85, .7, .55, .4, .3, .22, .15, .1],
                            "planning_budget_seconds": 60}}
    single = {"finite": None, "infinite": {"metrical_levels": [3]}}
    preset = {"max": {"finite": (1.75, 1.0), "infinite": (3.0, None)},
              "planner": {"finite": (1.0, None), "infinite": (1.0, None)}}[args.preset][args.mode]
    preset += ((single if args.no_variety else variety)[args.mode] if args.preset == "max" else None,)
    if args.amplitude_fraction is None: args.amplitude_fraction = preset[1]
    if args.box_scale is not None:
        # 3x the example box is ~34 deg, just inside rita.py's 35 deg local dance volume.
        if not 0 < args.box_scale <= 3: raise SystemExit("--box-scale must be in (0, 3]")
        scales = [args.box_scale]
    elif args.preset == "max":
        top = preset[0]
        scales = [x for x in (top, 2.75, 2.5, 2.25, 2.0, 1.75, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, .8, .6, .5) if x <= top]
    else:
        scales = [preset[0]]
    audio = find_audio(args.audio)
    analysis = analyze_audio(audio, ROOT/"outputs/analysis")
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else None
    if preset[2] is not None and not args.config:
        config = {**(config or {}), **preset[2]}
    if args.amplitude_fraction is not None:
        if args.mode != "finite": raise SystemExit("--amplitude-fraction applies to the finite planner only")
        config = {**(config or {}), "amplitude_fraction": args.amplitude_fraction}

    result, rejected = None, []
    for scale in scales:
        try:
            result = plan_once(args, snapshot, model, analysis, scale, config)
            args.box_scale = scale
            break
        except EnvelopeError as exc:
            rejected.append({"box_scale": scale, "reason": str(exc)})
            print(f"box x{scale}: {exc}", file=sys.stderr)
    if result is None:
        raise SystemExit("No box scale fits rita.py's envelopes from this robot pose; "
                         "move the arm to a more open start pose, run inspect, and export again")
    segments, times, q_deg, v_deg, a_deg, center_deg = (result[k] for k in ("segments", "times", "q", "v", "a", "center"))

    levels = Counter(int(s.metrical_level) for s in segments)
    summary = {
        "mode": args.mode, "seed": args.seed, "audio": str(audio), "bpm": analysis.bpm,
        "seconds_requested": args.seconds, "preset": args.preset, "box_scale": args.box_scale,
        "level_variety": args.preset == "max" and not args.no_variety,
        "box_scales_rejected": rejected, "planner_config_override": config,
        "amplitude_fraction": args.amplitude_fraction, "dance_start_s": segments[0].start,
        "dance_end_s": segments[-1].end, "reference_end_s": float(times[-1]),
        "segments": len(segments),
        "metrical_levels": {str(k): v for k, v in sorted(levels.items())},
        "segment_levels": [int(s.metrical_level) for s in segments],
        "rhythm_ids": [s.rhythm_id for s in segments],
        "motion_ids": [s.metadata.get("primitive_id") or s.metadata.get("motion_id") or s.metadata.get("candidate_id", "") for s in segments],
        "center_deg": center_deg.tolist(), "snapshot_q_deg": list(snapshot["q"]),
        "joint_offset_max_deg": np.max(np.abs(q_deg-center_deg), axis=0).tolist(),
        "peak_speed_deg_s": np.max(np.abs(v_deg), axis=0).tolist(),
        "peak_acceleration_deg_s2": np.max(np.abs(a_deg), axis=0).tolist(),
        "max_arm_origin_excursion_m": result["excursion"],
        "max_tool_distance_from_prerun_pose_m": result["tool_distance"],
        "planner_report_dir": str(result["out_dir"]),
        "scope": "Planner reference re-centred on the legacy interior posture; executed by rita.py's Kortex velocity loop. Infinite motion generated before playback.",
    }
    output = Path(args.output) if args.output else ROOT/f"data/reference_{args.mode}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, times=times, q=q_deg, v=v_deg, a=a_deg, center=center_deg,
             summary=json.dumps(summary))
    print(json.dumps({**summary, "rhythm_ids": summary["rhythm_ids"][:12],
                      "motion_ids": summary["motion_ids"][:12], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
