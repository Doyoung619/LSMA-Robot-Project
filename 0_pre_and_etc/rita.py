"""CLI: analyze, inspect, preview, run, stop. Importing this file never moves a robot."""
import argparse
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from rita_robot.audio import decode, analyze, AudioPlayer
from rita_robot.motion import RobotModel, create_plan, Plan, signed

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "data/robot/GEN3-6DOF_VISION_URDF_ARM_V01.urdf"


def analysis(args):
    wav = decode(ROOT/args.music, ROOT/"data/audio.wav")
    file = ROOT/"data/analysis.json"
    if args.command == "analyze" or not file.exists() or file.stat().st_mtime < wav.stat().st_mtime:
        info = analyze(wav, file)
    else:
        info = json.loads(file.read_text(encoding="utf-8"))
    return wav, info


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def snapshot(args):
    from rita_robot.robot import Robot
    with Robot(args.ip) as robot:
        result = robot.snapshot()
    write_json(ROOT/"data/robot_snapshot.json", result)
    return result


def load_reference(args):
    """Planner reference exported by export_reference.py (degrees, 100 Hz, music time)."""
    path = Path(args.reference)
    if not path.is_absolute(): path = ROOT/path if (ROOT/path).exists() else Path.cwd()/path
    data = np.load(path)
    summary = json.loads(str(data["summary"]))
    if Path(summary["audio"]).name != Path(args.music).name:
        raise ValueError(f"Reference was planned for {Path(summary['audio']).name}, not {args.music}")
    if args.start != 0: raise ValueError("--start must be 0 with --reference (reference is in music time)")
    return data, summary


def build(args, snap, info):
    if args.reference:
        data, summary = load_reference(args)
        drift = float(np.max(np.abs(signed(np.asarray(snap["q"])-np.asarray(summary["snapshot_q_deg"])))))
        if drift > 5:
            raise RuntimeError(f"Robot moved {drift:.1f} deg since the reference was exported; "
                               "run `rita.py inspect` and export_reference.py again")
        times, q, v, a = data["times"], data["q"], data["v"], data["a"]
        if np.any(np.max(np.abs(v), axis=0) > np.asarray(snap["speed_limits"])+1e-6):
            raise RuntimeError("Reference exceeds current firmware speed limits")
        model = RobotModel(MODEL)
        if np.any(q < model.lower+8) or np.any(q > model.upper-8):
            raise RuntimeError("Reference violates joint position margin")
        report = {**summary, "duration_seconds": float(times[-1]), "primitive_sequence":
                  [f"{m}/{r}/k{k}" for m, r, k in zip(summary["motion_ids"], summary["rhythm_ids"],
                   summary.get("segment_levels", [""]*len(summary["rhythm_ids"])))]}
        return model, Plan(times, q, v, a, np.asarray(data["center"]), [], 1., report)
    duration = min(args.seconds or info["duration_seconds"]-args.start,
                   info["duration_seconds"]-args.start)
    if duration < 6: raise ValueError("Choose at least six seconds of music")
    if args.start < 0: raise ValueError("Start must be nonnegative")
    model = RobotModel(MODEL)
    plan = create_plan(model, snap["q"], info, duration, args.start, args.seed, args.intensity,
                       snap["speed_limits"], snap["accel_limits"], args.dof)
    return model, plan


def preview(args):
    wav, info = analysis(args)
    snapfile = ROOT/"data/robot_snapshot.json"
    snap = json.loads(snapfile.read_text(encoding="utf-8")) if args.offline else snapshot(args)
    _, plan = build(args,snap,info)
    write_json(ROOT/"data/plan_report.json",plan.report)
    np.savez(ROOT/"data/plan.npz",times=plan.times,q=plan.q,v=plan.v,a=plan.a)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(13,9),sharex=True)
    for j in range(6):
        axes[0].plot(plan.times,plan.q[:,j]-plan.center[j],label=f"J{j+1}")
        axes[1].plot(plan.times,plan.v[:,j])
        axes[2].plot(plan.times,plan.a[:,j])
    for ax,label in zip(axes,["Offset (deg)","Speed (deg/s)","Acceleration (deg/s²)"]):
        ax.set_ylabel(label);ax.grid(alpha=.2)
        for b in np.arange(info["first_beat_seconds"]-args.start,plan.times[-1],info["period_seconds"]):
            ax.axvline(b,color="black",alpha=.06,linewidth=.5)
    axes[0].legend(ncol=6);axes[2].set_xlabel("Audio time (s)")
    fig.suptitle(f"Rita | {info['bpm']:.2f} BPM | continuous joint-space choreography")
    fig.tight_layout();fig.savefig(ROOT/"data/preview.png",dpi=130);plt.close(fig)
    print(json.dumps({"audio":info,"plan":plan.report},indent=2))


def run(args):
    from rita_robot.robot import Robot, Controller, watchdog
    if (ROOT/"STOP_RITA").exists(): raise RuntimeError("STOP_RITA exists; remove it deliberately before a new run")
    wav, info = analysis(args)
    print(f"Music: {info['bpm']:.2f} BPM; hardware playback clock; bounded joint-space motion",flush=True)
    if args.reference:
        data, summary = load_reference(args)
        # Play exactly as much music as the exported reference covers.
        args.seconds = min(float(data["times"][-1])+.5, info["duration_seconds"])
        print(f"Reference: {summary['mode']} planner, seed {summary['seed']}, {args.seconds:.1f}s, "
              f"metrical levels {summary['metrical_levels']}",flush=True)
    # Load all audio and initialize the audio device before arming any motion.
    player = AudioPlayer(wav,args.start,args.seconds,args.device,args.volume)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = ROOT/f"logs/{stamp}.jsonl"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    reportfile = ROOT/f"logs/{stamp}_summary.json"
    summary = {"completed":False,"bpm":info["bpm"],"log":str(logfile),"start_seconds":args.start}
    try:
        with watchdog(args.ip) as (pulse, armed, tripped):
            with Robot(args.ip,feedback_udp=True) as robot, logfile.open("w",encoding="utf-8",buffering=1) as log:
                control = None
                try:
                    snap = robot.snapshot();write_json(ROOT/"data/robot_snapshot.json",snap)
                    model,plan = build(args,snap,info)
                    write_json(ROOT/"data/plan_report.json",plan.report)
                    summary["plan"] = plan.report
                    print("Interior neutral posture:",np.round(plan.center,2).tolist(),flush=True)
                    print("Primitives:"," -> ".join(plan.report["primitive_sequence"]),flush=True)
                    print("Starting in 3 seconds. Q / Esc, STOP_RITA, or the physical emergency stop stops motion.",flush=True)
                    time.sleep(3)
                    control = Controller(robot,model,snap,plan,pulse,armed,tripped,log)
                    print("Moving slowly away from joint limits...",flush=True)
                    control.prepare();control.settle()
                    print("MUSIC + DANCE START",flush=True)
                    player.start();next_print=0;deadline=time.monotonic()
                    while True:
                        if player.finished.is_set() and player.clock() < plan.times[-1]-.15:
                            raise RuntimeError("Audio ended unexpectedly")
                        t=player.clock()
                        q,v=plan.at(t)
                        control.tick(q,v,"dance",t)
                        if t >= plan.times[-1]: break
                        if t >= next_print:
                            print(f"Audio {t:5.1f}s / {plan.times[-1]:.1f}s | beat {(args.start+t-info['first_beat_seconds'])/info['period_seconds']:6.1f}",flush=True)
                            next_print+=8
                        deadline+=.025;time.sleep(max(0,deadline-time.monotonic()))
                    control.settle()
                    summary.update(completed=True,max_tracking_error_deg=control.max_tracking_error,
                                   max_control_gap_seconds=control.max_loop_gap)
                except BaseException as exc:
                    summary["error"]=f"{type(exc).__name__}: {exc}"
                    if control is not None:
                        summary["max_tracking_error_deg"]=control.max_tracking_error
                        summary["max_control_gap_seconds"]=control.max_loop_gap
                    raise
                finally:
                    player.close()
                    try:
                        robot.stop()
                        armed.clear()
                        time.sleep(.15)
                        f=robot.feedback()
                        summary["final_joints_deg"]=[float((a.position+180)%360-180) for a in f.actuators]
                        summary["final_max_joint_speed_deg_s"]=max(abs(a.velocity) for a in f.actuators)
                        if summary["final_max_joint_speed_deg_s"] > .3:
                            summary["completed"]=False
                            summary["stop_verification_error"]="Measured joint speed remains above .3 deg/s"
                            robot.stop()
                    except Exception as exc:
                        summary["stop_error"]=str(exc)
                        summary["completed"]=False
                        # Keep the independent watchdog armed if this connection failed.
                        time.sleep(.6)
    finally:
        player.close()
        write_json(reportfile,summary)
        print("RUN REPORT",json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["analyze","inspect","preview","run","stop","devices"])
    p.add_argument("--ip",default="192.168.2.9")
    p.add_argument("--music",default="Billie Jean.mp3")
    p.add_argument("--seconds",type=float,default=None,help="Omit to play the whole track")
    p.add_argument("--start",type=float,default=0)
    p.add_argument("--intensity",type=float,default=1.)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--dof",type=int,default=6)
    p.add_argument("--offline",action="store_true",help="Preview from the last saved robot snapshot")
    p.add_argument("--volume",type=float,default=.65)
    p.add_argument("--device",type=int,default=None)
    p.add_argument("--reference",default=None,help="Dance a finite/infinite planner reference from export_reference.py instead of the legacy primitives")
    args=p.parse_args()
    import os
    os.chdir(ROOT)
    if args.seconds is not None and args.seconds < 6: p.error("--seconds must be at least 6")
    if not 0 <= args.volume <= 1: p.error("--volume must be in 0..1")
    if args.command=="analyze": print(json.dumps(analysis(args)[1],indent=2))
    elif args.command=="inspect": print(json.dumps(snapshot(args),indent=2))
    elif args.command=="preview": preview(args)
    elif args.command=="run":
        from rita_robot.robot import operator_keys
        with operator_keys(): run(args)
    elif args.command=="devices":
        import sounddevice as sd
        print(sd.query_devices())
    elif args.command=="stop":
        (ROOT/"STOP_RITA").touch()
        from rita_robot.robot import Robot
        with Robot(args.ip) as robot: robot.stop()
        print("Stop sent. STOP_RITA remains latched; remove it before deliberately restarting.")


if __name__=="__main__":
    multiprocessing.freeze_support()
    main()
