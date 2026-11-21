"""Reproducible analysis, safe default mock runs, evaluation and calibration."""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from .audio import analyze_audio,find_audio,fixture,synthetic_analysis,BASE
from .contracts import serial
from .profiles import load_profile,default_profile
from .runtime import run_session,check_feedback


def dump(obj): print(json.dumps(serial(obj),indent=2,allow_nan=False))


def configuration(args,mode=None):
    mode=mode or args.mode
    config=json.loads(Path(args.config).read_text(encoding="utf-8")) if getattr(args,"config",None) else {}
    ablation=getattr(args,"ablation",None)
    if mode in ("finite","fixed-loop","random"):
        if ablation=="no-tempo-adaptation":config["no_tempo_adaptation"]=True
        if ablation=="no-coverage-novelty":config["no_coverage"]=True
        if ablation=="single-rhythm":config["single_rhythm"]=True
    else:
        if ablation=="no-tempo-adaptation":config["metrical_levels"]=[0]
        if ablation=="no-coverage-novelty":config["no_coverage_novelty"]=True
        if ablation=="single-rhythm":config["single_rhythm"]=True
    return config


def output_dir(args,label):
    if getattr(args,"output",None): return Path(args.output).resolve()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return BASE/"outputs"/f"{label}_{stamp}"


def compact(report):
    return {"completed":report["run"]["completed"],"error":report["run"]["error"],"method":report["run"]["method"],
            "reference_seconds":report["reference_duration_seconds"],"segments":report["segments_executed"],
            "motion_diversity":report["motion_diversity"],"rhythm_diversity":report["rhythm_diversity"],
            "activity":report["activity"],"runtime":report["runtime"],
            "invalid_reference_segments":report["feasibility"]["invalid_segments"]}


def plot_comparison(rows,output):
    """Mark stopped rows and unequal completed durations directly on the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(max(13,1.25*len(rows)),10.5))
    labels=[f"{r['method']}\n{r.get('ablation') or r['condition']}\ns{r['seed']} / {r.get('reference_seconds',0):.1f}s"+
            ("\nSTOPPED" if not r['completed'] else "") for r in rows]
    colors=["#2676ad" if r["completed"] else "#d34a41" for r in rows]
    axes[0].bar(range(len(rows)),[r.get("motion_diversity",{}).get("shared_descriptor_feature_cells_visited",0) for r in rows],color=colors);axes[0].set_ylabel("Shared geometry cells / 8")
    axes[1].bar(range(len(rows)),[r.get("motion_diversity",{}).get("short_repeat_ngram_ratio",{}).get("2",0) for r in rows],color=colors);axes[1].set_ylabel("Repeated geometry-cell 2-gram fraction")
    axes[2].bar(range(len(rows)),[r.get("runtime",{}).get("planning_seconds_p95",0) or 0 for r in rows],color=colors);axes[2].set_ylabel("Planning p95 wall sec")
    for ax in axes:
        ax.set_xticks(range(len(rows)),labels,fontsize=6.5);ax.grid(axis="y",alpha=.2)
        for tick,r in zip(ax.get_xticklabels(),rows):
            if not r["completed"]:tick.set_color("#b8342b")
    fig.suptitle("Mock reference comparison — executed duration shown; red = stopped/failed",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(Path(output)/"comparison.png",dpi=120);plt.close(fig)


def benchmark(args):
    output=output_dir(args,"benchmark")
    output.mkdir(parents=True,exist_ok=True)
    duration=args.seconds or (660 if args.suite=="long" else 60)
    seeds=[int(x) for x in args.seeds.split(",")]
    analysis=analyze_audio(args.audio,output/"analysis") if args.audio else synthetic_analysis(duration=duration+180)
    duration=min(duration,analysis.duration-35)
    methods=["finite","infinite"] if args.suite=="long" else ["fixed-loop","random","finite","infinite"]
    scenarios=[(m,None,"normal") for m in methods]
    if args.suite in ("quick","full"):
        scenarios += [(m,a,"ablation") for m in ("finite","infinite") for a in ("no-tempo-adaptation","no-coverage-novelty","single-rhythm")]
    if args.suite in ("long","full"):
        scenarios += [(m,None,"tightened") for m in ("finite","infinite")]
        scenarios += [(m,None,c) for m in ("finite","infinite") for c in ("disconnect","feedback-dropout","delayed-feedback")]
    rows=[]
    for seed in seeds:
        for method,ablation,condition in scenarios:
            args.ablation=ablation
            profile=default_profile()
            if condition=="tightened":profile=replace(profile,vmax=profile.vmax*.3,amax=profile.amax*.2)
            options={}
            if condition=="disconnect":options["disconnect_at"]=12.
            if condition=="feedback-dropout":options["feedback_dropout"]=.03
            if condition=="delayed-feedback":options.update(command_delay=.025,tracking_delay=.05)
            label=f"{method}_{ablation or condition}_seed{seed}"
            print(f"BENCHMARK {label}",flush=True)
            try:
                report=run_session(analysis,profile,method,seed=seed,seconds=duration if condition in ("normal","tightened","ablation") else min(60,duration),
                                   output=output/label,config=configuration(args,method),mock_options=options,condition=condition+(":"+ablation if ablation else ""))
                row=compact(report)
                row.update(seed=seed,ablation=ablation,condition=condition)
            except Exception as exc:
                row={"method":method,"seed":seed,"ablation":ablation,"condition":condition,"completed":False,"error":f"{type(exc).__name__}: {exc}",
                     "note":"candidate-search failure is not a mathematical impossibility proof"}
            rows.append(row)
    (output/"comparison.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    import csv
    with (output/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["method","condition","ablation","seed","completed","reference_seconds","segments","descriptor_cells","rhythms","planning_p95_s","error"])
        for r in rows:w.writerow([r.get("method"),r.get("condition"),r.get("ablation"),r.get("seed"),r.get("completed"),r.get("reference_seconds"),r.get("segments"),
                                  r.get("motion_diversity",{}).get("shared_descriptor_feature_cells_visited"),len(r.get("rhythm_diversity",{}).get("rhythm_ids",[])),
                                  r.get("runtime",{}).get("planning_seconds_p95"),r.get("error")])
    plot_comparison(rows,output)
    dump({"output":str(output),"runs":len(rows),"completed":sum(r["completed"] for r in rows),"failures":[r for r in rows if not r["completed"]]})


def doctor(args):
    profile=load_profile(args.profile,args.backend)
    from .adapters import make_adapter
    adapter=make_adapter(args.backend,profile,execute=False)
    modules={"mock":[],"misty":["websocket"],"kinova":["rclpy","control_msgs","sensor_msgs","trajectory_msgs"]}[args.backend]
    result={"read_only":True,"motion_commands_sent":False,"profile":profile.to_dict(),"capabilities":adapter.get_capabilities(),
            "optional_dependencies":{m:importlib.util.find_spec(m) is not None for m in modules},"status":"implemented, hardware-unverified" if args.backend!="mock" else "mock ready"}
    if args.probe:
        try:
            adapter.connect();state=adapter.read_state()
            result["probe"]={"q_rad":state.q.tolist(),"fresh":state.fresh,"joint_names":state.joint_names,"metadata":state.metadata}
        finally:adapter.close()
    dump(result)


def calibrate(args):
    from .trajectory import make_segment,validate_segment,add_terminal_hold
    from .adapters import make_adapter
    profile=load_profile(args.profile,args.backend)
    if args.backend!="mock" and not args.execute: raise PermissionError("Calibration moves one joint: verified local profile and --execute required")
    if args.backend!="mock" and (BASE/"STOP_RITA").exists(): raise RuntimeError("STOP_RITA exists; calibration will not clear an operator stop latch")
    if args.backend!="mock" and args.headless: raise ValueError("Hardware calibration cannot use virtual time")
    duration=args.seconds or 8.
    if duration<4:raise ValueError("Calibration duration must be >=4 seconds")
    if not 0<args.amplitude<=.08:raise ValueError("Calibration amplitude must be in (0,.08] rad")
    if not 0<=args.joint<profile.nj:raise ValueError(f"Calibration joint must be 0..{profile.nj-1} in profile order")
    q=profile.neutral.copy();q[args.joint]+=args.amplitude
    seg=make_segment([0,duration/2,duration],[profile.neutral,q,profile.neutral],accent_times=[duration/2],accent_strengths=[1],metadata={"kind":"calibration","excluded_from_diversity":True})
    if args.backend!="mock":seg=add_terminal_hold(seg,max(.2,profile.dispatch_lead_seconds+.08))
    calibration_profile=replace(profile,min_excursion=min(profile.min_excursion,args.amplitude*.8),max_stationary_seconds=duration)
    if not validate_segment(seg,calibration_profile)["valid"]:raise ValueError(f"Calibration reference rejected: {seg.validation['errors']}")
    seg.validation["activity_exception"]="Deliberately slow calibration, excluded from normal dance/diversity; original q/dq/ddq bounds unchanged"
    options={"tracking_delay":args.tracking_delay} if args.backend=="mock" else {}
    adapter=make_adapter(args.backend,profile,execute=args.execute,**options);samples=[];times=[]
    try:
        adapter.connect();epoch=time.monotonic()+max(.3,profile.dispatch_lead_seconds+.15)
        if args.headless:adapter.advance(epoch)
        state=adapter.read_state();check_feedback(state,profile,epoch if args.headless else time.monotonic())
        if np.max(np.abs(state.q-profile.neutral))>.03:raise RuntimeError("Calibration requires actual initial neutral; no automatic home")
        receipt=adapter.execute_segment(seg,scheduled_monotonic=epoch,now_music=0)
        for t in np.arange(0,duration,profile.command_period):
            if args.backend!="mock" and (BASE/"STOP_RITA").exists():raise RuntimeError("STOP_RITA requested during calibration")
            if args.headless:adapter.advance(epoch+t)
            else:
                wait=epoch+t-time.monotonic()
                if wait>0:time.sleep(wait)
            state=adapter.read_state();check_feedback(state,profile,epoch+t if args.headless else time.monotonic())
            samples.append(state.q);times.append(float(state.t-epoch))
    finally:
        adapter.stop();adapter.close()
    times=np.array(times);samples=np.array(samples)
    lags=np.linspace(-.3,.5,801)
    errors=[np.mean((seg.at(np.clip(times-lag,0,duration))[0]-samples)**2) for lag in lags]
    best=int(np.argmin(errors))
    report={"backend":args.backend,"physical_measurement":args.backend!="mock","calibration":"joint-feedback trajectory lag fit, not HTTP RTT",
            "latency_seconds_estimate":float(lags[best]),"residual_mse":float(errors[best]),"sample_period_seconds":profile.command_period,
            "command_receipt":receipt,"profile":profile.to_dict(),"future_commands_only":True,
            "sample_timing":"actual feedback host-monotonic timestamps minus scheduled epoch, not nominal loop targets; feedback transport delay is included",
            "timestamp_policy":"For timestamp-capable controllers do not also subtract this lag from the execution header; dispatch lead is separate"}
    out=output_dir(args,"calibration");out.mkdir(parents=True,exist_ok=True)
    (out/"calibration.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savetxt(out/"feedback.csv",np.c_[times,samples],delimiter=",",header="actual_seconds,"+",".join(profile.joint_names),comments="")
    dump({**report,"output":str(out)})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("analyze");a.add_argument("--audio");a.add_argument("--output");a.add_argument("--bpm-override",type=float);a.add_argument("--beat-offset",type=float)
    f=sub.add_parser("fixture");f.add_argument("--output",required=True);f.add_argument("--bpm",type=float,default=120);f.add_argument("--seconds",type=float,default=660);f.add_argument("--phase",type=float,default=.17);f.add_argument("--tempo-change",type=float);f.add_argument("--silence",action="store_true")
    r=sub.add_parser("run");r.add_argument("--mode",choices=["finite","infinite","fixed-loop","random"],default="finite");r.add_argument("--backend",choices=["mock","misty","kinova"],default="mock");r.add_argument("--audio");r.add_argument("--synthetic",action="store_true");r.add_argument("--headless",action="store_true");r.add_argument("--seed",type=int,default=7);r.add_argument("--seconds",type=float,default=60);r.add_argument("--bpm",type=float,default=120);r.add_argument("--profile");r.add_argument("--config");r.add_argument("--output");r.add_argument("--execute",action="store_true");r.add_argument("--bpm-override",type=float);r.add_argument("--beat-offset",type=float)
    r.add_argument("--ablation",choices=["no-tempo-adaptation","no-coverage-novelty","single-rhythm"])
    r.add_argument("--command-delay",type=float,default=0);r.add_argument("--tracking-delay",type=float,default=0);r.add_argument("--feedback-dropout",type=float,default=0);r.add_argument("--disconnect-at",type=float)
    b=sub.add_parser("benchmark");b.add_argument("--suite",choices=["quick","long","full"],default="quick");b.add_argument("--audio");b.add_argument("--seconds",type=float);b.add_argument("--seeds",default="7");b.add_argument("--output");b.add_argument("--config")
    d=sub.add_parser("doctor");d.add_argument("--backend",choices=["mock","misty","kinova"],default="mock");d.add_argument("--profile");d.add_argument("--read-only",action="store_true",default=True);d.add_argument("--probe",action="store_true")
    c=sub.add_parser("calibrate");c.add_argument("--backend",choices=["mock","misty","kinova"],default="mock");c.add_argument("--profile");c.add_argument("--execute",action="store_true");c.add_argument("--headless",action="store_true");c.add_argument("--seconds",type=float,default=8);c.add_argument("--amplitude",type=float,default=.04);c.add_argument("--joint",type=int,default=0);c.add_argument("--tracking-delay",type=float,default=0);c.add_argument("--output")
    sub.add_parser("test")
    args=parser.parse_args(argv)
    try:
        if args.command=="analyze":
            out=output_dir(args,"analysis");a=analyze_audio(args.audio,out,args.bpm_override,args.beat_offset)
            dump({"bpm":a.bpm,"first_beat_seconds":a.beats[0],"beat_count":len(a.beats),"duration":a.duration,"audio_hash":a.audio_hash,"output":str(out),"settings":a.settings})
        elif args.command=="fixture":dump({"fixture":str(fixture(args.output,args.bpm,args.seconds,args.phase,args.tempo_change,args.silence))})
        elif args.command=="doctor":doctor(args)
        elif args.command=="calibrate":calibrate(args)
        elif args.command=="benchmark":benchmark(args)
        elif args.command=="test":
            suite=unittest.defaultTestLoader.discover(str(BASE/"tests_v2"))
            if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
        elif args.command=="run":
            out=output_dir(args,args.mode)
            a=synthetic_analysis(args.bpm,max(180,args.seconds+90)) if args.synthetic else analyze_audio(args.audio,BASE/"outputs/analysis",args.bpm_override,args.beat_offset)
            if args.synthetic and not args.headless:raise ValueError("--synthetic is a labeled silent planner fixture; use --headless")
            report=run_session(a,load_profile(args.profile,args.backend),args.mode,args.backend,args.seed,args.seconds,args.headless,out,configuration(args),
                               {"command_delay":args.command_delay,"tracking_delay":args.tracking_delay,"feedback_dropout":args.feedback_dropout,"disconnect_at":args.disconnect_at} if args.backend=="mock" else {},args.execute)
            dump({"output":str(out),**compact(report)})
            if not report["run"]["completed"]:raise SystemExit(2)
    except (ValueError,FileNotFoundError,PermissionError,RuntimeError,ImportError) as exc:
        parser.exit(2,f"{type(exc).__name__}: {exc}\n")


if __name__=="__main__":main()
