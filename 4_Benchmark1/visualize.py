"""Shared benchmark figures and an explicitly nonphysical head-pose animation.

The renderer only reads saved trajectories and the local audio file. It never
imports a robot SDK, creates a robot connection, or reruns/times a generator.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import textwrap

import numpy as np


METHODS = ("our_finite", "our_infinite", "reachy_adapted")
LABELS = {"our_finite": "Our finite", "our_infinite": "Our infinite",
          "reachy_adapted": "Reachy scheduler + fixed adaptation"}
COLORS = {"our_finite": "#35bfd3", "our_infinite": "#cf6bc7", "reachy_adapted": "#72c888"}
RGB = {"our_finite": (53, 191, 211), "our_infinite": (207, 107, 199),
       "reachy_adapted": (114, 200, 136)}
SUCCESS = {"success", "ok", "generated", "complete", "completed"}


def _rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _ok(row):
    return row.get("status", "success").lower() in SUCCESS


def _method(row):
    return row.get("method", row.get("method_id", ""))


def _summary_points(ax, rows, field, ylabel, *, percent=False):
    """Show every observed run and a mean; no missing value is turned into zero."""
    observed=[]
    for i, method in enumerate(METHODS):
        values = [_number(r.get(field)) for r in rows if _method(r) == method and _ok(r)]
        values = [v * (100 if percent else 1) for v in values if v is not None]
        if not values:
            ax.text(i, .04, "N/A", ha="center", transform=ax.get_xaxis_transform(), color="#666666")
            continue
        observed.extend(values)
        ax.bar(i, float(np.mean(values)), color=COLORS[method], alpha=.48, width=.58)
        jitter = np.linspace(-.13, .13, len(values)) if len(values) > 1 else [0]
        ax.scatter(i + np.asarray(jitter), values, s=16, color=COLORS[method], edgecolors="#20303b", linewidths=.3, zorder=3)
        ax.text(i, .96, f"n={len(values)}", ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xticks(range(len(METHODS)), ["Our finite", "Our infinite", "Reachy + adaptation"], fontsize=9)
    ax.set_xlim(-.5,len(METHODS)-.5)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,max(observed)*1.25 if observed and max(observed)>0 else 1.)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)


def make_plots(results: Path):
    """Render plots from saved CSV/JSON, returning generated figure paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = Path(results)
    target = results / "figures"
    target.mkdir(parents=True, exist_ok=True)
    rows = _rows(results / "same_robot_metrics.csv")
    latency = _rows(results / "latency_raw.csv")
    stage2 = [r for r in rows if r.get("stage") in ("stage2", "2", "small")]
    quality = stage2 or [r for r in rows if r.get("stage") in ("stage1", "1", "short")]
    if not quality:
        quality = rows
    quality_scope = "Three music excerpts; independent seeds where applicable" if stage2 else "Available saved runs (stage 2 not present)"
    paths = []
    manifest = {"scope": "Saved commanded trajectories; not physical robot measurements",
                "quality_scope": quality_scope, "methods": list(METHODS),
                "missing_values": "N/A, never converted to zero", "plots": []}

    def save(fig, name, note):
        path = target / name
        fig.tight_layout(rect=(0, .035, 1, .95))
        fig.text(.5, .012, note, ha="center", fontsize=8, color="#555555")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
        manifest["plots"].append({"file": path.name, "note": note})

    # Same output duration/crop across methods; initialization is kept separate.
    measured = [r for r in latency if r.get("kind", "measured") == "measured" and _ok(r)]
    def condition(r):
        return (r.get("stage", ""), r.get("start_beat", ""),
                r.get("duration_seconds", ""), r.get("seed", "0"))
    grouped = {}
    for row in measured:
        grouped.setdefault(condition(row), {}).setdefault(_method(row), []).append(row)
    common = sorted([key for key, value in grouped.items() if all(m in value for m in METHODS)])
    selected = grouped[common[0]] if common else {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    values_for_manifest = []
    for i, method in enumerate(METHODS):
        sample = selected.get(method, [])
        values = [_number(r.get("total_generation_seconds")) for r in sample]
        values = [v for v in values if v is not None]
        if values:
            mid = float(np.median(values)); lower = mid-min(values); upper = max(values)-mid
            axes[0].bar(i, mid, color=COLORS[method], alpha=.75)
            axes[0].errorbar(i, mid, yerr=[[lower], [upper]], color="#283541", capsize=5)
            axes[0].text(i, .97, f"n={len(values)}\nmedian/min/max", ha="center", va="top", fontsize=8, transform=axes[0].get_xaxis_transform())
            values_for_manifest.append({"method": method, "count": len(values), "median_seconds": mid,
                                        "min_seconds": min(values), "max_seconds": max(values)})
        else:
            axes[0].text(i, .05, "N/A", ha="center", transform=axes[0].get_xaxis_transform())
        bottom = 0.
        for field, color, label in (("generation_seconds", "#4583b4", "motion generation"),
                                    ("adaptation_seconds", "#db9b46", "adaptation"),
                                    ("limit_handling_seconds", "#699f77", "limit handling")):
            component = [_number(r.get(field)) for r in sample]
            component = [v for v in component if v is not None]
            if component:
                height = float(np.median(component))
                axes[1].bar(i, height, bottom=bottom, color=color, label=label if i == 0 else None)
                bottom += height
    for ax in axes:
        ax.set_xticks(range(3), ["Our finite", "Our infinite", "Reachy + adaptation"], fontsize=9)
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel("Generation + adaptation + limit handling (s)")
    observed_max=max((r["max_seconds"] for r in values_for_manifest),default=0.)
    if observed_max>0:axes[0].set_ylim(0,observed_max*1.28)
    axes[1].set_ylabel("Median processing components (s)")
    if axes[1].get_legend_handles_labels()[0]: axes[1].legend(fontsize=8)
    fig.suptitle("Equal-workload processing time — initialization/download/rendering excluded")
    note = f"Workload {common[0] if common else 'unavailable'}; warmups excluded. Five repeats do not establish a stable P95."
    save(fig, "generation_latency.png", note)
    manifest["latency_selection"] = {"condition": common[0] if common else None, "values": values_for_manifest}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    _summary_points(axes[0], quality, "window_distance_mean", "Nonadjacent window RMS / joint range")
    _summary_points(axes[1], quality, "recurrence_rate", "Nonadjacent recurrence (%)", percent=True)
    fig.suptitle("Geometry diversity and recurrence — identical physical-range normalization")
    save(fig, "diversity_recurrence.png", quality_scope + ". Dots are saved runs; bars are means. Read with activity/smoothness.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    _summary_points(axes[0], quality, "stationary_fraction", "Stationary time (%)", percent=True)
    _summary_points(axes[1], quality, "jerk_rms_rad_s3", "Sampled jerk RMS (rad/s³)")
    _summary_points(axes[2], quality, "max_sample_step_change_rad", "Largest adjacent q step (rad)")
    fig.suptitle("Activity and smoothness — motion magnitude is not hidden by per-run normalization")
    save(fig, "activity_smoothness.png", quality_scope + ". Sampling/finite-difference metrics are not continuous collision certificates.")

    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    _summary_points(axes[0],quality,"permitted_grid_median_error_ms","Motion-accent to permitted grid median (ms)")
    _summary_points(axes[1],quality,"original_beat_coverage","Original beat coverage (%)",percent=True)
    _summary_points(axes[2],quality,"motion_accent_rate_hz","Detected motion accents per second")
    fig.suptitle("Commanded motion accents versus the shared estimated music grid")
    save(fig,"music_alignment.png","Same fixed detector and half-beat vocabulary for every method. Insufficient activity/events gives N/A alignment, not zero error.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    fields = [("position_violation_rate", "position", "#4583b4"),
              ("velocity_violation_rate", "velocity", "#db9b46"),
              ("acceleration_violation_rate", "acceleration", "#b65c84")]
    for j, (field, label, color) in enumerate(fields):
        for i, method in enumerate(METHODS):
            vals = [_number(r.get(field)) for r in quality if _method(r) == method and _ok(r)]
            vals = [v for v in vals if v is not None]
            if vals: axes[0].bar(i+(j-1)*.23, 100*np.mean(vals), .22, color=color, label=label if i == 0 else None)
    fields2 = [("position_max_normalized_excess", "position", "#4583b4"),
               ("velocity_max_normalized_excess", "velocity", "#db9b46"),
               ("acceleration_max_normalized_excess", "acceleration", "#b65c84")]
    for j, (field, label, color) in enumerate(fields2):
        for i, method in enumerate(METHODS):
            vals = [_number(r.get(field)) for r in quality if _method(r) == method and _ok(r)]
            vals = [v for v in vals if v is not None]
            if vals: axes[1].bar(i+(j-1)*.23, max(vals), .22, color=color, label=label if i == 0 else None)
    for ax in axes:
        ax.set_xticks(range(3), ["Our finite", "Our infinite", "Reachy + adaptation"], fontsize=9)
        ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
        for i, method in enumerate(METHODS):
            if not any(_method(r) == method and _ok(r) for r in quality):
                ax.text(i, .04, "N/A", ha="center", transform=ax.get_xaxis_transform())
    axes[0].set_ylabel("Mean sampled violation rate (%)")
    axes[1].set_ylabel("Largest normalized excess across runs")
    fig.suptitle("Original-output kinematic checks — no unreported smoothing or clipping")
    save(fig, "feasibility.png", "No simulator/hardware feedback: balance, collision avoidance, and physical tracking are unverified.")

    lag_path = results / "lag_similarity.json"
    lag = json.loads(lag_path.read_text(encoding="utf-8")) if lag_path.is_file() else {}
    fig, ax = plt.subplots(figsize=(11, 4.7)); plotted = False
    for run_id, values in lag.items():
        if isinstance(values, dict): values = values.get("lag_similarity", [])
        if not isinstance(values, list) or not values: continue
        method = next((m for m in METHODS if m in run_id), None)
        if method is None: continue
        if "long" not in run_id and "stage3" not in run_id: continue
        x = [_number(v.get("lag_beats")) for v in values]
        y = [_number(v.get("mean_rms_joint_range_distance")) for v in values]
        valid = [(a,b) for a,b in zip(x,y) if a is not None and b is not None]
        if valid:
            ax.plot(*zip(*valid), color=COLORS[method], label=LABELS[method], lw=1.8);plotted = True
    if plotted: ax.legend(fontsize=8)
    else: ax.text(.5,.5,"Long-run lag data unavailable",ha="center",transform=ax.transAxes)
    ax.set(xlabel="Lag (music beats)",ylabel="Mean window distance / physical joint range")
    ax.grid(alpha=.2);fig.suptitle("Long-run recurrence structure — lower distance means more similar windows")
    save(fig,"long_lag_similarity.png","Uses saved beat-aligned windows. A short excerpt alone does not establish escape from long loops.")

    seeds = _rows(results / "seed_distances.csv")
    fig,ax=plt.subplots(figsize=(10,4.7))
    _summary_points(ax,seeds,"rms_joint_range_distance","Across-seed RMS / physical joint range")
    fig.suptitle("Different seeds on matching music/time grids")
    save(fig,"seed_diversity.png","A deterministic schedule is generated once per excerpt; identical repeats are not new dance samples.")

    (target / "plot_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return paths


def _font(size, bold=False):
    from PIL import ImageFont
    candidates=[Path("C:/Windows/Fonts")/("segoeuib.ttf" if bold else "segoeui.ttf"),
                Path("/usr/share/fonts/truetype/dejavu")/("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for path in candidates:
        if path.is_file():return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)


def _rotation(q):
    pitch,roll,yaw=q
    cp,sp=math.cos(pitch),math.sin(pitch);cr,sr=math.cos(roll),math.sin(roll);cy,sy=math.cos(yaw),math.sin(yaw)
    return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])


def _head(draw, center, q, accent, scale=122):
    """Pure orientation schematic; this is not a robot mesh/physics simulator."""
    cx,cy=center
    camera=np.array([4.,-6.,3.]);camera/=np.linalg.norm(camera)
    right=np.cross(np.array([0.,0.,1.]),camera);right/=np.linalg.norm(right)
    up=np.cross(camera,right)
    rotation=_rotation(q)
    def project(v):return (cx+float(v@right)*scale,cy-float(v@up)*scale)
    # Static neck and base establish the body reference frame.
    draw.ellipse((cx-143,cy+144,cx+143,cy+174),fill=(16,29,41),outline=(48,67,79),width=2)
    draw.rounded_rectangle((cx-31,cy+74,cx+31,cy+155),radius=18,fill=(38,50,61),outline=(64,80,91),width=2)
    vertices=np.array([[x,y,z] for x in (-1.05,1.05) for y in (-.78,.78) for z in (-.64,.64)])
    transformed=vertices@rotation.T
    faces=[([0,1,3,2],(36,52,66)),([4,6,7,5],accent),([0,4,5,1],(25,39,53)),
           ([2,3,7,6],(46,64,78)),([0,2,6,4],(23,33,45)),([1,5,7,3],(65,82,93))]
    faces.sort(key=lambda f:float(np.mean(transformed[f[0]]@camera)))
    for ids,color in faces:
        points=[project(transformed[j]) for j in ids]
        fill=tuple(int(c*.45) for c in color) if color==accent else color
        draw.polygon(points,fill=fill)
        draw.line(points+[points[0]],fill=accent if color==accent else (86,105,119),width=2)
    # Front display and its eyes rotate in the same frame as the sampled head.
    front=np.array([[1.066,y,z] for y,z in [(-.63,-.39),(.63,-.39),(.63,.38),(-.63,.38)]])@rotation.T
    draw.polygon([project(v) for v in front],fill=(5,13,22),outline=accent,width=2)
    for y in (-.29,.29):
        eye=np.array([[1.073,y-.105,-.04],[1.073,y+.105,-.04],[1.073,y+.105,.17],[1.073,y-.105,.17]])@rotation.T
        draw.polygon([project(v) for v in eye],fill=accent)
    for vec,col in (([.4,0,0],(239,116,111)),([0,.4,0],(130,214,126)),([0,0,.4],(106,156,235))):
        origin=np.array([-1.6,.9,-.8]);a=project(origin);b=project(origin+np.array(vec)@rotation.T)
        draw.line([a,b],fill=col,width=3)


def _load_stage1(results,crop_start,duration):
    rows=_rows(results/"same_robot_metrics.csv")
    loaded={};failures={}
    for method in METHODS:
        row=next((r for r in rows if _method(r)==method and r.get("stage")=="stage1" and (r.get("seed") or "0")=="0"),{})
        if row and not _ok(row):
            failures[method]=row.get("error") or row.get("failure_reason") or f"Saved run status: {row.get('status')}"
            continue
        candidates=sorted((results/"trajectories").glob(f"stage1__{method}__seed0__beat*.npz"))
        if not candidates:
            candidates=sorted((results/"trajectories").glob(f"stage1__{method}__*.npz"))
        if not candidates:
            failures[method]=row.get("error") or row.get("failure_reason") or "No successful saved stage-1 trajectory"
            continue
        try:
            with np.load(candidates[0],allow_pickle=False) as archive:
                times=np.asarray(archive["times_seconds"],dtype=float);q=np.asarray(archive["q"],dtype=float)
            if q.shape!=(len(times),3) or not np.isfinite(q).all() or len(times)<2 or np.any(np.diff(times)<=0):
                raise ValueError("Expected finite n×3 pitch/roll/yaw trajectory on increasing audio timestamps")
            dt=float(np.median(np.diff(times)))
            # The final saved control sample is held for at most one control
            # period, matching the sampled-command representation at the crop end.
            if crop_start<times[0]-1e-7 or crop_start+duration>times[-1]+dt+1e-7:
                raise ValueError("Requested interval extends outside the saved sampled-command coverage")
            loaded[method]=(times,q)
        except (OSError,KeyError,ValueError) as exc:failures[method]=str(exc)
    return loaded,failures


def render_video(results: Path,audio: Path,crop_start:float,duration=12.0,*,preview_only=False):
    """Encode a three-method commanded head-pose comparison with local music."""
    from PIL import Image,ImageDraw
    results=Path(results);audio=Path(audio)
    if not audio.is_file():raise FileNotFoundError(audio)
    if not math.isfinite(crop_start) or crop_start<0 or not 0<duration<=30:raise ValueError("Invalid audio crop/duration")
    loaded,failures=_load_stage1(results,crop_start,duration)
    target=results/"videos";target.mkdir(parents=True,exist_ok=True)
    # This preprocessing and all encoding are outside motion-generation timing.
    decoded=subprocess.run(["ffmpeg","-v","error","-ss",str(crop_start),"-t",str(duration),"-i",str(audio),
                            "-f","f32le","-ac","1","-ar","11025","pipe:1"],capture_output=True,check=True)
    waveform=np.frombuffer(decoded.stdout,dtype="<f4")
    if len(waveform)<duration*11025-.15*11025:raise ValueError("Source audio crop is too short")
    bins=np.linspace(0,len(waveform),900+1).astype(int)
    envelope=np.array([np.max(np.abs(waveform[a:max(a+1,b)]),initial=0) for a,b in zip(bins[:-1],bins[1:])])
    fonts={size:_font(size,size in (23,30,35,43)) for size in (15,17,19,21,23,26,30,35,43)}
    width,height,fps=1920,1080,30
    background=Image.new("RGB",(width,height),(7,13,22));draw=ImageDraw.Draw(background)
    draw.text((76,43),"MUSIC → MOTION",font=fonts[43],fill=(234,242,246))
    draw.text((1844,70),"SAME-ROBOT / SAME MUSIC / SAME LIMITS",font=fonts[21],fill=(137,158,177),anchor="ra")
    draw.text((78,104),"Misty head subset · pitch / roll / yaw · 3 DOF · raw commanded output",font=fonts[23],fill=(151,171,190))
    draw.line((76,151,1844,151),fill=(50,69,85),width=2)
    panel_x=[76,680,1284];panel_width=560
    metric_rows=_rows(results/"same_robot_metrics.csv")
    for x,method in zip(panel_x,METHODS):
        accent=RGB[method]
        draw.rounded_rectangle((x,180,x+panel_width,858),radius=23,fill=(12,23,35),outline=(46,65,80),width=2)
        draw.line((x+25,195,x+panel_width-25,195),fill=accent,width=3)
        title=LABELS[method]
        if method=="reachy_adapted":title="Reachy scheduler\n+ fixed adaptation"
        draw.multiline_text((x+26,220),title,font=fonts[30],fill=accent,spacing=2)
        draw.text((x+26,310),"COMMAND REFERENCE" if method in loaded else "UNAVAILABLE",font=fonts[17],fill=(145,164,183))
        row=next((r for r in metric_rows if _method(r)==method and r.get("stage")=="stage1"),{})
        if any((_number(row.get(field)) or 0)>0 for field in ("position_violation_rate","velocity_violation_rate","acceleration_violation_rate")):
            draw.text((x+panel_width-25,312),"LIMITS EXCEEDED",font=fonts[15],fill=(244,155,128),anchor="ra")
        if method in failures:
            reason="\n".join(textwrap.wrap(failures[method],width=42))
            draw.multiline_text((x+29,405),reason,font=fonts[21],fill=(220,163,145),spacing=10)
            continue
        times,q=loaded[method]
        for j,col in enumerate(((68,205,220),(214,131,207),(238,189,108))):
            ts=np.linspace(crop_start,crop_start+duration,400)
            vals=np.interp(ts,times,q[:,j]);scale=.5
            points=[(x+31+i/399*(panel_width-62),765-float(np.clip(v/scale,-1,1))*39) for i,v in enumerate(vals)]
            draw.line(points,fill=tuple(int(c*.6) for c in col),width=2)
        draw.line((x+29,765,x+panel_width-29,765),fill=(45,63,79),width=1)
        draw.text((x+28,817),"PITCH      ROLL      YAW  [rad]",font=fonts[17],fill=(145,164,183))
    draw.text((77,951),"Commanded trajectory animation — NOT physics simulation / no hardware",font=fonts[23],fill=(227,211,170))
    draw.text((77,993),"Head-rotation mapping only; XYZ/antenna DOFs omitted. All q traces use the same ±0.5 rad display scale.",font=fonts[21],fill=(146,167,185))

    def render_frame(elapsed):
        music_time=crop_start+elapsed
        frame=background.copy();draw=ImageDraw.Draw(frame)
        for x,method in zip(panel_x,METHODS):
            if method not in loaded:continue
            times,poses=loaded[method]
            q=np.array([np.interp(music_time,times,poses[:,j]) for j in range(3)])
            _head(draw,(x+panel_width/2,495),q,RGB[method])
            draw.text((x+30,679),f"{q[0]:+.3f}   {q[1]:+.3f}   {q[2]:+.3f}",font=fonts[26],fill=(230,240,244))
            cursor=x+31+elapsed/duration*(panel_width-62)
            draw.line((cursor,718,cursor,805),fill=RGB[method],width=2)
        for j,value in enumerate(envelope):
            x=330+j*1.45;hh=3+float(value)*65
            col=(94,217,220) if j/len(envelope)<=elapsed/duration else (48,72,91)
            draw.line((x,900-hh/2,x,900+hh/2),fill=col,width=1)
        draw.text((78,887),"BILLIE JEAN",font=fonts[23],fill=(229,240,245))
        draw.text((1841,888),f"{elapsed:04.1f} / {duration:g} s",font=fonts[23],fill=(229,240,245),anchor="ra")
        return frame

    if preview_only:
        path=target/"preview.jpg"
        render_frame(duration/2).save(path,quality=93)
        return path
    output=target/"same_robot_commanded_trajectory.mp4"
    command=["ffmpeg","-hide_banner","-loglevel","warning","-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{width}x{height}",
             "-r",str(fps),"-i","pipe:0","-ss",str(crop_start),"-i",str(audio),"-map","0:v:0","-map","1:a:0","-t",str(duration),
             "-c:v","libx264","-preset","fast","-crf","19","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k", "-movflags","+faststart",str(output)]
    log_path=target/"encode.log"
    with log_path.open("w",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for i in range(round(duration*fps)):
                frame=render_frame(i/fps)
                if i==round(fps*duration/2):frame.save(target/"preview.jpg",quality=92)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError(log_path.read_text(encoding="utf-8"))
        except BaseException:
            process.kill();process.wait();raise
    metadata={"output":output.name,"resolution":[width,height],"fps":fps,"duration_seconds":duration,
              "crop_start_seconds":crop_start,"scope":"commanded trajectory animation; not physics simulation or hardware feedback",
              "joint_order":["head_pitch","head_roll","head_yaw"],"units":"rad",
              "loaded_methods":list(loaded),"unavailable_methods":failures,
              "audio":"existing local Billie Jean source, no external audio download",
              "adaptation":"fixed head rotation mapping; translation and antennas omitted"}
    metadata["sample_visualization"]="Linear interpolation between saved control samples; final sample held for at most one control period."
    metadata["trace_display_scale_rad"]=[-.5,.5]
    metadata["trace_axis_policy"]="Common fixed display range; plotted lines clip at graph boundary only. Head orientation and numeric q remain raw, unscaled values."
    (target/"video_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    return output


def make_full_cycle_plot(results_path: Path):
    """Plot only the known-synthetic, same-head complete-waveform supplement.

    The period marker is read from the fixed experiment configuration; no peak,
    octave, or phase is fitted to the plotted results. Native Reachy xyz/antenna
    trajectories and G1 data are deliberately outside this plot's scope.
    """
    import hashlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results=Path(results_path)
    config_path=results/"config_snapshot.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    setup=config["full_cycle"]
    period=float(setup["head_waveform_period_beats"])
    if not math.isfinite(period) or period<=0:
        raise ValueError("A positive predeclared head-waveform period is required")
    if config.get("joint_subset")!=["head_pitch","head_roll","head_yaw"]:
        raise ValueError("Full-cycle plot requires the shared Misty head subset")
    rows=[r for r in _rows(results/"same_robot_metrics.csv")
          if r.get("stage")=="synthetic_full_cycle" and _method(r) in METHODS]
    lag_path=results/"lag_similarity.json"
    lag=json.loads(lag_path.read_text(encoding="utf-8"))
    target=results/"figures";target.mkdir(parents=True,exist_ok=True)
    fig,(ax,info)=plt.subplots(1,2,figsize=(14,5.8),gridspec_kw={"width_ratios":[2.7,1.25]})
    selected=[];table_rows=[];maximum_lag=period
    for method in METHODS:
        row=next((r for r in rows if _method(r)==method and str(r.get("seed","0"))==str(setup.get("seed",0))),None)
        values=lag.get(row.get("run_id"),[]) if row and _ok(row) else []
        valid=[]
        for value in values:
            x=_number(value.get("lag_beats"));y=_number(value.get("mean_rms_joint_range_distance"))
            if x is not None and y is not None:valid.append((x,y))
        if valid:
            x,y=zip(*valid)
            ax.plot(x,y,lw=1.6,color=COLORS[method],label=LABELS[method])
            maximum_lag=max(maximum_lag,max(x))
            at_period=next((value for value in values if abs(float(value["lag_beats"])-period)<1e-8),None)
            if at_period:
                ax.scatter([period],[at_period["mean_rms_joint_range_distance"]],s=35,
                           color=COLORS[method],edgecolors="#22313c",linewidths=.6,zorder=4)
        else:
            at_period=None
        def percentage(field):
            value=_number(row.get(field)) if row and _ok(row) else None
            return "N/A" if value is None else f"{value*100:.1f}%"
        short={"our_finite":"Our finite","our_infinite":"Our infinite","reachy_adapted":"Reachy adapted"}[method]
        table_rows.append([short,percentage("recurrence_rate"),percentage("stationary_fraction")])
        selected.append({"method":method,"run_id":row.get("run_id") if row else None,
                         "status":row.get("status") if row else "unavailable",
                         "duration_seconds":_number(row.get("duration_seconds")) if row else None,
                         "lag_at_predeclared_period":at_period})
    ax.axvline(period,color="#303c46",ls="--",lw=1.3,label=f"Predeclared head period: {period:g} beats")
    ax.set(xlim=(0,maximum_lag),xlabel="Lag (known synthetic music beats)",
           ylabel="Mean window RMS distance / physical joint range")
    ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=8,loc="upper right")
    ax.set_title("Same Misty head subset; original commanded outputs",fontsize=10)
    info.axis("off")
    table=info.table(cellText=table_rows,colLabels=["Method","Recurrence","Stationary"],
                    cellLoc="center",colLoc="center",bbox=[-.04,.60,1.08,.30])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#d4dce2")
        if r==0:cell.set_facecolor("#edf2f6")
    durations=[_number(r.get("duration_seconds")) for r in rows if _ok(r)]
    durations=[v for v in durations if v is not None]
    duration_text=f"{min(durations):.1f} s per method" if durations else "Duration N/A"
    lines=[f"Known synthetic {float(setup['bpm']):g} BPM",duration_text,
           f"{float(setup.get('cycles',2)):g} head-waveform cycles; seed {setup.get('seed',0)}",
           "Period is source/configuration-derived,","not fitted to a convenient result peak.","",
           "Raw kinematic violations are retained.","Physical tracking / balance / collision: N/A.",
           "No natural-music quality inference.","Native xyz / antenna channels excluded.",
           "Quality supplement; primary latency unchanged."]
    info.text(0,.50,"\n".join(lines),va="top",fontsize=9,linespacing=1.45,color="#37434e")
    fig.suptitle("Complete head-waveform recurrence — separate synthetic timing stress test",fontsize=13)
    fig.text(.5,.025,"Fixed beat-aligned windows and physical-range normalization. Read recurrence with activity and raw constraint violations.",
             ha="center",fontsize=8,color="#555555")
    fig.tight_layout(rect=(0,.065,1,.95))
    output=target/"full_cycle_lag_similarity.png"
    fig.savefig(output,dpi=150);plt.close(fig)
    manifest={"scope":"known-synthetic same-Misty-head commanded-trajectory quality supplement",
              "native_all_channel_data_included":False,"hardware_executed":False,
              "predeclared_head_waveform_period_beats":period,
              "marker_policy":"From config_snapshot.full_cycle, not post-hoc peak fitting",
              "selected_runs":selected,"config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "lag_json_sha256":hashlib.sha256(lag_path.read_bytes()).hexdigest(),
              "metrics_csv_sha256":hashlib.sha256((results/"same_robot_metrics.csv").read_bytes()).hexdigest(),
              "figure_sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
    (target/"full_cycle_plot_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return output
