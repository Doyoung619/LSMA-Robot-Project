"""Shared reference/feedback metrics; no physical results are fabricated."""
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy import signal
from .contracts import serial
from .trajectory import descriptor,descriptor_distance,rhythm_descriptor


def provenance():
    root=Path(__file__).resolve().parents[2]
    def git(*args):
        p=subprocess.run(["git","-C",str(root),*args],capture_output=True,text=True)
        return p.stdout.strip() if p.returncode==0 else None
    return {"git_sha":git("rev-parse","HEAD"),"git_dirty":bool(git("status","--porcelain")),"branch":git("branch","--show-current")}


def longest_run(seq):
    longest=0;current=0;previous=object()
    for item in seq:
        current=current+1 if item==previous else 1;longest=max(longest,current);previous=item
    return longest


def event_match(segments,times,positions,tolerance=.035):
    """Detect nearest local joint-pose passage in feedback around each target.

    Accent means the commanded waypoint passage, not every integer beat.
    Rests and subdivisions are respected. Duplicate detections are not reused.
    """
    errors=[];missed=0;used=set();matches=[]
    for seg in segments:
        events=list(seg.accent_times)
        for n,target in enumerate(events):
            q=seg.at(target)[0]
            separation=min([abs(target-x) for x in events if abs(x-target)>1e-7] or [.5])
            window=min(.24,separation*.45)
            ids=np.flatnonzero(np.abs(times-target)<=window)
            if len(ids)<3: missed+=1;continue
            # A stationary robot near a target is not an observed pose passage.
            if np.max(np.ptp(positions[ids],axis=0))<.002:
                missed+=1;continue
            distances=np.max(np.abs(positions[ids]-q),axis=1)
            # Choose the closest pose, then the closest time when plateau samples tie.
            order=np.lexsort((np.abs(times[ids]-target),np.round(distances,8)))
            chosen=next((int(ids[i]) for i in order if int(ids[i]) not in used),None)
            if chosen is None or np.max(np.abs(positions[chosen]-q))>tolerance:
                missed+=1;continue
            local=int(np.flatnonzero(ids==chosen)[0])
            # Require a localized distance minimum with observable approach/exit.
            if local==0 or local==len(ids)-1 or min(distances[0],distances[-1])<distances[local]+.0005:
                missed+=1;continue
            used.add(chosen);err=float(times[chosen]-target);errors.append(err)
            matches.append({"target_seconds":float(target),"measured_seconds":float(times[chosen]),"error_ms":err*1000,
                            "rhythm_id":seg.rhythm_id})
    return {"event":"localized joint-waypoint distance minimum with >=.002 rad window excursion and approach/exit; offbeat targets included",
            "median_abs_ms":float(np.median(np.abs(errors))*1000) if errors else None,
            "p95_abs_ms":float(np.percentile(np.abs(errors),95)*1000) if errors else None,
            "missed_events":missed,"matched_events":len(errors),
            "extra_events":None,"extra_events_note":"No independent full motion-event detector; unclassified extra passages are N/A",
            "matches":matches}


def evaluate(segments,profile,analysis,feedback,planner_stats,run_info,planning_times):
    normal=[s for s in segments if not (s.metadata.get("fallback") or s.metadata.get("recovery") or s.metadata.get("excluded_from_diversity") or s.metadata.get("generation_kind") in ("recovery","fallback_suffix"))]
    motion=[s.metadata.get("primitive_id",s.metadata.get("cell","continuous")) for s in normal]
    rhythm=[s.rhythm_id for s in normal]
    desc=[descriptor(s,profile) for s in normal]
    nearest=[min(descriptor_distance(desc[i],x) for x in desc[max(0,i-16):i]) for i in range(1,len(desc))]
    cells=[]
    for s in normal:
        q=s.at(np.linspace(s.start,s.end,33))[0]-profile.neutral
        exc=np.ptp(q,axis=0);dominant=int(np.argmax(exc));sign="+" if np.mean(q[:,dominant])>=0 else "-"
        cells.append(f"g{dominant%2}{sign}:"+("high" if np.max(exc/(profile.upper-profile.lower))>.18 else "low"))
    ngram={}
    for n in (2,3,4):
        # Shared geometry-cell sequences, never fresh infinite IDs/hashes.
        grams=[tuple(cells[i:i+n]) for i in range(max(0,len(cells)-n+1))]
        counts=Counter(grams);ngram[str(n)]=sum(max(0,c-1) for c in counts.values())/max(1,len(grams))
    visits={}
    for s,cell in zip(normal,cells): visits.setdefault(cell,[]).append(s.start)
    revisit=[b-a for values in visits.values() for a,b in zip(values[:-1],values[1:])]
    ts=[];qs=[];vs=[];acc=[];maxslow=0.;tail=0.
    for s in segments:
        t=np.linspace(s.start,s.end,max(3,int(s.duration/.025)+1));q,v,a=s.at(t)
        ts.extend(t[:-1]);qs.extend(q[:-1]);vs.extend(v[:-1]);acc.extend(a[:-1])
        val=s.validation
        maxslow=max(maxslow,val.get("max_stationary_seconds",0),tail+val.get("head_low_seconds",0))
        tail=val.get("tail_low_seconds",0)
    ts=np.array(ts);qs=np.array(qs);vs=np.array(vs);acc=np.array(acc)
    samples=np.array([f["q"] for f in feedback]) if feedback else np.empty((0,profile.nj))
    ft=np.array([f["music_time"] for f in feedback])
    measured=event_match(normal,ft,samples) if len(samples) else None
    measured_feas=None
    if len(samples)>8:
        dt=float(np.median(np.diff(ft)));window=min(9,len(samples)//2*2-1)
        filtered=signal.savgol_filter(samples,window,3,axis=0)
        velocity=np.gradient(filtered,ft,axis=0);acceleration=np.gradient(velocity,ft,axis=0)
        measured_feas={"position_excess":float(max(0,np.max(samples-profile.upper),np.max(profile.lower-samples))),
                       "velocity_excess_estimate":float(max(0,np.max(np.abs(velocity)-profile.vmax))),
                       "acceleration_excess_estimate":float(max(0,np.max(np.abs(acceleration)-profile.amax))),
                       "derivative_estimator":f"q Savitzky-Golay window={window}, poly=3, then gradients at actual sample timestamps; median dt={dt:.6g}s; estimates, not continuous-time certificates"}
    latency=np.array(planning_times)
    report={"provenance":{**provenance(),"seed":run_info["seed"],"audio_hash":analysis.audio_hash,
                          "profile":profile.to_dict(),"backend":run_info["backend"],"config":run_info,
                          "analysis_settings":analysis.settings},
            "run":run_info,"segments_executed":len(segments),"normal_dance_segments":len(normal),
            "reference_duration_seconds":float(segments[-1].end-segments[0].start) if segments else 0.,
            "synchronization":{"reference":{"event":"exact coefficient waypoint passage at declared rhythm targets","median_abs_ms":0. if normal else None,
                                                "missed_events":0 if normal else None,"scope":"reference only, not ground-truth music labels"},
                               "simulated_measured" if run_info["backend"]=="mock" else "hardware_measured":measured,
                               "physical_robot":None if run_info["backend"]=="mock" else measured},
            "feasibility":{"scope":"continuous reference control hull and C2; not collision-free or measured tracking certification",
                           "invalid_segments":sum(not s.validation.get("valid",False) for s in segments),
                           "max_position_excess_bound":max((s.validation.get("position_excess_bound",0) for s in segments),default=0.),
                           "max_velocity_excess_bound":max((s.validation.get("velocity_excess_bound",0) for s in segments),default=0.),
                           "max_acceleration_excess_bound":max((s.validation.get("acceleration_excess_bound",0) for s in segments),default=0.),
                           "max_C2_error":max((s.validation.get("C2_error",0) for s in segments),default=0.),
                           "collision_certified":False,"workspace":profile.workspace,
                           "simulated_estimates" if run_info["backend"]=="mock" else "measured_estimates":measured_feas},
            "motion_diversity":{"library_count":planner_stats.get("library_count"),"feasible_motion_count":planner_stats.get("feasible_motion_count"),
                                "scc_motion_count":planner_stats.get("scc_motion_count"),"motion_ids_observed":len(set(motion)),
                                "library_coverage":len(set(motion))/planner_stats["library_count"] if planner_stats.get("library_count") else None,
                                "shared_descriptor_feature_cells_visited":len(set(cells)),"shared_descriptor_feature_cells_total":8 if profile.nj>1 else 4,
                                "recent_window_distance_median":float(np.median(nearest)) if nearest else None,
                                "recent_window_distance_min":min(nearest,default=None),"short_repeat_ngram_ratio":ngram,
                                "repetition_basis":"shared geometry feature-cell sequence, not generated IDs or hashes",
                                "observed_max_revisit_seconds":max(revisit,default=None),"raw_hashes_not_used":True},
            "rhythm_diversity":{"event_interval_strength_patterns":len(set(json.dumps({k:v for k,v in rhythm_descriptor(s).items() if k!='rhythm_id'},sort_keys=True) for s in normal)),
                                "rhythm_ids":sorted(set(rhythm)),"longest_single_rhythm_run":longest_run(rhythm)},
            "activity":{"max_low_speed_seconds_continuous":maxslow,
                        "reference_stationary_fraction_sampled":float(np.mean(np.max(np.abs(vs),axis=1)<profile.speed_threshold)) if len(vs) else None,
                        "minimum_segment_excursion_rad":min((s.validation.get("min_excursion",0) for s in normal),default=None),
                        "low_speed_threshold_rad_s":profile.speed_threshold,"whole_robot_low_speed":True},
            "runtime":{"planning_seconds_median":float(np.median(latency)) if len(latency) else None,
                       "planning_seconds_p95":float(np.percentile(latency,95)) if len(latency) else None,
                       "planning_seconds_max":float(latency.max()) if len(latency) else None,
                       "deadline_misses":run_info.get("deadline_misses",0),"fallback_count":sum(bool(s.metadata.get("fallback")) for s in segments),
                       "recovery_count":len(segments)-len(normal),"novelty_relaxations":sum(s.metadata.get("novelty_relaxation_steps",0) for s in normal),
                       "k_changes":sum(a.metrical_level!=b.metrical_level for a,b in zip(segments[:-1],segments[1:])),
                       "certificate_resets":planner_stats.get("certificate_resets",0)},"planner":planner_stats}
    return serial(report),(ts,qs,vs,acc,ft,samples,analysis.onset_times,analysis.onset_strength,analysis.energy,analysis.beats,analysis.accent)


def save_report(output,report,series):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(serial(report),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    ts,qs,vs,acc,ft,samples,music_times,onset,energy,beats,accent=series
    with (output/"reference.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);n=qs.shape[1] if len(qs) else 0
        writer.writerow(["music_seconds"]+[f"{key}_{j}" for key in ("q_rad","dq_rad_s","ddq_rad_s2") for j in range(n)])
        for i in range(0,len(ts),4): writer.writerow([ts[i],*qs[i],*vs[i],*acc[i]])
    if not len(ts): return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(6,1,figsize=(12,13),sharex=True)
    mask=(music_times>=ts[0])&(music_times<=ts[-1])
    axes[0].plot(music_times[mask],onset[mask]/(np.max(onset)+1e-8),lw=.6,label="onset")
    axes[0].plot(music_times[mask],energy[mask],lw=.5,alpha=.6,label="energy")
    bm=(beats>=ts[0])&(beats<=ts[-1]);axes[0].vlines(beats[bm],0,accent[bm],color="orange",alpha=.3,lw=.5)
    axes[0].set_ylabel("Music / beat accent");axes[0].legend(fontsize=7)
    for ax,values,label in zip(axes[1:4],(qs,vs,acc),("q (rad)","dq (rad/s)","ddq (rad/s²)")):
        ax.plot(ts,values,lw=.7);ax.set_ylabel(label);ax.grid(alpha=.2)
    matches=next((v for k,v in report["synchronization"].items() if k in ("simulated_measured","hardware_measured")),None)
    if matches:
        events=matches["matches"];axes[4].scatter([m["target_seconds"] for m in events],[m["error_ms"] for m in events],s=5)
    axes[4].set_ylabel("Pose event Δ (ms)")
    axes[5].bar([ts[0],ts[-1]],[report["activity"]["max_low_speed_seconds_continuous"]]*2,width=1,label="max low-speed duration")
    axes[5].set(ylabel="Low-speed (s)",xlabel="Audio seconds (beat-origin blocks; not detected downbeats)")
    fig.suptitle(f"{report['run']['method']} | {report['run']['condition']} | {report['provenance']['backend']} | seed {report['run']['seed']}")
    fig.tight_layout();fig.savefig(output/"summary.png",dpi=110);plt.close(fig)
