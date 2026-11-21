"""Thin offline calls to the existing planners. No hardware adapter is imported."""
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
for folder in (REPO/"0_pre_and_etc",REPO/"1_finite_version",REPO/"2_infinite_version"):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))

from rita_core.contracts import BoundaryState,RobotProfile
from rita_core.profiles import default_profile


def robot_profile():
    p=default_profile("misty")
    return RobotProfile("misty-head-benchmark-UNVERIFIED",p.model,tuple(p.joint_names[:3]),
                        p.neutral[:3],p.lower[:3],p.upper[:3],p.vmax[:3],p.amax[:3],
                        command_mode="mock",command_period=p.command_period,
                        feedback_freshness=p.feedback_freshness,workspace=dict(p.workspace),
                        min_excursion=p.min_excursion,max_stationary_seconds=p.max_stationary_seconds,
                        speed_threshold=p.speed_threshold)


def create(mode,analysis,profile,seed):
    start=time.perf_counter()
    if mode=="our_finite":
        from rita_finite import FinitePlanner
        planner=FinitePlanner(analysis,profile,seed=seed)
    elif mode=="our_infinite":
        from rita_infinite import InfinitePlanner
        planner=InfinitePlanner(analysis,profile,seed=seed)
    else:raise ValueError(f"Unsupported existing planner: {mode}")
    return planner,time.perf_counter()-start


def generate(planner,analysis,profile,start_beat,times):
    """Generate actual coefficient segments, then sample the same requested crop."""
    start=time.perf_counter()
    state=BoundaryState(analysis.beat_time(start_beat),profile.neutral.copy(),np.zeros(profile.nj),np.zeros(profile.nj))
    requested_end=float(times[-1])+profile.command_period
    beat=start_beat;segments=[];latencies=[];deadline_misses=0;accent=[];strength=[];levels=[]
    previous=None;first_valid=None
    while state.t<requested_end-1e-8:
        before=time.perf_counter();segment=planner.next_segment(beat,state);elapsed=time.perf_counter()-before
        if first_valid is None:first_valid=time.perf_counter()-start
        if previous is not None:
            a=previous.end_state();b=segment.start_state()
            mismatch=max(abs(a.t-b.t),*[float(np.max(np.abs(getattr(a,k)-getattr(b,k)))) for k in ("q","dq","ddq")])
            if mismatch>1e-6:raise RuntimeError(f"Existing planner C2 join failed: {mismatch}")
            if elapsed>previous.duration:deadline_misses+=1
        if not segment.validation.get("valid",False):raise RuntimeError(f"Existing planner returned an invalid segment: {segment.validation}")
        segments.append(segment);latencies.append(elapsed);accent.extend(segment.accent_times);strength.extend(segment.accent_strengths)
        levels.append({"music_start_seconds":segment.start,"metrical_level":segment.metrical_level,
                       "motion_beat_ratio":1/(2**segment.metrical_level),"rhythm_id":segment.rhythm_id})
        state=segment.end_state();beat=int(segment.metadata["end_beat"]);previous=segment
    q=np.empty((len(times),profile.nj));dq=q.copy();ddq=q.copy()
    for i,t in enumerate(times):
        seg=next((s for s in segments if s.start-1e-8<=t<=s.end+1e-8),None)
        if seg is None:raise RuntimeError("Generated reference does not cover requested sample time")
        q[i],dq[i],ddq[i]=seg.at(float(t))
    normal=[s for s in segments if not (s.metadata.get("fallback") or s.metadata.get("recovery"))]
    return {"q":q,"analytic_dq":dq,"analytic_ddq":ddq,"generation_seconds":time.perf_counter()-start,
            "adaptation_seconds":0.,"limit_handling_seconds":0.,"input":"shared cached estimated beat timestamps and accents",
            "declared_accent_times":np.array(accent),"declared_accent_strengths":np.array(strength),
            "metadata":{"planner_source":str(Path(sys.modules[planner.__class__.__module__].__file__).relative_to(REPO)),
                        "existing_planner_unmodified":True,"generated_segment_count":len(segments),"normal_segment_count":len(normal),
                        "planned_until_music_seconds":segments[-1].end,"requested_until_music_seconds":requested_end,
                        "sampled_crop_may_end_inside_a_valid_segment":True,"first_valid_chunk_seconds_after_initialization":first_valid,
                        "chunk_planning_seconds":latencies,"chunk_deadline_misses":deadline_misses,"metrical_timeline":levels,
                        "offline_audio_grid_known":True,"motion_generation":"incremental planner calls; accelerated offline playback, not real-time listening",
                        "horizon_metadata":[{k:s.metadata[k] for k in ("horizon","horizon_requested","future_audio_lookahead_seconds") if k in s.metadata} for s in segments],
                        "planner_stats":planner.stats(),"hardware_executed":False}}
