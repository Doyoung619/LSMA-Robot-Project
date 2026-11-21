"""Frozen, shared commanded-trajectory metrics; physical range normalization."""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


def derivatives(times,q):
    v=np.gradient(q,times,axis=0,edge_order=2)
    a=np.gradient(v,times,axis=0,edge_order=2)
    j=np.gradient(a,times,axis=0,edge_order=2)
    return v,a,j


def beat_windows(times,q,analysis,cfg,profile):
    width=cfg["window_beats"];samples=cfg["window_samples"]
    phase=np.interp([times[0],times[-1]],analysis.beats,np.arange(len(analysis.beats)))
    first=int(np.ceil(phase[0]/width))*width
    descriptors=[];centers=[]
    for b in range(first,int(np.floor(phase[1]))-width+1,width):
        target=np.interp(np.linspace(b,b+width,samples),np.arange(len(analysis.beats)),analysis.beats)
        values=np.stack([np.interp(target,times,q[:,k]) for k in range(q.shape[1])],axis=1)
        descriptors.append(((values-profile.neutral)/(profile.upper-profile.lower)).ravel())
        centers.append(float(np.mean(target)))
    return np.array(descriptors),np.array(centers)


def evaluate(times,q,analysis,profile,cfg,declared=None):
    q=np.asarray(q);times=np.asarray(times)
    if q.shape!=(len(times),profile.nj) or not np.isfinite(q).all():raise ValueError("Invalid/nonfinite trajectory shape")
    if len(times)<5 or np.any(np.diff(times)<=0):raise ValueError("Need >=5 strictly increasing samples")
    v,a,jerk=derivatives(times,q);dt=float(np.median(np.diff(times)));duration=float(times[-1]-times[0]+dt)
    ranges=profile.upper-profile.lower
    excursion=np.ptp(q,axis=0);stationary=np.max(np.abs(v),axis=1)<cfg["stationary_speed_threshold_rad_s"]
    runs=np.diff(np.r_[False,stationary,False].astype(int));beg=np.flatnonzero(runs==1);end=np.flatnonzero(runs==-1)
    max_stop=float(max((b-a for a,b in zip(beg,end)),default=0)*dt)
    pos_excess=np.maximum(np.maximum(profile.lower-q,q-profile.upper),0)/ranges
    vel_excess=np.maximum(np.abs(v)/profile.vmax-1,0);acc_excess=np.maximum(np.abs(a)/profile.amax-1,0)
    desc,centers=beat_windows(times,q,analysis,cfg,profile)
    pairs=[]
    for i in range(len(desc)):
        for k in range(i+cfg["minimum_window_index_separation"],len(desc)):
            pairs.append(float(np.sqrt(np.mean((desc[i]-desc[k])**2))))
    lag=[]
    for shift in range(1,min(400,len(desc)-1)+1):
        d=np.sqrt(np.mean((desc[shift:]-desc[:-shift])**2,axis=1))
        lag.append({"lag_windows":shift,"lag_beats":shift*cfg["window_beats"],"mean_rms_joint_range_distance":float(np.mean(d))})
    # Fixed motion accent detector across all methods; no phase/octave search.
    energy=gaussian_filter1d(np.mean((v/ranges)**2,axis=1),.08/dt)
    peaks,_=signal.find_peaks(energy,prominence=cfg["accent_prominence_per_second_squared"],
                              distance=max(1,int(round(cfg["accent_minimum_interval_seconds"]/dt))))
    events=times[peaks]
    active=float(np.max(excursion))>=cfg["minimum_meaningful_excursion_rad"]
    alignment_valid=active and len(events)>=cfg["accent_minimum_count"] and len(events)/duration>=cfg["accent_minimum_rate_per_second"]
    beats=analysis.beats[(analysis.beats>=times[0])&(analysis.beats<=times[-1])]
    all_phase=np.arange(0,len(analysis.beats)-1,cfg["permitted_subdivision_beats"])
    vocabulary=np.interp(all_phase,np.arange(len(analysis.beats)),analysis.beats)
    vocabulary=vocabulary[(vocabulary>=times[0])&(vocabulary<=times[-1])]
    dist=np.min(np.abs(events[:,None]-vocabulary[None,:]),axis=1) if len(events) and len(vocabulary) else np.array([])
    beat_dist=np.min(np.abs(events[:,None]-beats[None,:]),axis=1) if len(events) and len(beats) else np.array([])
    beat_covered=np.min(np.abs(beats[:,None]-events[None,:]),axis=1)<=cfg["beat_tolerance_seconds"] if len(events) and len(beats) else np.zeros(len(beats),bool)
    declared=np.asarray([] if declared is None else declared)
    declared=declared[(declared>=times[0])&(declared<=times[-1])]
    declared_coverage=float(np.mean(np.min(np.abs(declared[:,None]-events[None,:]),axis=1)<=cfg["beat_tolerance_seconds"])) if len(events) and len(declared) else None
    result={"evaluation_scope":"uniformly sampled commanded reference, not simulator/hardware state or continuous collision certificate",
            "duration_seconds":duration,"control_period_seconds":dt,"joint_excursion_rad":excursion.tolist(),
            "range_normalized_excursion":(excursion/ranges).tolist(),"stationary_fraction":float(np.mean(stationary)),"max_stationary_seconds_sampled":max_stop,
            "jerk_rms_rad_s3":float(np.sqrt(np.mean(jerk**2))),"max_sample_step_change_rad":float(np.max(np.abs(np.diff(q,axis=0)))),
            "position_violation_rate":float(np.mean(pos_excess>1e-7)),"velocity_violation_rate":float(np.mean(vel_excess>1e-7)),"acceleration_violation_rate":float(np.mean(acc_excess>1e-7)),
            "position_max_normalized_excess":float(np.max(pos_excess)),"velocity_max_normalized_excess":float(np.max(vel_excess)),"acceleration_max_normalized_excess":float(np.max(acc_excess)),
            "window_count":len(desc),"nonadjacent_pair_count":len(pairs),"window_distance_mean":float(np.mean(pairs)) if pairs else None,
            "window_distance_min":min(pairs,default=None),"recurrence_rate":float(np.mean(np.array(pairs)<cfg["recurrence_rms_threshold_joint_range"])) if pairs else None,
            "motion_accent_count":len(events),"motion_accent_rate_hz":len(events)/duration,"alignment_valid":bool(alignment_valid),
            "permitted_grid_median_error_ms":float(np.median(dist)*1000) if alignment_valid and len(dist) else None,
            "permitted_grid_p95_error_ms":float(np.percentile(dist,95)*1000) if alignment_valid and len(dist) else None,
            "quarter_beat_median_error_ms":float(np.median(beat_dist)*1000) if alignment_valid and len(beat_dist) else None,
            "original_beat_coverage":float(np.mean(beat_covered)) if alignment_valid and len(beat_covered) else None,
            "declared_rhythm_event_coverage":declared_coverage if alignment_valid else None,
            "music_reference":"shared estimated beat reference, no annotated Billie Jean ground truth",
            "motion_accent_definition":cfg["accent_detector"],"offbeats_allowed_beats":cfg["permitted_subdivision_beats"],
            "physical_tracking":None,"balance":None,"collision_free":None,"nan_count":0}
    return result,{"descriptors":desc,"window_centers":centers,"lag_similarity":lag,"motion_events_seconds":events,"speed_energy":energy}


def seed_distances(trajectories,profile):
    values=[]
    for i,(seed,a) in enumerate(trajectories):
        for other_seed,b in trajectories[i+1:]:
            if a.shape!=b.shape:raise ValueError("Seed trajectories must use matching grids")
            values.append({"seed_a":seed,"seed_b":other_seed,"rms_joint_range_distance":float(np.sqrt(np.mean(((a-b)/(profile.upper-profile.lower))**2)))})
    return values
