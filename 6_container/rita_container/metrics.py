"""Shared trajectory metrics and conservative metrics for sampled rollout logs.

``evaluate_trajectory`` preserves Benchmark1's fixed definitions. ``evaluate_samples``
adds timestamp/freshness validation and resampling for command or measured streams;
the caller must keep those streams separate and label online windows provisional.
"""
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d


METRIC_VERSION = "benchmark1-v1"
DEFAULT_CONFIG = {
    "window_beats": 4,
    "window_samples": 32,
    "recurrence_rms_threshold_joint_range": 0.04,
    "minimum_window_index_separation": 2,
    "accent_detector": "peaks of mean squared joint-range-normalized velocity",
    "accent_prominence_per_second_squared": 0.0025,
    "accent_minimum_interval_seconds": 0.2,
    "accent_minimum_count": 2,
    "accent_minimum_rate_per_second": 0.1,
    "beat_tolerance_seconds": 0.1,
    "permitted_subdivision_beats": 0.5,
    "minimum_meaningful_excursion_rad": 0.03,
    "stationary_speed_threshold_rad_s": 0.005,
}


def _profile(value):
    """Accept a RobotProfile or its manifest representation, without mutation."""
    get = value.get if isinstance(value, Mapping) else lambda key, default=None: getattr(value, key, default)
    names = tuple(get("joint_names", ()))
    if not names or len(set(names)) != len(names):
        raise ValueError("Profile requires ordered unique joint_names")
    if get("units", "rad") != "rad":
        raise ValueError("Trajectory metrics require rad units")
    fields = {}
    for name in ("neutral", "lower", "upper", "vmax", "amax"):
        fields[name] = np.asarray(get(name), dtype=float)
        if fields[name].shape != (len(names),) or not np.isfinite(fields[name]).all():
            raise ValueError(f"Profile {name} must be a finite joint vector")
    if np.any(fields["upper"] <= fields["lower"]):
        raise ValueError("Profile joint ranges must be positive")
    if np.any(fields["vmax"] <= 0) or np.any(fields["amax"] <= 0):
        raise ValueError("Profile derivative limits must be positive")
    period = float(get("command_period", 0.1))
    if not np.isfinite(period) or period <= 0:
        raise ValueError("Profile command_period must be positive")
    return SimpleNamespace(**fields, nj=len(names), joint_names=names, command_period=period,
                           workspace=get("workspace", {}) or {})


def _analysis(value):
    if value is None:
        raise ValueError("No beat reference supplied")
    beats = np.asarray(value.get("beats") if isinstance(value, Mapping) else value.beats, dtype=float)
    if beats.ndim != 1 or len(beats) < 2 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        raise ValueError("Beat reference needs >=2 finite increasing timestamps")
    return SimpleNamespace(beats=beats)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sample_diagnostics(records, profile):
    """Optional sampled tracking and host receipt age; no inferred ground truth."""
    squared_error, max_errors, ages = [], [], []
    invalid_reference = invalid_age = 0
    for record in records:
        metadata = record.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        reference = metadata.get("reference_q")
        if reference is not None:
            try:
                reference = np.asarray(reference, dtype=float)
                if reference.shape != (profile.nj,) or not np.isfinite(reference).all():
                    raise ValueError("Invalid reference")
                difference = np.asarray(record["q"], dtype=float) - reference
                squared_error.extend((difference ** 2).tolist())
                max_errors.append(float(np.max(np.abs(difference))))
            except (TypeError, ValueError):
                invalid_reference += 1
        elif metadata.get("tracking_max_abs_error_rad") is not None:
            try:
                error = float(metadata["tracking_max_abs_error_rad"])
                if not np.isfinite(error) or error < 0:
                    raise ValueError("Invalid tracking error")
                max_errors.append(error)
            except (TypeError, ValueError):
                invalid_reference += 1
        observed, feedback = record.get("observed_monotonic"), record.get("feedback_monotonic")
        if observed is not None and feedback is not None:
            try:
                age = float(observed) - float(feedback)
                if not np.isfinite(age) or age < -1e-9:
                    raise ValueError("Inconsistent receipt clocks")
                ages.append(max(0., age))
            except (TypeError, ValueError):
                invalid_age += 1
    metrics = {
        "tracking_rmse_rad": float(np.sqrt(np.mean(squared_error))) if squared_error else None,
        "tracking_max_abs_error_rad": max(max_errors, default=None),
        "tracking_sample_count": len(max_errors),
        "tracking_reference_pair_count": len(squared_error) // profile.nj,
        "feedback_age_seconds_median": float(np.median(ages)) if ages else None,
        "feedback_age_seconds_p95": float(np.percentile(ages, 95)) if ages else None,
        "feedback_age_seconds_max": max(ages, default=None),
        "feedback_age_sample_count": len(ages),
    }
    quality = {"tracking_reference_invalid_count": invalid_reference,
               "feedback_age_invalid_count": invalid_age}
    return metrics, quality


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


def evaluate_trajectory(times,q,analysis,profile,cfg=None,declared=None):
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    profile = _profile(profile)
    analysis = _analysis(analysis)
    q=np.asarray(q);times=np.asarray(times)
    if q.shape!=(len(times),profile.nj) or not np.isfinite(q).all():raise ValueError("Invalid/nonfinite trajectory shape")
    if len(times)<5 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError("Need >=5 finite strictly increasing samples")
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
    for limit in profile.workspace.get("unavailable_limits", []):
        if limit in ("position", "velocity", "acceleration"):
            result[f"{limit}_violation_rate"] = None
            result[f"{limit}_max_normalized_excess"] = None
    return result,{"descriptors":desc,"window_centers":centers,"lag_similarity":lag,"motion_events_seconds":events,"speed_energy":energy}


def seed_distances(trajectories,profile):
    profile = _profile(profile)
    values=[]
    for i,(seed,a) in enumerate(trajectories):
        for other_seed,b in trajectories[i+1:]:
            if a.shape!=b.shape:raise ValueError("Seed trajectories must use matching grids")
            values.append({"seed_a":seed,"seed_b":other_seed,"rms_joint_range_distance":float(np.sqrt(np.mean(((a-b)/(profile.upper-profile.lower))**2)))})
    return values


def evaluate_samples(samples, profile, analysis=None, config=None):
    """Evaluate one command OR measured channel, with no interpolation over holes.

    Input records contain ``music_time`` and ``q`` in profile joint order. Optional
    ``fresh`` and ``feedback_monotonic`` fields remove stale/duplicate feedback.
    Backward clocks, nonfinite values, and gaps exceeding max(.25, 3*period)
    invalidate the window. Raw records remain the caller's responsibility to save.
    Pre/post-playback samples outside an audio beat reference are counted and
    excluded explicitly; gap validation applies to the evaluated overlap only.

    With no beat reference, diversity uses fixed two-second windows and all audio
    alignment scores are null. This is a distinct diversity reference and should
    not be pooled with beat-window results. Values derived from q are sampled
    kinematic diagnostics, not a continuous-time or physical safety certificate.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    result = {
        "metric_version": METRIC_VERSION,
        "configuration": dict(cfg),
        "status": "insufficient_data",
        "reason": "fewer_than_five_unique_fresh_samples",
        "quality": {"input_count": 0, "fresh_count": 0, "stale_count": 0,
                    "duplicate_count": 0, "sample_count": 0, "resampled_count": 0,
                    "out_of_reference_sample_count": 0, "evaluation_sample_count": 0,
                    "timestamp_snap_count": 0},
        "diversity_reference": "estimated_beats" if analysis is not None else "fixed_2_second_windows",
        "sync_reason": None if analysis is not None else "no_audio_reference",
        "metrics": {},
        "series": {},
    }
    quality = result["quality"]

    def invalid(reason, detail=None):
        result.update(status="invalid_data", reason=reason)
        if detail:
            result["detail"] = str(detail)
        return _json_value(result)

    try:
        profile = _profile(profile)
        period = profile.command_period
        quality.update(control_period_seconds=period, maximum_allowed_gap_seconds=max(.25, 3 * period))
        records = list(samples)
        quality["input_count"] = len(records)
        times, poses, retained = [], [], []
        previous_time = previous_feedback = None
        for record in records:
            if not isinstance(record, Mapping):
                return invalid("invalid_sample", "Each sample must be an object")
            fresh = record.get("fresh", True)
            if not isinstance(fresh, (bool, np.bool_)):
                return invalid("invalid_fresh_flag")
            if not fresh:
                quality["stale_count"] += 1
                continue
            quality["fresh_count"] += 1
            t = float(record["music_time"])
            q = np.asarray(record["q"], dtype=float)
            if not np.isfinite(t) or q.shape != (profile.nj,) or not np.isfinite(q).all():
                return invalid("nonfinite_or_invalid_sample")
            if "dq" in record and record["dq"] is not None:
                dq = np.asarray(record["dq"], dtype=float)
                if dq.shape != q.shape or not np.isfinite(dq).all():
                    return invalid("nonfinite_or_invalid_velocity")
            stamp = record.get("feedback_monotonic")
            if stamp is not None:
                stamp = float(stamp)
                if not np.isfinite(stamp):
                    return invalid("nonfinite_feedback_timestamp")
            if previous_time is not None and t < previous_time:
                return invalid("backward_music_timestamp")
            if stamp is not None and previous_feedback is not None and stamp < previous_feedback:
                return invalid("backward_feedback_timestamp")
            duplicate = (previous_time is not None and t == previous_time) or (
                stamp is not None and previous_feedback is not None and stamp == previous_feedback)
            previous_time = t
            if stamp is not None:
                previous_feedback = stamp
            if duplicate:
                quality["duplicate_count"] += 1
                continue
            times.append(t)
            poses.append(q)
            retained.append(record)
        quality["sample_count"] = len(times)
        reference = _analysis(analysis) if analysis is not None else None
        if reference is not None:
            in_reference = [reference.beats[0] - 1e-7 <= t <= reference.beats[-1] + 1e-7 for t in times]
            quality["out_of_reference_sample_count"] = len(times) - sum(in_reference)
            times = [t for t, keep in zip(times, in_reference) if keep]
            poses = [q for q, keep in zip(poses, in_reference) if keep]
            retained = [record for record, keep in zip(retained, in_reference) if keep]
            if times:
                # Mapping host-monotonic epochs to music seconds loses a few
                # ulps. Snap only sub-microsecond beat coincidences in this
                # working copy so frozen ceil/floor window boundaries are stable.
                values = np.asarray(times)
                right = np.searchsorted(reference.beats, values)
                candidates = reference.beats[np.clip(np.column_stack((right - 1, right)),
                                                     0, len(reference.beats) - 1)]
                nearest = candidates[np.arange(len(values)),
                                     np.argmin(np.abs(candidates - values[:, None]), axis=1)]
                snap = np.abs(nearest - values) <= 1e-7
                quality["timestamp_snap_count"] = int(np.count_nonzero(snap & (nearest != values)))
                values[snap] = nearest[snap]
                keep = np.r_[True, np.diff(values) > 0]
                quality["duplicate_count"] += int(np.count_nonzero(~keep))
                times = values[keep].tolist()
                poses = [q for q, use in zip(poses, keep) if use]
                retained = [record for record, use in zip(retained, keep) if use]
        quality["evaluation_sample_count"] = len(times)
        if len(times) > 1:
            gaps = np.diff(times)
            quality["maximum_observed_gap_seconds"] = float(np.max(gaps))
            if np.any(gaps > quality["maximum_allowed_gap_seconds"] + 1e-10):
                return invalid("sample_gap_exceeds_limit")
        if len(times) < 5:
            if analysis is not None:
                result["sync_reason"] = "insufficient_audio_overlap"
                if quality["out_of_reference_sample_count"]:
                    result["reason"] = "fewer_than_five_samples_in_beat_reference"
            return _json_value(result)
        t = np.asarray(times)
        q = np.asarray(poses)
        grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / period + 1e-9)) + 1) * period
        quality["resampled_count"] = len(grid)
        quality["window_start_music_seconds"] = float(t[0])
        quality["window_end_music_seconds"] = float(t[-1])
        quality["resampled_end_music_seconds"] = float(grid[-1])
        quality["resampling"] = "linear interpolation within validated gap limit; uniform profile.command_period"
        if len(grid) < 5:
            result["reason"] = "fewer_than_five_resampled_points"
            return _json_value(result)
        values = np.column_stack([np.interp(grid, t, q[:,j]) for j in range(profile.nj)])
        if analysis is None:
            # Reuse the descriptor implementation with a two-second physical grid.
            # The generated grid is never an audio reference for sync reporting.
            anchor = 2 * np.floor(grid[0] / 2)
            beats = np.arange(anchor, grid[-1] + 2.5, .5)
            reference = SimpleNamespace(beats=beats)
            evaluation_cfg = {**cfg, "window_beats": 4}
        else:
            evaluation_cfg = cfg
        declared = []
        for record in records:
            declared.extend(record.get("declared_accent_times", []) or [])
            metadata = record.get("metadata") or {}
            if isinstance(metadata, Mapping):
                declared.extend(metadata.get("declared_accent_times", []) or [])
        declared = np.asarray(declared, dtype=float)
        if not np.isfinite(declared).all():
            return invalid("nonfinite_declared_accent_time")
        declared = np.unique(declared)
        metrics, series = evaluate_trajectory(grid, values, reference, profile, evaluation_cfg, declared)
        diagnostics, diagnostic_quality = _sample_diagnostics(retained, profile)
        metrics.update(diagnostics)
        quality.update(diagnostic_quality)
        metrics["evaluation_scope"] = "uniformly resampled single logged channel; sampled kinematics, not continuous collision/balance certification"
        if analysis is None:
            metrics["alignment_valid"] = False
            for key in ("permitted_grid_median_error_ms", "permitted_grid_p95_error_ms",
                        "quarter_beat_median_error_ms", "original_beat_coverage",
                        "declared_rhythm_event_coverage"):
                metrics[key] = None
            metrics["music_reference"] = None
            metrics["offbeats_allowed_beats"] = None
        elif not metrics["alignment_valid"]:
            result["sync_reason"] = "insufficient_meaningful_motion_accents"
        result.update(status="ok", reason=None, metrics=metrics, series=series)
        return _json_value(result)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        return invalid("invalid_input", exc)


# Compatibility with the original offline Benchmark1 API.
evaluate = evaluate_trajectory
