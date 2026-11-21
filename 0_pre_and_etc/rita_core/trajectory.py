"""Piecewise quintic Bezier, continuous-time certificates, and shared descriptors."""
from math import comb
import numpy as np
from .contracts import DanceSegment


def bernstein(control, u):
    control = np.asarray(control)
    u = np.asarray(u)
    degree = len(control)-1
    weights = np.stack([comb(degree,i)*u**i*(1-u)**(degree-i) for i in range(degree+1)], axis=-1)
    return weights @ control


def evaluate(segment, t):
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(t).astype(float)
    if np.any(t < segment.start-1e-7) or np.any(t > segment.end+1e-7):
        raise ValueError("trajectory query outside absolute music-time interval")
    indices = np.clip(np.searchsorted(segment.knots,t,side="right")-1,0,len(segment.controls)-1)
    q = np.zeros((len(t),segment.controls.shape[-1])); v=q.copy(); a=q.copy()
    for i in np.unique(indices):
        mask = indices == i
        T = segment.knots[i+1]-segment.knots[i]
        u = np.clip((t[mask]-segment.knots[i])/T,0,1)
        b = segment.controls[i]
        q[mask] = bernstein(b,u)
        v[mask] = bernstein(5*np.diff(b,axis=0)/T,u)
        a[mask] = bernstein(20*np.diff(b,n=2,axis=0)/T**2,u)
    return (q[0],v[0],a[0]) if scalar else (q,v,a)


def make_segment(knots, waypoints, velocities=None, accelerations=None,
                 accent_times=None, accent_strengths=None, rhythm_id="", metrical_level=0, metadata=None):
    knots=np.asarray(knots,dtype=float); q=np.asarray(waypoints,dtype=float)
    if q.ndim != 2 or len(q) != len(knots): raise ValueError("one ordered joint waypoint per knot required")
    v=np.zeros_like(q) if velocities is None else np.asarray(velocities,dtype=float)
    a=np.zeros_like(q) if accelerations is None else np.asarray(accelerations,dtype=float)
    if v.shape != q.shape or a.shape != q.shape: raise ValueError("q/dq/ddq shapes must match")
    b=[]
    for i,T in enumerate(np.diff(knots)):
        if T<=0: raise ValueError("positive piece duration required")
        b.append([q[i], q[i]+T*v[i]/5, q[i]+2*T*v[i]/5+T*T*a[i]/20,
                  q[i+1]-2*T*v[i+1]/5+T*T*a[i+1]/20, q[i+1]-T*v[i+1]/5, q[i+1]])
    return DanceSegment(float(knots[0]),float(knots[-1]-knots[0]),knots,np.array(b),
                        np.array([] if accent_times is None else accent_times),
                        np.array([] if accent_strengths is None else accent_strengths),
                        rhythm_id, metrical_level, metadata or {})


def add_terminal_hold(seg, seconds):
    """Retiming only: finish the last transition early, then hold exact C2 rest.

    Original final pose and phrase end are unchanged. Callers MUST revalidate
    velocity, acceleration, excursion and combined stationary intervals afterward.
    Hardware may omit only this explicitly constant tail from its action message.
    """
    seconds=float(seconds)
    if seconds<=0: return seg
    qend,vend,aend=seg.at(seg.end)
    if max(np.max(np.abs(vend)),np.max(np.abs(aend)))>1e-7:
        raise ValueError("Terminal controller hold requires an existing rest boundary")
    last_start=seg.knots[-2]
    if seconds>=seg.end-last_start-1e-6:
        raise ValueError("Terminal hold consumes the final transition")
    qstart,vstart,astart=seg.at(last_start)
    endmotion=seg.end-seconds
    moving=make_segment([last_start,endmotion],[qstart,qend],
                        [vstart,vend],[astart,aend])
    controls=np.concatenate([seg.controls[:-1],moving.controls,np.repeat(qend[None,None,:],6,axis=1)])
    knots=np.r_[seg.knots[:-1],endmotion,seg.end]
    if np.any((seg.accent_times>endmotion+1e-8)&(seg.accent_times<seg.end-1e-8)):
        raise ValueError("Terminal hold would consume a declared accent")
    return DanceSegment(seg.start,seg.duration,knots,controls,seg.accent_times.copy(),
                        seg.accent_strengths.copy(),seg.rhythm_id,seg.metrical_level,
                        {**seg.metadata,"terminal_hold_seconds":seconds,"terminal_motion_end":endmotion}, {})


def power_coefficients(b):
    n=len(b)-1
    return np.array([comb(n,j)*np.diff(b,n=j,axis=0)[0] for j in range(n+1)])


def slow_intervals(segment, threshold):
    """Solve quartic dq=+/-threshold; intersect all-joint low-speed intervals."""
    intervals=[]
    for i,b in enumerate(segment.controls):
        T=segment.knots[i+1]-segment.knots[i]
        v=5*np.diff(b,axis=0)/T
        p=power_coefficients(v)
        critical=[0.,1.]
        for j in range(v.shape[1]):
            for sign in (-1,1):
                coef=p[:,j].copy();coef[0]-=sign*threshold
                for r in np.polynomial.polynomial.polyroots(coef):
                    if abs(r.imag)<1e-7 and 0<r.real<1: critical.append(float(r.real))
        cuts=np.unique(np.round(critical,12))
        mids=(cuts[:-1]+cuts[1:])/2
        low=np.max(np.abs(bernstein(v,mids)),axis=1)<=threshold+1e-10
        for l,r in zip(cuts[:-1][low],cuts[1:][low]):
            left=segment.knots[i]+l*T; right=segment.knots[i]+r*T
            if intervals and abs(intervals[-1][1]-left)<1e-8: intervals[-1][1]=right
            else: intervals.append([left,right])
    return intervals


def validate_segment(seg, profile, previous=None):
    errors=[]
    if seg.controls.shape[-1] != profile.nj: raise ValueError("trajectory joint count differs from profile order")
    qmax=np.max(seg.controls,axis=(0,1)); qmin=np.min(seg.controls,axis=(0,1))
    vmax=np.zeros(profile.nj);amax=vmax.copy(); continuity=0.
    for i,b in enumerate(seg.controls):
        T=seg.knots[i+1]-seg.knots[i]
        v=5*np.diff(b,axis=0)/T;a=20*np.diff(b,n=2,axis=0)/T**2
        vmax=np.maximum(vmax,np.max(np.abs(v),axis=0));amax=np.maximum(amax,np.max(np.abs(a),axis=0))
        if i:
            pb=seg.controls[i-1];pT=seg.knots[i]-seg.knots[i-1]
            continuity=max(continuity,float(np.max(np.abs(pb[-1]-b[0]))),
                           float(np.max(np.abs(5*(pb[-1]-pb[-2])/pT-v[0]))),
                           float(np.max(np.abs(20*(pb[-1]-2*pb[-2]+pb[-3])/pT**2-a[0]))))
    excess_q=float(max(0,np.max(qmax-profile.upper),np.max(profile.lower-qmin)))
    excess_v=float(max(0,np.max(vmax-profile.vmax)))
    excess_a=float(max(0,np.max(amax-profile.amax)))
    for key,val in (("position_control_hull",excess_q),("velocity_control_hull",excess_v),("acceleration_control_hull",excess_a),("C2_discontinuity",continuity)):
        if val>1e-7: errors.append(key)
    intervals=slow_intervals(seg,profile.speed_threshold)
    maxslow=max((r-l for l,r in intervals),default=0.)
    head=intervals[0][1]-seg.start if intervals and abs(intervals[0][0]-seg.start)<1e-7 else 0.
    tail=seg.end-intervals[-1][0] if intervals and abs(intervals[-1][1]-seg.end)<1e-7 else 0.
    excursion=np.ptp(np.vstack([seg.controls[:,0,:],seg.controls[-1:,-1,:]]),axis=0)
    # Waypoint range is a proven lower bound on the continuous path excursion.
    meaningful=float(np.max(excursion))
    if meaningful<profile.min_excursion-1e-9: errors.append("insufficient_excursion")
    if maxslow>profile.max_stationary_seconds+1e-8: errors.append("stationary_too_long")
    joined_slow=maxslow;join_error=0.
    if previous is not None:
        pstate=previous.end_state(); sstate=seg.start_state()
        join_error=max(abs(pstate.t-sstate.t),*[float(np.max(np.abs(getattr(pstate,k)-getattr(sstate,k)))) for k in ("q","dq","ddq")])
        if join_error>1e-6: errors.append("boundary_C2_discontinuity")
        pv=previous.validation or validate_segment(previous,profile)
        joined_slow=max(maxslow,pv["tail_low_seconds"]+head)
        if joined_slow>profile.max_stationary_seconds+1e-8: errors.append("joined_stationary_too_long")
    workspace=profile.workspace or {}
    workspace_status="not_checked"
    if workspace.get("joint_box_verified",False): workspace_status="local_verified_joint_box_only"
    if workspace.get("requires_collision_checker",False): errors.append("collision_checker_unavailable")
    result={"valid":not errors,"errors":errors,"scope":"continuous reference: Bezier control-hull q/dq/ddq; quartic-root activity",
            "position_excess_bound":excess_q,"velocity_excess_bound":excess_v,"acceleration_excess_bound":excess_a,
            "C2_error":max(continuity,join_error),"max_stationary_seconds":maxslow,"joined_stationary_seconds":joined_slow,
            "head_low_seconds":head,"tail_low_seconds":tail,"min_excursion":meaningful,
            "joint_excursion":excursion.tolist(),"workspace_status":workspace_status,
            "collision_free_certified":False}
    seg.validation=result
    return result


def descriptor(seg, profile):
    """Geometry only: normalized path, excursion, reversals, pair correlation."""
    q,v,_=seg.at(np.linspace(seg.start,seg.end,33))
    scale=np.maximum(profile.upper-profile.lower,1e-6)
    normalized=(q-profile.neutral)/scale
    # Time normalization removes duration; rhythm descriptors are kept separately.
    sampled=normalized[::4].ravel()
    excursion=np.ptp(q,axis=0)/scale
    direction=np.mean(np.diff(np.sign(np.diff(q,axis=0)),axis=0)!=0,axis=0)
    centered=q-q.mean(axis=0)
    norms=np.sqrt(np.sum(centered**2,axis=0))+1e-9
    corr=(centered.T@centered)/(norms[:,None]*norms[None,:])
    return np.r_[sampled,excursion,direction*.15,corr[np.triu_indices(profile.nj,1)]*.08]


def descriptor_distance(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape: raise ValueError("descriptor dimensions differ")
    return float(np.sqrt(np.mean((a-b)**2)))


def rhythm_descriptor(seg):
    times=np.asarray(seg.accent_times)
    return {"intervals":np.round(np.diff(times)/seg.duration,6).tolist(),
            "strengths":np.asarray(seg.accent_strengths).tolist(),"rhythm_id":seg.rhythm_id}
