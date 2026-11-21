"""Behavior-preserving compiled scoring for the independently frozen V2.

Features, parameters, candidate generation, and refinement are unchanged.
Near-tied scores fall back to the frozen NumPy implementation. Accuracy reuse
requires an exact full-output equivalence check, saved separately from timing.
"""
import hashlib
import math
from pathlib import Path
import numpy as np
from numba import njit
from .classical_v2 import ClassicalV2Estimator,features,estimate as reference_estimate
from .base import result


@njit(cache=True,fastmath=False)
def _score_grid(times,onset,periods,phases,duration,exponent,prior_strength):
    best=-np.inf
    second=-np.inf
    best_p=0
    best_phase=0
    max_beats=math.ceil(duration/np.min(periods))+1
    dt=times[1]-times[0]
    for pi in range(len(periods)):
        period=periods[pi]
        normalizer=period**exponent
        prior=math.exp(-.5*(math.log2((60/period)/120)/.8)**2)**prior_strength
        for si in range(len(phases)):
            total=0.
            count=0
            for bi in range(max_beats):
                position=period*(phases[si]+bi)
                if position < .25 or position > duration-.25:
                    continue
                index=min(max(int(position/dt),0),len(times)-2)
                while index>0 and times[index]>position:
                    index-=1
                while index<len(times)-2 and times[index+1]<position:
                    index+=1
                slope=(onset[index+1]-onset[index])/(times[index+1]-times[index])
                total+=slope*(position-times[index])+onset[index]
                count+=1
            value=(total/max(count,1))/normalizer*prior
            if value>best:
                second=best
                best=value
                best_p=pi
                best_phase=si
            elif value>second:
                second=value
    ambiguous=abs(best-second)<=1e-8*max(abs(best),1.)
    return best_p,best_phase,ambiguous


def estimate_fast(f,c):
    from scipy import signal
    times,dt=f["times"],float(f["dt"])
    onset=np.asarray(c["band_weights"])@f["bands"]
    if len(onset)<16 or not np.isfinite(onset).all() or onset.std()<1e-6:
        raise ValueError("Insufficient non-silent rhythmic content")
    y=onset-onset.mean()
    ac=signal.fftconvolve(y,y[::-1],mode="full")[len(y)-1:]
    ac/=np.maximum(len(y)-np.arange(len(y)),1)
    ac/=max(float(ac[0]),1e-9)
    lags=np.arange(len(ac))*dt
    peaks,_=signal.find_peaks(ac)
    peaks=peaks[(lags[peaks]>=60/c["max_bpm"])&(lags[peaks]<=60/c["min_bpm"])]
    if not len(peaks):
        raise ValueError("No periodic onset candidate")
    def ac_score(periods):
        return sum(w*np.interp(k*periods,lags,ac) for k,w in [(1,1.),(2,.5),(4,.25)])
    peaks=peaks[np.argsort(ac_score(lags[peaks]))[::-1][:c["candidate_count"]]]
    periods=[]
    for p in peaks:
        offset=0.
        if 0<p<len(ac)-1:
            denominator=ac[p-1]-2*ac[p]+ac[p+1]
            if abs(denominator)>1e-12:
                offset=float(np.clip(.5*(ac[p-1]-ac[p+1])/denominator,-.5,.5))
        base=(p+offset)*dt
        for multiple in (.5,1.,2.):
            value=base*multiple
            if 60/c["max_bpm"]<=value<=60/c["min_bpm"]:
                if not any(abs(value-old)/value<.008 for old in periods):
                    periods.append(value)
    periods=np.asarray(periods)
    periods=(periods[:,None]*np.linspace(.98,1.02,c["period_steps"])[None,:]).ravel()
    phases=np.arange(c["phase_steps"])/c["phase_steps"]
    pi,si,ambiguous=_score_grid(times,onset,periods,phases,float(f["duration"]),c["density_exponent"],c["tempo_prior_strength"])
    if ambiguous:
        return reference_estimate(f,c)
    period,phase=float(periods[pi]),float(phases[si]*periods[pi])
    initial_period=period
    peak_ids,_=signal.find_peaks(onset,distance=max(1,int(.08/dt)),prominence=.2)
    peak_times=times[peak_ids].copy()
    for i,p in enumerate(peak_ids):
        if 0<p<len(onset)-1:
            denominator=onset[p-1]-2*onset[p]+onset[p+1]
            if abs(denominator)>1e-12:
                peak_times[i]+=dt*float(np.clip(.5*(onset[p-1]-onset[p+1])/denominator,-.5,.5))
    for _ in range(3):
        ids=np.arange(math.ceil(f["duration"]/period)+1)
        expected=ids*period+phase
        mask=(expected>=.25)&(expected<=f["duration"]-.25)
        ids,expected=ids[mask],expected[mask]
        if not len(peak_times) or not len(expected):break
        delta=np.abs(expected[:,None]-peak_times[None,:])
        nearby=delta<min(.085,.22*period)
        strength=onset[peak_ids][None,:]-5*delta
        strength[~nearby]=-np.inf
        retained=nearby.any(axis=1)
        if retained.sum()<3:break
        observed=peak_times[np.argmax(strength[retained],axis=1)]
        x=ids[retained].astype(float)
        design=np.column_stack([x,np.ones(len(x))])
        parameters=np.linalg.lstsq(design,observed,rcond=None)[0]
        residual=observed-design@parameters
        keep=abs(residual)<=max(.025,2.5*float(np.median(abs(residual))))
        if keep.sum()>=3:parameters=np.linalg.lstsq(design[keep],observed[keep],rcond=None)[0]
        if abs(parameters[0]/initial_period-1)>.04:break
        period,phase=map(float,parameters)
    phase%=period
    return float(60/period),np.arange(phase,f["duration"],period).tolist()


class ClassicalV2FastEstimator(ClassicalV2Estimator):
    name="OUR_V2_FAST"

    def load(self):
        super().load()
        self.metadata.update({"checkpoint":"compiled implementation of frozen classical V2",
            "version":"v2-behavior-preserving-compiled","source_url":"adapters/classical_v2_fast.py",
            "reference_algorithm_sha256":self.metadata["sha256"],
            "sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "optimization":"Numba compiled period/phase scoring; near-ties use frozen NumPy scoring",
            "equivalence_requirement":"Exact BPM, beats and status against frozen V2 on every held-out clip"})

    def predict(self,audio_path):
        import soundfile as sf
        audio,sr=sf.read(audio_path,dtype="float32",always_2d=True)
        bpm,beats=estimate_fast(features(audio,sr,self.config),self.config)
        return result(self.name,bpm,beats)
