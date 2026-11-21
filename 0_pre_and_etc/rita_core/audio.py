"""Offline non-neural analysis, explicit beat timestamps, DAC/virtual clocks."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import threading
import numpy as np
from scipy import signal
import soundfile as sf
from .contracts import AudioAnalysis

BASE = Path(__file__).resolve().parents[1]


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def find_audio(path=None):
    if path is not None:
        p=Path(path).expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(f"Audio file not found: {p}")
        return p
    for p in (BASE/"Billie Jean.mp3", BASE/"data/fixture.wav"):
        if p.is_file(): return p
    raise FileNotFoundError("No local music found. Supply --audio /path/to/your/music.mp3 or create a fixture.")


def decode_audio(source, cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    target=cache/(file_hash(source)[:20]+".wav")
    if not target.exists():
        subprocess.run(["ffmpeg","-nostdin","-v","error","-y","-i",str(source),"-ar","44100","-ac","2",str(target)],check=True)
    return target


def features(wav):
    a,sr=sf.read(wav,dtype="float32",always_2d=True)
    if len(a)/sr<4: raise ValueError("At least four seconds of audio is required; short audio is not a stable tempo observation")
    if np.max(np.abs(a))<1e-6: raise ValueError("Silent audio: no tempo/beat support")
    mono=signal.resample_poly(a.mean(axis=1),1,4);rate=sr/4
    f,t,z=signal.stft(mono,rate,nperseg=1024,noverlap=896,boundary="zeros")
    mag=np.abs(z);white=np.log1p(mag/(mag.mean(axis=1,keepdims=True)+1e-6))
    flux=np.maximum(np.diff(white,axis=1,prepend=white[:,:1]),0)
    env=[]
    for lo,hi in [(35,180),(180,2000),(2000,5000)]:
        x=flux[(f>=lo)&(f<hi)].mean(axis=0);x=np.maximum(x-signal.medfilt(x,43),0)
        env.append(x/(x.std()+1e-8))
    onset=.55*env[0]+.30*env[1]+.15*env[2]
    energy=np.sqrt(np.mean(mag**2,axis=0));energy/=np.percentile(energy,85)+1e-8
    ac=signal.fftconvolve(onset-onset.mean(),(onset-onset.mean())[::-1],mode="full")[len(onset)-1:]
    return a,sr,mono,rate,t,onset,energy,ac


def sparse_transients(mono,rate):
    """Track genuinely sparse attacks, including changing-tempo click fixtures.

    This signal-based branch never reads fixture labels or the source filename.
    Dense music falls through to the existing robust spectral-flux grid fitter.
    """
    amplitude=np.abs(mono)
    threshold=float(np.max(amplitude)*.12)
    active=amplitude>threshold
    if np.mean(active)>.13: return None
    starts=np.flatnonzero(active & ~np.r_[False,active[:-1]])
    attacks=[]
    for idx in starts:
        value=idx/rate
        if not attacks or value-attacks[-1]>.15: attacks.append(value)
    attacks=np.array(attacks)
    if len(attacks)<8: return None
    intervals=np.diff(attacks)
    if np.mean((intervals>=.22)&(intervals<=1.4))<.96: return None
    return attacks


def analyze_audio(source=None, output=None, bpm_override=None, beat_offset=None):
    source=find_audio(source)
    out=Path(output or BASE/"outputs/analysis")
    out.mkdir(parents=True,exist_ok=True)
    digest=file_hash(source)
    key=hashlib.sha256(f"{digest}:{bpm_override}:{beat_offset}:v2.2".encode()).hexdigest()[:20]
    cached=out/(key+".json")
    if cached.exists(): return AudioAnalysis(**json.loads(cached.read_text(encoding="utf-8")))
    wav=decode_audio(source,out/"cache")
    audio,sr,mono,rate,t,onset,energy,ac=features(wav)
    duration=len(audio)/sr
    if beat_offset is not None and not 0<=beat_offset<duration:
        raise ValueError("--beat-offset must be a nonnegative absolute time inside the audio")
    beats=sparse_transients(mono,rate)
    diagnostics={}
    if bpm_override is not None:
        if not 30<=bpm_override<=300: raise ValueError("--bpm-override must be in 30..300")
        period=60/bpm_override
        if beat_offset is not None:phase=float(beat_offset)
        elif beats is not None:phase=float(beats[0])
        else:
            phases=np.linspace(0,period,96,endpoint=False)
            scores=[np.mean(np.interp(np.arange(p,duration,period),t,onset)) for p in phases]
            phase=float(phases[int(np.argmax(scores))])
        beats=np.arange(phase,duration,period)
        method="manual BPM grid with spectral onset phase estimate/correction"
        diagnostics={"manual_override":True,"automatic_beat_detection_accuracy":None}
    elif beats is not None:
        method="sparse transient onset tracking (no fixture labels)"
        diagnostics["tempo_changes_detected"]=bool(np.std(np.diff(beats))/np.mean(np.diff(beats))>.05)
    else:
        # Reuse the previously verified fitter and its beat offset/robust regression.
        from rita_robot.audio import analyze
        legacy=analyze(wav,out/(key+"_legacy.json"))
        beats=np.load(out/(key+"_legacy.npz"))["beats"]
        method=legacy["method"]
        diagnostics={"legacy_result":legacy,"tempo_changes_detected":False,
                     "global_tempo_assumption":True}
    if bpm_override is None and beat_offset is not None:
        beats=beats+(beat_offset-beats[0]);beats=beats[(beats>=0)&(beats<duration)]
    if len(beats)<4: raise ValueError("Insufficient supported beats after manual corrections")
    accent=np.interp(beats,t,onset)
    accent=np.clip(accent/(np.percentile(accent,90)+1e-8),0,1)
    settings={"method":method,"fft":1024,"hop":128,"analysis_rate":rate,
              "bands_hz":[[35,180],[180,2000],[2000,5000]],"weights":[.55,.3,.15],
              "bpm_override":bpm_override,"beat_offset":beat_offset,
              "origin":"estimated beat origin; four-beat blocks are not detected downbeats",
              "diagnostics":diagnostics,"decoded_wav":str(wav),
              "scope":"offline known audio analysis; online motion generation is separate",
              "ambiguous_octaves":[float(60/np.median(np.diff(beats))*x) for x in (.5,1,2)]}
    result=AudioAnalysis(str(source),digest,int(sr),duration,beats,t,onset,energy,accent,settings)
    cached.write_text(json.dumps(result.to_dict(),indent=2)+"\n",encoding="utf-8")
    (out/"latest.json").write_text(json.dumps({"analysis_file":str(cached),"audio_hash":digest,"bpm":result.bpm},indent=2),encoding="utf-8")
    diagnostic_plot(result,ac,out/"tempo_diagnostic.png")
    return result


def diagnostic_plot(analysis,ac,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6))
    lag=np.arange(len(ac))*128/analysis.settings["analysis_rate"]
    mask=(lag>=.18)&(lag<=2)
    axes[0].plot(60/lag[mask],ac[mask]/(ac[0]+1e-9))
    for ratio in (.5,1,2): axes[0].axvline(analysis.bpm*ratio,label=f"{ratio:g}x: {analysis.bpm*ratio:.2f} BPM",alpha=.6)
    axes[0].set(xlabel="Candidate BPM (octave alternatives are not confidence)",ylabel="Onset autocorrelation",xlim=(30,300));axes[0].legend()
    mask=analysis.onset_times<12
    axes[1].plot(analysis.onset_times[mask],analysis.onset_strength[mask])
    for b in analysis.beats[analysis.beats<12]: axes[1].axvline(b,color="orange",alpha=.6)
    axes[1].set(xlabel="Music seconds; estimated beat origin (no downbeat detection)",ylabel="Onset strength")
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def fixture(path,bpm=120,duration=660,phase=.17,tempo_change=None,silence=False,seed=7):
    if not 30<=bpm<=300 or (tempo_change is not None and not 30<=tempo_change<=300):
        raise ValueError("Fixture BPM must be in 30..300")
    if not np.isfinite(duration) or duration<=0 or not 0<=phase<duration:
        raise ValueError("Fixture needs positive duration and phase inside the audio")
    sr=44100;rng=np.random.default_rng(seed)
    audio=np.zeros(round(duration*sr),dtype=np.float32);beats=[]
    t=phase
    pulse=rng.normal(size=round(.025*sr))*np.exp(-np.arange(round(.025*sr))/(sr*.003))*.2
    while t<duration-.04:
        beats.append(t);idx=round(t*sr)
        if not silence: audio[idx:idx+len(pulse)]+=pulse
        current=bpm if tempo_change is None or t<duration/2 else tempo_change
        t+=60/current
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,audio,sr)
    labels={"beats":beats,"bpm":bpm,"phase":phase,"tempo_change":tempo_change,"duration":duration,"silence":silence,"seed":seed}
    path.with_suffix(".truth.json").write_text(json.dumps(labels,indent=2),encoding="utf-8")
    return path


def synthetic_analysis(bpm=120,duration=660,phase=.17,tempo_change=None):
    if not 30<=bpm<=300 or duration<=phase or phase<0:raise ValueError("Invalid synthetic tempo/duration/phase")
    beats=[];t=phase
    while t<duration:
        beats.append(t);t+=60/(bpm if tempo_change is None or t<duration/2 else tempo_change)
    ts=np.arange(0,duration,.025);ons=np.zeros(len(ts))
    for b in beats: ons+=np.exp(-((ts-b)/.025)**2)
    return AudioAnalysis("synthetic labeled grid",hashlib.sha256(f"{bpm}:{duration}:{phase}:{tempo_change}".encode()).hexdigest(),44100,duration,
                         np.array(beats),ts,ons,np.ones(len(ts)),np.tile([1,.5,.8,.5],len(beats)//4+1)[:len(beats)],
                         {"method":"labeled synthetic planner fixture, not an audio detector result","tempo_change":tempo_change})


class VirtualClock:
    """Explicit accelerated clock; never reported as an acoustic measurement."""
    def __init__(self,start=0): self.t=start;self.epoch=time.monotonic()
    def clock(self): return self.t
    def advance(self,t):
        if t<self.t-1e-9: raise ValueError("virtual clock cannot go backward")
        self.t=float(t)
    def music_to_monotonic(self,t): return self.epoch+t
    def mapping(self): return {"kind":"virtual","epoch":self.epoch,"acoustic_measurement":False}


class DACClock:
    """PortAudio outputBufferDacTime clock, with explicit stream→monotonic map."""
    def __init__(self,wav,start=0,duration=None,device=None,volume=.65):
        import sounddevice as sd
        a,self.sr=sf.read(wav,dtype="float32",always_2d=True)
        self.start_offset=start;last=len(a) if duration is None else round((start+duration)*self.sr)
        self.audio=np.ascontiguousarray(a[round(start*self.sr):last]*volume)
        self.position=0;self.dac_epoch=None;self.error=None;self.finished=threading.Event()
        self.stream=sd.OutputStream(samplerate=self.sr,channels=self.audio.shape[1],dtype="float32",device=device,
                                    blocksize=512,latency="low",callback=self._callback,finished_callback=self.finished.set)
    def _callback(self,outdata,frames,timing,status):
        import sounddevice as sd
        if status:
            self.error=str(status);outdata.fill(0);raise sd.CallbackAbort
        if self.dac_epoch is None: self.dac_epoch=float(timing.outputBufferDacTime)
        count=min(frames,len(self.audio)-self.position);outdata.fill(0)
        outdata[:count]=self.audio[self.position:self.position+count];self.position+=count
        if count<frames: raise sd.CallbackStop
    def start(self): self.stream.start()
    def clock(self):
        if self.error: raise RuntimeError("audio callback failed: "+self.error)
        if self.dac_epoch is None: return self.start_offset
        return self.start_offset+max(0,float(self.stream.time)-self.dac_epoch)
    def mapping(self):
        before=time.monotonic();st=float(self.stream.time);after=time.monotonic()
        return {"kind":"PortAudio DAC","monotonic_minus_stream":(before+after)/2-st,
                "sampling_uncertainty_seconds":(after-before)/2,"dac_epoch_stream":self.dac_epoch,
                "output_latency_seconds":float(self.stream.latency),"sample_position_not_used_as_clock":True}
    def music_to_monotonic(self,t):
        if self.dac_epoch is None: raise RuntimeError("DAC epoch has not been observed")
        m=self.mapping();return m["monotonic_minus_stream"]+self.dac_epoch+t-self.start_offset
    def close(self): self.stream.abort();self.stream.close()
