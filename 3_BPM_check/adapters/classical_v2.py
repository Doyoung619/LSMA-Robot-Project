"""Development version of the classical detector; original remains untouched.

No reference tempo, filename, genre, model output, or annotation is read here.
The original whitened multiband flux is retained, with coarser frames, vectorized
period/phase search, density-aware octave selection, and short-clip support.
"""
from pathlib import Path
import hashlib
import math
import json
from .base import BPMEstimator, result

DEFAULT_CONFIG = {"nfft": 512, "hop": 256, "band_weights": [.3, .5, .2],
                  "density_exponent": .5, "tempo_prior_strength": 0.,
                  "min_bpm": 40., "max_bpm": 240., "candidate_count": 6,
                  "period_steps": 7, "phase_steps": 24}


def features(audio, sr, config=None):
    import numpy as np
    import soxr
    from scipy import signal
    c = {**DEFAULT_CONFIG, **(config or {})}
    audio = np.asarray(audio, dtype=np.float32)
    mono = (.5*(audio[:,0]+audio[:,1]) if audio.ndim == 2 and audio.shape[1] == 2
            else audio.mean(axis=1) if audio.ndim == 2 else audio)
    if sr != 11025:
        mono = soxr.resample(mono, sr, 11025, quality="HQ")
    f, t, z = signal.stft(mono, 11025, nperseg=c["nfft"],
                          noverlap=c["nfft"]-c["hop"], boundary="zeros")
    magnitude = np.abs(z)
    whitened = np.log1p(magnitude/(magnitude.mean(axis=1, keepdims=True)+1e-6))
    flux = np.maximum(np.diff(whitened, axis=1, prepend=whitened[:, :1]), 0)
    envelopes = []
    width = int(round(.5*11025/c["hop"])) | 1
    for lo, hi in [(35,180), (180,2000), (2000,5000)]:
        x = flux[(f >= lo) & (f < hi)].mean(axis=0)
        x = np.maximum(x-signal.medfilt(x, width), 0)
        envelopes.append(x/(x.std()+1e-8))
    return {"times": t, "bands": np.asarray(envelopes),
            "duration": len(audio)/sr, "dt": c["hop"]/11025}


def estimate(f, config=None):
    import numpy as np
    from scipy import signal
    c = {**DEFAULT_CONFIG, **(config or {})}
    times, dt = f["times"], float(f["dt"])
    onset = np.asarray(c["band_weights"]) @ f["bands"]
    if len(onset) < 16 or not np.isfinite(onset).all() or onset.std() < 1e-6:
        raise ValueError("Insufficient non-silent rhythmic content")
    y = onset-onset.mean()
    ac = signal.fftconvolve(y, y[::-1], mode="full")[len(y)-1:]
    ac /= np.maximum(len(y)-np.arange(len(y)), 1)
    ac /= max(float(ac[0]), 1e-9)
    lags = np.arange(len(ac))*dt
    peaks, _ = signal.find_peaks(ac)
    peaks = peaks[(lags[peaks] >= 60/c["max_bpm"]) & (lags[peaks] <= 60/c["min_bpm"])]
    if not len(peaks):
        raise ValueError("No periodic onset candidate")
    def ac_score(periods):
        return sum(w*np.interp(k*periods, lags, ac) for k,w in [(1,1.),(2,.5),(4,.25)])
    peaks = peaks[np.argsort(ac_score(lags[peaks]))[::-1][:c["candidate_count"]]]
    periods = []
    for p in peaks:
        offset = 0.
        if 0 < p < len(ac)-1:
            denominator = ac[p-1]-2*ac[p]+ac[p+1]
            if abs(denominator) > 1e-12:
                offset = float(np.clip(.5*(ac[p-1]-ac[p+1])/denominator,-.5,.5))
        base = (p+offset)*dt
        for multiple in (.5,1.,2.):
            value = base*multiple
            if 60/c["max_bpm"] <= value <= 60/c["min_bpm"]:
                if not any(abs(value-old)/value < .008 for old in periods):
                    periods.append(value)
    periods = np.asarray(periods)
    periods = (periods[:,None]*np.linspace(.98,1.02,c["period_steps"])[None,:]).ravel()
    phases = np.arange(c["phase_steps"])/c["phase_steps"]
    beat_ids = np.arange(math.ceil(f["duration"]/periods.min())+1)
    positions = periods[:,None,None]*(phases[None,:,None]+beat_ids[None,None,:])
    valid = (positions >= .25) & (positions <= f["duration"]-.25)
    values = np.interp(positions.ravel(), times, onset).reshape(positions.shape)
    counts = valid.sum(axis=2)
    means = (values*valid).sum(axis=2)/np.maximum(counts,1)
    scores = means/periods[:,None]**c["density_exponent"]
    if c["tempo_prior_strength"]:
        prior = np.exp(-.5*(np.log2((60/periods)/120)/.8)**2)
        scores *= prior[:,None]**c["tempo_prior_strength"]
    best = np.unravel_index(np.argmax(scores), scores.shape)
    period, phase = float(periods[best[0]]), float(phases[best[1]]*periods[best[0]])
    initial_period = period
    peak_ids, _ = signal.find_peaks(onset, distance=max(1,int(.08/dt)), prominence=.2)
    peak_times = times[peak_ids].copy()
    for i,p in enumerate(peak_ids):
        if 0 < p < len(onset)-1:
            denominator = onset[p-1]-2*onset[p]+onset[p+1]
            if abs(denominator)>1e-12:
                peak_times[i] += dt*float(np.clip(.5*(onset[p-1]-onset[p+1])/denominator,-.5,.5))
    for _ in range(3):
        ids = np.arange(math.ceil(f["duration"]/period)+1)
        expected = ids*period+phase
        mask = (expected >= .25) & (expected <= f["duration"]-.25)
        ids, expected = ids[mask], expected[mask]
        if not len(peak_times) or not len(expected):
            break
        delta = np.abs(expected[:,None]-peak_times[None,:])
        nearby = delta < min(.085,.22*period)
        strength = onset[peak_ids][None,:]-5*delta
        strength[~nearby] = -np.inf
        retained = nearby.any(axis=1)
        if retained.sum() < 3:
            break
        observed = peak_times[np.argmax(strength[retained],axis=1)]
        x = ids[retained].astype(float)
        design = np.column_stack([x,np.ones(len(x))])
        parameters = np.linalg.lstsq(design,observed,rcond=None)[0]
        residual = observed-design@parameters
        keep = abs(residual) <= max(.025,2.5*float(np.median(abs(residual))))
        if keep.sum() >= 3:
            parameters = np.linalg.lstsq(design[keep],observed[keep],rcond=None)[0]
        if abs(parameters[0]/initial_period-1) > .04:
            break
        period, phase = map(float,parameters)
    phase %= period
    beats = np.arange(phase,f["duration"],period)
    return float(60/period), beats.tolist()


class ClassicalV2Estimator(BPMEstimator):
    name = "OUR_V2"

    def __init__(self, device="cpu", threads=1, config=None):
        super().__init__(device,threads)
        frozen = Path(__file__).resolve().parents[1]/"improvements/frozen_config.json"
        if config is None and frozen.exists():
            config = json.loads(frozen.read_text(encoding="utf-8"))
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def load(self):
        if self.device != "cpu":
            raise ValueError("Classical V2 is a CPU implementation")
        import numpy, scipy, soundfile, soxr
        self.metadata = {"pretrained_loaded":None,"weight_bytes":0,"checkpoint":"new classical development version",
            "source_url":"adapters/classical_v2.py","license":"project source","version":"development-v2",
            "sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"configuration":self.config,
            "family":"classical","output_type":"global BPM and fitted beat grid",
            "development_status":"Configuration selected on development tracks; independent holdout evaluated separately",
            "timing_scope":"WAV read, mono/resampling, onset features, period/phase search, fit and result",
            "runtime_versions":{"numpy":numpy.__version__,"scipy":scipy.__version__,"soundfile":soundfile.__version__,"soxr":soxr.__version__}}

    def predict(self, audio_path):
        import soundfile as sf
        audio,sr = sf.read(audio_path,dtype="float32",always_2d=True)
        bpm,beats = estimate(features(audio,sr,self.config),self.config)
        return result(self.name,bpm,beats)
