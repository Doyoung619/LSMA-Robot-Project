"""Small, model-free onset/tempo analysis and an audio-device playback clock."""
from pathlib import Path
import json
import subprocess
import threading
import numpy as np
from scipy import signal, optimize
import soundfile as sf


def decode(source, target):
    source, target = Path(source), Path(target)
    metadata = target.with_suffix(".source.json")
    identity = {"source": str(source.resolve()), "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns}
    previous = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None
    if not target.exists() or previous != identity:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                        "-ar", "44100", "-ac", "2", str(target)], check=True)
        metadata.write_text(json.dumps(identity),encoding="utf-8")
    return target


def analyze(wav, output):
    audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    mono = signal.resample_poly(audio.mean(axis=1), 1, 4)
    rate, hop, nfft = sr / 4, 128, 1024
    frequencies, times, stft = signal.stft(mono, rate, nperseg=nfft,
                                         noverlap=nfft-hop, boundary="zeros")
    # Whiten each frequency bin so the bass line cannot overwhelm drum attacks.
    magnitude = np.abs(stft)
    whitened = np.log1p(magnitude / (np.mean(magnitude, axis=1, keepdims=True)+1e-6))
    flux = np.maximum(np.diff(whitened, axis=1, prepend=whitened[:, :1]), 0)
    bands = [(frequencies >= lo) & (frequencies < hi) for lo, hi in
             [(35, 180), (180, 2000), (2000, 5000)]]
    envelopes = []
    for band in bands:
        x = flux[band].mean(axis=0)
        x = np.maximum(x - signal.medfilt(x, 43), 0)
        envelopes.append(x / (np.std(x)+1e-8))
    onset = .55*envelopes[0] + .3*envelopes[1] + .15*envelopes[2]
    dt = hop/rate
    ac = signal.fftconvolve(onset-onset.mean(), (onset-onset.mean())[::-1], mode="full")[len(onset)-1:]
    lags = np.arange(len(ac))*dt
    candidates, _ = signal.find_peaks(ac)
    candidates = [i for i in candidates if 60/180 <= lags[i] <= 60/70]
    # Aggregate repeated lags; this is explicit octave selection, not a known-song BPM.
    def lag_score(period):
        return sum(np.interp(k*period, lags, ac)*w for k, w in [(1,1), (2,.6), (4,.3)])
    candidates = sorted(candidates, key=lambda i: lag_score(lags[i]), reverse=True)[:8]
    if not candidates:
        raise RuntimeError("No stable tempo found. Supply a manual BPM after checking the music.")
    # Fit a global beat grid using 48 seconds, then refine against the entire song.
    grid_t = times[(times >= 1) & (times <= min(49, times[-1]-1))]
    grid_y = onset[(times >= 1) & (times <= min(49, times[-1]-1))]
    top = []
    for i in candidates:
        p0 = lags[i]
        best = (-np.inf, p0, 0)
        for period in np.linspace(p0*.985, p0*1.015, 121):
            for phase in np.linspace(0, period, 48, endpoint=False):
                beats = np.arange(phase, grid_t[-1], period)
                beats = beats[beats >= grid_t[0]]
                score = np.mean(np.interp(beats, grid_t, grid_y))
                if score > best[0]: best = (score, period, phase)
        top.append(best)
    best = max(top, key=lambda x: x[0])
    _, period, phase = best
    # Locate nearest attacks, regress their index to time, reject fills/outliers.
    peaks, _ = signal.find_peaks(onset, distance=int(.15/dt), prominence=.35)
    peak_t = times[peaks]
    for _ in range(4):
        expected = np.arange(phase, times[-1]-.5, period)
        expected = expected[expected > .5]
        ids = np.rint((expected-phase)/period).astype(int)
        observed, retained = [], []
        for k, beat in zip(ids, expected):
            near = np.flatnonzero(np.abs(peak_t-beat) < .085)
            if len(near):
                j = near[np.argmax(onset[peaks[near]] - 5*np.abs(peak_t[near]-beat))]
                observed.append(peak_t[j]); retained.append(k)
        if len(observed) < 16: raise RuntimeError("Insufficient beat-grid support")
        fit = optimize.least_squares(lambda x: x[0]*np.array(retained)+x[1]-observed,
                                     [period, phase], loss="soft_l1", f_scale=.015)
        period, phase = fit.x
    phase = float(phase % period)
    beats = np.arange(phase, len(audio)/sr, period)
    errors = np.array([np.min(np.abs(peak_t-b)) for b in beats[4:-4]])
    energy = np.sqrt(np.mean(np.abs(stft)**2, axis=0))
    energy /= np.percentile(energy, 85)+1e-8
    result = {"method": "multiband spectral flux + autocorrelation + robust beat-grid fit",
              "bpm": float(60/period), "period_seconds": float(period),
              "first_beat_seconds": phase, "duration_seconds": len(audio)/sr,
              "median_nearest_onset_error_ms": float(np.median(errors)*1000),
              "onset_support_within_50ms": float(np.mean(errors < .05)),
              "beat_count": len(beats),
              "tempo_candidates_bpm": [float(60/x[1]) for x in top]}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(Path(output).with_suffix(".npz"), times=times, onset=onset,
             energy=energy, beats=beats)
    return result


class AudioPlayer:
    """Clock measures elapsed audio at the DAC, not how long Python has run."""
    def __init__(self, wav, start=0, duration=None, device=None, volume=.65):
        import sounddevice as sd
        audio, self.sr = sf.read(wav, dtype="float32", always_2d=True)
        first = round(start*self.sr)
        last = len(audio) if duration is None else min(len(audio), first+round(duration*self.sr))
        self.audio = np.ascontiguousarray(audio[first:last]*volume)
        self.position = 0
        self.dac_start = None
        self.error = None
        self.finished = threading.Event()
        self.closed = False
        self.stream = sd.OutputStream(samplerate=self.sr, channels=self.audio.shape[1],
                                      dtype="float32", device=device, blocksize=512,
                                      latency="low", callback=self._callback,
                                      finished_callback=self.finished.set)

    def _callback(self, outdata, frames, timing, status):
        import sounddevice as sd
        if status:
            self.error = str(status)
            outdata.fill(0)
            raise sd.CallbackAbort
        if self.dac_start is None: self.dac_start = timing.outputBufferDacTime
        count = min(frames, len(self.audio)-self.position)
        outdata.fill(0)
        outdata[:count] = self.audio[self.position:self.position+count]
        self.position += count
        if count < frames: raise sd.CallbackStop

    def start(self): self.stream.start()

    def clock(self):
        if self.error: raise RuntimeError("Audio playback error: "+self.error)
        if self.dac_start is None: return 0.
        return max(0., self.stream.time-self.dac_start)

    def close(self):
        if not self.closed:
            self.closed = True
            self.stream.abort()
            self.stream.close()
