# Benchmark environment

The benchmark uses an isolated Python 3.11 virtual environment. The existing
robot Python environments are not modified.

On Windows, run `powershell -ExecutionPolicy Bypass -File setup_benchmark.ps1`
from this directory. Alternatively:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python download_models.py
.venv\Scripts\python run_benchmark.py --quick --device cpu
```

`requirements.txt` pins direct dependencies and CPU PyTorch wheels.
`requirements-lock.txt` records every installed package for the executed Windows
environment. Checkpoints are cached in `models/` and checked against SHA256
identities pinned in `adapters/model_cache.py`. Both downloader and adapter loads
reject a changed checkpoint. No checkpoint download occurs inside model loading
or timed inference.

The existing detector imports no neural packages. Its adapter calls the original
source function and leaves its algorithm and output files unchanged.

## TempoCNN runtime choice

Native Windows does not provide the Essentia TensorFlow wheels required by
`essentia.standard.TempoCNN`. The benchmark uses the official frozen
`deeptemp-k4-3.pb` from Essentia through a TensorFlow v1 Session, with the original
author's `tempocnn.feature.read_features` and `std_normalizer` frontend. The
global output follows the original author's argmax of mean patch probabilities.
This differs from Essentia's default majority vote over local estimates; timing
belongs to the measured frontend/runtime combination and must not be called
Essentia latency.

The author frontend uses 11025 Hz mono audio, 1024-point STFT, hop512, 40 mel
bands from 20–5000 Hz, and 256-frame patches with a 128-frame hop. It zero-pads clips shorter
than approximately 11.9 seconds, including this benchmark's 8 second condition.
The normalization and padding are those supplied by the original package.

DeepRhythm's official frontend consumes complete 8 second chunks and discards
trailing partial chunks: the 8/15/30 second files provide 8/8/24 seconds of effective
model input. Its file loader still reads the complete excerpt, which is included
in timing. No custom duration tuning is applied.

## Official sources

- [DeepRhythm](https://github.com/bleugreen/deeprhythm): model0.7; AGPL-3.0.
- [Essentia models](https://essentia.upf.edu/models.html): deeptemp-k4-3;
  CC BY-NC-SA 4.0. Original [TempoCNN frontend](https://github.com/hendriks73/tempo-cnn)
  is AGPL-3.0.
- [Beat This!](https://github.com/CPJKU/beat_this): small0, DBN disabled;
  code and weights MIT; GTZAN excluded from training per official model table.
- [BeatNet](https://github.com/mjhydri/BeatNet): optional checkpoint 1 is identified
  as GTZAN-out in the official source. The official online+PF configuration reads
  a complete audio file faster than wall-clock playback; offline mode requires
  DBN. Its old package dependency constraints are isolated from mandatory models.

The executed optional BeatNet 1.1.3 installation resolver fails because its
`numba==0.54.1` dependency requires Python below 3.10. It is marked unavailable,
and mandatory packages are left intact. The exact resolver output is in
`results/beatnet_install_attempt.txt`. The native Essentia attempt reports
`No matching distribution found for essentia-tensorflow`; its output is in
`results/essentia_install_attempt.txt`. `pip check` reports no broken requirements
in the executed mandatory environment.
