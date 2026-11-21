# Classical detector improvement and independent evaluation

`OUR_V2` is a separately developed detector. The original `OUR_EXISTING` adapter,
robot code, and saved `results/` comparison remain unchanged.

The goal is to improve accuracy and complete file-to-result latency under the
same conditions as pretrained neural methods. Joint superiority is assessed
against raw Acc1 and matched median latency; Acc2 is a separately labeled metric.

## Changes

- Retain the original whitened three-band positive spectral-flux representation.
- Use a 512-sample FFT, 256-sample hop, and SOXR resampling.
- Replace nested Python period/phase loops with a NumPy vectorized search.
- Search 40–240 BPM, add density-aware scoring, and fit short clips from three
  supported attacks instead of imposing the original fixed 16-attack requirement.
- Select scoring parameters only on development tracks and freeze them before
  evaluating new tracks. No annotations, genre, filename BPM, or other model's
  prediction is used by the inference code.

## Development and holdout

The original 50 evaluated GTZAN tracks are development data, not independent
evidence for the new version. All six initial configurations and the subsequent
tempo-prior comparison are retained under `results/`; the final search includes
22 configurations. `frozen_protocol.json` records the source/configuration hashes
before the remaining 50 official GTZAN-mini tracks are obtained. Both groups have
five tracks per genre, with zero track overlap.

```powershell
# Reproduce development selection (no held-out labels are used here).
.\.venv\Scripts\python.exe improvements\develop_v2.py

# Freeze selection and prepare the 50-track independent holdout.
.\.venv\Scripts\python.exe improvements\prepare_holdout.py

# All five methods, same new clips, one CPU thread, 5 warm-ups and
# 20 measured file-to-result repetitions; secondary preloaded stage: 5 repeats.
.\.venv\Scripts\python.exe run_benchmark.py --full --device cpu --manifest improvements\holdout_manifest.json --methods OUR_EXISTING OUR_V2 DeepRhythm TempoCNN BeatThis --preloaded-repetitions 5 --skip-download --results improvements\holdout_results

.\.venv\Scripts\python.exe validate_results.py --results improvements\holdout_results
.\.venv\Scripts\python.exe improvements\assess_holdout.py
```

An additional implementation-only experiment compiles period/phase scoring with
Numba. It keeps the frozen parameters, features and refinement; near-tied scores
use the original scoring. Before timing, `verify_compiled.py` must demonstrate
exact BPM, full beat-array, status and confidence equality on every held-out
clip. Both implementations and all earlier measurements are retained. This is
an implementation optimization, not another round of test-set accuracy tuning.

```powershell
.\.venv\Scripts\python.exe improvements\verify_compiled.py
.\.venv\Scripts\python.exe improvements\time_compiled.py
.\.venv\Scripts\python.exe improvements\measure_compiled_startup.py
.\.venv\Scripts\python.exe improvements\assess_compiled.py
.\.venv\Scripts\python.exe improvements\plot_improvement.py
```

The additional timing uses randomized method order within each repetition to
balance time-varying background load. It reports all methods on the same clips,
with five warm-ups and twenty file-to-result repetitions. Shared-process model
initialization is not compared with the separate-process cold measurements.

The final comparison is in [`compiled_comparison_report.md`](compiled_comparison_report.md)
with a [presentation figure](figures/improved_comparison.png). The fast variant's
empty-compiler-cache startup measurement is saved separately in
`compiled_startup.json`; warm latency does not represent first-use compilation.

Development feature caching is used only by `develop_v2.py`. The benchmark adapter
always reads audio and recomputes features. Downloaded audio and checkpoints stay
outside version control. Per-duration results, coverage, all model predictions,
and unsuccessful configurations are retained.

The neural models have their original pretrained checkpoints and official
preprocessing; TempoCNN retains the explicitly documented Windows TensorFlow
frontend fallback. GTZAN overlap with all neural training corpora cannot be
excluded, so this is an exploratory cross-model comparison. It is an independent
holdout with respect to the classical V2 development procedure.
