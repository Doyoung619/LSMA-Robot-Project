# Classical BPM detection vs. lightweight pretrained neural models

This report was generated from the saved CSV measurements. No BPM prediction is used as a reference annotation.

## Existing method

The real robot entry point is `../0_pre_and_etc/rita.py:analysis`, which calls `rita_robot.audio.decode` and `rita_robot.audio.analyze`. The benchmark imports and calls that unchanged `analyze(wav, output)` implementation through a thin adapter. It does not substitute a librosa beat tracker.

The standalone benchmark also contains a byte-identical source snapshot at `reference/rita_robot_audio.py` for checkouts without the rest of the robot project. The original project file takes precedence when present. Its SHA-256 is `c36c0ae462cc0a1c7793deb667ac0a27f6479e4cd75e78390ba4145a905ecaed`.

The normal pipeline decodes music with FFmpeg to 44.1 kHz stereo WAV, averages channels, and downsamples by four. A 1024-sample STFT with a 128-sample hop feeds log-whitened positive spectral flux. The onset bands are 35–180, 180–2000, and 2000–5000 Hz, weighted 0.55/0.30/0.15; 43-frame median filtering removes the local baseline. Autocorrelation scores 1×, 2×, and 4× lags with weights 1/0.6/0.3, keeps up to eight candidates initially between 70 and 180 BPM, and searches 121 period offsets × 48 phases per candidate. A soft-L1 least-squares beat-grid fit is repeated four times using attacks within 85 ms. The fit requires at least 16 supported beats. This is a global offline tempo and regular beat-grid estimator; it may fail on short clips or choose half tempo. The final fitted BPM is not clamped to the initial search range.

The robot consumes global BPM, beat period, and first-beat phase to plan moves before playback. The playback clock synchronizes motion after analysis; this benchmark does not evaluate motion control or streaming latency. `rita.py:analysis` normally reuses cached `data/analysis.json` when the decoded WAV is unchanged, unless the explicit analyze command requests recomputation. The benchmark calls `analyze` on every prediction, measuring detector computation rather than cached BPM retrieval. The original analysis also writes JSON and NPZ diagnostics. Those writes remain inside its file-to-result timing, preserving actual behavior.

## Separately developed classical V2

`OUR_V2` is a new method in `adapters/classical_v2.py`; the original robot implementation and its saved benchmark results remain unchanged. It retains whitened multiband spectral flux, uses a faster FFT grid and vectorized period/phase search, supports at least three matched attacks on short clips, and adds density-aware tempo scoring with a development-selected tempo prior. It has no neural weights. Runtime includes audio loading, resampling, features, fitting, and BPM/beat output; optional diagnostic file writes from the original implementation are not part of V2's required API.

Configuration was selected on the previously inspected 50 GTZAN tracks, then frozen in `improvements/frozen_protocol.json` before obtaining/evaluating the remaining 50 tracks. The holdout split has no track overlap with this classical development set. Both Acc1 and Acc2 are reported; an Acc2 advantage does not establish a raw-BPM Acc1 advantage.

## Compared methods

| Method | Type | Weight size | Output | Device | Checkpoint | Status | Pretrained verified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Existing classical | Classical | 0 (no weights) | Global BPM + fitted beat grid | cpu | unchanged rita_robot.audio.analyze | success | N/A |
| DeepRhythm | NN | 5.19 MiB | Global BPM | cpu | deeprhythm-0.7.pth | success | True |
| TempoCNN k4 | NN | 0.14 MiB | Global BPM | cpu | deeptemp-k4-3.pb | success | True |
| Beat This! small0 | NN | 8.06 MiB | Beats; BPM derived from median valid IBI | cpu | beat_this-small0.ckpt | success | True |
| Classical V2 (new) | Classical | 0 (no weights) | Global BPM + fitted beat grid | cpu | new classical development version | success | N/A |

For beat trackers, the BPM adapter uses ordinary beats, excludes nonfinite/nonpositive intervals, and computes 60 divided by the median valid inter-beat interval. Insufficient beats are a failure, not an invented estimate. Original beat timestamps are preserved. BPM-only models have no fabricated beat output.

## Data and protocol

Evaluated 50 real annotated source tracks (150 duration-specific excerpts), 0 synthetic source tracks (0 excerpts), and 0 Billie Jean excerpts. The same saved 44.1 kHz stereo WAV excerpts are supplied to all methods; each model then uses its own required preprocessing. The intended durations are 8, 15, and 30 seconds. File hashes, source track IDs, annotation sources, selection seed, and exact durations are retained in the dataset manifest/configuration.

Duration-specific accuracy groups use the requested (nominal) duration, so the nominal 30-second group includes every selected source track. Predictions, timing rows, the manifest, and RTF retain actual audio duration; latency calculations use actual durations. The nominal 30-second real group includes 1 original clips slightly shorter than 30 seconds (minimum 29.9818594104 s, at most 18.141 ms short). The dataset helper does not pad or alter these source lengths.

Synthetic metronome and musical patterns cover 60, 90, 117, 120, 150, and 180 BPM with known beat times. These diagnose implementation behavior and do not establish accuracy on real music. A reference audit found 10/150 real crops where median local annotated IBI implies a BPM more than 2% from the published global tempo. The global annotation remains the primary reference; this local/global discrepancy can affect prefix-duration accuracy. Billie Jean is a qualitative demonstration: no verified manual reference is supplied, and the repository's earlier 117.025 BPM analysis is a detector output, not ground truth.

Timing uses a high-resolution wall clock and serial CPU inference. Downloading and weight caching occur before initialization/inference measurements. Package import, model initialization/weight loading, and the first inference are recorded separately. Warm file-to-result time includes decoding/loading, feature extraction, inference, postprocessing, and BPM conversion. Preloaded-audio timings, where supported, form a separate stage and are never mixed into the main comparison.

Saved benchmark configuration:

```json
{
  "mode": "full",
  "device": "cpu",
  "threads": 1,
  "seed": 2026,
  "warmups": 5,
  "repetitions": 20,
  "methods": [
    "OUR_EXISTING",
    "OUR_V2",
    "DeepRhythm",
    "TempoCNN",
    "BeatThis"
  ],
  "timing_clip_ids": [
    "blues.00004_8s",
    "blues.00004_15s",
    "blues.00004_30s",
    "classical.00004_8s",
    "classical.00004_15s",
    "classical.00004_30s"
  ],
  "results_dir": "C:\\Users\\허도영\\Desktop\\Git-Research\\CMU Class\\Rita-Robot\\experiment\\3_BPM_check\\improvements\\holdout_results",
  "dataset": "all",
  "durations_seconds": [
    8,
    15,
    30
  ],
  "accuracy_denominator": "all annotated selected clips, including prediction failures; MAE successful predictions only",
  "accuracy_duration_grouping": "nominal requested duration; actual sample duration retained in predictions, manifest and RTF",
  "file_timing_scope": "file read to BPM and original beat outputs, all necessary preprocessing/postprocessing; no downloads",
  "timing_selection": "first two deterministic manifest real tracks, first synthetic track, and demo; all durations",
  "date": "2026-11-21",
  "timezone": "America/New_York",
  "preloaded_repetitions": 5,
  "input_manifest": "C:\\Users\\허도영\\Desktop\\Git-Research\\CMU Class\\Rita-Robot\\experiment\\3_BPM_check\\improvements\\holdout_manifest.json",
  "input_manifest_sha256": "7ffae493902f410cd26ce5a7cbba7881a07a3f8ee54667219f9995b29de3286c"
}
```

Recorded environment (the complete package list remains in `environment.json`):

```json
{
  "platform": "Windows-10-10.0.26200-SP0",
  "os": "Windows",
  "python": "3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]",
  "executable": "C:\\Users\\허도영\\Desktop\\Git-Research\\CMU Class\\Rita-Robot\\experiment\\3_BPM_check\\.venv\\Scripts\\python.exe",
  "logical_cpu_count": 8,
  "inference_threads": 1,
  "thread_environment": {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1"
  },
  "ram_bytes": 16687734784,
  "physical_cpu_count": 8,
  "cpu_name": "Intel(R) Core(TM) Ultra 5 226V",
  "gpu_names": [
    "Intel(R) Arc(TM) 130V GPU (8GB)"
  ],
  "torch_version": "2.6.0+cpu",
  "cuda_available": false
}
```

## Accuracy results

Acc1 requires raw BPM within 2% of the reference. Acc2 additionally allows a factor of 0.5 or 2 applied to the prediction. Predictions are not normalized to the reference range. Accuracies use all valid-reference cases in the denominator, including prediction failures as incorrect; MAE and median error are computed over successful predictions only. Coverage is shown to expose missing outputs. A model that never successfully loaded is labeled Unavailable in these tables and omitted from accuracy figures; its accounting rows in CSV do not represent measured 0% model accuracy.

### Real music, all durations pooled

| Method | Annotated clips | Successful | Coverage (%) | Acc1 (%) | Acc2 (%) | MAE (BPM) | Median error (BPM) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Beat This! small0 | 150 | 150 | 100.0 | 64.0 | 76.7 | 18.06 | 1.54 |
| DeepRhythm | 150 | 150 | 100.0 | 58.7 | 82.7 | 26.48 | 1.37 |
| Existing classical | 150 | 95 | 63.3 | 32.7 | 52.7 | 30.80 | 1.12 |
| Classical V2 (new) | 150 | 150 | 100.0 | 62.7 | 86.7 | 24.88 | 0.95 |
| TempoCNN k4 | 150 | 150 | 100.0 | 61.3 | 80.0 | 22.87 | 1.13 |

| Method | Nominal duration (s) | Annotated clips | Successful | Acc1 (%) | Acc2 (%) | MAE (BPM) |
| --- | --- | --- | --- | --- | --- | --- |
| Beat This! small0 | 8.0 | 50 | 50 | 58.0 | 68.0 | 18.34 |
| Beat This! small0 | 15.0 | 50 | 50 | 62.0 | 78.0 | 19.70 |
| Beat This! small0 | 30.0 | 50 | 50 | 72.0 | 84.0 | 16.14 |
| DeepRhythm | 8.0 | 50 | 50 | 58.0 | 80.0 | 26.03 |
| DeepRhythm | 15.0 | 50 | 50 | 58.0 | 80.0 | 26.03 |
| DeepRhythm | 30.0 | 50 | 50 | 60.0 | 88.0 | 27.36 |
| Existing classical | 8.0 | 50 | 2 | 2.0 | 4.0 | 34.27 |
| Existing classical | 15.0 | 50 | 43 | 46.0 | 74.0 | 30.56 |
| Existing classical | 30.0 | 50 | 50 | 50.0 | 80.0 | 30.86 |
| Classical V2 (new) | 8.0 | 50 | 50 | 54.0 | 84.0 | 29.53 |
| Classical V2 (new) | 15.0 | 50 | 50 | 68.0 | 88.0 | 20.29 |
| Classical V2 (new) | 30.0 | 50 | 50 | 66.0 | 88.0 | 24.83 |
| TempoCNN k4 | 8.0 | 50 | 50 | 54.0 | 74.0 | 25.00 |
| TempoCNN k4 | 15.0 | 50 | 50 | 64.0 | 86.0 | 24.01 |
| TempoCNN k4 | 30.0 | 50 | 50 | 66.0 | 80.0 | 19.61 |

### Synthetic diagnostics, all durations pooled

No valid-reference results available.

Duration-specific crops from the same track are correlated; pooled clip counts are not independent song counts. Per-track dependence and a small real subset limit statistical certainty. See `predictions.csv` for unaltered BPMs, error statuses, timestamp counts, and any available beat F1 at ±70 ms tolerance.

## Latency results

The primary warm CPU table uses the same 4 successfully measured real excerpts for every listed method. This intersection avoids comparing fast failures with valid estimates or using different excerpt durations for different methods. A method that fails a short excerpt is still penalized in the accuracy/coverage tables. These timings represent this subset and machine only.

| Method | Clips | Repeats total | Median (ms) | Mean (ms) | P95 (ms) | SD (ms) | Median RTF | Init (ms) | First call (ms) | Time / existing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing classical | 4 | 80 | 427.35 | 433.73 | 638.25 | 136.36 | 0.0193 | 1,660.03 | 562.27 | 1.00× |
| DeepRhythm | 4 | 80 | 243.47 | 224.03 | 323.34 | 86.21 | 0.0091 | 8,501.98 | 8,033.04 | 0.57× |
| TempoCNN k4 | 4 | 80 | 29.11 | 27.02 | 44.62 | 10.02 | 0.0012 | 16,128.47 | 2,839.87 | 0.07× |
| Beat This! small0 | 4 | 80 | 1,418.64 | 1,709.75 | 3,074.72 | 1,151.62 | 0.0583 | 7,680.91 | 3,388.72 | 3.32× |
| Classical V2 (new) | 4 | 80 | 39.87 | 35.70 | 49.28 | 11.24 | 0.0015 | 368.13 | 1,668.75 | 0.09× |

`Time / existing` is the method's median processing time divided by the existing method's median; a value above one means the existing method is faster by that factor. RTF is measured processing time divided by clip duration; RTF below one means faster than playback for an already available file, and does not establish low-latency streaming behavior. `latency_summary.csv` contains per-duration, per-tier, and separate preloaded-audio measurements. `model_metadata.csv` also records package import/process startup details and checkpoint hashes.

Common real timing clip IDs: `blues.00004_15s`, `blues.00004_30s`, `classical.00004_15s`, `classical.00004_30s`.

The first inference can itself fail on the selected clip; its recorded status is shown below. Initialization includes framework imports performed inside the adapter's `load`; package-import time covers adapter construction. Process startup is measured before the worker's first marker. First-call time is fresh-process inference, but filesystem and Numba on-disk caches may already be warm from the smoke test. Initial smoke/JIT measurements are retained separately in `../results/neural_smoke_verification.json` (or `neural_smoke_verification.json` in the default results folder). Neither first-call nor initialization time includes checkpoint downloads.

| Method | Process start (ms) | Adapter import (ms) | Initialization (ms) | First call (ms) | First call status |
| --- | --- | --- | --- | --- | --- |
| Beat This! small0 | 215.24 | 89.33 | 7,680.91 | 3,388.72 | success |
| DeepRhythm | 262.36 | 78.53 | 8,501.98 | 8,033.04 | success |
| Existing classical | 232.55 | 16.06 | 1,660.03 | 562.27 | success |
| Classical V2 (new) | 185.33 | 7.15 | 368.13 | 1,668.75 | success |
| TempoCNN k4 | 164.40 | 77.24 | 16,128.47 | 2,839.87 | success |

Preloaded-audio stage availability: Existing classical: unavailable / no successful measurement; DeepRhythm: measured; TempoCNN k4: unavailable / no successful measurement; Beat This! small0: measured; Classical V2 (new): unavailable / no successful measurement. Unavailable stages have no substitute timing.

## Failures and availability

| Method | Clip failures | Recorded error |
| --- | --- | --- |
| Existing classical | 55 | RuntimeError: Insufficient beat-grid support |


## Fairness and limitations

- **Training overlap:** Beat This! small0's authors explicitly exclude GTZAN from its training set ([official model description](https://github.com/CPJKU/beat_this#available-models)). TempoCNN uses training data that include Extended Ballroom; an Extended Ballroom test would be contaminated ([official Essentia model catalog](https://essentia.upf.edu/models.html)). The published TempoCNN metadata names Extended Ballroom, LMDTempo, and MTGTempo; GTZAN membership has not been independently verified for every underlying source. DeepRhythm's exact training membership is not established by its public inference documentation ([official repository](https://github.com/bleugreen/deeprhythm)). No universal overlap-free claim is made across all models; real-music results are exploratory unless checkpoint-specific provenance excludes overlap.
- **TempoCNN runtime:** Native Windows has no Essentia TensorFlow wheel in this environment. The adapter loads the official `deeptemp-k4-3.pb` graph in a TensorFlow v1 session and uses the original author's `tempocnn` librosa frontend and standard normalization. This is a named runtime/frontend fallback, not an Essentia `TempoCNN` timing result. Global BPM follows the original author's argmax of mean patch probabilities, which differs from Essentia's default majority aggregation. The frontend pads short inputs to a 256-frame patch (about 11.9 seconds); 8-second results use zero padding and should not be interpreted as a naturally supported full-context input. The checkpoint and graph node names are verified before prediction.
- **Preprocessing and output:** Native feature extraction/sample rates differ. All main numbers include full file-to-result work. Beat-derived median-IBI BPM is a different target from a direct global tempo classifier; tempo changes, half-time feel, subdivisions, and missed beats can change its output.
- **DeepRhythm effective context:** The official implementation keeps only complete nonoverlapping 8-second windows and discards the remainder. Thus nominal 8/15/30-second files provide 8/8/24 seconds to the network, while reading/resampling the full file remains timed. No adapter-specific padding or context extension is introduced; duration curves need not scale smoothly or monotonically.
- **Classical assumptions:** A single regular grid, initial 70–180 BPM search, and at least 16 supported attacks constrain coverage. Short or weak-onset excerpts can fail. The implementation was not tuned to this test data.
- **Timing scope:** Models run serially with the saved CPU thread configuration. OS scheduling, file caching, JIT warm-up, and background activity can affect results. Repetition distributions characterize this run; they are not confidence intervals across hardware. Initialization excludes downloading. GPU results, if recorded separately, must not be mixed with this CPU table.
- **Offline scope:** Inputs are already recorded excerpts. RTF does not measure the time needed to acquire 8–30 seconds of future music, audio-device delay, robot response time, or causal beat-tracking latency.
- **Ground truth and sample size:** Synthetic references are exact; real BPM uses the dataset's published global tempo annotations unchanged for every prefix duration. Local median-IBI tempo is retained separately as a diagnostic and does not replace the published reference. Annotation interpretation and tempo variability remain sources of uncertainty. A small selected subset and correlated crops cannot support a general superiority claim.
- **Model validity:** Only adapters that successfully load official pretrained weights enter valid predictions. Unavailable models and failed excerpts remain explicitly recorded; no untrained substitutes or synthetic outputs are reported.

## Why we chose classical signal processing instead of a neural network

On the matched real timing subset, the existing detector is faster than Beat This! small0 by 3.32×.
On the same subset, DeepRhythm is 1.76× faster than the existing detector; TempoCNN k4 is 14.68× faster than the existing detector.
The latency advantage depends on the neural implementation: the classical detector is not the fastest tested method, so a general claim that classical BPM detection is faster than pretrained networks is not supported by this experiment.
For nominal 30-second real excerpts, existing Acc1 is 50.0% and the best measured neural Acc1 is 72.0% (Beat This! small0); the neural-minus-existing gap is +22.0 percentage points.

These measurements support using the classical method only when its observed accuracy, coverage, tempo assumptions, and processing budget fit the robot demonstration. They do not by themselves justify preferring it for general music or short excerpts. A neural alternative should be selected when its measured accuracy/coverage or processing time better meets the deployment requirement.

## Presentation-ready English text

- **Method selection:** Reused the unchanged multiband spectral-flux/autocorrelation detector; on nominal 30-second real excerpts it achieved 50.0% Acc1 with 100.0% output coverage.
- **Latency comparison:** Existing median CPU file-to-result time was 427.35 ms; the fastest measured method was TempoCNN k4 at 29.11 ms on 4 matched real excerpts.
- **Accuracy-efficiency trade-off:** Best neural Acc1 was 72.0% (Beat This! small0) versus 50.0% classical; the small subset and short-clip failures limit generalization.

## Validation

Final artifact, measurement-consistency, checkpoint, and source-preservation checks are recorded in [`validation.json`](validation.json). The benchmark only writes within its own directory and does not change robot-control behavior. Figures are generated from CSV values; unavailable methods and failed predictions remain visible in the saved results.

## Artifacts and reproduction

From `experiment/3_BPM_check`, follow the environment setup in `README.md`, then run:

```powershell
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe run_benchmark.py --quick --device cpu
.\.venv\Scripts\python.exe run_benchmark.py --full --device cpu
.\.venv\Scripts\python.exe plot_results.py
.\.venv\Scripts\python.exe report_results.py
```

The runner's CLI help documents dataset/audio-path and output options. Figures are regenerated from saved CSVs, not manually entered measurements. `plot_data.csv` records the exact plotted values. Model weights, cached audio, virtual environments, and intermediate diagnostics are excluded from version control.

- [Latency figure](figures/latency_comparison.png)
- [Real-music accuracy figure](figures/accuracy_comparison.png)
- [Synthetic diagnostic figure](figures/accuracy_comparison_synthetic.png)
- [Accuracy-latency trade-off](figures/accuracy_latency_tradeoff.png)
- [Duration vs. latency](figures/duration_vs_latency.png)
- [Billie Jean qualitative comparison](figures/billie_jean_comparison.png)
