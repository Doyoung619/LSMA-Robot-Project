# Classical BPM detection vs. lightweight pretrained neural models

This report was generated from the saved CSV measurements. No BPM prediction is used as a reference annotation.

## Existing method

The real robot entry point is `../0_pre_and_etc/rita.py:analysis`, which calls `rita_robot.audio.decode` and `rita_robot.audio.analyze`. The benchmark imports and calls that unchanged `analyze(wav, output)` implementation through a thin adapter. It does not substitute a librosa beat tracker.

The standalone benchmark also contains a byte-identical source snapshot at `reference/rita_robot_audio.py` for checkouts without the rest of the robot project. The original project file takes precedence when present. Its SHA-256 is `c36c0ae462cc0a1c7793deb667ac0a27f6479e4cd75e78390ba4145a905ecaed`.

The normal pipeline decodes music with FFmpeg to 44.1 kHz stereo WAV, averages channels, and downsamples by four. A 1024-sample STFT with a 128-sample hop feeds log-whitened positive spectral flux. The onset bands are 35–180, 180–2000, and 2000–5000 Hz, weighted 0.55/0.30/0.15; 43-frame median filtering removes the local baseline. Autocorrelation scores 1×, 2×, and 4× lags with weights 1/0.6/0.3, keeps up to eight candidates initially between 70 and 180 BPM, and searches 121 period offsets × 48 phases per candidate. A soft-L1 least-squares beat-grid fit is repeated four times using attacks within 85 ms. The fit requires at least 16 supported beats. This is a global offline tempo and regular beat-grid estimator; it may fail on short clips or choose half tempo. The final fitted BPM is not clamped to the initial search range.

The robot consumes global BPM, beat period, and first-beat phase to plan moves before playback. The playback clock synchronizes motion after analysis; this benchmark does not evaluate motion control or streaming latency. `rita.py:analysis` normally reuses cached `data/analysis.json` when the decoded WAV is unchanged, unless the explicit analyze command requests recomputation. The benchmark calls `analyze` on every prediction, measuring detector computation rather than cached BPM retrieval. The original analysis also writes JSON and NPZ diagnostics. Those writes remain inside its file-to-result timing, preserving actual behavior.

## Compared methods

| Method | Type | Weight size | Output | Device | Checkpoint | Status | Pretrained verified |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Existing classical | Classical | 0 (no weights) | Global BPM + fitted beat grid | cpu | unchanged rita_robot.audio.analyze | success | N/A |
| DeepRhythm | NN | 5.19 MiB | Global BPM | cpu | deeprhythm-0.7.pth | success | True |
| TempoCNN k4 | NN | 0.14 MiB | Global BPM | cpu | deeptemp-k4-3.pb | success | True |
| Beat This! small0 | NN | 8.06 MiB | Beats; BPM derived from median valid IBI | cpu | beat_this-small0.ckpt | success | True |
| BeatNet | NN | Unavailable | Beats; BPM derived from median valid IBI | cpu |  | unavailable |  |

For beat trackers, the BPM adapter uses ordinary beats, excludes nonfinite/nonpositive intervals, and computes 60 divided by the median valid inter-beat interval. Insufficient beats are a failure, not an invented estimate. Original beat timestamps are preserved. BPM-only models have no fabricated beat output.

## Data and protocol

Evaluated 50 real annotated source tracks (150 duration-specific excerpts), 12 synthetic source tracks (36 excerpts), and 3 Billie Jean excerpts. The same saved 44.1 kHz stereo WAV excerpts are supplied to all methods; each model then uses its own required preprocessing. The intended durations are 8, 15, and 30 seconds. File hashes, source track IDs, annotation sources, selection seed, and exact durations are retained in the dataset manifest/configuration.

Duration-specific accuracy groups use the requested (nominal) duration, so the nominal 30-second group includes every selected source track. Predictions, timing rows, the manifest, and RTF retain actual audio duration; latency calculations use actual durations. The nominal 30-second real group includes 2 original clips slightly shorter than 30 seconds (minimum 29.9958276644 s, at most 4.172 ms short). The dataset helper does not pad or alter these source lengths.

Synthetic metronome and musical patterns cover 60, 90, 117, 120, 150, and 180 BPM with known beat times. These diagnose implementation behavior and do not establish accuracy on real music. A reference audit found 7/150 real crops where median local annotated IBI implies a BPM more than 2% from the published global tempo. The global annotation remains the primary reference; this local/global discrepancy can affect prefix-duration accuracy. Billie Jean is a qualitative demonstration: no verified manual reference is supplied, and the repository's earlier 117.025 BPM analysis is a detector output, not ground truth.

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
    "DeepRhythm",
    "TempoCNN",
    "BeatThis",
    "BeatNet"
  ],
  "timing_clip_ids": [
    "synthetic_metronome_60_8s",
    "synthetic_metronome_60_15s",
    "synthetic_metronome_60_30s",
    "blues.00000_8s",
    "blues.00000_15s",
    "blues.00000_30s",
    "classical.00002_8s",
    "classical.00002_15s",
    "classical.00002_30s",
    "billie_jean_8s",
    "billie_jean_15s",
    "billie_jean_30s"
  ],
  "results_dir": "C:\\Users\\허도영\\Desktop\\Git-Research\\CMU Class\\Rita-Robot\\experiment\\3_BPM_check\\results",
  "dataset": "all",
  "durations_seconds": [
    8,
    15,
    30
  ],
  "accuracy_denominator": "all annotated selected clips, including prediction failures; MAE successful predictions only",
  "file_timing_scope": "file read to BPM and original beat outputs, all necessary preprocessing/postprocessing; no downloads",
  "timing_selection": "first two deterministic manifest real tracks, first synthetic track, and demo; all durations",
  "date": "2026-11-21",
  "timezone": "America/New_York",
  "accuracy_duration_grouping": "Nominal requested duration; actual sample duration retained in predictions, manifest, timings and RTF."
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
| BeatNet | 150 | 0 | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable |
| Beat This! small0 | 150 | 150 | 100.0 | 73.3 | 83.3 | 19.49 | 1.44 |
| DeepRhythm | 150 | 150 | 100.0 | 66.0 | 80.7 | 19.36 | 1.10 |
| Existing classical | 150 | 93 | 62.0 | 40.7 | 53.3 | 21.37 | 0.43 |
| TempoCNN k4 | 150 | 150 | 100.0 | 65.3 | 79.3 | 18.10 | 0.87 |

| Method | Nominal duration (s) | Annotated clips | Successful | Acc1 (%) | Acc2 (%) | MAE (BPM) |
| --- | --- | --- | --- | --- | --- | --- |
| BeatNet | 8.0 | 50 | 0 | Unavailable | Unavailable | Unavailable |
| BeatNet | 15.0 | 50 | 0 | Unavailable | Unavailable | Unavailable |
| BeatNet | 30.0 | 50 | 0 | Unavailable | Unavailable | Unavailable |
| Beat This! small0 | 8.0 | 50 | 50 | 68.0 | 76.0 | 21.32 |
| Beat This! small0 | 15.0 | 50 | 50 | 78.0 | 88.0 | 20.59 |
| Beat This! small0 | 30.0 | 50 | 50 | 74.0 | 86.0 | 16.57 |
| DeepRhythm | 8.0 | 50 | 50 | 66.0 | 80.0 | 19.56 |
| DeepRhythm | 15.0 | 50 | 50 | 66.0 | 80.0 | 19.56 |
| DeepRhythm | 30.0 | 50 | 50 | 66.0 | 82.0 | 18.96 |
| Existing classical | 8.0 | 50 | 0 | 0.0 | 0.0 | Unavailable |
| Existing classical | 15.0 | 50 | 43 | 60.0 | 78.0 | 20.99 |
| Existing classical | 30.0 | 50 | 50 | 62.0 | 82.0 | 21.71 |
| TempoCNN k4 | 8.0 | 50 | 50 | 62.0 | 74.0 | 19.54 |
| TempoCNN k4 | 15.0 | 50 | 50 | 64.0 | 80.0 | 17.63 |
| TempoCNN k4 | 30.0 | 50 | 50 | 70.0 | 84.0 | 17.13 |

### Synthetic diagnostics, all durations pooled

| Method | Annotated clips | Successful | Coverage (%) | Acc1 (%) | Acc2 (%) | MAE (BPM) | Median error (BPM) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BeatNet | 36 | 0 | 0.0 | Unavailable | Unavailable | Unavailable | Unavailable |
| Beat This! small0 | 36 | 36 | 100.0 | 80.6 | 97.2 | 13.53 | 1.62 |
| DeepRhythm | 36 | 36 | 100.0 | 66.7 | 100.0 | 25.58 | 1.00 |
| Existing classical | 36 | 22 | 61.1 | 33.3 | 61.1 | 35.46 | 0.03 |
| TempoCNN k4 | 36 | 36 | 100.0 | 69.4 | 100.0 | 21.83 | 0.00 |

| Method | Nominal duration (s) | Annotated clips | Successful | Acc1 (%) | Acc2 (%) | MAE (BPM) |
| --- | --- | --- | --- | --- | --- | --- |
| BeatNet | 8.0 | 12 | 0 | Unavailable | Unavailable | Unavailable |
| BeatNet | 15.0 | 12 | 0 | Unavailable | Unavailable | Unavailable |
| BeatNet | 30.0 | 12 | 0 | Unavailable | Unavailable | Unavailable |
| Beat This! small0 | 8.0 | 12 | 12 | 75.0 | 91.7 | 14.32 |
| Beat This! small0 | 15.0 | 12 | 12 | 83.3 | 100.0 | 13.14 |
| Beat This! small0 | 30.0 | 12 | 12 | 83.3 | 100.0 | 13.14 |
| DeepRhythm | 8.0 | 12 | 12 | 66.7 | 100.0 | 25.58 |
| DeepRhythm | 15.0 | 12 | 12 | 66.7 | 100.0 | 25.58 |
| DeepRhythm | 30.0 | 12 | 12 | 66.7 | 100.0 | 25.58 |
| Existing classical | 8.0 | 12 | 0 | 0.0 | 0.0 | Unavailable |
| Existing classical | 15.0 | 12 | 11 | 50.0 | 91.7 | 35.46 |
| Existing classical | 30.0 | 12 | 11 | 50.0 | 91.7 | 35.46 |
| TempoCNN k4 | 8.0 | 12 | 12 | 66.7 | 100.0 | 23.83 |
| TempoCNN k4 | 15.0 | 12 | 12 | 75.0 | 100.0 | 17.75 |
| TempoCNN k4 | 30.0 | 12 | 12 | 66.7 | 100.0 | 23.92 |

Duration-specific crops from the same track are correlated; pooled clip counts are not independent song counts. Per-track dependence and a small real subset limit statistical certainty. See `predictions.csv` for unaltered BPMs, error statuses, timestamp counts, and any available beat F1 at ±70 ms tolerance.

## Latency results

The primary warm CPU table uses the same 4 successfully measured real excerpts for every listed method. This intersection avoids comparing fast failures with valid estimates or using different excerpt durations for different methods. A method that fails a short excerpt is still penalized in the accuracy/coverage tables. These timings represent this subset and machine only.

| Method | Clips | Repeats total | Median (ms) | Mean (ms) | P95 (ms) | SD (ms) | Median RTF | Init (ms) | First call (ms) | Time / existing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing classical | 4 | 80 | 231.55 | 222.76 | 256.44 | 27.97 | 0.0099 | 1,310.98 | 221.21 | 1.00× |
| DeepRhythm | 4 | 80 | 204.31 | 192.60 | 301.05 | 83.35 | 0.0082 | 11,247.44 | 4,160.13 | 0.88× |
| TempoCNN k4 | 4 | 80 | 26.38 | 21.52 | 30.87 | 7.09 | 0.0009 | 14,010.75 | 2,321.73 | 0.11× |
| Beat This! small0 | 4 | 80 | 1,377.10 | 1,674.93 | 3,689.29 | 1,271.64 | 0.0550 | 5,094.00 | 2,585.36 | 5.95× |

`Time / existing` is the method's median processing time divided by the existing method's median; a value above one means the existing method is faster by that factor. RTF is measured processing time divided by clip duration; RTF below one means faster than playback for an already available file, and does not establish low-latency streaming behavior. `latency_summary.csv` contains per-duration, per-tier, and separate preloaded-audio measurements. `model_metadata.csv` also records package import/process startup details and checkpoint hashes.

Common real timing clip IDs: `blues.00000_15s`, `blues.00000_30s`, `classical.00002_15s`, `classical.00002_30s`.

The first inference can itself fail on the selected clip; its recorded status is shown below. Initialization includes framework imports performed inside the adapter's `load`; package-import time covers adapter construction. Process startup is measured before the worker's first marker. First-call time is fresh-process inference, but filesystem and Numba on-disk caches may already be warm from the smoke test. Initial smoke/JIT measurements are retained separately in `../results/neural_smoke_verification.json` (or `neural_smoke_verification.json` in the default results folder). Neither first-call nor initialization time includes checkpoint downloads.

| Method | Process start (ms) | Adapter import (ms) | Initialization (ms) | First call (ms) | First call status |
| --- | --- | --- | --- | --- | --- |
| BeatNet | 152.78 | 63.76 | 2,330.90 | Unavailable | Unavailable |
| Beat This! small0 | 160.90 | 69.98 | 5,094.00 | 2,585.36 | success |
| DeepRhythm | 195.34 | 90.22 | 11,247.44 | 4,160.13 | success |
| Existing classical | 156.37 | 8.73 | 1,310.98 | 221.21 | success |
| TempoCNN k4 | 185.49 | 95.61 | 14,010.75 | 2,321.73 | success |

Preloaded-audio stage availability: Existing classical: unavailable / no successful measurement; DeepRhythm: measured; TempoCNN k4: unavailable / no successful measurement; Beat This! small0: measured; BeatNet: unavailable / no successful measurement. Unavailable stages have no substitute timing.

## Failures and availability

| Method | Clip failures | Recorded error |
| --- | --- | --- |
| BeatNet | 189 | ModuleNotFoundError: No module named 'BeatNet' |
| Existing classical | 72 | RuntimeError: Insufficient beat-grid support |

- BeatNet initialization/availability: `ModuleNotFoundError: No module named 'BeatNet'`

BeatNet was not evaluated: its dependency resolver requested `numba==0.54.1`, which declares `Requires-Python >=3.7,<3.10` and cannot install in the isolated Python 3.11 environment. The root installation error is preserved in [`beatnet_install_attempt.txt`](beatnet_install_attempt.txt). The later unavailable-module message is a consequence of this installation failure, not a measured tempo-prediction failure. It is excluded from model accuracy rankings.

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

On the matched real timing subset, the existing detector is faster than Beat This! small0 by 5.95×.
On the same subset, DeepRhythm is 1.13× faster than the existing detector; TempoCNN k4 is 8.78× faster than the existing detector.
The latency advantage depends on the neural implementation: the classical detector is not the fastest tested method, so a general claim that classical BPM detection is faster than pretrained networks is not supported by this experiment.
For nominal 30-second real excerpts, existing Acc1 is 62.0% and the best measured neural Acc1 is 74.0% (Beat This! small0); the neural-minus-existing gap is +12.0 percentage points.

These measurements support using the classical method only when its observed accuracy, coverage, tempo assumptions, and processing budget fit the robot demonstration. They do not by themselves justify preferring it for general music or short excerpts. A neural alternative should be selected when its measured accuracy/coverage or processing time better meets the deployment requirement.

## Presentation-ready English text

- **Method selection:** Reused the unchanged multiband spectral-flux/autocorrelation detector; on nominal 30-second real excerpts it achieved 62.0% Acc1 with 100.0% output coverage.
- **Latency comparison:** Existing median CPU file-to-result time was 231.55 ms; the fastest measured method was TempoCNN k4 at 26.38 ms on 4 matched real excerpts.
- **Accuracy-efficiency trade-off:** Best neural Acc1 was 74.0% (Beat This! small0) versus 62.0% classical; the small subset and short-clip failures limit generalization.

## Validation

The unchanged original detector was rerun on the full existing Billie Jean file and reproduced 117.02546482192 BPM with 573 fitted beats. This is a regression check, not a ground-truth accuracy score. The source hash remained unchanged. 15 automated adapter/metric checks passed under Python 3.11.9. The source snapshot and original pipeline source are byte-identical.

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
