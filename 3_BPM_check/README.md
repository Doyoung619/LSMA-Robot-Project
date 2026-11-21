# BPM detector benchmark

Executed comparison of the robot's original classical detector, pretrained DeepRhythm, official TempoCNN `deeptemp-k4-3`, and Beat This! `small0`. BeatNet is optional and records an explicit availability error if its dependencies cannot run. Saved numerical results and generated figures are under [`results`](results/benchmark_report.md).

In the saved environment, optional BeatNet could not install because its resolver requested `numba==0.54.1`, which supports Python `>=3.7,<3.10`, while this benchmark uses Python 3.11. The exact error is saved in [`results/beatnet_install_attempt.txt`](results/beatnet_install_attempt.txt); it is excluded from accuracy rankings. Native Essentia has no matching Windows wheel here, so the explicitly labeled TempoCNN runtime fallback below is used. See [`INSTALLATION.md`](INSTALLATION.md) for installation details.

This location follows the requested `experiment/3_BPM_check` layout. The benchmark does not change dance or robot-control code.

## Setup

Use Python **3.11** and FFmpeg available on `PATH`. The benchmark has a separate environment and does not install dependencies into the robot environment. From this directory on Windows PowerShell:

```powershell
.\setup_benchmark.ps1
```

Equivalent explicit setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
```

`requirements-lock.txt` records the environment used for the saved experiment. CPU-only Torch is deliberate: the primary comparison runs all supported methods on the same CPU with one inference thread by default. Do not run other model-inference jobs concurrently with timing.

The downloader caches official weights under `models/`, verifies file sizes, and records SHA-256 hashes and source URLs. Adapters require local weights; downloads never occur in a timed prediction. Adapter-module import, initialization including framework imports/model construction/weight loading, first inference, warm file-to-result latency, and supported preloaded-waveform latency are separate measurements.

First-call latency uses a fresh worker process; filesystem and Numba on-disk caches may already be warm from setup/smoke inference. The initial smoke/JIT measurements are retained separately in `results/neural_smoke_verification.json`. These measurements exclude network downloads and are not a fully cold machine/cache restart.

## Execute

```powershell
# 20 deterministic real tracks, 12 synthetic patterns, existing song demo;
# 8/15/30-second excerpts; 2 warm-ups and 5 timed repeats per representative clip.
.\.venv\Scripts\python.exe run_benchmark.py --quick --device cpu

# 50 real tracks if available; 5 warm-ups and 20 timed repeats.
.\.venv\Scripts\python.exe run_benchmark.py --full --device cpu

# Regenerate figures/report from the saved CSVs without inference.
.\.venv\Scripts\python.exe plot_results.py
.\.venv\Scripts\python.exe report_results.py

# Lightweight adapter/metric validation.
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Results are overwritten in the selected output directory. Preserve a prior run with `--results results_quick` or a different directory. The runner preserves completed method outputs even if another model fails. Examples:

```powershell
.\.venv\Scripts\python.exe run_benchmark.py --quick --real-count 20 --threads 1 --results results_quick
.\.venv\Scripts\python.exe run_benchmark.py --quick --dataset synthetic --methods OUR_EXISTING
.\.venv\Scripts\python.exe run_benchmark.py --quick --audio-path "C:\music\example.wav"
.\.venv\Scripts\python.exe run_benchmark.py --help
```

`--audio-path` adds/replaces the qualitative project-audio demo; it does not infer ground truth from a filename or a model's output. A synthetic-only run does not establish real-music accuracy. `--skip-download` reuses cached weights. `--device cuda` is a secondary supported-adapter experiment and should use a separate results directory and a CUDA-enabled Torch installation; unsupported classical/TensorFlow GPU adapters remain explicitly unavailable. CPU and GPU numbers must not be pooled.

## What is measured

- **Existing implementation:** `../0_pre_and_etc/rita_robot/audio.py:analyze`, called by `../0_pre_and_etc/rita.py:analysis` before playback. The robot normally reuses cached `data/analysis.json` when its decoded WAV is unchanged unless the explicit analyze command is used. This benchmark calls `analyze` every time and measures computation rather than cache retrieval. The adapter directly imports this source and retains its JSON/NPZ writes inside the timing. If the original project file is absent in a standalone checkout, it imports the byte-identical snapshot [`reference/rita_robot_audio.py`](reference/rita_robot_audio.py). Source SHA-256: `c36c0ae462cc0a1c7793deb667ac0a27f6479e4cd75e78390ba4145a905ecaed`.
- **Algorithm:** weighted multiband spectral flux, repeated-lag autocorrelation, exhaustive local beat-grid search, and robust global period/phase regression. No replacement with `librosa.beat.beat_track`, benchmark-specific retuning, or neural dependency is introduced into this adapter. The original initial search range is 70–180 BPM and fitting requires at least 16 supported attacks; some 8-second clips legitimately fail.
- **DeepRhythm:** pretrained official `deeprhythm-0.7.pth`, complete file-to-BPM pipeline. Its native preloaded-waveform API is timed separately. The official splitter retains only complete nonoverlapping 8-second windows and discards the remainder: 8/15/30-second files provide 8/8/24 seconds of effective network context. The full file is still read/resampled inside file-to-result timing.
- **TempoCNN on Windows:** official Essentia `deeptemp-k4-3.pb` weights loaded into a TensorFlow v1 session, with the original author's `tempocnn` librosa feature frontend and standard normalizer. This explicit compatibility fallback is **not an Essentia runtime measurement**. It uses the author's argmax of mean patch probabilities rather than Essentia's default majority aggregation. Its 256-frame context is about 11.9 seconds; the original frontend zero-pads 8-second audio. The report labels this distinction.
- **Beat This!:** official `small0` checkpoint, CPU, `dbn=False`; ordinary beat timestamps are kept. Global BPM is derived from median finite positive inter-beat intervals. Downbeats are not used as ordinary beat predictions.
- **Timing:** the main comparison measures local file input through final BPM/beat output, using `perf_counter_ns`, serial isolated model workers, and fixed thread settings. Models are loaded once and warmed before repeated representative-clip measurements. RTF below one means offline processing faster than playback, not streaming look-ahead or robot synchronization latency.

## Data and references

The dataset helper uses [mirdata's published GTZAN-mini index](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/gtzan_genre.py) and downloads only the selected tracks and published tempo/beat annotations. Selection uses seed `2026`, a seeded shuffle within genre, and round-robin genre selection. Repository revisions, source URLs, original MD5s, local SHA-256s, and exact track IDs are recorded in the manifests. Published **global tempo** annotations are the primary reference for each duration; local beat-derived median tempo is a separate diagnostic. Audio is retained locally for evaluation and excluded from Git.

Twelve generated source tracks cover 60, 90, 117, 120, 150, and 180 BPM, each as metronome and musical drum/bass patterns with exact quarter-note beat annotations. All methods receive matching 44.1 kHz stereo PCM WAV prefixes at 8, 15, and 30 seconds; model-specific preprocessing remains part of the measurement. The existing `Billie Jean.mp3` is used without downloading another copy and has no asserted ground-truth BPM.

Duration-specific accuracy uses the requested nominal duration. A source slightly shorter than 30 seconds stays in the nominal 30-second accuracy group and is not padded by the dataset helper; the report states the count and minimum actual duration. Predictions, timing rows, manifests, and RTF retain actual durations. In the expanded saved run, two country excerpts are about 4.17 ms shorter than 30 seconds, so the nominal 30-second accuracy comparison still includes all 50 selected real tracks.

[Beat This! small0 excludes GTZAN from training](https://github.com/CPJKU/beat_this#available-models). TempoCNN metadata names Extended Ballroom, LMDTempo, and MTGTempo. Exact DeepRhythm training membership is not fully documented in its inference repository. The report therefore treats cross-model real-music results as exploratory and discusses possible overlap and small-sample uncertainty.

## Outputs

| File | Contents |
| --- | --- |
| `results/predictions.csv` | Raw computed BPMs, beats, references, statuses, errors, and metrics |
| `results/accuracy_summary.csv` | Acc1, Acc2, success coverage, MAE, median error by tier/duration |
| `results/latency_raw.csv` | Every measured repetition, with file/preloaded stages separate |
| `results/latency_summary.csv` | Median, mean, P95, standard deviation, and RTF |
| `results/model_metadata.csv` | Versions, official checkpoints, bytes, hashes, licenses, initialization and first call |
| `results/environment.json` | Hardware, Python/packages, OS, thread configuration |
| `results/benchmark_config.json` | Exact run settings and representative timing clip IDs |
| `results/dataset_manifest.json` | Exact excerpts, source/annotation provenance, reference beats and hashes |
| `results/plot_data.csv` | Exact values plotted, regenerated from the CSV measurements |
| `results/figures/` | Presentation-ready 300 dpi PNGs; real and synthetic accuracy shown separately |
| `results/benchmark_report.md` | Objective findings, limitations, and three English slide bullets |

Acc1 requires raw predicted BPM within 2%. Acc2 separately allows half/double tempo equivalence. Failed predictions from a successfully loaded method count as incorrect in accuracy; MAE uses successful outputs and is accompanied by coverage. A model that never loads is labeled unavailable in the report and excluded from accuracy figures; its CSV accounting rows do not imply measured 0% model accuracy. The primary latency figure/report uses the intersection of successful real timing clips across measured methods, while duration curves show each method's valid timings. This avoids rewarding failures or comparing different content in the main latency ranking.

Virtual environments, checkpoints, audio files, source-audio caches, cropped WAVs, intermediate diagnostics, and worker artifacts are ignored. Compact JSON manifests/provenance/validation, CSV summaries, figures, source snapshots, and the report can be versioned. The `data/` directory is not ignored wholesale. Model licenses remain separate: follow the exact license fields in `model_metadata.csv`, especially the Essentia weights' CC BY-NC-SA terms.
