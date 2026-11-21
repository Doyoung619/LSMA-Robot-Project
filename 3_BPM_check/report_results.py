"""Write an evidence-based report from the benchmark's saved measurements."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from plot_results import (accuracy_rows, common_real_latency, is_demo, is_real,
                          is_synthetic, method_order, number, pretty, read_csv, successful)


def fmt(value, places=2):
    value = number(value)
    return f"{value:,.{places}f}" if np.isfinite(value) else "Unavailable"


def percent(value):
    return fmt(100*number(value), 1)


def table(headers, rows):
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(str(value).replace("\n", " ").replace("|", "/") for value in row) + " |" for row in rows)
    return "\n".join(result)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def generate(results_dir="results"):
    root = Path(results_dir)
    predictions = read_csv(root / "predictions.csv")
    accuracy = read_csv(root / "accuracy_summary.csv")
    latency_raw = read_csv(root / "latency_raw.csv")
    metadata = read_csv(root / "model_metadata.csv")
    environment = read_json(root / "environment.json")
    config = read_json(root / "benchmark_config.json")
    original_validation = read_json(root / "original_detector_validation.json")
    if not original_validation:
        original_validation = read_json(root.parent / "data" / "original_detector_validation.json")
    manifest = read_json(root / "dataset_manifest.json")
    manifest = manifest if isinstance(manifest, list) else manifest.get("clips", [])
    local_audit = [row for row in manifest if is_real(row.get("tier"))
                   and np.isfinite(number(row.get("local_annotation_median_bpm")))
                   and number(row.get("reference_bpm")) > 0]
    changed_local = [row for row in local_audit
                     if abs(number(row["local_annotation_median_bpm"])-number(row["reference_bpm"]))/number(row["reference_bpm"]) > .02]
    short_real = [row for row in manifest if is_real(row.get("tier"))
                  and number(row.get("requested_duration_seconds")) == 30
                  and number(row.get("duration_seconds")) < 30 - 1e-6]
    short_note = ""
    if short_real:
        minimum = min(number(row["duration_seconds"]) for row in short_real)
        short_note = (f"The nominal 30-second real group includes {len(short_real)} original clips slightly shorter "
                      f"than 30 seconds (minimum {minimum:.10f} s, at most {(30-minimum)*1000:.3f} ms short). "
                      "The dataset helper does not pad or alter these source lengths. ")
    environment_brief = {key: value for key, value in environment.items() if key != "packages"}
    measured, common = common_real_latency(latency_raw)
    model_lookup = {row["method"]: row for row in metadata}
    loaded_methods = {row["method"] for row in metadata if row.get("status") == "success"}
    loaded_methods.update(row["method"] for row in predictions if successful(row))
    methods = method_order([row["method"] for row in metadata] + [row["method"] for row in predictions])
    real = [row for row in predictions if is_real(row.get("tier"))]
    synth = [row for row in predictions if is_synthetic(row.get("tier"))]
    n_real_tracks = len({row["track_id"] for row in real})
    n_synth_tracks = len({row["track_id"] for row in synth})
    n_real_clips = len({row["clip_id"] for row in real})
    n_synth_clips = len({row["clip_id"] for row in synth})
    n_demo_clips = len({row["clip_id"] for row in predictions if is_demo(row.get("tier"))})
    heading = ["# Classical BPM detection vs. lightweight pretrained neural models", "",
               "This report was generated from the saved CSV measurements. No BPM prediction is used as a reference annotation.", "",
               "## Existing method", "",
               "The real robot entry point is `../0_pre_and_etc/rita.py:analysis`, which calls "
               "`rita_robot.audio.decode` and `rita_robot.audio.analyze`. The benchmark imports and calls that "
               "unchanged `analyze(wav, output)` implementation through a thin adapter. It does not substitute a librosa beat tracker.", "",
               "The standalone benchmark also contains a byte-identical source snapshot at `reference/rita_robot_audio.py` "
               "for checkouts without the rest of the robot project. The original project file takes precedence when present. "
               "Its SHA-256 is `c36c0ae462cc0a1c7793deb667ac0a27f6479e4cd75e78390ba4145a905ecaed`.", "",
               "The normal pipeline decodes music with FFmpeg to 44.1 kHz stereo WAV, averages channels, and downsamples "
               "by four. A 1024-sample STFT with a 128-sample hop feeds log-whitened positive spectral flux. "
               "The onset bands are 35–180, 180–2000, and 2000–5000 Hz, weighted 0.55/0.30/0.15; "
               "43-frame median filtering removes the local baseline. Autocorrelation scores 1×, 2×, and 4× lags "
               "with weights 1/0.6/0.3, keeps up to eight candidates initially between 70 and 180 BPM, "
               "and searches 121 period offsets × 48 phases per candidate. A soft-L1 least-squares beat-grid "
               "fit is repeated four times using attacks within 85 ms. The fit requires at least 16 supported beats. "
               "This is a global offline tempo and regular beat-grid estimator; it may fail on short clips or choose half tempo. "
               "The final fitted BPM is not clamped to the initial search range.", "",
               "The robot consumes global BPM, beat period, and first-beat phase to plan moves before playback. "
               "The playback clock synchronizes motion after analysis; this benchmark does not evaluate motion control or streaming latency. "
               "`rita.py:analysis` normally reuses cached `data/analysis.json` when the decoded WAV is unchanged, "
               "unless the explicit analyze command requests recomputation. The benchmark calls `analyze` on every "
               "prediction, measuring detector computation rather than cached BPM retrieval. "
               "The original analysis also writes JSON and NPZ diagnostics. Those writes remain inside its file-to-result timing, preserving actual behavior.", "",
               "## Compared methods", ""]
    model_rows = []
    for method in methods:
        row = model_lookup.get(method, {})
        size = number(row.get("weight_bytes"))
        classical = method.startswith("OUR_")
        output = "Global BPM + fitted beat grid" if classical else ("Beats; BPM derived from median valid IBI" if "beat" in method.lower() else "Global BPM")
        model_rows.append([pretty(method), "Classical" if classical else "NN", "0 (no weights)" if classical else (f"{size/1024**2:.2f} MiB" if np.isfinite(size) else "Unavailable"),
                           output, config.get("device", "cpu"), row.get("checkpoint", "—"), row.get("status", "unknown"), "N/A" if classical else row.get("pretrained_loaded", "unknown")])
    heading += [table(["Method", "Type", "Weight size", "Output", "Device", "Checkpoint", "Status", "Pretrained verified"], model_rows), "",
                "For beat trackers, the BPM adapter uses ordinary beats, excludes nonfinite/nonpositive intervals, "
                "and computes 60 divided by the median valid inter-beat interval. Insufficient beats are a failure, "
                "not an invented estimate. Original beat timestamps are preserved. BPM-only models have no fabricated beat output.", "",
                "## Data and protocol", "",
                f"Evaluated {n_real_tracks} real annotated source tracks ({n_real_clips} duration-specific excerpts), "
                f"{n_synth_tracks} synthetic source tracks ({n_synth_clips} excerpts), and {n_demo_clips} Billie Jean excerpts. "
                "The same saved 44.1 kHz stereo WAV excerpts are supplied to all methods; each model then uses its own required preprocessing. "
                "The intended durations are 8, 15, and 30 seconds. File hashes, source track IDs, annotation sources, "
                "selection seed, and exact durations are retained in the dataset manifest/configuration.", "",
                "Duration-specific accuracy groups use the requested (nominal) duration, so the nominal 30-second "
                "group includes every selected source track. Predictions, timing rows, the manifest, and RTF retain "
                "actual audio duration; latency calculations use actual durations. " + short_note, "",
                "Synthetic metronome and musical patterns cover 60, 90, 117, 120, 150, and 180 BPM with known beat times. "
                "These diagnose implementation behavior and do not establish accuracy on real music. "
                f"A reference audit found {len(changed_local)}/{len(local_audit)} real crops where median local annotated "
                "IBI implies a BPM more than 2% from the published global tempo. The global annotation remains the "
                "primary reference; this local/global discrepancy can affect prefix-duration accuracy. "
                "Billie Jean is a qualitative demonstration: no verified manual reference is supplied, "
                "and the repository's earlier 117.025 BPM analysis is a detector output, not ground truth.", "",
                "Timing uses a high-resolution wall clock and serial CPU inference. Downloading and weight caching occur "
                "before initialization/inference measurements. Package import, model initialization/weight loading, "
                "and the first inference are recorded separately. Warm file-to-result time includes decoding/loading, "
                "feature extraction, inference, postprocessing, and BPM conversion. Preloaded-audio timings, where "
                "supported, form a separate stage and are never mixed into the main comparison.", "",
                "Saved benchmark configuration:", "", "```json", json.dumps(config, indent=2, ensure_ascii=False), "```", "",
                "Recorded environment (the complete package list remains in `environment.json`):", "", "```json", json.dumps(environment_brief, indent=2, ensure_ascii=False), "```", "",
                "## Accuracy results", "",
                "Acc1 requires raw BPM within 2% of the reference. Acc2 additionally allows a factor of 0.5 or 2 "
                "applied to the prediction. Predictions are not normalized to the reference range. "
                "Accuracies use all valid-reference cases in the denominator, including prediction failures as incorrect; "
                "MAE and median error are computed over successful predictions only. Coverage is shown to expose missing outputs. "
                "A model that never successfully loaded is labeled Unavailable in these tables and omitted from "
                "accuracy figures; its accounting rows in CSV do not represent measured 0% model accuracy.", ""]
    for test, label in [(is_real, "Real music"), (is_synthetic, "Synthetic diagnostics")]:
        selected = accuracy_rows(accuracy, test)
        rows = [[pretty(row["method"]), row.get("n_reference", ""), row.get("n_success", ""), percent(row.get("coverage")),
                 percent(row.get("acc1")) if row["method"] in loaded_methods else "Unavailable",
                 percent(row.get("acc2")) if row["method"] in loaded_methods else "Unavailable",
                 fmt(row.get("mae_bpm")), fmt(row.get("median_absolute_error_bpm"))] for row in selected]
        heading += [f"### {label}, all durations pooled", "",
                    table(["Method", "Annotated clips", "Successful", "Coverage (%)", "Acc1 (%)", "Acc2 (%)", "MAE (BPM)", "Median error (BPM)"], rows) if rows else "No valid-reference results available.", ""]
        by_duration = [row for row in accuracy if test(row.get("tier")) and row.get("duration_seconds") != "all"]
        if by_duration:
            heading += [table(["Method", "Nominal duration (s)", "Annotated clips", "Successful", "Acc1 (%)", "Acc2 (%)", "MAE (BPM)"],
                               [[pretty(row["method"]), row["duration_seconds"], row["n_reference"], row["n_success"],
                                 percent(row["acc1"]) if row["method"] in loaded_methods else "Unavailable",
                                 percent(row["acc2"]) if row["method"] in loaded_methods else "Unavailable", fmt(row["mae_bpm"])] for row in by_duration]), ""]
    heading += ["Duration-specific crops from the same track are correlated; pooled clip counts are not independent song counts. "
                "Per-track dependence and a small real subset limit statistical certainty. See `predictions.csv` for "
                "unaltered BPMs, error statuses, timestamp counts, and any available beat F1 at ±70 ms tolerance.", "",
                "## Latency results", "",
                f"The primary warm CPU table uses the same {len(common)} successfully measured real excerpts for every listed method. "
                "This intersection avoids comparing fast failures with valid estimates or using different excerpt durations "
                "for different methods. A method that fails a short excerpt is still penalized in the accuracy/coverage tables. "
                "These timings represent this subset and machine only.", ""]
    base = next((row["median_ms"] for row in measured if row["method"] == "OUR_EXISTING"), np.nan)
    rows = []
    for row in measured:
        meta = model_lookup.get(row["method"], {})
        ratio = row["median_ms"]/base if np.isfinite(base) and base > 0 else np.nan
        rows.append([pretty(row["method"]), row["n_clips"], row["n"], fmt(row["median_ms"]), fmt(row["mean_ms"]),
                     fmt(row["p95_ms"]), fmt(row["std_ms"]), fmt(row["rtf"], 4), fmt(meta.get("initialization_ms")),
                     fmt(meta.get("first_inference_ms")), f"{ratio:.2f}×" if np.isfinite(ratio) else "Unavailable"])
    heading += [table(["Method", "Clips", "Repeats total", "Median (ms)", "Mean (ms)", "P95 (ms)", "SD (ms)", "Median RTF", "Init (ms)", "First call (ms)", "Time / existing"], rows) if rows else "No common valid real latency comparison available.", "",
                "`Time / existing` is the method's median processing time divided by the existing method's median; "
                "a value above one means the existing method is faster by that factor. "
                "RTF is measured processing time divided by clip duration; RTF below one means faster than playback "
                "for an already available file, and does not establish low-latency streaming behavior. "
                "`latency_summary.csv` contains per-duration, per-tier, and separate preloaded-audio measurements. "
                "`model_metadata.csv` also records package import/process startup details and checkpoint hashes.", "",
                "Common real timing clip IDs: " + (", ".join(f"`{clip}`" for clip in common) if common else "None") + ".", "",
                "The first inference can itself fail on the selected clip; its recorded status is shown below. "
                "Initialization includes framework imports performed inside the adapter's `load`; package-import time "
                "covers adapter construction. Process startup is measured before the worker's first marker. "
                "First-call time is fresh-process inference, but filesystem and Numba on-disk caches may already be "
                "warm from the smoke test. Initial smoke/JIT measurements are retained separately in "
                "`../results/neural_smoke_verification.json` (or `neural_smoke_verification.json` in the default results folder). "
                "Neither first-call nor initialization time includes checkpoint downloads.", ""]
    startup_rows = []
    for meta in metadata:
        try:
            extra = json.loads(meta.get("extra_json") or "{}")
        except json.JSONDecodeError:
            extra = {}
        startup_rows.append([pretty(meta["method"]), fmt(meta.get("process_start_ms")), fmt(meta.get("package_import_ms")),
                             fmt(meta.get("initialization_ms")), fmt(meta.get("first_inference_ms")), extra.get("first_inference_status", "Unavailable")])
    heading += [table(["Method", "Process start (ms)", "Adapter import (ms)", "Initialization (ms)", "First call (ms)", "First call status"], startup_rows), "",
                "Preloaded-audio stage availability: " + "; ".join(
                    f"{pretty(method)}: " + ("measured" if any(row["method"] == method and row.get("stage") == "preloaded" and successful(row) for row in latency_raw)
                    else "unavailable / no successful measurement") for method in methods) + ". "
                "Unavailable stages have no substitute timing.", "",
                "## Failures and availability", ""]
    failures = Counter((row["method"], row.get("error", "")) for row in predictions if not successful(row))
    failure_rows = [[pretty(method), count, error] for (method, error), count in sorted(failures.items())]
    heading += [table(["Method", "Clip failures", "Recorded error"], failure_rows) if failure_rows else "No prediction failures recorded.", ""]
    for row in metadata:
        if row.get("error"):
            heading.append(f"- {pretty(row['method'])} initialization/availability: `{row['error']}`")
    if "BeatNet" in methods and "BeatNet" not in loaded_methods:
        log_name = "beatnet_install_attempt.txt" if root.name == "results" else "../results/beatnet_install_attempt.txt"
        heading += ["", "BeatNet was not evaluated: its dependency resolver requested `numba==0.54.1`, "
                    "which declares `Requires-Python >=3.7,<3.10` and cannot install in the isolated Python 3.11 environment. "
                    f"The root installation error is preserved in [`{log_name}`]({log_name}). "
                    "The later unavailable-module message is a consequence of this installation failure, "
                    "not a measured tempo-prediction failure. It is excluded from model accuracy rankings."]
    heading += ["", "## Fairness and limitations", "",
                "- **Training overlap:** Beat This! small0's authors explicitly exclude GTZAN from its training set "
                "([official model description](https://github.com/CPJKU/beat_this#available-models)). "
                "TempoCNN uses training data that include Extended Ballroom; an Extended Ballroom test would be contaminated "
                "([official Essentia model catalog](https://essentia.upf.edu/models.html)). "
                "The published TempoCNN metadata names Extended Ballroom, LMDTempo, and MTGTempo; GTZAN membership "
                "has not been independently verified for every underlying source. "
                "DeepRhythm's exact training membership is not established by its public inference documentation "
                "([official repository](https://github.com/bleugreen/deeprhythm)). "
                "No universal overlap-free claim is made across all models; real-music results are exploratory unless checkpoint-specific provenance excludes overlap.",
                "- **TempoCNN runtime:** Native Windows has no Essentia TensorFlow wheel in this environment. "
                "The adapter loads the official `deeptemp-k4-3.pb` graph in a TensorFlow v1 session and uses "
                "the original author's `tempocnn` librosa frontend and standard normalization. This is a named "
                "runtime/frontend fallback, not an Essentia `TempoCNN` timing result. Global BPM follows the original author's "
                "argmax of mean patch probabilities, which differs from Essentia's default majority aggregation. The frontend pads short inputs "
                "to a 256-frame patch (about 11.9 seconds); 8-second results use zero padding and should not be "
                "interpreted as a naturally supported full-context input. The checkpoint and graph node names "
                "are verified before prediction.",
                "- **Preprocessing and output:** Native feature extraction/sample rates differ. All main numbers include full "
                "file-to-result work. Beat-derived median-IBI BPM is a different target from a direct global tempo classifier; "
                "tempo changes, half-time feel, subdivisions, and missed beats can change its output.",
                "- **DeepRhythm effective context:** The official implementation keeps only complete nonoverlapping "
                "8-second windows and discards the remainder. Thus nominal 8/15/30-second files provide 8/8/24 seconds "
                "to the network, while reading/resampling the full file remains timed. No adapter-specific padding "
                "or context extension is introduced; duration curves need not scale smoothly or monotonically.",
                "- **Classical assumptions:** A single regular grid, initial 70–180 BPM search, and at least 16 supported "
                "attacks constrain coverage. Short or weak-onset excerpts can fail. The implementation was not tuned to this test data.",
                "- **Timing scope:** Models run serially with the saved CPU thread configuration. OS scheduling, file caching, "
                "JIT warm-up, and background activity can affect results. Repetition distributions characterize this run; "
                "they are not confidence intervals across hardware. Initialization excludes downloading. GPU results, "
                "if recorded separately, must not be mixed with this CPU table.",
                "- **Offline scope:** Inputs are already recorded excerpts. RTF does not measure the time needed to acquire "
                "8–30 seconds of future music, audio-device delay, robot response time, or causal beat-tracking latency.",
                "- **Ground truth and sample size:** Synthetic references are exact; real BPM uses the dataset's published "
                "global tempo annotations unchanged for every prefix duration. Local median-IBI tempo is retained "
                "separately as a diagnostic and does not replace the published reference. Annotation interpretation and tempo variability remain "
                "sources of uncertainty. A small selected subset and correlated crops cannot support a general superiority claim.",
                "- **Model validity:** Only adapters that successfully load official pretrained weights enter valid predictions. "
                "Unavailable models and failed excerpts remain explicitly recorded; no untrained substitutes or synthetic outputs are reported.", "",
                "## Why we chose classical signal processing instead of a neural network", ""]
    real30 = accuracy_rows(accuracy, is_real, "30.0") or accuracy_rows(accuracy, is_real, "30")
    primary_acc = real30 or accuracy_rows(accuracy, is_real)
    acc_lookup = {row["method"]: row for row in primary_acc}
    ours = acc_lookup.get("OUR_EXISTING")
    comparisons = [row for row in measured if not row["method"].startswith("OUR_")]
    statements = []
    if np.isfinite(base) and comparisons:
        faster = [row for row in comparisons if row["median_ms"] > base]
        slower = [row for row in comparisons if row["median_ms"] < base]
        if faster:
            statements.append("On the matched real timing subset, the existing detector is faster than " + "; ".join(f"{pretty(row['method'])} by {row['median_ms']/base:.2f}×" for row in faster) + ".")
        if slower:
            statements.append("On the same subset, " + "; ".join(f"{pretty(row['method'])} is {base/row['median_ms']:.2f}× faster than the existing detector" for row in slower) + ".")
        if slower and not faster:
            statements.append("For the tested implementations on this CPU, the measurements do not support lower "
                              "processing latency as a reason to choose the classical detector over the neural models.")
        elif slower and faster:
            statements.append("The latency advantage depends on the neural implementation: the classical detector "
                              "is not the fastest tested method, so a general claim that classical BPM detection "
                              "is faster than pretrained networks is not supported by this experiment.")
    if ours and primary_acc:
        ours_acc = number(ours.get("acc1"))
        valid_nn = [row for row in primary_acc if not row["method"].startswith("OUR_") and number(row.get("n_success"), 0) > 0]
        if valid_nn:
            best = max(valid_nn, key=lambda row: number(row.get("acc1"), -1))
            delta = 100*(number(best["acc1"])-ours_acc)
            scope = "nominal 30-second real excerpts" if real30 else "pooled real excerpts"
            statements.append(f"For {scope}, existing Acc1 is {percent(ours_acc)}% and the best measured neural Acc1 is "
                              f"{percent(best['acc1'])}% ({pretty(best['method'])}); the neural-minus-existing gap is {delta:+.1f} percentage points.")
    if statements:
        heading += statements + ["", "These measurements support using the classical method only when its observed accuracy, "
                                 "coverage, tempo assumptions, and processing budget fit the robot demonstration. "
                                 "They do not by themselves justify preferring it for general music or short excerpts. "
                                 "A neural alternative should be selected when its measured accuracy/coverage or processing time "
                                 "better meets the deployment requirement.", ""]
    else:
        heading += ["The available measurements do not establish a justified preference. A method-selection claim requires "
                    "valid matched latency and annotated real-music results; missing results must not be filled with assumptions.", ""]
    heading += ["## Presentation-ready English text", ""]
    if ours:
        heading.append(f"- **Method selection:** Reused the unchanged multiband spectral-flux/autocorrelation detector; "
                       f"on {'nominal 30-second' if real30 else 'pooled'} real excerpts it achieved {percent(ours['acc1'])}% Acc1 "
                       f"with {percent(ours.get('coverage'))}% output coverage.")
    else:
        heading.append("- **Method selection:** Reused the original offline classical BPM detector; annotated real-music accuracy is unavailable.")
    if np.isfinite(base) and comparisons:
        fastest = min(measured, key=lambda row: row["median_ms"])
        heading.append(f"- **Latency comparison:** Existing median CPU file-to-result time was {fmt(base)} ms; "
                       f"the fastest measured method was {pretty(fastest['method'])} at {fmt(fastest['median_ms'])} ms "
                       f"on {len(common)} matched real excerpts.")
    else:
        heading.append("- **Latency comparison:** No valid common real excerpt latency comparison is available.")
    if ours and any(not row["method"].startswith("OUR_") and number(row.get("n_success"), 0) > 0 for row in primary_acc):
        best = max((row for row in primary_acc if not row["method"].startswith("OUR_") and number(row.get("n_success"), 0) > 0), key=lambda row: number(row.get("acc1"), -1))
        heading.append(f"- **Accuracy-efficiency trade-off:** Best neural Acc1 was {percent(best['acc1'])}% "
                       f"({pretty(best['method'])}) versus {percent(ours['acc1'])}% classical; "
                       "the small subset and short-clip failures limit generalization.")
    else:
        heading.append("- **Accuracy-efficiency trade-off:** Missing annotated results prevent an evidence-based trade-off claim.")
    heading += ["", "## Validation", ""]
    if original_validation.get("status") == "success":
        heading += [f"The unchanged original detector was rerun on the full existing Billie Jean file and reproduced "
                    f"{number(original_validation.get('full_song_bpm')):.11f} BPM with "
                    f"{original_validation.get('beat_count')} fitted beats. This is a regression check, not a ground-truth accuracy score. "
                    f"The source hash remained unchanged. {original_validation.get('unit_tests_passed')} automated adapter/metric checks "
                    f"passed under Python {original_validation.get('unit_test_python')}. "
                    "The source snapshot and original pipeline source are byte-identical.", ""]
    heading += ["Final artifact, measurement-consistency, checkpoint, and source-preservation checks are recorded in "
                "[`validation.json`](validation.json). The benchmark only writes within its own directory and does not "
                "change robot-control behavior. Figures are generated from CSV values; unavailable methods and failed "
                "predictions remain visible in the saved results.", "",
                "## Artifacts and reproduction", "",
                "From `experiment/3_BPM_check`, follow the environment setup in `README.md`, then run:", "",
                "```powershell", ".\\.venv\\Scripts\\python.exe download_models.py",
                ".\\.venv\\Scripts\\python.exe run_benchmark.py --quick --device cpu",
                ".\\.venv\\Scripts\\python.exe run_benchmark.py --full --device cpu",
                ".\\.venv\\Scripts\\python.exe plot_results.py",
                ".\\.venv\\Scripts\\python.exe report_results.py", "```", "",
                "The runner's CLI help documents dataset/audio-path and output options. Figures are regenerated from saved CSVs, "
                "not manually entered measurements. `plot_data.csv` records the exact plotted values. "
                "Model weights, cached audio, virtual environments, and intermediate diagnostics are excluded from version control.", "",
                "- [Latency figure](figures/latency_comparison.png)",
                "- [Real-music accuracy figure](figures/accuracy_comparison.png)",
                "- [Synthetic diagnostic figure](figures/accuracy_comparison_synthetic.png)",
                "- [Accuracy-latency trade-off](figures/accuracy_latency_tradeoff.png)",
                "- [Duration vs. latency](figures/duration_vs_latency.png)",
                "- [Billie Jean qualitative comparison](figures/billie_jean_comparison.png)", ""]
    if "OUR_V2" in methods:
        position=heading.index("## Compared methods")
        heading[position:position]=["## Separately developed classical V2", "",
            "`OUR_V2` is a new method in `adapters/classical_v2.py`; the original robot implementation and its saved "
            "benchmark results remain unchanged. It retains whitened multiband spectral flux, uses a faster FFT grid "
            "and vectorized period/phase search, supports at least three matched attacks on short clips, and adds "
            "density-aware tempo scoring with a development-selected tempo prior. It has no neural weights. "
            "Runtime includes audio loading, resampling, features, fitting, and BPM/beat output; optional diagnostic "
            "file writes from the original implementation are not part of V2's required API.", "",
            "Configuration was selected on the previously inspected 50 GTZAN tracks, then frozen in "
            "`improvements/frozen_protocol.json` before obtaining/evaluating the remaining 50 tracks. "
            "The holdout split has no track overlap with this classical development set. Both Acc1 and Acc2 "
            "are reported; an Acc2 advantage does not establish a raw-BPM Acc1 advantage.", ""]
    destination = root / "benchmark_report.md"
    destination.write_text("\n".join(line.rstrip() for line in "\n".join(heading).splitlines()) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    print(f"Saved {generate(args.results)}")
