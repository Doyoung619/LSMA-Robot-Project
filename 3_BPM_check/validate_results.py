"""Audit saved benchmark provenance, metrics, timing summaries, and plot values.

This performs no inference and makes no network calls. It checks consistency
against recorded inputs and raw measurements, rather than claiming an
independent replication of the pretrained neural network predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def optional(value):
    return None if value is None or value == "" else float(value)


def nominal_duration(row):
    """Accuracy groups use requested context; RTF retains actual sample length."""
    requested = optional(row.get("requested_duration_seconds"))
    return requested if requested is not None else float(row["duration_seconds"])


def hash_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tempo(bpm, reference):
    if reference is None:
        return {"absolute_error": None, "acc1": None, "acc2": None}
    if bpm is None or not math.isfinite(bpm) or bpm <= 0:
        return {"absolute_error": None, "acc1": 0, "acc2": 0}
    error = abs(bpm - reference)
    return {"absolute_error": error, "acc1": int(error <= .02 * reference + 1e-10),
            "acc2": int(min(abs(bpm / 2 - reference), error, abs(bpm * 2 - reference))
                        <= .02 * reference + 1e-10)}


def latency_stats(rows):
    times = [float(row["latency_ms"]) for row in rows]
    return {"n": len(times), "n_clips": len({row["clip_id"] for row in rows}),
            "median_ms": statistics.median(times), "mean_ms": statistics.mean(times),
            "p95_ms": float(np.percentile(times, 95)), "std_ms": statistics.pstdev(times),
            "rtf": statistics.median(float(row["latency_ms"]) / 1000 /
                                      float(row["duration_seconds"]) for row in rows)}


def validate(results):
    results = Path(results).resolve()
    errors, warnings = [], []
    checks = 0

    def check(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    def equal(observed, expected, label):
        if isinstance(expected, (int, float)):
            try:
                matches = math.isfinite(float(observed)) and math.isclose(
                    float(observed), float(expected), rel_tol=1e-9, abs_tol=1e-8)
            except (ValueError, TypeError):
                matches = False
        else:
            matches = observed in (None, "") if expected is None else observed == expected
        check(matches, f"{label}: observed={observed!r}, expected={expected!r}")

    required = ["dataset_manifest.json", "benchmark_config.json", "predictions.csv",
                "latency_raw.csv", "accuracy_summary.csv", "latency_summary.csv",
                "model_metadata.csv", "plot_data.csv"]
    for name in required:
        check((results / name).is_file(), f"Missing required result file: {name}")
    if errors:
        outcome = {"status": "failed", "checks": checks, "errors": errors, "warnings": warnings}
        (results / "validation.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
        return outcome
    manifest = json.loads((results / "dataset_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((results / "benchmark_config.json").read_text(encoding="utf-8"))
    clips = {row["clip_id"]: row for row in manifest}
    predictions = read_csv(results / "predictions.csv")
    raw = read_csv(results / "latency_raw.csv")
    accuracy = read_csv(results / "accuracy_summary.csv")
    latency = read_csv(results / "latency_summary.csv")
    metadata = read_csv(results / "model_metadata.csv")
    plot = read_csv(results / "plot_data.csv")
    check(len(clips) == len(manifest), "Duplicate clip IDs in dataset manifest")
    real_index = {}
    annotation_cache = {}
    if any(clip["tier"] == "real" for clip in manifest):
        index_path = ROOT / "data" / "gtzan_mini" / "mirdata_mini_index.json"
        check(index_path.is_file(), "Cached official GTZAN-mini annotation index missing")
        if index_path.is_file():
            equal(hash_file(index_path, "md5"), "ac97f5a783d7843cf92ed8875d85af3d", "Official mirdata mini index MD5")
            real_index = json.loads(index_path.read_text(encoding="utf-8"))["tracks"]

    for clip in manifest:
        label = clip["clip_id"]
        path = ROOT / clip["audio_path"]
        check(path.is_file(), f"Missing local audio: {label}")
        if not path.is_file():
            continue
        equal(hash_file(path), clip["pcm_sha256"], f"{label} WAV SHA256")
        with wave.open(str(path), "rb") as wav:
            equal(wav.getframerate(), clip["sample_rate"], f"{label} sample rate")
            equal(wav.getnchannels(), clip["channels"], f"{label} channel count")
            equal(wav.getnframes() / wav.getframerate(), clip["duration_seconds"], f"{label} duration")
        if clip["tier"] == "demo":
            equal(clip["reference_bpm"], None, f"{label} demo reference BPM")
            equal(clip["reference_beats"], None, f"{label} demo reference beats")
        if clip["tier"] == "real":
            check(bool(clip.get("annotation_source")), f"{label} real annotation provenance missing")
            check(bool(clip.get("source_audio_md5")), f"{label} real source MD5 missing")
            track = clip["track_id"]
            check(track in real_index, f"{label} track absent from official mini index")
            if track in real_index and track not in annotation_cache:
                item = real_index[track]
                paths = {kind: ROOT / "data" / "gtzan_mini" / kind / Path(identity[0]).name
                         for kind, identity in item.items()}
                for kind, path in paths.items():
                    check(path.is_file(), f"{track} cached source {kind} missing")
                    if path.is_file():
                        equal(hash_file(path, "md5"), item[kind][1], f"{track} official {kind} MD5")
                if all(path.is_file() for path in paths.values()):
                    annotation_cache[track] = {"bpm": float(np.loadtxt(paths["tempo"])),
                                              "beats": np.loadtxt(paths["beats"], ndmin=2)[:, 0]}
            if track in annotation_cache:
                original = annotation_cache[track]
                equal(clip["reference_bpm"], original["bpm"], f"{label} published tempo annotation")
                beats = original["beats"]
                cropped = beats[(beats >= 0) & (beats < clip["duration_seconds"])]
                check(np.array_equal(np.asarray(clip["reference_beats"]), cropped),
                      f"{label} reference beats match published annotation prefix")

    counts = Counter((row["method"], row["clip_id"]) for row in predictions)
    for method in config["methods"]:
        for clip_id in clips:
            equal(counts[(method, clip_id)], 1, f"{method}/{clip_id} exactly one prediction record")
    for row in predictions:
        label = f"{row['method']}/{row['clip_id']}"
        check(row["clip_id"] in clips, f"{label} is absent from manifest")
        if row["clip_id"] not in clips:
            continue
        clip = clips[row["clip_id"]]
        equal(row["track_id"], clip["track_id"], f"{label} track ID")
        equal(row["tier"], clip["tier"], f"{label} tier")
        equal(row["duration_seconds"], clip["duration_seconds"], f"{label} duration")
        equal(nominal_duration(row), nominal_duration(clip), f"{label} requested duration")
        equal(row["reference_bpm"], clip["reference_bpm"], f"{label} reference BPM")
        bpm = optional(row.get("bpm"))
        if row["status"] == "success":
            check(bpm is not None and math.isfinite(bpm) and bpm > 0, f"{label} success requires valid BPM")
        else:
            equal(bpm, None, f"{label} failed BPM must be absent")
        for key, value in expected_tempo(bpm, clip["reference_bpm"]).items():
            equal(row.get(key), value, f"{label} {key}")
        beats = json.loads(row["beat_times_json"]) if row.get("beat_times_json") else None
        if beats is not None:
            equal(row["beat_count"], len(beats), f"{label} beat count")
            check(all(math.isfinite(float(x)) for x in beats), f"{label} beat timestamps finite")
            check(all(b > a for a, b in zip(beats, beats[1:])), f"{label} beat timestamps strictly increasing")
            if row["method"] in ("BeatThis", "BeatNet") and row["status"] == "success":
                intervals = [b - a for a, b in zip(beats, beats[1:])
                             if math.isfinite(a) and math.isfinite(b) and b > a]
                check(bool(intervals), f"{label} derived BPM requires valid ordinary beat intervals")
                if intervals:
                    equal(bpm, 60 / statistics.median(intervals), f"{label} BPM derived from preserved beat timestamps")
            from metrics import beat_metrics
            for key, value in beat_metrics(beats, clip["reference_beats"]).items():
                equal(row.get(key), value, f"{label} {key}")
        else:
            equal(row.get("beat_count"), None, f"{label} unavailable beats not invented")
            equal(row.get("beat_f1"), None, f"{label} unavailable beat metric")

    for row in accuracy:
        duration = row["duration_seconds"]
        subset = [p for p in predictions if p["method"] == row["method"] and p["tier"] == row["tier"]
                  and (duration == "all" or nominal_duration(p) == float(duration))]
        refs = [p for p in subset if optional(p["reference_bpm"]) is not None]
        valid = [p for p in refs if p["status"] == "success"]
        absolute = [abs(float(p["bpm"]) - float(p["reference_bpm"])) for p in valid]
        expected = {"n_total": len(subset), "n_success": sum(p["status"] == "success" for p in subset),
                    "n_reference": len(refs), "coverage": len(valid) / len(refs) if refs else None,
                    "mae_bpm": statistics.mean(absolute) if absolute else None,
                    "median_absolute_error_bpm": statistics.median(absolute) if absolute else None}
        for key in ("acc1", "acc2"):
            expected[key] = sum(expected_tempo(optional(p["bpm"]), float(p["reference_bpm"]))[key]
                                for p in refs) / len(refs) if refs else None
        for key, value in expected.items():
            equal(row[key], value, f"accuracy {row['method']}/{row['tier']}/{duration} {key}")

    good_times = [row for row in raw if row["status"] == "success"]
    for row in good_times:
        check(math.isfinite(float(row["latency_ms"])) and float(row["latency_ms"]) > 0,
              f"Nonpositive/nonfinite measured latency: {row['method']}/{row['clip_id']}")
        equal(row["duration_seconds"], clips[row["clip_id"]]["duration_seconds"],
              f"latency {row['clip_id']} manifest duration")
    for row in latency:
        subset = [p for p in good_times if p["method"] == row["method"] and p["tier"] == row["tier"]
                  and p["stage"] == row["stage"] and (row["duration_seconds"] == "all" or
                      float(p["duration_seconds"]) == float(row["duration_seconds"]))]
        check(bool(subset), f"Latency summary has no raw source: {row}")
        if subset:
            for key, value in latency_stats(subset).items():
                if key != "n_clips":
                    equal(row[key], value, f"latency {row['method']}/{row['tier']}/{row['duration_seconds']}/{row['stage']} {key}")

    for row in metadata:
        method = row["method"]
        successes = sum(p["method"] == method and p["status"] == "success" for p in predictions)
        if method == "OUR_EXISTING":
            from adapters.existing import SOURCE
            equal(hash_file(SOURCE), row["sha256"], "Classical source hash unchanged")
            equal(row["weight_bytes"], 0, "Classical weight size")
        elif successes:
            from adapters.model_cache import EXPECTED_SHA256
            check(row["pretrained_loaded"].lower() == "true", f"{method} pretrained weights not verified")
            if row["checkpoint"] in EXPECTED_SHA256:
                equal(row["sha256"], EXPECTED_SHA256[row["checkpoint"]], f"{method} official checkpoint identity")
            path = ROOT / "models" / row["checkpoint"]
            check(path.is_file(), f"{method} local checkpoint missing")
            if path.is_file():
                equal(path.stat().st_size, optional(row["weight_bytes"]), f"{method} checkpoint bytes")
                equal(hash_file(path), row["sha256"], f"{method} checkpoint SHA256")
            check(bool(row["source_url"] and row["license"]), f"{method} checkpoint source/license missing")
        else:
            warnings.append(f"{method}: no successful predictions; excluded from valid model comparisons")

    real_times = [row for row in good_times if row["tier"] == "real" and row["stage"] == "file_to_result"]
    methods = sorted({row["method"] for row in real_times})
    sets = [{row["clip_id"] for row in real_times if row["method"] == method} for method in methods]
    common = set.intersection(*sets) if sets else set()
    matched = {method: latency_stats([row for row in real_times if row["method"] == method and row["clip_id"] in common])
               for method in methods} if common else {}
    for row in plot:
        method, figure = row["method"], row["figure"]
        expected = {}
        if figure == "latency_comparison":
            expected = matched.get(method, {})
        elif figure.startswith("accuracy_comparison"):
            tier = "synthetic" if figure.endswith("_synthetic") else "real"
            source = next((x for x in accuracy if x["method"] == method and x["tier"] == tier
                           and x["duration_seconds"] == "all"), None)
            check(source is not None, f"{figure}/{method} accuracy source missing")
            if source:
                expected = {key: optional(source[key]) for key in ("n_reference", "n_success", "n_total", "coverage",
                            "acc1", "acc2", "mae_bpm", "median_absolute_error_bpm")}
        elif figure == "accuracy_latency_tradeoff":
            source = next((x for x in accuracy if x["method"] == method and x["tier"] == "real" and
                           ((row["duration_seconds"] == "all" and x["duration_seconds"] == "all") or
                            (row["duration_seconds"] != "all" and x["duration_seconds"] != "all" and
                             float(row["duration_seconds"]) == float(x["duration_seconds"])))), None)
            check(source is not None and method in matched, f"Trade-off source missing for {method}")
            if source and method in matched:
                expected = {"acc1": optional(source["acc1"]), "median_ms": matched[method]["median_ms"]}
        elif figure == "duration_vs_latency":
            subset = [x for x in real_times if x["method"] == method and float(x["duration_seconds"]) == float(row["duration_seconds"])]
            check(bool(subset), f"Duration curve raw source missing for {method}")
            if subset:
                expected = latency_stats(subset)
        elif figure == "billie_jean_comparison":
            source = next((x for x in predictions if x["method"] == method and x["tier"] == "demo" and
                           float(x["duration_seconds"]) == float(row["duration_seconds"]) and x["status"] == "success"), None)
            check(source is not None, f"Billie Jean source missing for {method}")
            if source:
                values = [float(x["latency_ms"]) for x in good_times if x["method"] == method and
                          x["clip_id"] == source["clip_id"] and x["stage"] == "file_to_result"]
                expected = {"bpm": float(source["bpm"]), "median_ms": statistics.median(values) if values else None}
        else:
            errors.append(f"Unknown plot-data figure: {figure}")
        for key, value in expected.items():
            equal(row[key], value, f"plot {figure}/{method} {key}")
    for name in ("latency_comparison", "accuracy_comparison", "accuracy_latency_tradeoff", "duration_vs_latency"):
        path = results / "figures" / f"{name}.png"
        check(path.exists() and path.stat().st_size > 1024, f"Required figure missing/empty: {name}")

    outcome = {"status": "passed" if not errors else "failed", "checks": checks,
               "clips": len(manifest), "prediction_records": len(predictions),
               "successful_predictions": sum(row["status"] == "success" for row in predictions),
               "measured_timing_records": len(good_times), "plot_records": len(plot),
               "matched_real_timing_clip_ids": sorted(common), "errors": errors, "warnings": warnings,
               "scope": "Input hashes and recorded results consistency; no inference rerun or independent accuracy replication"}
    (results / "validation.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    output = validate(args.results)
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output["status"] == "passed" else 1)
