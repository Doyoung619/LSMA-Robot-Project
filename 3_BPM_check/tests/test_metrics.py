"""Independent numerical checks for the benchmark's scientific accounting."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.base import bpm_from_beats, result
from metrics import beat_metrics, tempo_metrics
from run_benchmark import summarize


class TempoMetricTests(unittest.TestCase):
    def test_absolute_error_preserves_octave_error(self):
        value = tempo_metrics(60, 120)
        self.assertEqual(value, {"absolute_error": 60, "acc1": 0, "acc2": 1})
        self.assertEqual(tempo_metrics(240, 120)["acc2"], 1)

    def test_only_single_half_or_double_equivalence_is_allowed(self):
        self.assertEqual(tempo_metrics(30, 120)["acc2"], 0)
        self.assertEqual(tempo_metrics(480, 120)["acc2"], 0)

    def test_two_percent_boundary_is_inclusive(self):
        for predicted in (98, 102):
            self.assertEqual(tempo_metrics(predicted, 100)["acc1"], 1)
        for predicted in (97.999, 102.001):
            self.assertEqual(tempo_metrics(predicted, 100)["acc1"], 0)

    def test_failure_counts_as_incorrect_with_no_fabricated_error(self):
        for predicted in (None, float("nan"), float("inf"), 0, -120):
            self.assertEqual(tempo_metrics(predicted, 120),
                             {"absolute_error": None, "acc1": 0, "acc2": 0})

    def test_unknown_reference_has_no_accuracy(self):
        self.assertEqual(tempo_metrics(117, None),
                         {"absolute_error": None, "acc1": None, "acc2": None})

    def test_invalid_reference_is_rejected(self):
        for reference in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                tempo_metrics(120, reference)


class BeatMetricTests(unittest.TestCase):
    def test_one_prediction_cannot_match_multiple_reference_beats(self):
        value = beat_metrics([1.03], [1.0, 1.06])
        self.assertAlmostEqual(value["beat_f1"], 2 / 3)

    def test_unmatched_beats_reduce_precision_and_recall(self):
        value = beat_metrics([.03, 1.06, 9], [0, 1, 2])
        self.assertAlmostEqual(value["beat_f1"], 2 / 3)
        self.assertAlmostEqual(value["beat_timing_error_ms"], 45)

    def test_missing_timestamps_do_not_become_zero_accuracy(self):
        self.assertEqual(beat_metrics(None, [0, 1]),
                         {"beat_f1": None, "beat_timing_error_ms": None})

    def test_derived_bpm_uses_finite_positive_consecutive_intervals(self):
        beats = [0, .5, 1, 1.8, 2.3, 2.3, float("nan"), 4, 4.5]
        self.assertAlmostEqual(bpm_from_beats(beats), 120)
        with self.assertRaises(ValueError):
            bpm_from_beats([1])
        with self.assertRaises(ValueError):
            bpm_from_beats([1, 1, 0, float("nan")])

    def test_result_rejects_invalid_success_bpm(self):
        for bpm in (None, float("nan"), 0):
            value = result("test", bpm)
            self.assertEqual(value["status"], "error")
            self.assertIsNone(value["bpm"])


class AggregationTests(unittest.TestCase):
    def test_nominal_30_second_group_includes_slightly_short_source(self):
        short = 29.995827664399094
        predictions = []
        clips = []
        for clip_id, actual, bpm in [("exact_30s", 30.0, 120.0), ("short_30s", short, 116.0)]:
            # Simulate existing worker files from before the nominal-duration
            # column was added: summarize must recover it from the manifest.
            predictions.append({"method": "Fixture", "clip_id": clip_id, "tier": "real",
                "track_id": clip_id, "duration_seconds": actual, "bpm": bpm,
                "reference_bpm": 120.0, "status": "success", **tempo_metrics(bpm, 120.0)})
            clips.append({"clip_id": clip_id, "duration_seconds": actual, "requested_duration_seconds": 30.0})
        timings = [{"method": "Fixture", "clip_id": "short_30s", "tier": "real",
                    "duration_seconds": short, "stage": "file_to_result", "repetition": 0,
                    "latency_ms": 30.0, "status": "success", "error": None}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "dataset_manifest.json").write_text(json.dumps(clips), encoding="utf-8")
            (folder / "worker_Fixture.json").write_text(json.dumps({
                "metadata": {"method": "Fixture", "status": "success"},
                "predictions": predictions, "timings": timings}), encoding="utf-8")
            summarize(folder, {})
            with (folder / "accuracy_summary.csv").open(newline="", encoding="utf-8") as stream:
                accuracy = [row for row in csv.DictReader(stream) if row["duration_seconds"] != "all"]
            with (folder / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                saved = {row["clip_id"]: row for row in csv.DictReader(stream)}
            with (folder / "latency_summary.csv").open(newline="", encoding="utf-8") as stream:
                latency = next(row for row in csv.DictReader(stream) if row["duration_seconds"] == "all")
        self.assertEqual(len(accuracy), 1)
        self.assertEqual(float(accuracy[0]["duration_seconds"]), 30.0)
        self.assertEqual(int(accuracy[0]["n_reference"]), 2)
        self.assertEqual(float(accuracy[0]["acc1"]), .5)
        self.assertEqual(float(accuracy[0]["mae_bpm"]), 2.0)
        self.assertEqual(float(saved["short_30s"]["duration_seconds"]), short)
        self.assertEqual(float(saved["short_30s"]["requested_duration_seconds"]), 30.0)
        self.assertAlmostEqual(float(latency["rtf"]), .030 / short)

    def test_failure_denominator_unknown_labels_and_latency_statistics(self):
        predictions = []
        for index, (bpm, reference, status) in enumerate([
                (120, 120, "success"), (60, 120, "success"),
                (None, 120, "error"), (117, None, "success")]):
            predictions.append({"method": "Fixture", "clip_id": str(index),
                "tier": "real", "track_id": str(index), "duration_seconds": 30,
                "bpm": bpm, "reference_bpm": reference, "status": status,
                **tempo_metrics(bpm, reference)})
        timings = [{"method": "Fixture", "clip_id": "0", "tier": "real",
                    "duration_seconds": 30, "stage": "file_to_result", "repetition": i,
                    "latency_ms": value, "status": "success", "error": None}
                   for i, value in enumerate((10, 20, 30))]
        timings.append({**timings[0], "status": "error", "latency_ms": 999})
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "worker_Fixture.json").write_text(json.dumps({
                "metadata": {"method": "Fixture", "status": "success"},
                "predictions": predictions, "timings": timings}), encoding="utf-8")
            summarize(folder, {"Fixture": 10})
            with (folder / "accuracy_summary.csv").open(newline="", encoding="utf-8") as stream:
                accuracy = next(row for row in csv.DictReader(stream) if row["duration_seconds"] == "all")
            with (folder / "latency_summary.csv").open(newline="", encoding="utf-8") as stream:
                latency = next(row for row in csv.DictReader(stream) if row["duration_seconds"] == "all")
        self.assertEqual(int(accuracy["n_total"]), 4)
        self.assertEqual(int(accuracy["n_reference"]), 3)
        self.assertAlmostEqual(float(accuracy["coverage"]), 2 / 3)
        self.assertAlmostEqual(float(accuracy["acc1"]), 1 / 3)
        self.assertAlmostEqual(float(accuracy["acc2"]), 2 / 3)
        self.assertAlmostEqual(float(accuracy["mae_bpm"]), 30)
        self.assertEqual(int(latency["n"]), 3)
        self.assertEqual(float(latency["median_ms"]), 20)
        self.assertAlmostEqual(float(latency["p95_ms"]), 29)
        self.assertAlmostEqual(float(latency["rtf"]), 20 / 1000 / 30)


if __name__ == "__main__":
    unittest.main()
