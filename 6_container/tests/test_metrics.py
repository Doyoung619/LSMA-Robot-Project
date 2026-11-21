"""Metric contracts for real rollout timestamps and shared offline definitions."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rita_container.metrics import DEFAULT_CONFIG, evaluate_samples, evaluate_trajectory


def profile():
    return SimpleNamespace(
        joint_names=("a", "b"), neutral=np.zeros(2), lower=-np.ones(2),
        upper=np.ones(2), vmax=np.ones(2) * 5, amax=np.ones(2) * 20,
        command_period=.1, units="rad", workspace={},
    )


def records(times, stationary=False):
    return [{"music_time": float(t), "q": [0 if stationary else .2 * np.sin(2 * np.pi * t), 0],
             "fresh": True} for t in times]


class RolloutMetricContracts(unittest.TestCase):
    def setUp(self):
        self.profile = profile()
        self.analysis = {"beats": np.arange(0, 61, .5).tolist()}
        self.times = np.arange(0, 20, .1)

    def test_fixed_defaults_match_benchmark1(self):
        config = Path(__file__).resolve().parents[2] / "4_Benchmark1/configs/benchmark.json"
        self.assertEqual(DEFAULT_CONFIG, json.loads(config.read_text(encoding="utf-8"))["metrics"])

    def test_stationary_motion_has_no_perfect_sync(self):
        result = evaluate_samples(records(self.times, stationary=True), self.profile, self.analysis)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["metrics"]["alignment_valid"])
        self.assertIsNone(result["metrics"]["permitted_grid_median_error_ms"])
        self.assertEqual(result["sync_reason"], "insufficient_meaningful_motion_accents")
        self.assertEqual(result["metrics"]["stationary_fraction"], 1)

    def test_periodic_motion_recurs_and_uses_physical_joint_range(self):
        samples = records(self.times)
        large = evaluate_samples(samples, self.profile, self.analysis)
        tiny = evaluate_samples([{**s, "q": [s["q"][0] * 1e-6, 0]} for s in samples],
                                self.profile, self.analysis)
        self.assertGreater(large["metrics"]["recurrence_rate"], .99)
        self.assertTrue(large["metrics"]["alignment_valid"])
        self.assertFalse(tiny["metrics"]["alignment_valid"])
        self.assertGreater(np.max(large["series"]["descriptors"]),
                           np.max(tiny["series"]["descriptors"]) * 1000)

    def test_irregular_timestamps_resample_without_silent_reordering(self):
        t = self.times.copy()
        t[1::2] += .015
        result = evaluate_samples(records(t), self.profile, self.analysis)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["quality"]["resampled_count"], len(self.times))
        samples = records(self.times)
        samples[5], samples[6] = samples[6], samples[5]
        bad = evaluate_samples(samples, self.profile, self.analysis)
        self.assertEqual(bad["status"], "invalid_data")
        self.assertEqual(bad["reason"], "backward_music_timestamp")
        self.assertEqual(bad["metrics"], {})

    def test_duplicate_sensor_updates_are_not_counted_as_new_measurements(self):
        samples = records(self.times)
        for i, sample in enumerate(samples):
            sample["feedback_monotonic"] = 50 + (i // 2) * .2
        result = evaluate_samples(samples, self.profile, self.analysis)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["quality"]["sample_count"], len(samples) // 2)
        self.assertEqual(result["quality"]["duplicate_count"], len(samples) // 2)
        samples[10]["feedback_monotonic"] = 49
        bad = evaluate_samples(samples, self.profile, self.analysis)
        self.assertEqual(bad["reason"], "backward_feedback_timestamp")

    def test_no_audio_keeps_diversity_but_never_reports_sync(self):
        result = evaluate_samples(records(self.times), self.profile)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["diversity_reference"], "fixed_2_second_windows")
        self.assertEqual(result["sync_reason"], "no_audio_reference")
        self.assertGreater(result["metrics"]["recurrence_rate"], .99)
        self.assertFalse(result["metrics"]["alignment_valid"])
        self.assertIsNone(result["metrics"]["original_beat_coverage"])
        self.assertIsNone(result["metrics"]["permitted_grid_median_error_ms"])
        json.dumps(result, allow_nan=False)

    def test_preplayback_and_postplayback_samples_trim_to_audio_overlap(self):
        # Long waiting times outside the audio reference do not create a false
        # gap inside the measured dance window.
        samples = records([-10, -9, *self.times, 62, 63])
        result = evaluate_samples(samples, self.profile, self.analysis)
        direct = evaluate_samples(records(self.times), self.profile, self.analysis)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["quality"]["out_of_reference_sample_count"], 4)
        self.assertEqual(result["quality"]["evaluation_sample_count"], len(self.times))
        self.assertEqual(result["metrics"], direct["metrics"])

    def test_short_or_absent_audio_overlap_returns_no_scores(self):
        result = evaluate_samples(records([-2, -1, 0, .1]), self.profile, self.analysis)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["quality"]["out_of_reference_sample_count"], 2)
        self.assertEqual(result["sync_reason"], "insufficient_audio_overlap")
        self.assertEqual(result["metrics"], {})

    def test_host_clock_roundoff_does_not_drop_complete_beat_windows(self):
        times = np.arange(0, 20.01, .1)
        exact = evaluate_samples(records(times), self.profile, self.analysis)
        offset_samples = records(times + 1e-11)
        noisy = evaluate_samples(offset_samples, self.profile, self.analysis)
        self.assertEqual(noisy["status"], "ok")
        self.assertEqual(noisy["metrics"]["window_count"], exact["metrics"]["window_count"])
        self.assertEqual(noisy["metrics"]["nonadjacent_pair_count"], exact["metrics"]["nonadjacent_pair_count"])
        self.assertGreater(noisy["quality"]["timestamp_snap_count"], 0)
        self.assertEqual(offset_samples[0]["music_time"], 1e-11)

    def test_gaps_and_stale_feedback_cannot_produce_valid_scores(self):
        samples = records(self.times)
        for sample in samples[40:50]:
            sample["fresh"] = False
        result = evaluate_samples(samples, self.profile, self.analysis)
        self.assertEqual(result["status"], "invalid_data")
        self.assertEqual(result["reason"], "sample_gap_exceeds_limit")
        self.assertEqual(result["quality"]["stale_count"], 10)
        self.assertEqual(result["metrics"], {})

    def test_absent_acceleration_limits_do_not_report_placeholder_violations(self):
        self.profile.workspace["unavailable_limits"] = ["acceleration"]
        self.profile.amax = np.ones(2) * .001
        result = evaluate_samples(records(self.times), self.profile, self.analysis)
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["metrics"]["acceleration_violation_rate"])
        self.assertIsNone(result["metrics"]["acceleration_max_normalized_excess"])
        self.assertIsInstance(result["metrics"]["jerk_rms_rad_s3"], float)

    def test_optional_tracking_and_receipt_age_use_actual_sample_metadata(self):
        samples = records(self.times)
        for sample in samples:
            sample["metadata"] = {"reference_q": [sample["q"][0] - .03, -.04]}
            sample["feedback_monotonic"] = sample["music_time"] + 10
            sample["observed_monotonic"] = sample["feedback_monotonic"] + .05
        result = evaluate_samples(samples, self.profile, self.analysis)
        metrics = result["metrics"]
        self.assertEqual(metrics["tracking_sample_count"], len(samples))
        self.assertAlmostEqual(metrics["tracking_rmse_rad"], np.sqrt((.03 ** 2 + .04 ** 2) / 2))
        self.assertAlmostEqual(metrics["tracking_max_abs_error_rad"], .04)
        self.assertAlmostEqual(metrics["feedback_age_seconds_median"], .05)
        self.assertEqual(metrics["feedback_age_sample_count"], len(samples))
        self.assertIsNone(metrics["physical_tracking"])

    def test_scalar_tracking_error_cannot_invent_full_joint_rmse(self):
        samples = records(self.times)
        for sample in samples:
            sample["metadata"] = {"tracking_max_abs_error_rad": .02}
        result = evaluate_samples(samples, self.profile)
        self.assertAlmostEqual(result["metrics"]["tracking_max_abs_error_rad"], .02)
        self.assertIsNone(result["metrics"]["tracking_rmse_rad"])
        self.assertIsNone(result["metrics"]["feedback_age_seconds_median"])
        self.assertEqual(result["metrics"]["tracking_reference_pair_count"], 0)

    def test_declared_accents_accept_segment_metadata(self):
        samples = records(self.times)
        samples[0]["metadata"] = {"declared_accent_times": np.arange(0, 20, .5).tolist()}
        result = evaluate_samples(samples, self.profile, self.analysis)
        self.assertIsNotNone(result["metrics"]["declared_rhythm_event_coverage"])

    def test_short_or_nonfinite_streams_remain_explicitly_unscored(self):
        short = evaluate_samples(records([0, .1, .2]), self.profile, self.analysis)
        self.assertEqual(short["status"], "insufficient_data")
        self.assertEqual(short["metrics"], {})
        samples = records(self.times)
        samples[20]["q"][0] = float("nan")
        bad = evaluate_samples(samples, self.profile, self.analysis)
        self.assertEqual(bad["status"], "invalid_data")
        json.dumps(bad, allow_nan=False)

    def test_uniform_samples_match_direct_shared_metric_evaluation(self):
        samples = records(self.times)
        direct, _ = evaluate_trajectory(self.times, np.asarray([s["q"] for s in samples]),
                                        self.analysis, self.profile)
        sampled = evaluate_samples(samples, self.profile, self.analysis)["metrics"]
        for key in ("recurrence_rate", "window_distance_mean", "stationary_fraction",
                    "permitted_grid_median_error_ms", "jerk_rms_rad_s3"):
            self.assertAlmostEqual(sampled[key], direct[key], places=10)


if __name__ == "__main__":
    unittest.main()
