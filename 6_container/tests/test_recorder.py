"""Common rollout format, persistence, and reader contracts without hardware."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT / "0_pre_and_etc", ROOT / "0_pre_and_etc" / "tests_v2", ROOT / "6_container"):
    sys.path.insert(0, str(folder))

from rita_core.audio import synthetic_analysis
from rita_core.profiles import default_profile
from rita_container import RolloutRecorder, TelemetryError
from rita_container.reader import iter_records, load_run, recompute_run
from rita_core import runtime
from rita_core.adapters.mock import MockAdapter
from test_runtime import SmallPlanner


class RecorderContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def recorder(self, backend="mock", **options):
        return RolloutRecorder(self.root, profile=default_profile(backend), **options)

    def test_misty_and_kinova_share_reader_and_keep_joint_order(self):
        paths = []
        for backend in ("misty", "kinova"):
            profile = default_profile(backend)
            with self.recorder(backend, metadata={"backend": backend}) as log:
                log.record_sample("feedback", 0., profile.neutral,
                                  dq=np.zeros(profile.nj), feedback_monotonic=12.)
                paths.append(log.run_dir)
        self.assertNotEqual(paths[0], paths[1])
        for backend, path in zip(("misty", "kinova"), paths):
            run = load_run(path)
            self.assertEqual(run["manifest"]["schema_version"], 1)
            self.assertEqual(len(run["samples"][0]["q"]), default_profile(backend).nj)
            self.assertEqual(run["samples"][0]["source"], "feedback")
            self.assertIsNotNone(run["summary"])

    def test_reference_and_measured_channels_remain_separate(self):
        profile = default_profile()
        with self.recorder() as log:
            log.record_sample("reference", .1, profile.neutral)
            log.record_sample("feedback", .1, profile.neutral + .01,
                              feedback_monotonic=100.1, fresh=False,
                              metadata={"reason": "injected stale feedback"})
            log.record_event("execution_failed", music_time=.1, error="stale")
        run = load_run(log.run_dir)
        self.assertEqual([s["source"] for s in run["samples"]], ["reference", "feedback"])
        self.assertFalse(run["samples"][1]["fresh"])
        self.assertEqual(run["samples"][1]["metadata"]["reason"], "injected stale feedback")

    def test_exception_retains_partial_log_and_failure_status(self):
        profile = default_profile()
        with self.assertRaisesRegex(RuntimeError, "injected disconnect"):
            with self.recorder() as log:
                log.record_sample("feedback", .2, profile.neutral)
                raise RuntimeError("injected disconnect")
        run = load_run(log.run_dir)
        self.assertEqual(len(run["samples"]), 1)
        saved = json.dumps({"manifest": run["manifest"], "summary": run["summary"]})
        self.assertIn("failed", saved)
        self.assertIn("injected disconnect", saved)

    def test_reader_recovers_complete_lines_from_interrupted_tail(self):
        file = self.root / "interrupted.jsonl"
        file.write_text('{"index": 1}\n{"index": 2}\n{"index":', encoding="utf-8")
        self.assertEqual(list(iter_records(file)), [{"index": 1}, {"index": 2}])

    def test_invalid_vector_does_not_become_a_valid_sample(self):
        profile = default_profile()
        with self.recorder() as log:
            for values in ([0.] * (profile.nj + 1), [float("nan")] * profile.nj):
                with self.assertRaises((ValueError, TypeError)):
                    log.record_sample("feedback", .1, values)
            with self.assertRaises((ValueError, TypeError)):
                log.record_sample("", .1, profile.neutral)
        self.assertEqual(load_run(log.run_dir)["samples"], [])

    def test_custom_metric_and_replay_keep_raw_samples(self):
        profile = default_profile()
        def custom(samples, robot, analysis):
            return {"observations": len(samples), "tag": "custom_metric_test"}
        with self.recorder(analysis=synthetic_analysis(duration=15),
                           metrics={"example": custom}) as log:
            for t in np.arange(0., 3., profile.command_period):
                log.record_sample("feedback", float(t),
                                  profile.neutral + .04 * np.sin(2 * np.pi * t),
                                  feedback_monotonic=100. + float(t))
        run = load_run(log.run_dir)
        self.assertIn("custom_metric_test", json.dumps(run["summary"]))
        replay = recompute_run(log.run_dir)
        self.assertIsInstance(replay, dict)
        self.assertEqual(len(load_run(log.run_dir)["samples"]), len(run["samples"]))

    def test_runtime_records_completed_reference_and_mock_observations(self):
        profile = default_profile()
        analysis = synthetic_analysis(duration=30)
        report = runtime.run_session(analysis, profile, seconds=3,
            planner=SmallPlanner(analysis, profile), container_root=self.root)
        self.assertTrue(report["run"]["completed"], report["run"])
        run = load_run(report["run"]["container"]["path"])
        self.assertEqual({s["source"] for s in run["samples"]},
                         {"reference", "simulated_feedback"})
        kinds = [event["kind"] for event in run["events"]]
        self.assertEqual(kinds.count("segment_planned"), 2)
        self.assertEqual(kinds.count("segment_dispatched"), 2)
        self.assertEqual(kinds.count("segment_completed"), 2)
        self.assertEqual(run["summary"]["status"], "completed")

    def test_runtime_preserves_stale_failure_sample_and_partial_segment(self):
        profile = default_profile()
        analysis = synthetic_analysis(duration=30)
        adapter = MockAdapter(profile)
        original = adapter.read_state
        calls = 0
        def stale_after_start():
            nonlocal calls
            calls += 1
            state = original()
            if calls >= 3:
                state.fresh = False
                state.metadata["injected"] = "stale_after_dispatch"
            return state
        with patch.object(adapter, "read_state", side_effect=stale_after_start), \
             patch.object(runtime, "make_adapter", return_value=adapter):
            report = runtime.run_session(analysis, profile, seconds=3,
                planner=SmallPlanner(analysis, profile), container_root=self.root)
        self.assertFalse(report["run"]["completed"])
        self.assertTrue(adapter.stopped)
        self.assertEqual(report["segments_executed"], 0)
        run = load_run(report["run"]["container"]["path"])
        self.assertEqual(run["summary"]["status"], "failed")
        offending = [s for s in run["samples"] if not s["fresh"]]
        self.assertEqual(len(offending), 1)
        self.assertEqual(offending[0]["metadata"]["injected"], "stale_after_dispatch")
        kinds = [event["kind"] for event in run["events"]]
        self.assertIn("segment_planned", kinds)
        self.assertIn("segment_dispatched", kinds)
        self.assertNotIn("segment_completed", kinds)
        self.assertLess(kinds.index("rollout_error"), kinds.index("stop_result"))
        self.assertIn("cleanup_result", kinds)

    def test_full_queue_reports_failure_instead_of_silent_loss(self):
        profile = default_profile()
        log = self.recorder(queue_size=1)
        entered, release = threading.Event(), threading.Event()
        original = log._snapshot
        def slow_snapshot(*args, **kwargs):
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test worker did not resume")
            return original(*args, **kwargs)
        try:
            with patch.object(log, "_snapshot", side_effect=slow_snapshot):
                log.record_sample("feedback", 0., profile.neutral)
                self.assertTrue(entered.wait(timeout=5))
                log.record_sample("feedback", .1, profile.neutral)
                with self.assertRaisesRegex(TelemetryError, "queue full"):
                    log.record_sample("feedback", .2, profile.neutral)
                release.set()
                with self.assertRaisesRegex(TelemetryError, "queue full"):
                    log.close()
        finally:
            release.set()
            try:
                log.close()
            except TelemetryError:
                pass
        run = load_run(log.run_dir)
        self.assertEqual(run["summary"]["status"], "failed")
        self.assertIn("queue full", run["summary"]["telemetry_error"])


if __name__ == "__main__":
    unittest.main()
