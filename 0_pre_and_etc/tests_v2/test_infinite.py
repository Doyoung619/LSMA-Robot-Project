"""Behavioral regression checks for continuous generation and safe suffix use."""
from pathlib import Path
import sys
import unittest

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[2]
for source in (EXPERIMENT / "0_pre_and_etc", EXPERIMENT / "2_infinite_version"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from rita_core.contracts import AudioAnalysis, BoundaryState, RobotProfile
from rita_core.trajectory import validate_segment
from rita_infinite import InfiniteConfig, InfinitePlanner, InfinitePlanningError


def fixture(bpm=120, speed=1.2, acceleration=6, joints=6):
    beats = np.arange(5000)*60/bpm+.16
    analysis = AudioAnalysis("synthetic", "fixture", 44100, float(beats[-1]+1),
                             beats, beats, np.ones(len(beats)), np.ones(len(beats)),
                             np.tile([1, .5, .8, .5], 1250), {"fixture": True})
    profile = RobotProfile("test", "mock", tuple(f"j{i}" for i in range(joints)),
                           np.zeros(joints), np.full(joints, -.8), np.full(joints, .8),
                           np.full(joints, speed), np.full(joints, acceleration))
    state = BoundaryState(analysis.beat_time(0), profile.neutral.copy(),
                          np.zeros(joints), np.zeros(joints))
    return analysis, profile, state


class InfinitePlannerTests(unittest.TestCase):
    def test_continuous_new_coefficients_C2_and_terminal_stop(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8})
        previous = None
        fingerprints, boundaries = [], []
        beat = 0
        for _ in range(5):
            seg = planner.next_segment(beat, state)
            self.assertTrue(validate_segment(seg, profile, previous)["valid"])
            self.assertEqual(seg.metadata["generation_kind"], "continuous")
            self.assertEqual(seg.metadata["horizon"], 2)
            self.assertGreaterEqual(seg.metadata["novelty_distance"]+1e-10,
                                    seg.metadata["novelty_threshold_used"])
            self.assertEqual(seg.metadata["fallback"], False)
            # Compare meaningful rounded coefficients; raw hash jitter is not a metric.
            fingerprints.append(tuple(np.round(seg.controls.ravel(), 3)))
            boundaries.append(np.max(np.abs(seg.end_state().dq)))
            terminal = planner._suffix[-1].end_state()
            self.assertTrue(np.allclose(terminal.q, profile.neutral))
            self.assertLess(np.max(np.abs(terminal.dq)), 1e-8)
            self.assertLess(np.max(np.abs(terminal.ddq)), 1e-8)
            beat, state, previous = seg.metadata["end_beat"], seg.end_state(), seg
        self.assertEqual(len(set(fingerprints)), 5)
        self.assertGreater(max(boundaries), .01)
        self.assertGreater(len(planner.stats()["rhythm_visits"]), 1)

    def test_failure_uses_suffix_once_then_requests_stop(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8})
        first = planner.next_segment(0, state)
        normal_count = planner.stats()["generated_segments"]
        planner.config.force_failure = True
        fallback = planner.next_segment(first.metadata["end_beat"], first.end_state())
        self.assertTrue(validate_segment(fallback, profile, first)["valid"])
        self.assertEqual(fallback.metadata["generation_kind"], "fallback_suffix")
        self.assertFalse(fallback.metadata["novelty_success"])
        self.assertFalse(fallback.metadata["coverage_success"])
        self.assertEqual(planner.stats()["generated_segments"], normal_count)
        with self.assertRaises(InfinitePlanningError) as error:
            planner.next_segment(fallback.metadata["end_beat"], fallback.end_state())
        self.assertTrue(error.exception.stop_required)
        self.assertFalse(error.exception.diagnostic["mathematical_infeasibility_proven"])

    def test_timeout_preserves_exact_reserved_suffix(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8})
        first = planner.next_segment(0, state)
        reserved = planner._suffix[0].controls.copy()
        planner.config.force_timeout = True
        fallback = planner.next_segment(first.metadata["end_beat"], first.end_state())
        self.assertTrue(np.array_equal(fallback.controls, reserved))
        self.assertEqual(fallback.metadata["fallback_reason"], "planning_timeout")
        self.assertEqual(planner.stats()["timeouts"], 1)

    def test_displaced_feedback_never_snaps_to_suffix(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8})
        first = planner.next_segment(0, state)
        wrong = first.end_state()
        wrong.q = wrong.q+.04
        planner.config.force_failure = True
        with self.assertRaises(InfinitePlanningError):
            planner.next_segment(first.metadata["end_beat"], wrong)

    def test_novelty_relaxation_and_witness_queue_are_explicit(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8,
                                                            "novelty_threshold": 2})
        seg = planner.next_segment(0, state)
        self.assertGreater(seg.metadata["novelty_relaxation_steps"], 0)
        self.assertFalse(seg.metadata["novelty_success"])
        stats = planner.stats()
        self.assertGreater(stats["feature_cells_ever_witnessed"], 0)
        self.assertEqual(stats["feature_cells_total"], 8)
        for witness in planner.witnesses.values():
            self.assertEqual(witness["boundary_key"][0], 0)
            self.assertTrue(witness["validated"])
        pending = planner.service[0]
        # If current search does not rediscover a target, it must not vanish.
        planner.service.appendleft("g0:never:a_witness")
        next_seg = planner.next_segment(seg.metadata["end_beat"], seg.end_state())
        self.assertFalse(next_seg.metadata["service_target_witnessed_now"])
        self.assertEqual(planner.service[0], "g0:never:a_witness")
        self.assertIn(pending, planner.service)

    def test_slow_actuator_adapts_but_keeps_meaningful_motion(self):
        analysis, profile, state = fixture(bpm=180, speed=.30, acceleration=1.2)
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 12})
        seg = planner.next_segment(0, state)
        self.assertGreater(seg.metrical_level, 0)
        self.assertTrue(seg.validation["valid"])
        self.assertGreaterEqual(seg.validation["min_excursion"], profile.min_excursion)
        self.assertLessEqual(seg.validation["max_stationary_seconds"], profile.max_stationary_seconds)

    def test_bad_contracts_and_horizon_configuration(self):
        for value in (0, 5):
            with self.assertRaises(ValueError):
                InfiniteConfig.from_value({"horizon": value})
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile)
        state.t += .1
        with self.assertRaises(ValueError):
            planner.next_segment(0, state)

    def test_hardware_stop_boundaries_keep_internal_flow(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8,
                                                            "stop_to_stop": True})
        seg = planner.next_segment(0, state)
        self.assertTrue(np.allclose(seg.end_state().dq, 0, atol=1e-8))
        self.assertTrue(np.allclose(seg.end_state().ddq, 0, atol=1e-8))
        internal = [np.max(np.abs(seg.at(float(t))[1])) for t in seg.knots[1:-1]]
        self.assertGreater(max(internal), .01)
        next_seg = planner.next_segment(seg.metadata["end_beat"], seg.end_state())
        self.assertTrue(validate_segment(next_seg, profile, seg)["valid"])

    def test_supported_horizons_and_disabled_novelty_are_honest(self):
        for horizon in (1, 4):
            analysis, profile, state = fixture()
            planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8,
                                                                "horizon": horizon,
                                                                "single_rhythm": True,
                                                                "no_coverage_novelty": True})
            seg = planner.next_segment(0, state)
            self.assertEqual(seg.metadata["horizon"], horizon)
            self.assertEqual(len(planner._suffix), horizon-1)
            self.assertEqual(seg.metadata["novelty_threshold_used"], 0)
            self.assertFalse(seg.metadata["novelty_policy_enabled"])
            self.assertEqual(seg.rhythm_id, "straight")

    def test_real_budget_exhaustion_without_suffix_requests_stop(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 1e-9})
        with self.assertRaises(InfinitePlanningError) as error:
            planner.next_segment(0, state)
        self.assertEqual(error.exception.reason, "planning_timeout")

    def test_hardware_terminal_hold_is_exact_and_join_certified(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8,
                                                            "stop_to_stop": True,
                                                            "terminal_hold_seconds": .20})
        first = planner.next_segment(0, state)
        q, dq, ddq = first.at(first.end-.1)
        self.assertTrue(np.allclose(q, first.end_state().q, atol=1e-9))
        self.assertTrue(np.allclose(dq, 0, atol=1e-9))
        self.assertTrue(np.allclose(ddq, 0, atol=1e-9))
        self.assertAlmostEqual(first.end, analysis.beat_time(first.metadata["end_beat"]))
        second = planner.next_segment(first.metadata["end_beat"], first.end_state())
        self.assertTrue(validate_segment(second, profile, first)["valid"])
        self.assertLessEqual(second.validation["joined_stationary_seconds"], profile.max_stationary_seconds)
        with self.assertRaises(ValueError):
            InfiniteConfig.from_value({"terminal_hold_seconds": .2})

    def test_session_end_drains_suffix_without_search_failure_or_diversity_credit(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8,
                                                            "horizon": 3})
        first = planner.next_segment(0, state)
        generated = planner.stats()["generated_segments"]
        previous = first
        for remaining in (1, 0):
            suffix = planner.terminal_suffix(previous.metadata["end_beat"], previous.end_state())
            self.assertTrue(validate_segment(suffix, profile, previous)["valid"])
            self.assertEqual(suffix.metadata["generation_kind"], "terminal_suffix")
            self.assertTrue(suffix.metadata["excluded_from_diversity"])
            self.assertTrue(suffix.metadata["recovery"])
            self.assertFalse(suffix.metadata["coverage_success"])
            self.assertEqual(suffix.metadata["reserved_suffix_remaining"], remaining)
            previous = suffix
        self.assertTrue(np.allclose(previous.end_state().q, profile.neutral))
        self.assertLess(np.max(np.abs(previous.end_state().dq)), 1e-8)
        self.assertIsNone(planner.terminal_suffix(previous.metadata["end_beat"], previous.end_state()))
        self.assertEqual(planner.stats()["generated_segments"], generated)
        self.assertEqual(planner.stats().get("search_failures", 0), 0)
        self.assertEqual(planner.stats().get("timeouts", 0), 0)

    def test_missing_flowing_terminal_suffix_is_an_explicit_error(self):
        analysis, profile, state = fixture()
        planner = InfinitePlanner(analysis, profile, config={"planning_budget_seconds": 8})
        seg = planner.next_segment(0, state)
        self.assertGreater(np.max(np.abs(seg.end_state().dq)), .001)
        planner._suffix = []
        with self.assertRaises(InfinitePlanningError) as error:
            planner.terminal_suffix(seg.metadata["end_beat"], seg.end_state())
        self.assertEqual(error.exception.reason, "terminal_suffix_unavailable")


if __name__ == "__main__":
    unittest.main()
