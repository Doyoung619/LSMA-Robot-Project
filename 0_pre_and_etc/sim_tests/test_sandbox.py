"""Checks against actual models and independently evaluated motion derivatives."""
import unittest
import numpy as np
from rita_sim.assets import ROBOT_IDS
from rita_sim.core import load, graph, Dance


class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environments = {robot: load(robot) for robot in ROBOT_IDS}

    def test_four_distinct_morphologies_load_without_contacts(self):
        counts = {'kinova': 6, 'g1': 29, 'misty': 5, 'go2': 12}
        for robot, env in self.environments.items():
            self.assertEqual(env.model.nq, counts[robot])
            for pose in env.poses:
                env.put(pose)
                self.assertEqual(env.data.ncon, 0)
                self.assertTrue(np.all(np.isfinite(env.data.xpos)))

    def test_allowed_transitions_obey_speed_and_acceleration_parameters(self):
        u = np.linspace(0, 1, 1001)[:, None]
        first = 30*u**2*(1-u)**2
        second = 60*u - 180*u**2 + 120*u**3
        for env in self.environments.values():
            adjacency, _ = graph(env, 90)
            for i, j in zip(*np.nonzero(adjacency)):
                delta = env.poses[j] - env.poses[i]
                velocity = first * delta / (60/90)
                acceleration = second * delta / (60/90)**2
                self.assertTrue(np.all(np.abs(velocity) <= env.speed + 1e-9))
                self.assertTrue(np.all(np.abs(acceleration) <= env.acceleration + 1e-9))

    def test_increasing_bpm_only_removes_edges_and_capacity(self):
        for env in self.environments.values():
            previous, previous_h = graph(env, 40)
            for bpm in range(41, 241):
                current, h = graph(env, bpm)
                self.assertTrue(np.all(current <= previous))
                self.assertLessEqual(h, previous_h + 1e-10)
                self.assertFalse(np.any(np.diag(current)))
                previous, previous_h = current, h

    def test_sequences_are_reproducible_and_use_only_allowed_edges(self):
        for env in self.environments.values():
            dance = Dance(env, 90, seed=7)
            clone = Dance(env, 90, seed=7)
            for t in np.linspace(0, 30, 101):
                q = dance.at(t)
                np.testing.assert_allclose(q, clone.at(t))
            result = dance.report(30)
            self.assertEqual(result['completed_transitions'], 45)
            self.assertGreater(result['visited_poses'], 2)
            for i, j in zip(dance.sequence[:-1], dance.sequence[1:]):
                self.assertTrue(dance.adjacency[i, j])

    def test_partial_transition_is_not_counted_as_arrived(self):
        dance = Dance(self.environments['kinova'], 60)
        dance.at(.9)
        report = dance.report(.9)
        self.assertEqual(report['completed_transitions'], 0)
        self.assertEqual(report['visited_poses'], 1)

    def test_impossible_tempo_is_reported_without_forcing_transitions(self):
        for env in self.environments.values():
            with self.assertRaisesRegex(ValueError, 'no sustainable'):
                Dance(env, 1e9)
            for bpm in (0, -1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    graph(env, bpm)


if __name__ == '__main__':
    unittest.main()
