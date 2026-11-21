"""Expanded motion remains bounded and arrives on musical beat boundaries."""
from types import SimpleNamespace
import unittest
import numpy as np
from rita_sim.assets import ROBOT_IDS
from rita_sim.core import load
from rita_sim.expressive import PhraseDance
from render_dance_video import beat_state


class ExpressiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.envs = {r: load(r, 'expressive') for r in ROBOT_IDS}
        cls.bpm = 117.02546482191966

    def test_larger_pose_ranges_with_the_same_motion_limits(self):
        for robot, env in self.envs.items():
            original = load(robot)
            self.assertEqual(len(env.poses), 16)
            gain = np.linalg.norm(np.ptp(env.poses, axis=0))/np.linalg.norm(np.ptp(original.poses, axis=0))
            self.assertGreater(gain, 1.8)
            np.testing.assert_allclose(env.speed, original.speed)
            np.testing.assert_allclose(env.acceleration, original.acceleration)

    def test_all_sixteen_poses_visited_in_the_video(self):
        for i, env in enumerate(self.envs.values()):
            dance = PhraseDance(env, self.bpm, 10+i)
            result = dance.report(60-.33592642533486466)
            self.assertEqual(result['visited_poses'], 16)
            self.assertEqual(result['reachable_pose_count'], 16)
            self.assertTrue(set(np.diff(result['arrival_beats'])).issubset({1, 2, 4}))

    def test_transition_derivatives_and_audio_phase(self):
        u = np.linspace(0, 1, 1001)[:, None]
        first = 30*u*u*(1-u)**2
        second = 60*u-180*u*u+120*u**3
        for env in self.envs.values():
            dance = PhraseDance(env, self.bpm, 10)
            dance.report(60)
            for i, j in zip(*np.nonzero(dance.adjacency)):
                duration = dance.durations[i, j]*dance.period
                delta = env.poses[j]-env.poses[i]
                self.assertTrue(np.all(np.abs(first*delta/duration) <= env.speed+1e-9))
                self.assertTrue(np.all(np.abs(second*delta/duration**2) <= env.acceleration+1e-9))
            for index, beat in enumerate(dance.arrival_beats[:-1]):
                t = .33592642533486466+beat*dance.period
                elapsed, actual_index, current, _, _ = beat_state(dance, t, .33592642533486466)
                self.assertEqual(actual_index, index)
                np.testing.assert_allclose(dance.at(elapsed), env.poses[current], atol=1e-9)

    def test_capacity_accounts_for_multi_beat_duration(self):
        dance = PhraseDance.__new__(PhraseDance)
        dance.env = SimpleNamespace(poses=np.zeros((4, 1)))
        dance.adjacency = ~np.eye(4, dtype=bool)
        dance.durations = dance.adjacency.astype(int)*2
        self.assertAlmostEqual(dance._capacity(), np.log(3)/2, places=10)

    def test_sequence_is_independent_of_sampling_density(self):
        env = self.envs['g1']
        dense, sparse = PhraseDance(env, self.bpm, 10), PhraseDance(env, self.bpm, 10)
        for t in np.linspace(0, 60, 2001):
            dense.at(t)
        np.testing.assert_allclose(dense.at(60), sparse.at(60))
        self.assertEqual(dense.report(60), sparse.report(60))


if __name__ == '__main__':
    unittest.main()
