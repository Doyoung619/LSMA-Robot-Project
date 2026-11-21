"""Video pose arrivals are anchored to the measured audio beat phase."""
import json
import unittest
import numpy as np
from rita_sim.assets import ROOT, ROBOT_IDS
from rita_sim.core import load, Dance
from render_dance_video import beat_state


class VideoSyncTests(unittest.TestCase):
    def test_audio_phase_and_pose_arrivals_for_each_morphology(self):
        info = json.loads((ROOT/'data/analysis.json').read_text(encoding='utf-8'))
        for robot in ROBOT_IDS:
            env = load(robot)
            dance = Dance(env, info['bpm'], 10)
            first, period = info['first_beat_seconds'], info['period_seconds']
            for t in (0., first/2, first):
                elapsed, index, _, _, completed = beat_state(dance, t, first)
                self.assertEqual(index, 0)
                np.testing.assert_allclose(dance.at(elapsed), env.poses[dance.sequence[0]], atol=1e-12)
                self.assertEqual(len(completed), 1)
            for n in range(1, 118):
                t = first+n*period
                elapsed, index, current, _, completed = beat_state(dance, t, first)
                self.assertEqual(index, n)
                self.assertEqual(current, dance.sequence[n])
                self.assertEqual(len(completed), n+1)
                np.testing.assert_allclose(dance.at(elapsed), env.poses[current], atol=1e-10)

    def test_music_time_mapping_is_independent_of_video_clip_start(self):
        info = json.loads((ROOT/'data/analysis.json').read_text(encoding='utf-8'))
        env = load('kinova')
        full = Dance(env, info['bpm'], 10)
        trimmed = Dance(env, info['bpm'], 10)
        for t in np.arange(0, 18, .17):
            beat_state(full, t, info['first_beat_seconds'])
        a = beat_state(full, 18., info['first_beat_seconds'])
        b = beat_state(trimmed, 18., info['first_beat_seconds'])
        np.testing.assert_allclose(full.at(a[0]), trimmed.at(b[0]))
        self.assertEqual(a[1:], b[1:])


if __name__ == '__main__':
    unittest.main()
