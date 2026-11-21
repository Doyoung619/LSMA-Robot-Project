import sys
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
for p in (BASE,BASE.parent/"1_finite_version",BASE.parent/"2_infinite_version"):
    sys.path.insert(0,str(p))
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from rita_core.audio import fixture,analyze_audio,synthetic_analysis,DACClock
from rita_core.contracts import RobotProfile,BoundaryState
from rita_core.trajectory import make_segment,validate_segment,descriptor,descriptor_distance


def profile():
    return RobotProfile("test","mock",("j0","j1"),[0,0],[-.6,-.6],[.6,.6],[1.2,1.2],[6,6])


class CommonTests(unittest.TestCase):
    def test_continuous_control_hull_and_c2(self):
        p=profile()
        s=make_segment([0,1,2],[[0,0],[.15,-.1],[0,0]])
        self.assertTrue(validate_segment(s,p)["valid"])
        s.controls[0,2,0]=.9
        self.assertFalse(validate_segment(s,p)["valid"])

    def test_join_checks_and_no_micro_motion(self):
        p=profile()
        a=make_segment([0,1,2],[[0,0],[.15,-.1],[0,0]])
        b=make_segment([2,3,4],[[.02,0],[.15,-.1],[0,0]])
        self.assertIn("boundary_C2_discontinuity",validate_segment(b,p,a)["errors"])
        tiny=make_segment([0,1,2],[[0,0],[.001,-.001],[0,0]])
        self.assertIn("insufficient_excursion",validate_segment(tiny,p)["errors"])

    def test_descriptor_does_not_count_float_jitter(self):
        p=profile()
        a=make_segment([0,1,2],[[0,0],[.15,-.1],[0,0]])
        b=make_segment([0,1,2],[[0,0],[.15000001,-.1],[0,0]])
        self.assertLess(descriptor_distance(descriptor(a,p),descriptor(b,p)),1e-5)

    def test_contract_refuses_degrees_and_order_shape(self):
        with self.assertRaises(ValueError):
            RobotProfile("bad","mock",("a",),[0],[-1],[1],[1],[1],units="degree")
        with self.assertRaises(ValueError): BoundaryState(0,[0,1],[0],[0])

    def test_audio_clock_uses_dac_not_queued_samples(self):
        c=DACClock.__new__(DACClock);c.error=None;c.start_offset=7;c.dac_epoch=20
        c.position=999999
        c.stream=type("FakeStream",(),{"time":21.25})()
        self.assertAlmostEqual(c.clock(),8.25)


class AudioTests(unittest.TestCase):
    def test_manual_grid_bypasses_failed_automatic_fitter(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            t=np.arange(6*44100)/44100
            p=Path(tmp)/'tone.wav';sf.write(p,.1*np.sin(2*np.pi*220*t),44100)
            with patch('rita_robot.audio.analyze',side_effect=RuntimeError('automatic fit failed')):
                a=analyze_audio(p,Path(tmp)/'manual',bpm_override=120,beat_offset=.2)
            self.assertAlmostEqual(a.bpm,120)
            self.assertAlmostEqual(a.beats[0],.2)
            with self.assertRaisesRegex(ValueError,'beat-offset'):
                analyze_audio(p,Path(tmp)/'bad',bpm_override=120,beat_offset=-.2)

    def test_actual_audio_fixtures_60_120_180_240_and_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bpm in (60,120,180,240):
                with self.subTest(bpm=bpm):
                    p=fixture(Path(tmp)/f"click_{bpm}.wav",bpm,12)
                    a=analyze_audio(p,Path(tmp)/f"analysis_{bpm}")
                    self.assertAlmostEqual(a.bpm,bpm,delta=.05)
                    self.assertAlmostEqual(a.beats[0],.17,delta=.015)
                    self.assertTrue((Path(tmp)/f"analysis_{bpm}/tempo_diagnostic.png").exists())

    def test_tempo_change_manual_override_and_silence_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=fixture(Path(tmp)/"changing.wav",120,16,tempo_change=180)
            a=analyze_audio(p,Path(tmp)/"a")
            self.assertAlmostEqual(np.median(np.diff(a.beats[a.beats<7])),.5,delta=.01)
            self.assertAlmostEqual(np.median(np.diff(a.beats[a.beats>10])),1/3,delta=.01)
            override=analyze_audio(p,Path(tmp)/"b",bpm_override=90,beat_offset=.3)
            self.assertAlmostEqual(override.bpm,90,delta=.01)
            self.assertAlmostEqual(override.beats[0],.3)
            silent=fixture(Path(tmp)/"silence.wav",120,8,silence=True)
            with self.assertRaisesRegex(ValueError,"Silent"): analyze_audio(silent,Path(tmp)/"s")
            short=fixture(Path(tmp)/"short.wav",120,2)
            with self.assertRaisesRegex(ValueError,"four seconds"): analyze_audio(short,Path(tmp)/"short")


if __name__=="__main__": unittest.main()
