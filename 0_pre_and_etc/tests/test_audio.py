import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from rita_robot.audio import analyze


class TempoTests(unittest.TestCase):
    def test_unknown_synthetic_click_track(self):
        # Independent input, deliberately different from Billie Jean's tempo.
        sr=44100; bpm=132.; period=60/bpm; offset=.16
        audio=np.zeros(24*sr,dtype=np.float32)
        rng=np.random.default_rng(4)
        pulse=rng.normal(size=round(.025*sr))*np.exp(-np.arange(round(.025*sr))/(sr*.004))
        for beat in np.arange(offset,23.8,period):
            i=round(beat*sr);audio[i:i+len(pulse)]+=pulse*.2
        with tempfile.TemporaryDirectory() as directory:
            wav=Path(directory)/'click.wav';sf.write(wav,audio,sr)
            result=analyze(wav,Path(directory)/'analysis.json')
        self.assertAlmostEqual(result['bpm'],bpm,delta=.4)
        self.assertLess(result['median_nearest_onset_error_ms'],20)


if __name__=='__main__':unittest.main()
