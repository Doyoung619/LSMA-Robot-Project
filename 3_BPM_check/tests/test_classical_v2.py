"""Independent short-clip regression checks; no public-dataset labels used."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class ClassicalV2Tests(unittest.TestCase):
    def make_audio(self,bpm,seconds):
        import numpy as np
        sr=44100
        audio=np.zeros(round(sr*seconds),dtype=np.float32)
        rng=np.random.default_rng(808)
        t=np.arange(round(.025*sr))/sr
        pulse=rng.normal(size=len(t))*np.exp(-t/.004)
        for beat in np.arange(.31,seconds-.05,60/bpm):
            i=round(sr*beat);n=min(len(pulse),len(audio)-i)
            audio[i:i+n]+=.15*pulse[:n]
        return np.repeat(audio[:,None],2,axis=1),sr

    def test_short_clip_support_and_no_filename_tempo_leakage(self):
        import soundfile as sf
        from adapters.classical_v2 import ClassicalV2Estimator
        model=ClassicalV2Estimator(config={"band_weights":[.55,.3,.15],"density_exponent":.25,"tempo_prior_strength":.5})
        model.load()
        with tempfile.TemporaryDirectory() as folder:
            audio,sr=self.make_audio(132,8)
            paths=[Path(folder)/"tempo_60.wav",Path(folder)/"tempo_240.wav"]
            for path in paths:sf.write(path,audio,sr)
            values=[model.predict(p) for p in paths]
        self.assertAlmostEqual(values[0]["bpm"],132,delta=2.64)
        self.assertEqual(values[0],values[1])
        self.assertGreater(len(values[0]["beat_times"]),10)
        self.assertEqual(model.metadata["weight_bytes"],0)
        self.assertIsNone(model.metadata["pretrained_loaded"])

    def test_silence_is_not_reported_as_a_success(self):
        import numpy as np
        from adapters.classical_v2 import features,estimate
        with self.assertRaises(ValueError):
            estimate(features(np.zeros((44100*8,2),dtype=np.float32),44100))

    def test_compiled_scoring_preserves_complete_outputs(self):
        import soundfile as sf
        from adapters.classical_v2 import ClassicalV2Estimator
        from adapters.classical_v2_fast import ClassicalV2FastEstimator
        reference,compiled=ClassicalV2Estimator(),ClassicalV2FastEstimator()
        reference.load();compiled.load()
        with tempfile.TemporaryDirectory() as folder:
            for bpm in (66,153):
                audio,sr=self.make_audio(bpm,8)
                path=Path(folder)/f"click_{bpm}.wav"
                sf.write(path,audio,sr)
                a,b=reference.predict(path),compiled.predict(path)
                for key in ("bpm","beat_times","confidence","status","error"):
                    self.assertEqual(a[key],b[key])

    def test_compiled_classical_load_has_no_neural_framework_dependency(self):
        import subprocess
        code=("import sys; from adapters import get_estimator; "
              "m=get_estimator('OUR_V2_FAST'); m.load(); "
              "assert 'torch' not in sys.modules; assert 'tensorflow' not in sys.modules")
        subprocess.run([sys.executable,"-c",code],cwd=ROOT,check=True,capture_output=True,text=True)


if __name__ == "__main__":
    unittest.main()
