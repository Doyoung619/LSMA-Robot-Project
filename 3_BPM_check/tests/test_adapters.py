"""Contract smoke test executing the unchanged detector on independent audio."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import get_estimator
from adapters.base import BPMEstimator
from adapters.existing import SOURCE


class ExistingAdapterTests(unittest.TestCase):
    def test_classical_registry_does_not_import_neural_framework(self):
        command = [sys.executable, "-c", (
            "import sys; from adapters import get_estimator; "
            "estimator=get_estimator('OUR_EXISTING'); "
            "assert 'torch' not in sys.modules; assert 'tensorflow' not in sys.modules")]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

    def test_original_detector_contract_on_known_132_bpm_audio(self):
        import numpy as np
        import soundfile as sf
        sample_rate, bpm = 44100, 132.0
        audio = np.zeros(24 * sample_rate, dtype=np.float32)
        rng = np.random.default_rng(4)
        samples = np.arange(round(.025 * sample_rate))
        pulse = rng.normal(size=len(samples)) * np.exp(-samples / (sample_rate * .004))
        for beat in np.arange(.16, 23.8, 60 / bpm):
            first = round(beat * sample_rate)
            audio[first:first + len(pulse)] += pulse * .2
        before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        estimator = get_estimator("OUR_EXISTING", device="cpu", threads=1)
        self.assertIsInstance(estimator, BPMEstimator)
        estimator.load()
        try:
            self.assertEqual(Path(estimator.analyze.__code__.co_filename), SOURCE)
            self.assertEqual(estimator.metadata["sha256"], before)
            with tempfile.TemporaryDirectory() as directory:
                wav = Path(directory) / "independent_click.wav"
                sf.write(wav, np.repeat(audio[:, None], 2, axis=1), sample_rate)
                value = estimator.predict(wav)
            self.assertEqual(set(value), {"method", "bpm", "beat_times", "confidence", "status", "error"})
            self.assertEqual(value["status"], "success")
            self.assertEqual(value["method"], "OUR_EXISTING")
            self.assertAlmostEqual(value["bpm"], bpm, delta=.4)
            beats = np.asarray(value["beat_times"])
            self.assertGreater(len(beats), 45)
            self.assertTrue(np.isfinite(beats).all())
            self.assertTrue((np.diff(beats) > 0).all())
            self.assertGreaterEqual(beats[0], 0)
            self.assertLess(beats[-1], 24)
            self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), before)
        finally:
            estimator.directory.cleanup()


if __name__ == "__main__":
    unittest.main()
