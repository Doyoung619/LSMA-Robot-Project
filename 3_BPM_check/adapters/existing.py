"""Call the actual robot detector unchanged, including its JSON/NPZ writes."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from .base import BPMEstimator, result

PROJECT_SOURCE = Path(__file__).resolve().parents[2] / "0_pre_and_etc/rita_robot/audio.py"
SNAPSHOT = Path(__file__).resolve().parents[1] / "reference/rita_robot_audio.py"
SOURCE = PROJECT_SOURCE if PROJECT_SOURCE.exists() else SNAPSHOT


class ExistingEstimator(BPMEstimator):
    name = "OUR_EXISTING"

    def load(self):
        if self.device != "cpu":
            raise ValueError("The original classical detector supports CPU only")
        spec = importlib.util.spec_from_file_location("rita_original_audio", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.analyze = module.analyze
        self.directory = tempfile.TemporaryDirectory(prefix="rita_bpm_")
        self.output = Path(self.directory.name) / "analysis.json"
        self.metadata = {
            "pretrained_loaded": None, "weight_bytes": 0,
            "checkpoint": "unchanged rita_robot.audio.analyze", "version": "repository source",
            "source_url": "../0_pre_and_etc/rita_robot/audio.py", "license": "project source",
            "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "actual_source": str(SOURCE), "using_snapshot_fallback": SOURCE == SNAPSHOT,
            "configuration": {"input": "44100 Hz stereo WAV", "downsample_factor": 4,
                "stft_nfft": 1024, "hop": 128, "bands_hz": [[35,180],[180,2000],[2000,5000]],
                "band_weights": [.55,.30,.15], "median_filter_frames": 43,
                "candidate_bpm_range": [70,180], "top_candidates": 8,
                "repeated_lags_weights": [[1,1],[2,.6],[4,.3]],
                "grid_period_steps": 121, "grid_phase_steps": 48,
                "refine_iterations": 4, "minimum_supported_beats": 16,
                "peak_match_seconds": .085, "loss": "soft_l1", "f_scale": .015},
            "timing_scope": "WAV read + unmodified analyze + JSON/NPZ writes + beat readback",
            "preloaded_support": "unavailable: original interface accepts file path only"}

    def predict(self, audio_path):
        import numpy as np
        info = self.analyze(audio_path, self.output)
        with np.load(self.output.with_suffix(".npz")) as archive:
            beats = archive["beats"].tolist()
        return result(self.name, info["bpm"], beats)
