"""Official DeepRhythm predictor using a mandatory pre-downloaded checkpoint."""
from .base import BPMEstimator, result
from .model_cache import MODEL_DIR, checkpoint_metadata, configure_torch_threads


class DeepRhythmEstimator(BPMEstimator):
    name = "DeepRhythm"

    def load(self):
        import torch
        import deeprhythm.model.predictor as predictor_module
        configure_torch_threads(self.threads)
        self.metadata = checkpoint_metadata(
            "deeprhythm-0.7.pth", "deeprhythm",
            "https://raw.githubusercontent.com/bleugreen/deeprhythm/main/weights/deeprhythm-0.7.pth",
            "AGPL-3.0", output_type="global BPM", device=self.device,
            training_overlap="Training corpus not fully documented; exploratory GTZAN evaluation",
            preprocessing="Official librosa 22050 Hz; complete nonoverlapping 8 s chunks; trailing remainder discarded")
        # Predictor has no checkpoint-path argument. Override only its path resolver,
        # then restore it; model architecture and strict state loading are untouched.
        resolver = predictor_module.get_weights
        try:
            predictor_module.get_weights = lambda **kwargs: str(MODEL_DIR / "deeprhythm-0.7.pth")
            self.model = predictor_module.DeepRhythmPredictor(device=self.device, quiet=True)
        finally:
            predictor_module.get_weights = resolver
        self.metadata["pretrained_loaded"] = True
        self.metadata["parameter_count"] = sum(p.numel() for p in self.model.model.parameters())

    def predict(self, audio_path):
        bpm, confidence = self.model.predict(str(audio_path), include_confidence=True)
        return result(self.name, bpm, confidence=confidence)

    def predict_preloaded(self, audio, sr):
        import numpy as np
        import librosa
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        bpm, confidence = self.model.predict_from_audio(audio, 22050, include_confidence=True)
        return result(self.name, bpm, confidence=confidence)
