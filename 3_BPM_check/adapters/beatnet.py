"""Optional BeatNet adapter; dependency failures are isolated by the runner."""
from .base import BPMEstimator, result, bpm_from_beats
from .model_cache import configure_torch_threads


class BeatNetEstimator(BPMEstimator):
    name = "BeatNet"

    def load(self):
        import torch
        from BeatNet.BeatNet import BeatNet as BeatNetModel
        from pathlib import Path
        import BeatNet
        import hashlib
        import importlib.metadata
        configure_torch_threads(self.threads)
        # "online" consumes the whole file immediately; "offline" requires DBN.
        self.model = BeatNetModel(1, mode="online", inference_model="PF", plot=[], thread=False,
                                  device=self.device)
        files = list(Path(BeatNet.__file__).parent.rglob("model_1_weights.pt"))
        self.metadata = {"checkpoint": "BeatNet pretrained model 1", "version": importlib.metadata.version("BeatNet"),
                         "source_url": "https://github.com/mjhydri/BeatNet", "license": "CC BY 4.0 (official repository)",
                         "device": "cpu", "output_type": "beats; BPM derived via median IBI",
                         "training_overlap": "Model 1: GTZAN-out training, per official source; other training corpora not independently verified",
                         "pretrained_loaded": True, "weight_bytes": sum(p.stat().st_size for p in files),
                         "sha256": ";".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in files)}

    def predict(self, audio_path):
        output = self.model.process(str(audio_path))
        beats = output[:, 0]
        return result(self.name, bpm_from_beats(beats), beat_times=beats)
