"""Official Beat This! small0, minimal decoding; BPM is derived from beats."""
from .base import BPMEstimator, result, bpm_from_beats
from .model_cache import MODEL_DIR, checkpoint_metadata, configure_torch_threads


class BeatThisEstimator(BPMEstimator):
    name = "BeatThis"

    def load(self):
        import torch
        from beat_this.inference import File2Beats
        configure_torch_threads(self.threads)
        self.metadata = checkpoint_metadata(
            "beat_this-small0.ckpt", "beat-this",
            "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt",
            "MIT (code and published weights, per official repository)", device=self.device,
            output_type="beats; BPM derived as 60 / median positive finite inter-beat interval",
            training_overlap="small0 excludes GTZAN per official model table", dbn=False,
            preprocessing="Official soundfile/torchaudio loader, soxr to 22050 Hz, log-mel spectrogram")
        self.model = File2Beats(checkpoint_path=str(MODEL_DIR / "beat_this-small0.ckpt"),
                               device=self.device, dbn=False)
        self.metadata["pretrained_loaded"] = True
        self.metadata["parameter_count"] = sum(p.numel() for p in self.model.model.parameters())

    def _result(self, outputs):
        beats, downbeats = outputs
        try:
            bpm = bpm_from_beats(beats)
        except ValueError as exc:
            return result(self.name, beat_times=beats, status="error", error=str(exc))
        answer = result(self.name, bpm, beat_times=beats)
        answer["downbeat_times"] = [float(x) for x in downbeats]
        return answer

    def predict(self, audio_path):
        return self._result(self.model(str(audio_path)))

    def predict_preloaded(self, audio, sr):
        from beat_this.inference import Audio2Beats
        return self._result(Audio2Beats.__call__(self.model, audio, sr))
