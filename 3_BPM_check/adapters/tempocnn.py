"""Official deeptemp-k4-3 graph with original author's librosa frontend.

Native Windows has no Essentia TensorFlow wheel. This explicit fallback uses
tempocnn.feature.read_features and std_normalizer, not an emulation of Essentia.
Its latency must be attributed to this frontend/runtime combination.
"""
from .base import BPMEstimator, result
from .model_cache import MODEL_DIR, checkpoint_metadata


class TempoCNNEstimator(BPMEstimator):
    name = "TempoCNN"

    def load(self):
        if self.device != "cpu":
            raise ValueError("This TensorFlow frozen-graph adapter supports CPU only")
        import tensorflow as tf
        from tempocnn.feature import read_features
        from tempocnn.classifier import std_normalizer
        self.read_features, self.normalize = read_features, std_normalizer
        self.metadata = checkpoint_metadata(
            "deeptemp-k4-3.pb", "tensorflow",
            "https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k4-3.pb",
            "CC BY-NC-SA 4.0 (Essentia model); AGPL-3.0 (original frontend)",
            output_type="global BPM", device="cpu",
            runtime="TensorFlow v1 Session; original tempocnn librosa frontend (Essentia unavailable on Windows)",
            preprocessing="11025 Hz, 1024 STFT, hop 512, 40 mel bands 20-5000 Hz, 256-frame patches hop 128; original std_normalizer",
            aggregation="Original author: argmax of mean patch probabilities + 30 BPM; no interpolation",
            minimum_context="256 frames (~11.9 s); original frontend zero-pads 8 s inputs",
            training_overlap="Extended Ballroom, LMDTempo, MTGTempo per official metadata; full GTZAN overlap not independently verified")
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString((MODEL_DIR / "deeptemp-k4-3.pb").read_bytes())
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=self.threads,
                                         inter_op_parallelism_threads=1,
                                         device_count={"GPU": 0})
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        self.input = self.graph.get_tensor_by_name("input:0")
        self.output = self.graph.get_tensor_by_name("output:0")
        self.metadata["pretrained_loaded"] = True
        self.metadata["input_shape"] = self.input.shape.as_list()
        self.metadata["graph_nodes"] = len(graph_def.node)

    def predict(self, audio_path):
        import numpy as np
        features = self.normalize(self.read_features(str(audio_path)))
        probabilities = self.session.run(self.output, feed_dict={self.input: features})
        if probabilities.ndim != 2 or probabilities.shape[1] != 256:
            raise ValueError(f"Unexpected pretrained model output shape: {probabilities.shape}")
        mean = np.mean(probabilities, axis=0)
        index = int(np.argmax(mean))
        return result(self.name, 30 + index, confidence=float(mean[index]))
