# Classical V2 compiled: measured comparison

Accuracy was evaluated on 50 tracks withheld from classical development (150 matching 8/15/30-second prefixes). The configuration was frozen beforehand. The compiled implementation then reproduced every BPM, beat array and status exactly, so its accuracy is unchanged. Timing was subsequently measured for every method in randomized interleaved order on the same successful representative files.

| Method | Successful /150 | Acc1 | Acc2 | MAE BPM | Median ms | P95 ms | Compiled speedup over method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Existing classical | 95 | 32.7% | 52.7% | 30.80 | 300.71 | 423.36 | 15.00x |
| Classical V2 (new) | 150 | 62.7% | 86.7% | 24.88 | 22.89 | 38.21 | 1.14x |
| Classical V2 compiled | 150 | 62.7% | 86.7% | 24.88 | 20.05 | 35.42 | 1.00x |
| DeepRhythm | 150 | 58.7% | 82.7% | 26.48 | 141.41 | 224.18 | 7.05x |
| TempoCNN k4 | 150 | 61.3% | 80.0% | 22.87 | 20.52 | 30.31 | 1.02x |
| Beat This! small0 | 150 | 64.0% | 76.7% | 18.06 | 1310.58 | 2316.39 | 65.38x |

Acc1 uses raw BPM within 2%; Acc2 separately allows half/double tempo. Failed predictions count as incorrect. MAE is successful-output only. Main timings include file reading, preprocessing, inference, postprocessing and BPM output. Compiler/JIT warm-up is excluded from warm timing. Model download and initialization are excluded. They are measured separately where available.

## Supported interpretation

- Faster than every measured neural method: **True**.
- Higher raw BPM Acc1 than every neural method: **False**.
- Higher separately labeled octave-tolerant Acc2 than every neural method: **True**.

The full raw-accuracy-and-speed superiority claim requires both first conditions; it is **False** for this measured comparison. Acc2 must not be silently substituted for Acc1. Accuracy confidence intervals overlap for some comparisons; point-estimate differences do not establish universal or statistically certain superiority.

TempoCNN latency is comparable: 20.52 ms versus 20.05 ms for the compiled classical detector. This sub-millisecond median gap and overlapping variability do not establish a robust speed advantage over TempoCNN.

## Presentation-ready English text

- Our classical detector achieved 62.7% raw tempo accuracy and 86.7% octave-tolerant accuracy on 50 held-out GTZAN tracks.
- Warm CPU file-to-result latency was 20.05 ms: 7.05x faster than DeepRhythm, 1.02x versus TempoCNN, and 65.38x versus Beat This!.
- Raw accuracy was 62.7% versus 64.0% for Beat This!; octave-tolerant accuracy and latency are separate strengths, with no pretrained neural weights.

## Evidence

- `frozen_protocol.json`, `frozen_config.json`, `holdout_split.json`: development/test separation and hashes.
- `holdout_results/validation.json`: audio, published annotations, weights, metrics and figures checked.
- `compiled_equivalence.json`, `compiled_predictions.csv`: all 150 complete outputs recomputed and identical.
- `compiled_timing/`: actual interleaved timing rows, protocol, model identities and figure.
- `compiled_startup.json`: separate first-call measurement with an empty compiler cache.
- `holdout_results/claim_assessment.json`: track-level paired bootstrap uncertainty.

TempoCNN uses the documented Windows TensorFlow frontend fallback. Neural training overlap remains partly unknown. The original robot pipeline has not been replaced by this separately developed version.

[Presentation comparison figure](figures/improved_comparison.png)
