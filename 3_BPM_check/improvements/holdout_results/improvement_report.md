# Classical V2: independent evaluation

The original detector and its saved results are preserved. V2 was selected on the previously inspected 50 tracks, then frozen before the remaining 50 GTZAN-mini tracks were obtained. No test result was used to change V2.

## Measurements

Accuracy uses all 150 held-out prefixes (8/15/30 seconds); failed predictions count as incorrect. MAE uses successful predictions. Main latency uses identical successful representative clips, one CPU thread, five warm-ups and twenty file-to-result repetitions.

| Method | Successful /150 | Acc1 | Acc2 | MAE BPM | Median ms | P95 ms | V2 speedup over method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Existing classical | 95 | 32.7% | 52.7% | 30.80 | 427.35 | 638.25 | 10.72x |
| Classical V2 (new) | 150 | 62.7% | 86.7% | 24.88 | 39.87 | 49.28 | 1.00x |
| DeepRhythm | 150 | 58.7% | 82.7% | 26.48 | 243.47 | 323.34 | 6.11x |
| TempoCNN k4 | 150 | 61.3% | 80.0% | 22.87 | 29.11 | 44.62 | 0.73x |
| Beat This! small0 | 150 | 64.0% | 76.7% | 18.06 | 1418.64 | 3074.72 | 35.58x |

## Predeclared claim assessment

The claim requires BOTH higher raw-BPM Acc1 and lower matched median latency than every compared neural method.

- Higher Acc1 than every neural method: **False**.
- Lower matched median latency than every neural method: **False**.
- Both conditions measured: **False**.
- Higher separately labeled Acc2 than every neural method: **True**.

The measurements do not establish the requested joint claim. Improvements over the original detector and any separately measured advantages are reported without substituting Acc2 for Acc1.

## Paired accuracy uncertainty

Tracks, rather than their correlated prefixes, are resampled 10,000 times with seed2026. Intervals are exploratory 95% paired bootstrap intervals without multiplicity correction.

| Neural comparison | V2 minus NN Acc1, pp | 95% CI | V2 minus NN Acc2, pp | 95% CI |
| --- | --- | --- | --- | --- |
| Beat This! small0 | -1.3 | [-14.0, +11.3] | +10.0 | [-0.0, +20.7] |
| DeepRhythm | +4.0 | [-8.0, +16.0] | +4.0 | [-4.7, +12.7] |
| TempoCNN k4 | +1.3 | [-10.0, +12.7] | +6.7 | [-0.7, +15.3] |

## Reproduction and scope

See `../README.md`, `benchmark_config.json`, `model_metadata.csv`, `environment.json`, `validation.json`, and `claim_assessment.json`. This is a new classical variant with no neural weights. TempoCNN uses the documented Windows TensorFlow frontend fallback. The held-out set is independent of classical development; unknown neural training overlap remains a limitation.
