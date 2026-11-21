"""Raw tempo metrics and one-to-one beat matching against actual annotations."""
import math
import statistics


def tempo_metrics(predicted, reference):
    if reference is None:
        return {"absolute_error": None, "acc1": None, "acc2": None}
    if not math.isfinite(float(reference)) or reference <= 0:
        raise ValueError("Reference BPM must be finite and positive")
    if predicted is None or not math.isfinite(float(predicted)) or predicted <= 0:
        return {"absolute_error": None, "acc1": 0, "acc2": 0}
    error = abs(predicted-reference)
    return {"absolute_error": error, "acc1": int(error/reference <= .02 + 1e-12),
            "acc2": int(min(abs(2**k*predicted-reference)/reference for k in (-1,0,1)) <= .02 + 1e-12)}


def beat_metrics(predicted, reference, tolerance=.07):
    if predicted is None or reference is None:
        return {"beat_f1": None, "beat_timing_error_ms": None}
    p = sorted(float(x) for x in predicted if math.isfinite(float(x)))
    r = sorted(float(x) for x in reference if math.isfinite(float(x)))
    i = j = 0
    errors = []
    while i < len(p) and j < len(r):
        delta = p[i]-r[j]
        if abs(delta) <= tolerance:
            errors.append(abs(delta)); i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return {"beat_f1": 2*len(errors)/(len(p)+len(r)) if p or r else 1.,
            "beat_timing_error_ms": statistics.mean(errors)*1000 if errors else None}
