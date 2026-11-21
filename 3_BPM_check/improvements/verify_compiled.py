"""Require exact output equivalence before evaluating a compiled implementation.

This implementation optimization does not select tempo parameters or change the
frozen detector. Every fast prediction is recomputed from audio and retained.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from adapters.classical_v2 import ClassicalV2Estimator
    from adapters.classical_v2_fast import ClassicalV2FastEstimator
    from metrics import tempo_metrics
    out=ROOT/"improvements"
    manifest=json.loads((out/"holdout_manifest.json").read_text(encoding="utf-8"))
    protocol=json.loads((out/"frozen_protocol.json").read_text(encoding="utf-8"))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(ROOT/"adapters/classical_v2.py")==protocol["algorithm_sha256"]
    reference=ClassicalV2Estimator();reference.load()
    fast=ClassicalV2FastEstimator();fast.load()
    failures,rows=[],[]
    for clip in manifest:
        path=ROOT/clip["audio_path"]
        a,b=reference.predict(path),fast.predict(path)
        equal=all(a[k]==b[k] for k in ("bpm","beat_times","confidence","status","error"))
        if not equal:failures.append(clip["clip_id"])
        rows.append({"method":"OUR_V2_FAST","clip_id":clip["clip_id"],"track_id":clip["track_id"],
                     "tier":"real","duration_seconds":clip["duration_seconds"],
                     "requested_duration_seconds":clip["requested_duration_seconds"],
                     "reference_bpm":clip["reference_bpm"],"bpm":b["bpm"],"status":b["status"],
                     "beat_times_json":json.dumps(b["beat_times"]),"exact_equivalence":equal,
                     **tempo_metrics(b["bpm"],clip["reference_bpm"])})
    outcome={"status":"passed" if not failures else "failed","n_clips":len(manifest),
             "exact_equal_clips":len(manifest)-len(failures),"failures":failures,
             "reference_algorithm_sha256":protocol["algorithm_sha256"],
             "compiled_source_sha256":digest(ROOT/"adapters/classical_v2_fast.py"),
             "configuration_sha256":digest(out/"frozen_config.json"),
             "scope":"Full BPM, beat arrays, status, confidence and error exactly equal; all fast predictions recomputed from audio",
             "change":"Implementation-only compiled scoring, with original scoring used for near ties; no accuracy parameter changes"}
    (out/"compiled_equivalence.json").write_text(json.dumps(outcome,indent=2),encoding="utf-8")
    with (out/"compiled_predictions.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(outcome,indent=2))
    if failures:raise SystemExit(1)


if __name__=="__main__":main()
