"""Evaluate declared classical changes on the previously inspected 50 tracks.

Feature caching is DEVELOPMENT ONLY and is never used by the inference adapter.
No independent-test result is read or used to select the configuration.
"""
import csv
import itertools
import json
import os
from pathlib import Path
import statistics
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from adapters.classical_v2 import DEFAULT_CONFIG, features, estimate
from metrics import tempo_metrics


def main():
    manifest = json.loads((ROOT/"results/dataset_manifest.json").read_text(encoding="utf-8"))
    clips = [c for c in manifest if c["tier"] == "real"]
    cache = ROOT/"data/cache/v2_features"
    cache.mkdir(parents=True,exist_ok=True)
    out = ROOT/"improvements/results"
    out.mkdir(parents=True,exist_ok=True)
    prepared = []
    for clip in clips:
        path = cache/f"{clip['clip_id']}_fft{DEFAULT_CONFIG['nfft']}_hop{DEFAULT_CONFIG['hop']}.npz"
        if path.exists():
            with np.load(path) as archive:
                f = {k:archive[k] for k in archive.files}
        else:
            audio,sr = sf.read(ROOT/clip["audio_path"],dtype="float32",always_2d=True)
            f = features(audio,sr)
            np.savez(path,**f)
        prepared.append((clip,f))
    summaries,predictions = [],[]
    configurations = []
    for weights,exponent in itertools.product(([.55,.3,.15],[.3,.5,.2]),(0.,.5,1.)):
        configurations.append({**DEFAULT_CONFIG,"band_weights":weights,"density_exponent":exponent})
    for weights,exponent,prior in itertools.product(([.55,.3,.15],[.3,.5,.2]),(.25,.5),(.5,1.,2.,4.)):
        configurations.append({**DEFAULT_CONFIG,"band_weights":weights,"density_exponent":exponent,
                               "tempo_prior_strength":prior})
    (out/"declared_configurations.json").write_text(json.dumps(configurations,indent=2),encoding="utf-8")
    for index,config in enumerate(configurations):
        rows = []
        for clip,f in prepared:
            try:
                bpm,_ = estimate(f,config)
                status,error = "success",None
            except Exception as exc:
                bpm,status,error = None,"error",str(exc)
            rows.append({"configuration_id":index,"clip_id":clip["clip_id"],"track_id":clip["track_id"],
                         "duration_seconds":clip["requested_duration_seconds"],"reference_bpm":clip["reference_bpm"],
                         "bpm":bpm,"status":status,"error":error,**tempo_metrics(bpm,clip["reference_bpm"])})
        predictions.extend(rows)
        summary={"configuration_id":index,"n":len(rows),"n_success":sum(r["status"]=="success" for r in rows),
                 "acc1":statistics.mean(r["acc1"] for r in rows),"acc2":statistics.mean(r["acc2"] for r in rows),
                 "mae_bpm":statistics.mean(r["absolute_error"] for r in rows if r["absolute_error"] is not None),
                 "band_weights":json.dumps(config["band_weights"]),"density_exponent":config["density_exponent"],
                 "tempo_prior_strength":config["tempo_prior_strength"]}
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    for name,rows in (("development_scores.csv",summaries),("development_predictions.csv",predictions)):
        with (out/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    best=max(summaries,key=lambda r:(r["acc1"],r["acc2"],-r["mae_bpm"]))
    selected=configurations[best["configuration_id"]]
    (out/"selected_development_config.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    (out/"development_split.json").write_text(json.dumps({"track_ids":sorted({c["track_id"] for c in clips}),
        "source":"Previously inspected original benchmark; development only, not independent evidence",
        "selection_rule":"highest development Acc1, then Acc2, then lowest MAE",
        "configuration_count":len(configurations),"selected_configuration_id":best["configuration_id"]},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
