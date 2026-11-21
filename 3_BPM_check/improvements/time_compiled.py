"""Randomized interleaved CPU latency comparison after exact-output checking."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS"):
    os.environ[key]="1"
os.environ.update(CUDA_VISIBLE_DEVICES="",TF_CPP_MIN_LOG_LEVEL="2")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from adapters import get_estimator
    from run_benchmark import predict,environment
    from plot_results import common_real_latency
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=ROOT/"improvements/compiled_timing"
    out.mkdir(parents=True,exist_ok=True)
    equivalence=json.loads((ROOT/"improvements/compiled_equivalence.json").read_text(encoding="utf-8"))
    assert equivalence["status"]=="passed"
    assert hashlib.sha256((ROOT/"adapters/classical_v2_fast.py").read_bytes()).hexdigest()==equivalence["compiled_source_sha256"]
    manifest=json.loads((ROOT/"improvements/holdout_manifest.json").read_text(encoding="utf-8"))
    tracks=list(dict.fromkeys(c["track_id"] for c in manifest))[:2]
    clips=[c for c in manifest if c["track_id"] in tracks]
    names=["OUR_EXISTING","OUR_V2","OUR_V2_FAST","DeepRhythm","TempoCNN","BeatThis"]
    protocol={"seed":2026,"threads":1,"device":"cpu","warmups_per_clip_method":5,"repetitions_per_clip_method":20,
              "order":"Randomized interleaving of methods within each repetition for each common clip",
              "clip_ids":[c["clip_id"] for c in clips],"methods":names,"primary_stage":"file_to_result",
              "scope":"All required file read, preprocessing, inference, decoding and BPM output; no downloads or feature cache",
              "cold_initialization":"Not compared in shared process; separate process cold measurements retained in holdout_results",
              "secondary_preloaded":"Already measured separately in holdout_results; not pooled with file timings",
              "compiled_source_sha256":equivalence["compiled_source_sha256"]}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    (out/"environment.json").write_text(json.dumps(environment(1),indent=2),encoding="utf-8")
    (out/"dataset_manifest.json").write_text(json.dumps(clips,indent=2),encoding="utf-8")
    models={}
    for name in names:
        model=get_estimator(name,device="cpu",threads=1);model.load();models[name]=model
        print(name,"loaded",flush=True)
    (out/"model_metadata.json").write_text(json.dumps({k:v.metadata for k,v in models.items()},indent=2),encoding="utf-8")
    rows=[]
    rng=random.Random(2026)
    for clip in clips:
        path=ROOT/clip["audio_path"]
        supported=[]
        for name in names:
            value=None
            for _ in range(5):value=predict(models[name],path)
            if value["status"]=="success":supported.append(name)
            else:rows.append({"method":name,"clip_id":clip["clip_id"],"tier":"real","duration_seconds":clip["duration_seconds"],
                "stage":"file_to_result","repetition":-1,"order_in_repetition":-1,"latency_ms":None,"status":value["status"],"error":value.get("error")})
        for repetition in range(20):
            order=supported.copy();rng.shuffle(order)
            for position,name in enumerate(order):
                stamp=time.perf_counter_ns();value=predict(models[name],path);elapsed=(time.perf_counter_ns()-stamp)/1e6
                rows.append({"method":name,"clip_id":clip["clip_id"],"tier":"real","duration_seconds":clip["duration_seconds"],
                    "stage":"file_to_result","repetition":repetition,"order_in_repetition":position,"latency_ms":elapsed,
                    "status":value["status"],"error":value.get("error")})
        with (out/"latency_raw.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print("Interleaved timing",clip["clip_id"],"done",flush=True)
    summary,common=common_real_latency(rows)
    with (out/"latency_summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (out/"matched_clips.json").write_text(json.dumps(common,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(10,5))
    labels={"OUR_EXISTING":"Original classical","OUR_V2":"V2 frozen","OUR_V2_FAST":"V2 compiled", "DeepRhythm":"DeepRhythm","TempoCNN":"TempoCNN k4","BeatThis":"Beat This!"}
    for index,row in enumerate(summary):
        ax.bar(index,row["median_ms"],color="#C66464" if row["method"]=="OUR_V2_FAST" else "#335C81")
        ax.errorbar(index,row["median_ms"],yerr=[[0],[max(0,row["p95_ms"]-row["median_ms"])]],color="black",capsize=3)
    ax.set_yscale("log");ax.set_xticks(range(len(summary)),[labels[r["method"]] for r in summary],rotation=15,ha="right")
    ax.set_ylabel("Median file-to-result latency (ms, log scale)");ax.set_title("Interleaved CPU timing, fixed predictions")
    ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.2)
    fig.text(.12,.01,f"Same {len(common)} successful excerpts;5warmups;20repetitions;randomized method order. Whiskers: P95.",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(out/"latency_comparison.png",dpi=300);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
