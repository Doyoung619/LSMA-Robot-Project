"""Assess the predeclared joint-superiority claim using saved actual results."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from plot_results import common_real_latency,pretty


def read_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8",newline="")))


def generate():
    base=ROOT/"improvements"
    results=base/"holdout_results"
    protocol=json.loads((base/"frozen_protocol.json").read_text(encoding="utf-8"))
    split=json.loads((base/"holdout_split.json").read_text(encoding="utf-8"))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(ROOT/"adapters/classical_v2.py")==protocol["algorithm_sha256"]
    assert digest(base/"frozen_config.json")==protocol["configuration_sha256"]
    assert not set(split["development_track_ids"])&set(split["holdout_track_ids"])
    accuracy=[r for r in read_csv(results/"accuracy_summary.csv") if r["tier"]=="real" and r["duration_seconds"]=="all"]
    scores={r["method"]:r for r in accuracy}
    predictions=[r for r in read_csv(results/"predictions.csv") if r["tier"]=="real"]
    latency,common=common_real_latency(read_csv(results/"latency_raw.csv"))
    times={r["method"]:r for r in latency}
    neural=[m for m in scores if not m.startswith("OUR_") and int(scores[m]["n_success"])>0]
    assert set(scores)=={"OUR_EXISTING","OUR_V2","DeepRhythm","TempoCNN","BeatThis"}
    assert all(int(r["n_reference"])==150 for r in scores.values())
    tracks=split["holdout_track_ids"]
    rng=np.random.default_rng(2026)
    bootstrap=rng.integers(0,len(tracks),size=(10000,len(tracks)))
    paired={}
    for m in neural:
        paired[m]={}
        for metric in ("acc1","acc2"):
            by_track=[]
            for track in tracks:
                ours=[float(r[metric]) for r in predictions if r["method"]=="OUR_V2" and r["track_id"]==track]
                other=[float(r[metric]) for r in predictions if r["method"]==m and r["track_id"]==track]
                assert len(ours)==len(other)==3
                by_track.append(np.mean(ours)-np.mean(other))
            delta=np.asarray(by_track)
            distribution=delta[bootstrap].mean(axis=1)
            paired[m][metric]={"difference_percentage_points":float(delta.mean()*100),
                              "paired_track_bootstrap_95_percent_ci":(np.percentile(distribution,[2.5,97.5])*100).tolist()}
        paired[m]["v2_speedup"]=times[m]["median_ms"]/times["OUR_V2"]["median_ms"]
    better_acc1=all(float(scores["OUR_V2"]["acc1"])>float(scores[m]["acc1"]) for m in neural)
    better_acc2=all(float(scores["OUR_V2"]["acc2"])>float(scores[m]["acc2"]) for m in neural)
    faster=all(times["OUR_V2"]["median_ms"]<times[m]["median_ms"] for m in neural)
    clear_acc1=all(paired[m]["acc1"]["paired_track_bootstrap_95_percent_ci"][0]>0 for m in neural)
    outcome={"n_development_tracks":50,"n_holdout_tracks":50,"n_holdout_excerpts":150,"track_overlap":0,
             "algorithm_sha256":protocol["algorithm_sha256"],"configuration_sha256":protocol["configuration_sha256"],
             "higher_raw_acc1_than_every_nn":better_acc1,"higher_acc2_than_every_nn":better_acc2,
             "lower_matched_median_latency_than_every_nn":faster,
             "joint_superiority_point_estimate":better_acc1 and faster,
             "raw_accuracy_differences_all_positive_95_percent_ci":clear_acc1,
             "paired_comparisons":paired,"matched_timing_clip_ids":common,
             "limitations":"50 independent tracks with correlated prefixes; bootstrap is exploratory and not multiplicity-adjusted; neural training overlap partly unknown"}
    (results/"claim_assessment.json").write_text(json.dumps(outcome,indent=2),encoding="utf-8")
    order=["OUR_EXISTING","OUR_V2","DeepRhythm","TempoCNN","BeatThis"]
    lines=["# Classical V2: independent evaluation", "",
           "The original detector and its saved results are preserved. V2 was selected on the previously inspected 50 tracks, "
           "then frozen before the remaining 50 GTZAN-mini tracks were obtained. No test result was used to change V2.", "",
           "## Measurements", "",
           "Accuracy uses all 150 held-out prefixes (8/15/30 seconds); failed predictions count as incorrect. "
           "MAE uses successful predictions. Main latency uses identical successful representative clips, "
           "one CPU thread, five warm-ups and twenty file-to-result repetitions.", "",
           "| Method | Successful /150 | Acc1 | Acc2 | MAE BPM | Median ms | P95 ms | V2 speedup over method |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in order:
        a,t=scores[m],times[m]
        lines.append(f"| {pretty(m)} | {a['n_success']} | {100*float(a['acc1']):.1f}% | {100*float(a['acc2']):.1f}% | "
                     f"{float(a['mae_bpm']):.2f} | {t['median_ms']:.2f} | {t['p95_ms']:.2f} | {t['median_ms']/times['OUR_V2']['median_ms']:.2f}x |")
    lines += ["", "## Predeclared claim assessment", "",
              "The claim requires BOTH higher raw-BPM Acc1 and lower matched median latency than every compared neural method.", "",
              f"- Higher Acc1 than every neural method: **{better_acc1}**.",
              f"- Lower matched median latency than every neural method: **{faster}**.",
              f"- Both conditions measured: **{better_acc1 and faster}**.",
              f"- Higher separately labeled Acc2 than every neural method: **{better_acc2}**.", ""]
    if better_acc1 and faster:
        lines.append("The held-out point estimates support the joint claim for this dataset and hardware. "
                     "The paired accuracy intervals below show the statistical uncertainty; this is not a universal superiority claim.")
    else:
        lines.append("The measurements do not establish the requested joint claim. Improvements over the original detector "
                     "and any separately measured advantages are reported without substituting Acc2 for Acc1.")
    lines += ["", "## Paired accuracy uncertainty", "",
              "Tracks, rather than their correlated prefixes, are resampled 10,000 times with seed2026. "
              "Intervals are exploratory 95% paired bootstrap intervals without multiplicity correction.", "",
              "| Neural comparison | V2 minus NN Acc1, pp | 95% CI | V2 minus NN Acc2, pp | 95% CI |",
              "| --- | --- | --- | --- | --- |"]
    for m in neural:
        p=paired[m]
        fmt_ci=lambda metric:"["+", ".join(f"{x:+.1f}" for x in p[metric]["paired_track_bootstrap_95_percent_ci"])+"]"
        lines.append(f"| {pretty(m)} | {p['acc1']['difference_percentage_points']:+.1f} | {fmt_ci('acc1')} | "
                     f"{p['acc2']['difference_percentage_points']:+.1f} | {fmt_ci('acc2')} |")
    lines += ["", "## Reproduction and scope", "",
              "See `../README.md`, `benchmark_config.json`, `model_metadata.csv`, `environment.json`, "
              "`validation.json`, and `claim_assessment.json`. This is a new classical variant with no neural weights. "
              "TempoCNN uses the documented Windows TensorFlow frontend fallback. The held-out set is independent "
              "of classical development; unknown neural training overlap remains a limitation.", ""]
    (results/"improvement_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(outcome,indent=2))
    return outcome


if __name__ == "__main__":
    generate()
