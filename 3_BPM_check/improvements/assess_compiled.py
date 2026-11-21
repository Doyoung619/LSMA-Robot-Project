"""Combine independently evaluated accuracy with interleaved implementation timing."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from plot_results import common_real_latency,pretty


def read(path):return list(csv.DictReader(Path(path).open(encoding="utf-8",newline="")))


def main():
    base=ROOT/"improvements"
    evidence=json.loads((base/"compiled_equivalence.json").read_text(encoding="utf-8"))
    assert evidence["status"]=="passed" and evidence["exact_equal_clips"]==150
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(ROOT/"adapters/classical_v2_fast.py")==evidence["compiled_source_sha256"]
    fast=read(base/"compiled_predictions.csv")
    original=read(base/"holdout_results/predictions.csv")
    reference={r["clip_id"]:r for r in original if r["method"]=="OUR_V2"}
    assert len(fast)==150
    for row in fast:
        other=reference[row["clip_id"]]
        assert row["exact_equivalence"]=="True"
        assert float(row["bpm"])==float(other["bpm"])
        assert json.loads(row["beat_times_json"])==json.loads(other["beat_times_json"])
        assert row["acc1"]==other["acc1"] and row["acc2"]==other["acc2"]
    summaries=[r for r in read(base/"holdout_results/accuracy_summary.csv") if r["tier"]=="real" and r["duration_seconds"]=="all"]
    scores={r["method"]:r for r in summaries}
    scores["OUR_V2_FAST"]={"n_reference":len(fast),"n_success":sum(r["status"]=="success" for r in fast),
                          "acc1":statistics.mean(float(r["acc1"]) for r in fast),
                          "acc2":statistics.mean(float(r["acc2"]) for r in fast),
                          "mae_bpm":statistics.mean(abs(float(r["bpm"])-float(r["reference_bpm"])) for r in fast)}
    rows=read(base/"compiled_timing/latency_raw.csv")
    times,common=common_real_latency(rows)
    times={r["method"]:r for r in times}
    protocol=json.loads((base/"compiled_timing/protocol.json").read_text(encoding="utf-8"))
    assert protocol["warmups_per_clip_method"]==5 and protocol["repetitions_per_clip_method"]==20
    for method in times:
        for clip in common:
            observed=[r for r in rows if r["method"]==method and r["clip_id"]==clip and r["status"]=="success"]
            assert len(observed)==20
            assert {int(r["repetition"]) for r in observed}==set(range(20))
    metadata=json.loads((base/"compiled_timing/model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["OUR_V2_FAST"]["sha256"]==evidence["compiled_source_sha256"]
    assert metadata["OUR_V2"]["sha256"]==evidence["reference_algorithm_sha256"]
    from adapters.model_cache import EXPECTED_SHA256
    nn=["DeepRhythm","TempoCNN","BeatThis"]
    for name in nn:
        m=metadata[name]
        assert m["pretrained_loaded"] is True
        assert m["sha256"]==EXPECTED_SHA256[m["checkpoint"]]
    faster=all(times["OUR_V2_FAST"]["median_ms"]<times[n]["median_ms"] for n in nn)
    better_raw=all(float(scores["OUR_V2_FAST"]["acc1"])>float(scores[n]["acc1"]) for n in nn)
    better_octave=all(float(scores["OUR_V2_FAST"]["acc2"])>float(scores[n]["acc2"]) for n in nn)
    outcome={"status":"passed","n_accuracy_tracks":50,"n_accuracy_clips":150,"exact_equivalent_predictions":150,
             "primary_timing":"Randomized interleaved matched file-to-result; CPU1thread;5warmups20repetitions",
             "matched_timing_clips":common,"lower_latency_than_every_nn":faster,
             "higher_acc1_than_every_nn":better_raw,"higher_acc2_than_every_nn":better_octave,
             "joint_raw_accuracy_and_latency_superiority":faster and better_raw,
             "compiled_median_ms":times["OUR_V2_FAST"]["median_ms"],
             "speedups_over_nn":{n:times[n]["median_ms"]/times["OUR_V2_FAST"]["median_ms"] for n in nn},
             "uncertainty":"Accuracy point estimates on 50 tracks; paired bootstrap intervals in holdout_results/claim_assessment.json; no universal superiority claim"}
    (base/"compiled_assessment.json").write_text(json.dumps(outcome,indent=2),encoding="utf-8")
    lines=["# Classical V2 compiled: measured comparison", "",
           "Accuracy was evaluated on 50 tracks withheld from classical development (150 matching 8/15/30-second prefixes). "
           "The configuration was frozen beforehand. The compiled implementation then reproduced every BPM, beat array "
           "and status exactly, so its accuracy is unchanged. Timing was subsequently measured for every method in "
           "randomized interleaved order on the same successful representative files.", "",
           "| Method | Successful /150 | Acc1 | Acc2 | MAE BPM | Median ms | P95 ms | Compiled speedup over method |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in ("OUR_EXISTING","OUR_V2","OUR_V2_FAST","DeepRhythm","TempoCNN","BeatThis"):
        a,t=scores[name],times[name]
        lines.append(f"| {pretty(name)} | {a['n_success']} | {100*float(a['acc1']):.1f}% | {100*float(a['acc2']):.1f}% | "
                     f"{float(a['mae_bpm']):.2f} | {t['median_ms']:.2f} | {t['p95_ms']:.2f} | {t['median_ms']/times['OUR_V2_FAST']['median_ms']:.2f}x |")
    lines += ["", "Acc1 uses raw BPM within 2%; Acc2 separately allows half/double tempo. "
              "Failed predictions count as incorrect. MAE is successful-output only. Main timings include file reading, "
              "preprocessing, inference, postprocessing and BPM output. Compiler/JIT warm-up is excluded from warm timing. "
              "Model download and initialization are excluded. They are measured separately where available.", "",
              "## Supported interpretation", "",
              f"- Faster than every measured neural method: **{faster}**.",
              f"- Higher raw BPM Acc1 than every neural method: **{better_raw}**.",
              f"- Higher separately labeled octave-tolerant Acc2 than every neural method: **{better_octave}**.", "",
              "The full raw-accuracy-and-speed superiority claim requires both first conditions; "
              f"it is **{faster and better_raw}** for this measured comparison. Acc2 must not be silently substituted for Acc1. "
              "Accuracy confidence intervals overlap for some comparisons; point-estimate differences do not establish "
              "universal or statistically certain superiority.", "",
              f"TempoCNN latency is comparable: {times['TempoCNN']['median_ms']:.2f} ms versus "
              f"{times['OUR_V2_FAST']['median_ms']:.2f} ms for the compiled classical detector. "
              "This sub-millisecond median gap and overlapping variability do not establish a robust speed advantage over TempoCNN.", "",
              "## Presentation-ready English text", ""]
    speed=outcome["speedups_over_nn"]
    lines += [f"- Our classical detector achieved {100*float(scores['OUR_V2_FAST']['acc1']):.1f}% raw tempo accuracy "
              f"and {100*float(scores['OUR_V2_FAST']['acc2']):.1f}% octave-tolerant accuracy on 50 held-out GTZAN tracks.",
              f"- Warm CPU file-to-result latency was {times['OUR_V2_FAST']['median_ms']:.2f} ms: "
              f"{speed['DeepRhythm']:.2f}x faster than DeepRhythm, {speed['TempoCNN']:.2f}x versus TempoCNN, "
              f"and {speed['BeatThis']:.2f}x versus Beat This!.",
              f"- Raw accuracy was {100*float(scores['OUR_V2_FAST']['acc1']):.1f}% versus "
              f"{100*float(scores['BeatThis']['acc1']):.1f}% for Beat This!; octave-tolerant accuracy and latency "
              "are separate strengths, with no pretrained neural weights.", "",
              "## Evidence", "",
              "- `frozen_protocol.json`, `frozen_config.json`, `holdout_split.json`: development/test separation and hashes.",
              "- `holdout_results/validation.json`: audio, published annotations, weights, metrics and figures checked.",
              "- `compiled_equivalence.json`, `compiled_predictions.csv`: all 150 complete outputs recomputed and identical.",
              "- `compiled_timing/`: actual interleaved timing rows, protocol, model identities and figure.",
              "- `compiled_startup.json`: separate first-call measurement with an empty compiler cache.",
              "- `holdout_results/claim_assessment.json`: track-level paired bootstrap uncertainty.", "",
              "TempoCNN uses the documented Windows TensorFlow frontend fallback. Neural training overlap remains partly unknown. "
              "The original robot pipeline has not been replaced by this separately developed version.", ""]
    lines += ["[Presentation comparison figure](figures/improved_comparison.png)", ""]
    (base/"compiled_comparison_report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(outcome,indent=2))


if __name__=="__main__":main()
