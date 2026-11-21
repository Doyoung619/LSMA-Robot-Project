"""Bundle completed measurements for two review figures; no inference or tuning."""
import csv
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
EXPERIMENT=BASE.parents[2]
SOURCE=EXPERIMENT/"3_BPM_check/improvements"
METHODS=("OUR_V2_FAST","DeepRhythm","TempoCNN","BeatThis")


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def write_csv(path,rows,columns):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")


def text_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def main():
    data=BASE/"data";data.mkdir(parents=True,exist_ok=True)
    proof=json.loads((SOURCE/"compiled_equivalence.json").read_text(encoding="utf-8"))
    assert proof["status"]=="passed" and proof["exact_equal_clips"]==150
    models=json.loads((SOURCE/"compiled_timing/model_metadata.json").read_text(encoding="utf-8"))
    assert models["OUR_V2_FAST"]["sha256"]==proof["compiled_source_sha256"]
    for name in METHODS[1:]:assert models[name]["pretrained_loaded"] is True
    fast=read_csv(SOURCE/"compiled_predictions.csv")
    original=read_csv(SOURCE/"holdout_results/predictions.csv")
    predictions=[r for r in original if r["method"] in METHODS[1:]]+fast
    predictions.sort(key=lambda r:(METHODS.index(r["method"]),r["clip_id"]))
    assert len(predictions)==600
    columns=["method","clip_id","track_id","duration_seconds","requested_duration_seconds","reference_bpm","bpm","status","error","absolute_error","acc1","acc2"]
    write_csv(data/"predictions.csv",predictions,columns)
    timing=read_csv(SOURCE/"compiled_timing/latency_raw.csv")
    timing=[r for r in timing if r["method"] in METHODS]
    assert len(timing)==480 and all(r["status"]=="success" for r in timing)
    write_csv(data/"latency_raw.csv",timing,list(timing[0]))
    clips=json.loads((SOURCE/"holdout_manifest.json").read_text(encoding="utf-8"))
    write_csv(data/"input_clips.csv",clips,["clip_id","track_id","audio_path","duration_seconds","requested_duration_seconds","sample_rate","channels","pcm_sha256"])
    references={}
    for clip in clips:
        row={key:clip.get(key) for key in ("track_id","genre","reference_bpm","annotation_source","beat_annotation_source","source_url","source_audio_md5")}
        if clip["track_id"] in references:assert references[clip["track_id"]]==row
        references[clip["track_id"]]=row
    write_csv(data/"track_references.csv",[references[k] for k in sorted(references)],list(next(iter(references.values()))))
    for name,relative in {
        "environment.json":"compiled_timing/environment.json",
        "timing_protocol.json":"compiled_timing/protocol.json",
        "holdout_split.json":"holdout_split.json",
        "compiler_equivalence.json":"compiled_equivalence.json",
        "compiler_startup.json":"compiled_startup.json",
        "previous_four_clip_summary.json":"compiled_assessment.json",
        "frozen_protocol.json":"frozen_protocol.json",
        "source_validation.json":"holdout_results/validation.json",
    }.items():write_json(data/name,json.loads((SOURCE/relative).read_text(encoding="utf-8")))
    # This record is byte-hashed by holdout_split.json; preserve its original bytes.
    (data/"frozen_protocol.json").write_bytes((SOURCE/"frozen_protocol.json").read_bytes())
    write_json(data/"model_metadata.json",{name:models[name] for name in METHODS})
    provenance={"source":"Completed independently held-out evaluation; no new predictions or parameter selection in this packaging step",
        "selected_methods":list(METHODS),
        "selection_reason":"Retain the three originally requested, successfully executed neural baselines: direct tempo CNN, lightweight TempoCNN, and beat tracker",
        "accuracy_scope":"All 50 held-out tracks and all 150 nominal 8/15/30-second excerpts; both raw Acc1 and half/double-tolerant Acc2 shown",
        "latency_scope":"All six excerpts supported by the four selected methods; 20 repetitions each, 120 timings per method",
        "previous_latency_scope":"Earlier six-method table included legacy OUR_EXISTING, which failed on 8-second clips; its common set had four 15/30-second excerpts. This package uses all six common clips for the four displayed methods. Earlier results are preserved.",
        "hash_definition":"SHA256 after CRLF-to-LF normalization for cross-platform text snapshots",
        "original_source_files":{relative:text_hash(SOURCE/relative) for relative in ["compiled_predictions.csv","holdout_results/predictions.csv","compiled_timing/latency_raw.csv","compiled_timing/model_metadata.json","compiled_equivalence.json","holdout_split.json","holdout_manifest.json"]},
        "packaged_input_hashes":{p.name:text_hash(p) for p in sorted(data.iterdir()) if p.is_file() and p.name not in {"summary.csv","duration_summary.csv"}},
        "excluded":"Original robot baseline and uncompiled V2 remain available in the complete benchmark; no baseline was replaced to obtain a favorable ranking"}
    write_json(BASE/"source_provenance.json",provenance)
    print("Bundled 600 accuracy records and 480 measured timing records.")


if __name__=="__main__":main()
