"""Regenerate exactly two measured-result figures from the bundled CSV records."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE=Path(__file__).resolve().parent
OUTPUT=BASE.parent
DATA=BASE/"data"
METHODS=("OUR_V2_FAST","DeepRhythm","TempoCNN","BeatThis")
LABELS=("Ours\nClassical V2","DeepRhythm","TempoCNN\nk4 (TF)","Beat This!\nsmall0")
NAMES={"OUR_V2_FAST":"Ours (Classical V2)","DeepRhythm":"DeepRhythm","TempoCNN":"TempoCNN k4 (TF)","BeatThis":"Beat This! small0"}
TEAL="#147D72"
BLUE="#58758D"
GREY="#C7D0D7"
INK="#23333F"


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8",newline="\n")


def write_csv(path,rows,columns=None):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns or list(rows[0]),extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def load_and_summarize():
    provenance=json.loads((BASE/"source_provenance.json").read_text(encoding="utf-8"))
    for name,expected in provenance["packaged_input_hashes"].items():
        observed=hashlib.sha256((DATA/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        assert observed==expected,f"Input changed: {name}"
    predictions=read_csv(DATA/"predictions.csv")
    raw=read_csv(DATA/"latency_raw.csv")
    split=json.loads((DATA/"holdout_split.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((DATA/"frozen_protocol.json").read_bytes()).hexdigest()==split["protocol_sha256"]
    references={r["track_id"]:r for r in read_csv(DATA/"track_references.csv")}
    clips={r["clip_id"]:r for r in read_csv(DATA/"input_clips.csv")}
    assert len(clips)==150
    protocol=json.loads((DATA/"timing_protocol.json").read_text(encoding="utf-8"))
    assert protocol["device"]=="cpu" and protocol["threads"]==1
    assert protocol["warmups_per_clip_method"]==5 and protocol["repetitions_per_clip_method"]==20
    assert all(r["stage"]=="file_to_result" for r in raw)
    tracks=sorted(split["holdout_track_ids"])
    assert len(tracks)==50 and not set(tracks)&set(split["development_track_ids"])
    expected=set(tracks)
    rng=np.random.default_rng(2026)
    resamples=rng.integers(0,50,size=(10000,50))
    success_sets=[{r["clip_id"] for r in raw if r["method"]==m and r["status"]=="success"} for m in METHODS]
    common=set.intersection(*success_sets)
    assert len(common)==6
    summary=[];duration_rows=[];checks=0
    for method in METHODS:
        records=[r for r in predictions if r["method"]==method]
        assert len(records)==150 and {r["track_id"] for r in records}==expected
        assert len({r["clip_id"] for r in records})==150
        assert {r["clip_id"] for r in records}==set(clips)
        for track in tracks:
            cases=[r for r in records if r["track_id"]==track]
            assert len(cases)==3 and {float(r["requested_duration_seconds"]) for r in cases}=={8.,15.,30.}
        measured=[]
        for row in records:
            reference=float(row["reference_bpm"])
            assert math.isfinite(reference) and reference>0
            assert reference==float(references[row["track_id"]]["reference_bpm"])
            assert references[row["track_id"]]["annotation_source"]
            assert row["track_id"]==clips[row["clip_id"]]["track_id"]
            assert float(row["duration_seconds"])==float(clips[row["clip_id"]]["duration_seconds"])
            valid=row["status"]=="success" and bool(row["bpm"])
            predicted=float(row["bpm"]) if valid else None
            if predicted is not None:assert math.isfinite(predicted) and predicted>0
            absolute=None if predicted is None else abs(predicted-reference)
            a1=int(predicted is not None and absolute/reference<=.02+1e-12)
            a2=int(predicted is not None and min(abs(predicted*factor-reference)/reference for factor in (.5,1.,2.))<=.02+1e-12)
            assert a1==int(row["acc1"]) and a2==int(row["acc2"])
            measured.append({**row,"a1":a1,"a2":a2,"absolute":absolute});checks+=1
        values={key:np.asarray([np.mean([r[key] for r in measured if r["track_id"]==track]) for track in tracks]) for key in ("a1","a2")}
        intervals={key:np.percentile(values[key][resamples].mean(axis=1),[2.5,97.5]) for key in values}
        timing=[r for r in raw if r["method"]==method and r["clip_id"] in common and r["status"]=="success"]
        assert len(timing)==120
        for clip in common:
            cases=[r for r in timing if r["clip_id"]==clip]
            assert len(cases)==20 and {int(r["repetition"]) for r in cases}==set(range(20))
            checks+=1
        elapsed=np.asarray([float(r["latency_ms"]) for r in timing])
        assert np.isfinite(elapsed).all() and (elapsed>0).all()
        accuracy1=float(values["a1"].mean());accuracy2=float(values["a2"].mean())
        summary.append({"method":method,"label":NAMES[method],"accuracy_tracks":50,"accuracy_clips":150,
            "successful_predictions":sum(r["status"]=="success" for r in measured),
            "acc1":accuracy1,"acc1_ci_low":float(intervals["a1"][0]),"acc1_ci_high":float(intervals["a1"][1]),
            "acc2":accuracy2,"acc2_ci_low":float(intervals["a2"][0]),"acc2_ci_high":float(intervals["a2"][1]),
            "mae_bpm":statistics.mean(r["absolute"] for r in measured if r["absolute"] is not None),
            "latency_clips":6,"latency_repetitions":20,"latency_measurements":120,
            "median_ms":float(np.median(elapsed)),"mean_ms":float(elapsed.mean()),
            "p95_ms":float(np.percentile(elapsed,95)),"std_ms":float(elapsed.std()),
            "median_rtf":float(np.median([float(r["latency_ms"])/1000/float(r["duration_seconds"]) for r in timing]))})
        for duration in (8.,15.,30.):
            chosen=[r for r in measured if float(r["requested_duration_seconds"])==duration]
            assert len(chosen)==50
            times=[float(r["latency_ms"]) for r in timing if float(r["duration_seconds"])==duration]
            assert len(times)==40
            duration_rows.append({"method":method,"nominal_duration_seconds":duration,"accuracy_clips":50,
                "acc1":statistics.mean(r["a1"] for r in chosen),"acc2":statistics.mean(r["a2"] for r in chosen),
                "median_ms":statistics.median(times),"p95_ms":float(np.percentile(times,95))})
    write_csv(DATA/"summary.csv",summary)
    write_csv(DATA/"duration_summary.csv",duration_rows)
    return summary,sorted(common),checks


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"text.color":INK,
        "axes.labelcolor":INK,"axes.edgecolor":"#ADB8BF","xtick.color":INK,"ytick.color":INK,
        "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white",
        "savefig.facecolor":"white","svg.hashsalt":"rita-bpm-review-v1","svg.fonttype":"none"})


def save(fig,stem,preview_dir):
    fig.savefig(OUTPUT/f"{stem}.png",dpi=300,facecolor="white")
    vector=BASE/"vector";vector.mkdir(exist_ok=True)
    svg=vector/f"{stem}.svg"
    fig.savefig(svg,metadata={"Date":None,"Creator":"Rita BPM measured benchmark"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
    if preview_dir:
        preview_dir.mkdir(parents=True,exist_ok=True)
        fig.savefig(preview_dir/f"{stem}.png",dpi=100,facecolor="white")
    plt.close(fig)


def accuracy_figure(summary,preview_dir):
    fig=plt.figure(figsize=(13.333333,7.5))
    ax=fig.add_axes([.09,.235,.88,.50])
    fig.text(.09,.93,"TEMPO ACCURACY",fontsize=12,color="#61727D",weight="bold")
    fig.text(.09,.868,"Highest observed half/double-tolerant accuracy",fontsize=23,weight="bold")
    fig.text(.09,.818,"50 held-out tracks · 150 audio excerpts · identical evaluation files for every method",fontsize=12,color="#61727D")
    x=np.arange(4);width=.30
    a1=np.asarray([r["acc1"] for r in summary])*100
    a2=np.asarray([r["acc2"] for r in summary])*100
    ax.axvspan(-.5,.5,color="#EFF7F5",zorder=0)
    ax.bar(x-width/2,a1,width,color=GREY,zorder=3)
    ax.bar(x+width/2,a2,width,color=TEAL,zorder=3)
    for shift,values,metric in ((-width/2,a1,"acc1"),(width/2,a2,"acc2")):
        low=np.asarray([r[f"{metric}_ci_low"] for r in summary])*100
        high=np.asarray([r[f"{metric}_ci_high"] for r in summary])*100
        ax.errorbar(x+shift,values,yerr=[values-low,high-values],fmt="none",ecolor=INK,capsize=4,linewidth=1.2,zorder=4)
        for index,value in enumerate(values):
            ax.text(index+shift,high[index]+1.7,f"{value:.1f}",ha="center",va="bottom",fontsize=12,
                    weight="bold" if index==0 else "normal")
    ax.set_ylim(0,108);ax.set_yticks(np.arange(0,101,20));ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x,LABELS);ax.get_xticklabels()[0].set_weight("bold")
    ax.set_axisbelow(True);ax.grid(axis="y",color="#DDE3E7",alpha=.7)
    fig.legend([Patch(facecolor=GREY),Patch(facecolor=TEAL)],
        ["Acc1: raw BPM within ±2%","Acc2: also allows half / double BPM"],
        frameon=False,ncol=2,loc="upper left",bbox_to_anchor=(.08,.79),fontsize=11,columnspacing=2)
    fig.text(.09,.14,"Acc2 is a separate tempo-family metric; it is not raw-BPM accuracy. Acc1 is shown alongside it.",fontsize=10)
    fig.text(.09,.105,"Error bars: 95% track-bootstrap intervals (10,000 resamples). Crops from one track are resampled together.",fontsize=10)
    fig.text(.09,.058,"Ours = the improved classical V2 compiled implementation. Point estimates do not establish statistically significant superiority.",fontsize=9,color="#61727D")
    save(fig,"01_BPM_accuracy",preview_dir)


def latency_figure(summary,preview_dir):
    fig=plt.figure(figsize=(13.333333,7.5))
    ax=fig.add_axes([.09,.235,.88,.51])
    ours=summary[0];lookup={r["method"]:r for r in summary}
    fig.text(.09,.93,"CPU PROCESSING LATENCY",fontsize=12,color="#61727D",weight="bold")
    fig.text(.09,.868,"Comparable to the fastest neural baseline",fontsize=23,weight="bold")
    fig.text(.09,.818,f"Ours: {ours['median_ms']:.2f} ms  ·  TempoCNN: {lookup['TempoCNN']['median_ms']:.2f} ms  ·  warm file-to-BPM processing",fontsize=12,color="#61727D")
    x=np.arange(4)
    values=np.asarray([r["median_ms"] for r in summary]);upper=np.asarray([r["p95_ms"] for r in summary])
    ax.axvspan(-.5,.5,color="#EFF7F5",zorder=0)
    ax.bar(x,values,color=[TEAL,BLUE,BLUE,BLUE],width=.58,zorder=3)
    ax.errorbar(x,values,yerr=[np.zeros(4),upper-values],fmt="none",ecolor=INK,capsize=5,linewidth=1.3,zorder=4)
    for i,value in enumerate(values):ax.text(i,upper[i]*1.15,f"{value:.2f} ms",ha="center",va="bottom",fontsize=13,weight="bold" if i==0 else "normal")
    ax.set_yscale("log");ax.set_ylim(5,6000);ax.set_ylabel("Processing time (ms, log scale)")
    ax.set_xticks(x,LABELS);ax.get_xticklabels()[0].set_weight("bold")
    ax.set_axisbelow(True);ax.grid(axis="y",which="major",color="#DDE3E7",alpha=.7)
    ax.text(.025,.96,f"Ours is {lookup['DeepRhythm']['median_ms']/ours['median_ms']:.2f}× faster than DeepRhythm\n"
            f"and {lookup['BeatThis']['median_ms']/ours['median_ms']:.2f}× faster than Beat This!",
            transform=ax.transAxes,ha="left",va="top",fontsize=12,
            bbox={"facecolor":"white","edgecolor":"#DEE8E5","boxstyle":"round,pad=.6"})
    fig.text(.09,.14,"Bars: medians. Whiskers: P95. Same 6 excerpts (2 tracks × 8/15/30 s), 1 CPU thread, 5 warm-ups + 20 repetitions each.",fontsize=10)
    fig.text(.09,.105,"Full file read, resampling, features, inference, decoding and BPM output included; downloads, initialization and JIT warm-up excluded.",fontsize=9.5)
    fig.text(.09,.058,"TempoCNN k4 uses the documented TensorFlow frontend on Windows. The small median gap to Ours is a near tie, not a proven speed advantage.",fontsize=9,color="#61727D")
    save(fig,"02_BPM_latency",preview_dir)


def documentation(summary,common,checks):
    ours=summary[0];lookup={r["method"]:r for r in summary}
    review=(f"Our improved classical detector achieved the highest observed octave-tolerant tempo accuracy "
        f"(Acc2: {100*ours['acc2']:.1f}%) among the four evaluated methods, while matching the fastest neural baseline's "
        f"observed CPU processing latency ({ours['median_ms']:.2f} ms vs. {lookup['TempoCNN']['median_ms']:.2f} ms for TempoCNN).")
    lines=["# BPM 비교: 발표용 Figure 두 개", "",
        "이 폴더의 자료는 새 알고리즘 실험이 아니라, 이미 실행한 독립 평가의 원시 측정값을 정리한 것입니다.", "",
        "- [Figure 1: 정확도](../01_BPM_accuracy.png)",
        "- [Figure 2: CPU 처리 시간](../02_BPM_latency.png)",
        "- PowerPoint용 SVG는 `vector/`에 같은 Figure 두 개가 있습니다.", "",
        "## 리뷰에 쓸 수 있는 문장", "",review,"",
        "우리 방법이 가장 높았다는 정확도 주장은 **Acc2(반·두 배 BPM 허용)** 기준입니다. "
        "원래 BPM 자체를 비교하는 Acc1은 함께 표시했습니다. Acc1 최고는 Beat This!이며, "
        "TempoCNN과 우리 방법의 처리 시간은 비슷합니다. 오차 구간이 겹치므로 보편적·통계적으로 유의한 우월성이나 "
        "정식 동등성 검정을 주장하지 않습니다.", "",
        "| 방법 | Acc1 | Acc2 | 중앙값 ms | 평균 ms | P95 ms |",
        "| --- | --- | --- | --- | --- | --- |"]
    for row in summary:
        lines.append(f"| {row['label']} | {100*row['acc1']:.1f}% | {100*row['acc2']:.1f}% | {row['median_ms']:.2f} | {row['mean_ms']:.2f} | {row['p95_ms']:.2f} |")
    lines += ["", "## 비교군과 평가 조건", "",
        "비교군은 처음부터 요청됐고 실제 pretrained 가중치를 로딩한 세 모델을 유지했습니다: "
        "DeepRhythm 0.7, TempoCNN deeptemp-k4-3, Beat This! small0. 더 약한 모델로 교체하지 않았습니다.", "",
        "**Ours는 `OUR_V2_FAST` 개선 버전**입니다. 원본 `OUR_EXISTING`과 다르며, 기존 로봇 제어 코드는 교체하지 않았습니다. "
        "개발에 사용한 50곡과 다른 50곡을 사용했고 곡 중복은 0입니다. 8·15·30초의 150구간을 모두 정확도에 포함했습니다. "
        "개선 버전의 템포 설정은 평가 전에 고정했으며, 컴파일 최적화 전후 150구간의 BPM과 박자 배열이 정확히 같음을 확인했습니다.", "",
        "Acc1은 raw BPM 상대오차가 2% 이하일 때 성공입니다. Acc2는 예측값에 0.5, 1, 2를 각각 곱한 경우도 허용합니다. "
        "실패한 예측을 정확도 분모에서 제외하지 않았습니다. MAE 등 전체 수치는 `data/summary.csv`에 있습니다.", "",
        "정확도 오차 막대는 50곡을 곡 단위로 10,000회 bootstrap한 95% 구간입니다. 같은 곡의 세 구간을 독립 표본으로 간주하지 않았습니다. "
        "구간이 겹치므로 최고 점수는 이번 표본의 관측값을 뜻합니다. 신경망의 학습 자료 중복 여부는 일부 확인되지 않았습니다.", "",
        "속도는 네 방법이 모두 처리한 **대표 입력 6개**(두 곡 × 8·15·30초)를 전부 사용했습니다. "
        "각 입력에서 CPU 1 thread, 준비 실행 5회, 측정 20회이며 모델당 측정값은 120개입니다. "
        "모델 실행 순서는 매 반복마다 무작위로 섞었습니다. 파일 읽기부터 최종 BPM까지의 전체 처리 비용을 포함합니다. "
        "오디오를 모으는 시간이나 로봇의 실제 반응 지연을 측정한 값은 아닙니다.", "",
        "**이전 20.05 ms 표와의 차이:** 이전 표는 원본 검출기까지 포함한 6개 방법의 공통 성공 입력을 사용해 "
        "8초 입력이 제외된 15·30초 4구간이었습니다. 이번 그림은 표시하는 네 방법의 공통 입력 6구간을 모두 사용하므로 "
        "중앙값이 달라집니다. 새 속도 값을 만들거나 유리한 반복만 선택한 것이 아닙니다. 이전 표는 "
        "`data/previous_four_clip_summary.json`와 원래 실험 폴더에 보존돼 있습니다.", "",
        "TempoCNN은 Windows에서 Essentia가 설치되지 않아 공식 가중치를 TensorFlow와 원저자의 feature frontend로 실행했습니다. "
        "Beat This!의 BPM은 ordinary beat 간격의 중앙값에서 유도했습니다. 다운비트로 BPM을 계산하지 않았습니다. "
        "최초 컴파일·모델 로딩 비용은 warm latency에 포함되지 않으며 `data/compiler_startup.json`에 별도로 보존했습니다.", "",
        "DeepRhythm의 공식 분할은 완성된 8초 단위만 사용하므로 8·15·30초 파일의 실제 신경망 문맥은 8·8·24초입니다. "
        "파일 전체를 읽고 resampling하는 시간은 포함됩니다. TempoCNN의 공식 feature frontend는 8초 입력을 "
        "256-frame 문맥(약 11.9초)으로 zero-pad합니다. 이 처리를 임의로 바꾸지 않았고, 모델마다 요구하는 전처리를 사용했습니다.", "",
        "## Figure 재생성", "",
        "```powershell", "# experiment/result/1_BPM/benchmark 폴더에서 실행",
        "..\\..\\..\\3_BPM_check\\.venv\\Scripts\\python.exe make_figures.py", "```", "",
        "독립 환경에서는 Python 3.11에서 `pip install -r requirements.txt` 후 `python make_figures.py`를 실행하면 됩니다. "
        "다운로드나 모델 추론 없이 포함된 CSV에서 그림을 다시 만듭니다. 원래 실험 CSV를 다시 묶으려면 "
        "같은 폴더에서 `python collect_inputs.py`를 먼저 실행합니다.", "",
        "원시 예측 600개와 속도 측정 480개, 모델/하드웨어/분할/검증 metadata는 `data/`에 있습니다. "
        "`track_references.csv`에 50곡의 공개 BPM 주석 출처를, `input_clips.csv`에 실제 구간 길이와 WAV 해시를 보존했습니다. "
        "`source_provenance.json`은 선택 기준, 측정 범위와 원본 파일의 해시를 기록합니다. "
        "전체 코드와 측정은 `../../../3_BPM_check/improvements/`에서 확인할 수 있습니다. "
        "기존 `1_BPM` 영상과 분석 자산은 유지했습니다.", ""]
    (BASE/"README.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    captions=["# Figure captions and slide text", "", "## Figure 1", "",
        "Tempo estimation accuracy on 50 held-out GTZAN-mini tracks (150 matched 8/15/30-second excerpts). "
        "Acc1 requires the raw estimate within 2% of the published reference; Acc2 additionally permits half/double tempo. "
        "Error bars are 95% track-bootstrap intervals. Ours has the highest observed Acc2; raw Acc1 is shown separately.", "",
        "## Figure 2", "",
        "Warm CPU file-to-BPM processing time on all six representative excerpts supported by the four displayed methods. "
        "Bars show medians and whiskers show P95 over 20 measurements per excerpt after five warm-ups, with one CPU thread "
        "and randomized interleaved method order. Ours and TempoCNN have comparable observed latency. "
        "All input loading, preprocessing and required output processing are included; startup and JIT warm-up are excluded.", "",
        "## Review sentence", "",review,"",
        "## Raw-accuracy qualification", "",
        f"Ours scored {100*ours['acc1']:.1f}% Acc1 versus {100*lookup['BeatThis']['acc1']:.1f}% for Beat This!; "
        "the highest-accuracy statement applies specifically to Acc2, not to every accuracy metric.", ""]
    (BASE/"CAPTIONS.md").write_text("\n".join(captions),encoding="utf-8",newline="\n")
    verification={"status":"passed","prediction_metric_checks":checks,"accuracy_records":600,"timing_records":480,
        "accuracy_tracks":50,"accuracy_clips_per_method":150,"latency_clips":common,"latency_repetitions_per_clip":20,
        "bootstrap_replicates":10000,"bootstrap_seed":2026,
        "highest_observed_acc2":ours["acc2"]==max(r["acc2"] for r in summary),
        "lowest_observed_median":ours["median_ms"]==min(r["median_ms"] for r in summary),
        "raw_acc1_best_method":max(summary,key=lambda r:r["acc1"])["method"],
        "interpretation":"Highest observed Acc2 and comparable observed latency to TempoCNN; no generic raw-accuracy or statistically proven superiority claim",
        "output_hashes":{name:hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() for name in ("01_BPM_accuracy.png","02_BPM_latency.png")}}
    write_json(BASE/"verification.json",verification)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir",type=Path)
    args=parser.parse_args()
    summary,common,checks=load_and_summarize()
    style();accuracy_figure(summary,args.preview_dir);latency_figure(summary,args.preview_dir)
    documentation(summary,common,checks)
    print(json.dumps({"figures":[str(OUTPUT/"01_BPM_accuracy.png"),str(OUTPUT/"02_BPM_latency.png")],
                      "summary":summary},indent=2))


if __name__=="__main__":main()
