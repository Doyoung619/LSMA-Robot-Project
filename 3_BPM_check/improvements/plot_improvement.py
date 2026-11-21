"""Presentation figure regenerated only from measured accuracy/timing CSVs."""
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_results import common_real_latency


def read(path):return list(csv.DictReader(Path(path).open(encoding="utf-8",newline="")))


def main():
    base=ROOT/"improvements"
    scores={r["method"]:r for r in read(base/"holdout_results/accuracy_summary.csv") if r["tier"]=="real" and r["duration_seconds"]=="all"}
    fast=read(base/"compiled_predictions.csv")
    scores["OUR_V2_FAST"]={"acc1":np.mean([float(r["acc1"]) for r in fast]),
                           "acc2":np.mean([float(r["acc2"]) for r in fast])}
    latency,common=common_real_latency(read(base/"compiled_timing/latency_raw.csv"))
    times={r["method"]:r for r in latency}
    methods=["OUR_EXISTING","OUR_V2_FAST","DeepRhythm","TempoCNN","BeatThis"]
    labels=["Original\nclassical","Our\nimproved\nclassical","Deep\nRhythm","TempoCNN\nk4","Beat This!\nsmall0"]
    colors=["#A8B1B8","#217A73","#D08C37","#5D9279","#9966A5"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.2))
    x=np.arange(len(methods))
    for ax,metric,title in zip(axes[:2],["acc1","acc2"],["Raw tempo accuracy · Acc1","Half/double-tolerant accuracy · Acc2"]):
        values=[100*float(scores[m][metric]) for m in methods]
        ax.bar(x,values,color=colors,width=.66)
        ax.set_ylim(0,104);ax.set_ylabel("Accuracy (%)");ax.set_title(title,loc="left",fontsize=12,pad=13)
        ax.set_xticks(x,labels,fontsize=9);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
        for i,value in enumerate(values):ax.text(i,value+1.4,f"{value:.1f}",ha="center",fontsize=9)
    ax=axes[2]
    values=np.array([times[m]["median_ms"] for m in methods])
    upper=np.array([times[m]["p95_ms"] for m in methods])
    ax.bar(x,values,color=colors,width=.66)
    ax.errorbar(x,values,yerr=[np.zeros(len(x)),np.maximum(0,upper-values)],fmt="none",ecolor="#27313B",capsize=3,linewidth=1)
    ax.set_yscale("log");ax.set_ylim(10,max(upper)*1.8);ax.set_ylabel("Processing latency (ms, log scale)")
    ax.set_title("Warm CPU file-to-result latency",loc="left",fontsize=12,pad=13)
    ax.set_xticks(x,labels,fontsize=9);ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
    for i,value in enumerate(values):ax.text(i,upper[i]*1.12,f"{value:.2f}",ha="center",fontsize=9)
    fig.suptitle("Classical BPM detector improvement · 50 held-out tracks",fontsize=16,x=.06,ha="left",y=.98)
    fig.text(.06,.025,"Accuracy: 150 held-out 8/15/30-second excerpts; failures count as incorrect. Latency: identical 4 representative excerpts, 1 CPU thread, 5 warm-ups, 20 interleaved repetitions. Whiskers: P95.",fontsize=8)
    fig.subplots_adjust(left=.06,right=.99,bottom=.23,top=.80,wspace=.34)
    destination=base/"figures"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination/"improved_comparison.png",dpi=300,facecolor="white")
    preview=ROOT/"data/cache/improved_comparison_preview.png"
    preview.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(preview,dpi=100,facecolor="white")
    plt.close(fig)
    with (base/"figure_data.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=["method","accuracy_clips","acc1","acc2","median_ms","p95_ms","timing_clips","timing_repetitions"])
        writer.writeheader()
        for m in methods:writer.writerow({"method":m,"accuracy_clips":150,"acc1":scores[m]["acc1"],"acc2":scores[m]["acc2"],
            "median_ms":times[m]["median_ms"],"p95_ms":times[m]["p95_ms"],"timing_clips":len(common),"timing_repetitions":20})
    print(destination/"improved_comparison.png")


if __name__=="__main__":main()
