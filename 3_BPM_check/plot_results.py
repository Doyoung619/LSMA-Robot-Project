"""Regenerate presentation figures directly from saved benchmark CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["OUR_EXISTING", "DeepRhythm", "TempoCNN", "BeatThis", "BeatNet", "OUR_V2", "OUR_V2_FAST"]
COLORS = ["#335C81", "#D08C37", "#5D9279", "#9966A5", "#C66464", "#217A73", "#956C44"]


def read_csv(path):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, default=np.nan):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def pretty(method):
    return {"OUR_EXISTING": "Existing classical", "OUR_V2": "Classical V2 (new)", "OUR_V2_FAST": "Classical V2 compiled", "BeatThis": "Beat This! small0",
            "BeatThisSmall0": "Beat This! small0", "BEAT_THIS": "Beat This! small0",
            "TEMPOCNN": "TempoCNN k4", "TempoCNN": "TempoCNN k4",
            "DEEPRHYTHM": "DeepRhythm", "Beat This!": "Beat This! small0"}.get(method, method)


def method_order(methods):
    return sorted(set(methods), key=lambda value: (ORDER.index(value) if value in ORDER else 99, value))


def is_real(tier):
    value = str(tier).lower()
    return value in {"real", "gtzan", "gtzan_mini", "public", "c"} or "gtzan" in value


def is_synthetic(tier):
    return str(tier).lower() in {"synthetic", "synth", "b"}


def is_demo(tier):
    return str(tier).lower() in {"demo", "billie_jean", "billie", "a", "project"}


def successful(row):
    return row.get("status", "success") == "success"


def summarize_latencies(rows):
    output = []
    for method in method_order(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method and successful(row)]
        values = np.array([number(row["latency_ms"]) for row in selected])
        valid = np.isfinite(values)
        values = values[valid]
        if not len(values):
            continue
        durations = np.array([number(row["duration_seconds"]) for row in selected])[valid]
        output.append({"method": method, "n": len(values),
                       "n_clips": len({row["clip_id"] for row in selected}),
                       "median_ms": float(np.median(values)), "mean_ms": float(np.mean(values)),
                       "p95_ms": float(np.percentile(values, 95)), "std_ms": float(np.std(values)),
                       "rtf": float(np.median(values / durations / 1000))})
    return output


def common_real_latency(rows):
    """Use identical successful real clips for the primary latency comparison."""
    selected = [row for row in rows if is_real(row.get("tier")) and successful(row)
                and row.get("stage") == "file_to_result" and np.isfinite(number(row.get("latency_ms")))]
    methods = method_order(row["method"] for row in selected)
    clip_sets = [{row["clip_id"] for row in selected if row["method"] == method} for method in methods]
    common = set.intersection(*clip_sets) if clip_sets else set()
    selected = [row for row in selected if row["clip_id"] in common]
    return summarize_latencies(selected), sorted(common)


def accuracy_rows(rows, tier_test, duration="all"):
    return [row for row in rows if tier_test(row.get("tier"))
            and str(row.get("duration_seconds")) == str(duration)]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#27313B",
                         "text.color": "#27313B", "axes.edgecolor": "#A6ADB3",
                         "savefig.facecolor": "white", "figure.facecolor": "white",
                         "axes.axisbelow": True})


def save(fig, destination):
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def empty_figure(destination, title, reason):
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    ax.axis("off")
    ax.set_title(title, loc="left", pad=15)
    ax.text(.5, .5, reason, ha="center", va="center", wrap=True, transform=ax.transAxes)
    save(fig, destination)


def generate(results_dir="results"):
    root = Path(results_dir)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    style()
    raw = read_csv(root / "latency_raw.csv")
    accuracy = read_csv(root / "accuracy_summary.csv")
    predictions = read_csv(root / "predictions.csv")
    metadata = read_csv(root / "model_metadata.csv")
    loaded_methods = {row["method"] for row in metadata if row.get("status") == "success"}
    loaded_methods.update(row["method"] for row in predictions if successful(row))
    plot_data = []
    latency, common = common_real_latency(raw)

    if latency:
        fig, ax = plt.subplots(figsize=(8.7, 5.2))
        x = np.arange(len(latency))
        medians = np.array([row["median_ms"] for row in latency])
        p95 = np.array([row["p95_ms"] for row in latency])
        bars = ax.bar(x, medians, color=COLORS[:len(x)], width=.62)
        ax.errorbar(x, medians, yerr=[np.zeros(len(x)), np.maximum(0, p95-medians)],
                    fmt="none", ecolor="#27313B", capsize=5, linewidth=1.2, label="Median to P95")
        ax.set_xticks(x, [pretty(row["method"]) for row in latency], rotation=12, ha="right")
        ax.set_ylabel("Warm file-to-result latency (ms)")
        ax.set_title("BPM detection latency · CPU", loc="left", pad=18)
        ax.grid(axis="y", alpha=.18)
        ax.set_ylim(0, max(p95)*1.22)
        for bar, value, upper in zip(bars, medians, p95):
            ax.text(bar.get_x()+bar.get_width()/2, upper+max(p95)*.025,
                    f"{value:,.1f}", ha="center", va="bottom", fontsize=10)
        ax.legend(frameon=False, loc="upper left")
        fig.text(.12, -.02, f"Matched successful real excerpts: {len(common)}. Bars: medians; whiskers: P95. Downloads excluded.", fontsize=8)
        fig.tight_layout()
        save(fig, figures / "latency_comparison.png")
        for row in latency:
            plot_data.append({"figure": "latency_comparison", "tier": "real_matched", **row})
    else:
        empty_figure(figures / "latency_comparison.png", "BPM detection latency · CPU",
                     "No common successful real excerpts are available across the measured methods.")

    for tier_test, suffix, label in [(is_real, "", "Real music"), (is_synthetic, "_synthetic", "Synthetic diagnostics")]:
        selected = [row for row in accuracy_rows(accuracy, tier_test) if row["method"] in loaded_methods]
        if not selected:
            empty_figure(figures / f"accuracy_comparison{suffix}.png", f"BPM accuracy · {label}", "No annotated results available.")
            continue
        selected = sorted(selected, key=lambda row: method_order(r["method"] for r in selected).index(row["method"]))
        fig, ax = plt.subplots(figsize=(8.7, 5.2))
        x = np.arange(len(selected))
        for offset, key, color, legend in [(-.18, "acc1", "#335C81", "Acc1: within 2%"),
                                             (.18, "acc2", "#80AF9A", "Acc2: half/double allowed")]:
            values = [100*number(row.get(key)) for row in selected]
            bars = ax.bar(x+offset, values, width=.34, color=color, label=legend)
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    ax.text(bar.get_x()+bar.get_width()/2, value+1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        labels = [f"{pretty(row['method'])}\nN={row['n_reference']}, valid={row['n_success']}" for row in selected]
        ax.set_xticks(x, labels, rotation=8, ha="right")
        ax.set_ylim(0, 118)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"BPM accuracy · {label}", loc="left", pad=18)
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False, fontsize=9)
        fig.text(.12, -.02, "All tested durations pooled; repeated excerpts are correlated. Failed predictions count as incorrect.", fontsize=8)
        fig.tight_layout()
        save(fig, figures / f"accuracy_comparison{suffix}.png")
        for row in selected:
            plot_data.append({"figure": f"accuracy_comparison{suffix}", **row})

    real30 = accuracy_rows(accuracy, is_real, "30.0") or accuracy_rows(accuracy, is_real, "30")
    trade_accuracy = real30 or accuracy_rows(accuracy, is_real)
    by_method = {row["method"]: row for row in trade_accuracy}
    points = [row for row in latency if row["method"] in by_method]
    if points:
        fig, ax = plt.subplots(figsize=(8.7, 5.2))
        for index, row in enumerate(points):
            value = 100*number(by_method[row["method"]]["acc1"])
            ax.scatter(row["median_ms"], value, s=100, color=COLORS[index], edgecolor="white", linewidth=.8)
            ax.annotate(pretty(row["method"]), (row["median_ms"], value), xytext=(7, 8),
                        textcoords="offset points", fontsize=9)
            plot_data.append({"figure": "accuracy_latency_tradeoff", "method": row["method"],
                              "median_ms": row["median_ms"], "acc1": value/100,
                              "duration_seconds": "30" if real30 else "all", "tier": "real"})
        ax.set_xscale("log")
        xs = [row["median_ms"] for row in points]
        ax.set_xlim(min(xs)/1.45, max(xs)*2.2)
        ax.set_ylim(-4, 110)
        ax.set_ylabel("Real music Acc1 (%)")
        ax.set_xlabel("Matched real excerpt median file-to-result latency (ms, log scale)")
        ax.set_title("Accuracy-latency trade-off · CPU", loc="left", pad=18)
        ax.grid(alpha=.18)
        fig.text(.12, -.02, f"Accuracy: {'nominal 30 s excerpts' if real30 else 'all durations'}. Latency: {len(common)} common successful representative excerpts.", fontsize=8)
        fig.tight_layout()
        save(fig, figures / "accuracy_latency_tradeoff.png")
    else:
        empty_figure(figures / "accuracy_latency_tradeoff.png", "Accuracy-latency trade-off · CPU", "No paired real accuracy and latency measurements available.")

    real_raw = [row for row in raw if is_real(row.get("tier")) and successful(row) and row.get("stage") == "file_to_result"]
    if real_raw:
        fig, ax = plt.subplots(figsize=(8.7, 5.2))
        methods = method_order(row["method"] for row in real_raw)
        for index, method in enumerate(methods):
            durations, medians = [], []
            for duration in sorted({number(row["duration_seconds"]) for row in real_raw if row["method"] == method}):
                selected = [row for row in real_raw if row["method"] == method and number(row["duration_seconds"]) == duration]
                stats = summarize_latencies(selected)[0]
                durations.append(duration)
                medians.append(stats["median_ms"])
                plot_data.append({"figure": "duration_vs_latency", "tier": "real", "duration_seconds": duration, **stats})
            ax.plot(durations, medians, marker="o", linewidth=2, color=COLORS[index % len(COLORS)], label=pretty(method))
        ax.set_xticks([8, 15, 30])
        ax.set_xlabel("Audio duration (s)")
        ax.set_ylabel("Median file-to-result latency (ms)")
        ax.set_title("Audio duration vs. processing time · CPU", loc="left", pad=18)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
        fig.text(.12, -.02, "Successful representative real excerpts only. Missing points indicate failures or unavailable results.", fontsize=8)
        fig.tight_layout()
        save(fig, figures / "duration_vs_latency.png")
    else:
        empty_figure(figures / "duration_vs_latency.png", "Audio duration vs. processing time · CPU", "No real excerpt timings available.")

    demo = [row for row in predictions if is_demo(row.get("tier")) and successful(row) and np.isfinite(number(row.get("bpm")))]
    if demo:
        longest = max(number(row["duration_seconds"]) for row in demo)
        selected = [row for row in demo if number(row["duration_seconds"]) == longest]
        demo_method_order = method_order(row["method"] for row in selected)
        selected.sort(key=lambda row: demo_method_order.index(row["method"]))
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.axis("off")
        ax.set_title(f"Billie Jean · {longest:g} s qualitative demo", loc="left", pad=22)
        cells = []
        for row in selected:
            times = [number(t["latency_ms"]) for t in raw if t["method"] == row["method"] and t["clip_id"] == row["clip_id"]
                     and t.get("stage") == "file_to_result" and successful(t)]
            timing = f"{np.median(times):,.1f}" if times else "Unavailable"
            cells.append([pretty(row["method"]), f"{number(row['bpm']):.2f}", timing, row.get("beat_count") or "—"])
            plot_data.append({"figure": "billie_jean_comparison", "method": row["method"],
                              "duration_seconds": longest, "bpm": row["bpm"], "median_ms": np.median(times) if times else ""})
        table = ax.table(cellText=cells, colLabels=["Method", "Estimated BPM", "Median latency (ms)", "Detected beats"],
                         loc="center", cellLoc="center", colLoc="center", colWidths=[.29, .20, .28, .18])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.0)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#DBE0E4")
            if r == 0:
                cell.set_facecolor("#EAF0F5")
                cell.set_text_props(weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F6F8FA")
        fig.text(.12, .08, "No verified reference BPM or manual beat annotations: predictions are not ground truth.", fontsize=9)
        save(fig, figures / "billie_jean_comparison.png")
    else:
        (figures / "billie_jean_comparison.png").unlink(missing_ok=True)

    fields = ["figure", "method", "tier", "duration_seconds", "n", "n_clips", "n_reference", "n_success",
              "median_ms", "mean_ms", "p95_ms", "std_ms", "rtf", "acc1", "acc2", "bpm", "mae_bpm", "median_absolute_error_bpm", "n_total", "coverage"]
    with (root / "plot_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(plot_data)
    return {"figures": str(figures), "matched_real_clip_ids": common, "matched_latency": latency}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    outcome = generate(args.results)
    print(f"Saved figures to {outcome['figures']}; matched real latency clips: {len(outcome['matched_real_clip_ids'])}")
