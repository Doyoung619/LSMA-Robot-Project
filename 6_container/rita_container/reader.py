"""One reader for mock, physical robots, simulation and future benchmarks."""
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .metrics import evaluate_samples


def iter_records(path):
    """Read a live journal; tolerate only an unfinished final JSON line."""
    path = Path(path)
    if not path.exists(): return
    with path.open(encoding="utf-8") as file:
        for line in file:
            try: yield json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith("\n") and not file.read(1): return
                raise


def load_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1: raise ValueError("Unsupported rollout schema version")
    summary_path = path / "summary.json"
    return {"manifest": manifest, **{name: list(iter_records(path / (name + ".jsonl")))
            for name in ("samples", "events", "metrics")},
            "summary": json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None}


def recompute_run(path, metrics=None):
    run = load_run(path)
    manifest = run["manifest"]
    profile_data = dict(manifest["robot"])
    for name in ("neutral", "lower", "upper", "vmax", "amax"):
        profile_data[name] = np.asarray(profile_data[name])
    profile_data["nj"] = len(profile_data["joint_names"])
    profile = SimpleNamespace(**profile_data)
    analysis = manifest["audio"]
    if analysis is not None:
        analysis = SimpleNamespace(**{**analysis, "beats": np.asarray(analysis["beats"])})
    grouped = defaultdict(list)
    for sample in run["samples"]: grouped[sample["source"]].append(sample)
    channels = {}
    for source, samples in grouped.items():
        channel = {"source": source, "provisional": False,
                   **evaluate_samples(samples, profile, analysis, manifest["metric_configuration"])}
        channel["extensions"] = {}
        for name, metric in (metrics or {}).items():
            try:
                value = metric(samples, profile, analysis)
                from .recorder import serial
                value = serial(value)
                json.dumps(value, allow_nan=False)
                channel["extensions"][name] = {"status": "ok", "value": value}
            except Exception as exc:
                channel["extensions"][name] = {"status": "error", "value": None, "error": str(exc)}
        channels[source] = channel
    return {"schema_version": 1, "run_id": manifest["run_id"], "status": manifest["status"],
            "error": manifest.get("error"), "recomputed": True,
            "sample_counts": {k: len(v) for k, v in grouped.items()},
            "channels": channels}


def flatten_runs(paths, recompute=False):
    rows = []
    for path in paths:
        run = load_run(path)
        m, summary = run["manifest"], run["summary"]
        if recompute: summary = recompute_run(path)
        for source, channel in (summary or {}).get("channels", {}).items():
            row = {"run_id": m["run_id"], "robot": m["robot"]["model"],
                   **m["metadata"], "source": source, "status": summary["status"],
                   "metric_status": channel.get("status"), "run_dir": str(Path(path).resolve())}
            row.update({k: v for k, v in channel.get("metrics", {}).items()
                        if v is None or isinstance(v, (str, int, float, bool))})
            rows.append(row)
        if summary is None:
            rows.append({"run_id": m["run_id"], "status": m["status"], "run_dir": str(path)})
    return rows
