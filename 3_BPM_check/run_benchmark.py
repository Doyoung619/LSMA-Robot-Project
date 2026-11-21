"""Sequential, isolated-process CPU benchmark. Run from this directory."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
DEFAULT_METHODS = ["OUR_EXISTING", "DeepRhythm", "TempoCNN", "BeatThis", "BeatNet"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding="utf-8")


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def error_result(method, exc):
    return {"method": method, "bpm": None, "beat_times": None, "confidence": None,
            "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def predict(estimator, path):
    try:
        value = estimator.predict(path)
        bpm = value.get("bpm")
        if value.get("status") == "success" and (bpm is None or not math.isfinite(float(bpm)) or bpm <= 0):
            raise ValueError("Adapter returned nonfinite/nonpositive BPM")
        return value
    except Exception as exc:
        return error_result(estimator.name, exc)


def worker(args):
    # Flush immediately so the parent can measure interpreter+stdlib startup.
    print("WORKER_STARTED", flush=True)
    config = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
    out = Path(config["results_dir"])
    clips = json.loads((out/"dataset_manifest.json").read_text(encoding="utf-8"))
    name = args.worker
    from metrics import tempo_metrics, beat_metrics
    metadata = {"method": name, "status": "unavailable", "error": None,
                "initialization_ms": None, "first_inference_ms": None,
                "package_import_ms": None, "process_start_ms": None}
    predictions, timings = [], []
    stamp = time.perf_counter_ns()
    try:
        from adapters import get_estimator
        estimator = get_estimator(name, config["device"], config["threads"])
        metadata["package_import_ms"] = (time.perf_counter_ns()-stamp)/1e6
        stamp = time.perf_counter_ns()
        estimator.load()
        metadata["initialization_ms"] = (time.perf_counter_ns()-stamp)/1e6
        metadata.update(estimator.metadata)
        metadata["status"] = "success"
        try:
            if name not in ("DeepRhythm", "BeatThis", "BeatNet"):
                raise ImportError
            import torch
            metadata["torch_version"] = torch.__version__
            metadata["torch_num_threads"] = torch.get_num_threads()
            metadata["torch_num_interop_threads"] = torch.get_num_interop_threads()
            metadata["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            pass
        def clip_path(clip):
            return str((ROOT / clip["audio_path"]).resolve())
        # Separate first call, even when the detector rejects this excerpt.
        first = next((c for c in clips if c["tier"] == "demo" and c["duration_seconds"] >= 29),
                     next((c for c in clips if c["duration_seconds"] >= 29), clips[0]))
        if config["device"] == "cuda":
            import torch
            torch.cuda.synchronize()
        stamp = time.perf_counter_ns()
        cold = predict(estimator, clip_path(first))
        if config["device"] == "cuda":
            torch.cuda.synchronize()
        metadata["first_inference_ms"] = (time.perf_counter_ns()-stamp)/1e6
        metadata["first_inference_status"] = cold["status"]
        metadata["first_inference_clip_id"] = first["clip_id"]
        for index, clip in enumerate(clips):
            value = predict(estimator, clip_path(clip))
            bpm = value.get("bpm") if value.get("status") == "success" else None
            beats = value.get("beat_times")
            row = {"method": name, **{k: clip.get(k) for k in
                ("clip_id", "tier", "track_id", "duration_seconds", "requested_duration_seconds", "reference_bpm")},
                "bpm": bpm, "beat_count": len(beats) if beats is not None else None,
                "beat_times_json": json.dumps(beats) if beats is not None else None,
                "confidence": value.get("confidence"), "status": value["status"], "error": value.get("error"),
                **tempo_metrics(bpm, clip.get("reference_bpm")),
                **beat_metrics(beats, clip.get("reference_beats"))}
            predictions.append(row)
            if index % 20 == 0:
                print(f"{name}: accuracy {index+1}/{len(clips)} BPM={bpm} status={value['status']}", flush=True)
            # Preserve completed results if an optional backend fails later.
            write_json(out/f"worker_{name}.json", {"metadata": metadata, "predictions": predictions, "timings": timings})
        selected_ids = set(config["timing_clip_ids"])
        for clip in clips:
            if clip["clip_id"] not in selected_ids:
                continue
            row = next(p for p in predictions if p["clip_id"] == clip["clip_id"])
            if row["status"] != "success":
                timings.append({"method": name, "clip_id": clip["clip_id"], "tier": clip["tier"],
                    "duration_seconds": clip["duration_seconds"], "stage": "file_to_result", "repetition": -1,
                    "latency_ms": None, "status": row["status"], "error": row["error"]})
                continue
            for _ in range(config["warmups"]):
                predict(estimator, clip_path(clip))
            for repetition in range(config["repetitions"]):
                if config["device"] == "cuda":
                    import torch
                    torch.cuda.synchronize()
                stamp = time.perf_counter_ns()
                value = predict(estimator, clip_path(clip))
                if config["device"] == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns()-stamp)/1e6
                timings.append({"method": name, "clip_id": clip["clip_id"], "tier": clip["tier"],
                    "duration_seconds": clip["duration_seconds"], "stage": "file_to_result",
                    "repetition": repetition, "latency_ms": elapsed,
                    "status": value["status"], "error": value.get("error")})
            print(f"{name}: timing {clip['clip_id']} done", flush=True)
            # Preloaded measurement includes feature extraction, inference, decoding.
            if hasattr(estimator, "predict_preloaded"):
                import soundfile as sf
                audio, sr = sf.read(clip_path(clip), dtype="float32", always_2d=True)
                try:
                    for _ in range(config["warmups"]):
                        value = estimator.predict_preloaded(audio, sr)
                    for repetition in range(config["repetitions"]):
                        if config["device"] == "cuda":
                            torch.cuda.synchronize()
                        stamp = time.perf_counter_ns()
                        value = estimator.predict_preloaded(audio, sr)
                        if config["device"] == "cuda":
                            torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns()-stamp)/1e6
                        timings.append({"method": name, "clip_id": clip["clip_id"], "tier": clip["tier"],
                            "duration_seconds": clip["duration_seconds"], "stage": "preloaded",
                            "repetition": repetition, "latency_ms": elapsed,
                            "status": value.get("status", "error"), "error": value.get("error")})
                except Exception as exc:
                    metadata["preloaded_error"] = str(exc)
            write_json(out/f"worker_{name}.json", {"metadata": metadata, "predictions": predictions, "timings": timings})
    except Exception as exc:
        metadata["status"] = "unavailable"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if metadata["initialization_ms"] is None:
            metadata["initialization_ms"] = (time.perf_counter_ns()-stamp)/1e6
        print(traceback.format_exc(), flush=True)
        for clip in clips:
            if not any(p["clip_id"] == clip["clip_id"] for p in predictions):
                predictions.append({"method": name, **{k: clip.get(k) for k in
                    ("clip_id", "tier", "track_id", "duration_seconds", "requested_duration_seconds", "reference_bpm")},
                    "status": "unavailable", "error": metadata["error"],
                    **tempo_metrics(None, clip.get("reference_bpm"))})
    write_json(out/f"worker_{name}.json", {"metadata": metadata, "predictions": predictions, "timings": timings})


def summarize(results, process_starts):
    all_predictions, all_timings, all_metadata = [], [], []
    manifest_path = results / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    nominal_durations = {c["clip_id"]: c.get("requested_duration_seconds", c["duration_seconds"]) for c in manifest}
    for path in sorted(results.glob("worker_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        meta = value["metadata"]
        meta["process_start_ms"] = process_starts.get(meta["method"])
        all_predictions.extend(value["predictions"])
        all_timings.extend(value["timings"])
        common_keys = ["method","status","error","initialization_ms","first_inference_ms","package_import_ms",
                       "process_start_ms","pretrained_loaded","weight_bytes","checkpoint","source_url","license","version","sha256"]
        all_metadata.append({**meta, "extra_json": json.dumps({k: v for k,v in meta.items() if k not in common_keys})})
    for row in all_predictions:
        row["requested_duration_seconds"] = nominal_durations.get(row["clip_id"], row.get("requested_duration_seconds", row["duration_seconds"]))
    write_csv(results/"predictions.csv", all_predictions, ["method","clip_id","tier","track_id","duration_seconds","requested_duration_seconds",
        "reference_bpm","bpm","beat_count","beat_times_json","confidence","status","error","absolute_error","acc1","acc2","beat_f1","beat_timing_error_ms"])
    write_csv(results/"latency_raw.csv", all_timings, ["method","clip_id","tier","duration_seconds","stage","repetition","latency_ms","status","error"])
    write_csv(results/"model_metadata.csv", all_metadata, common_keys + ["extra_json"])
    accuracy, latency = [], []
    for method in sorted({p["method"] for p in all_predictions}):
        for tier in sorted({p["tier"] for p in all_predictions}):
            rows = [p for p in all_predictions if p["method"] == method and p["tier"] == tier]
            for duration in ["all"] + sorted({p["requested_duration_seconds"] for p in rows}):
                subset = [p for p in rows if duration == "all" or p["requested_duration_seconds"] == duration]
                refs = [p for p in subset if p.get("reference_bpm") is not None]
                valid = [p for p in refs if p["status"] == "success"]
                errors = [p["absolute_error"] for p in valid]
                accuracy.append({"method": method,"tier":tier,"duration_seconds":duration,"n_total":len(subset),
                    "n_success": sum(p["status"] == "success" for p in subset), "n_reference":len(refs),
                    "mae_bpm":statistics.mean(errors) if errors else None,
                    "median_absolute_error_bpm":statistics.median(errors) if errors else None,
                    "acc1":sum(p.get("acc1",0) for p in refs)/len(refs) if refs else None,
                    "acc2":sum(p.get("acc2",0) for p in refs)/len(refs) if refs else None,
                    "coverage":len(valid)/len(refs) if refs else None})
    import numpy as np
    for method in sorted({p["method"] for p in all_timings}):
        for tier in sorted({p["tier"] for p in all_timings}):
            for stage in ["file_to_result", "preloaded"]:
                rows = [p for p in all_timings if p["method"]==method and p["tier"]==tier and p["stage"]==stage and p["status"]=="success"]
                if not rows:
                    continue
                for duration in ["all"] + sorted({p["duration_seconds"] for p in rows}):
                    subset = [p for p in rows if duration == "all" or p["duration_seconds"] == duration]
                    times = [p["latency_ms"] for p in subset]
                    latency.append({"method":method,"tier":tier,"duration_seconds":duration,"stage":stage,"n":len(times),
                        "median_ms":statistics.median(times),"mean_ms":statistics.mean(times),"p95_ms":float(np.percentile(times,95)),
                        "std_ms":statistics.pstdev(times),"rtf":statistics.median(p["latency_ms"]/1000/p["duration_seconds"] for p in subset)})
    write_csv(results/"accuracy_summary.csv", accuracy,["method","tier","duration_seconds","n_total","n_success","n_reference","mae_bpm","median_absolute_error_bpm","acc1","acc2","coverage"])
    write_csv(results/"latency_summary.csv", latency,["method","tier","duration_seconds","stage","n","median_ms","mean_ms","p95_ms","std_ms","rtf"])


def environment(threads):
    info = {"platform":platform.platform(),"os":platform.system(),"python":sys.version,"executable":sys.executable,
            "logical_cpu_count":os.cpu_count(),"inference_threads":threads,"thread_environment":{k:os.environ.get(k) for k in
            ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS")}}
    try:
        import psutil
        info.update(ram_bytes=psutil.virtual_memory().total,physical_cpu_count=psutil.cpu_count(logical=False))
    except ImportError:
        pass
    if platform.system() == "Windows":
        response = subprocess.run(["powershell","-NoProfile","-Command","Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],capture_output=True,text=True,encoding="utf-8",errors="replace")
        info["cpu_name"] = response.stdout.strip()
        response = subprocess.run(["powershell","-NoProfile","-Command","Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],capture_output=True,text=True,encoding="utf-8",errors="replace")
        info["gpu_names"] = response.stdout.strip().splitlines()
    else:
        info["cpu_name"] = platform.processor()
    response = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,encoding="utf-8",errors="replace")
    info["packages"] = response.stdout.splitlines()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick",action="store_true")
    mode.add_argument("--full",action="store_true")
    mode.add_argument("--smoke",action="store_true")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--real-count",type=int,default=None)
    parser.add_argument("--dataset",choices=["all","synthetic","real","demo"],default="all")
    parser.add_argument("--audio-path",type=Path)
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--methods",nargs="+",default=DEFAULT_METHODS)
    parser.add_argument("--skip-download",action="store_true")
    parser.add_argument("--repetitions",type=int)
    parser.add_argument("--worker",help=argparse.SUPPRESS)
    parser.add_argument("--worker-config",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.threads < 1:
        parser.error("--threads must be positive")
    for var in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS"):
        os.environ[var] = str(args.threads)
    os.environ.update(CUDA_VISIBLE_DEVICES="" if args.device=="cpu" else "0",TF_CPP_MIN_LOG_LEVEL="2",PYTHONUTF8="1")
    results = args.results.resolve()
    results.mkdir(parents=True,exist_ok=True)
    # Remove only this runner's prior worker files, never other outputs or source data.
    for path in results.glob("worker_*.json"):
        path.unlink()
    from dataset import prepare_dataset
    real_count = args.real_count if args.real_count is not None else (50 if args.full else 20)
    clips = prepare_dataset(ROOT, durations=(8,15,30), seed=2026,
                            real_count=real_count if args.dataset in ("all", "real") else 0,
                            audio_path=args.audio_path)
    if args.dataset != "all":
        clips = [c for c in clips if c["tier"] == args.dataset]
    if args.smoke:
        clips = [next((c for c in clips if c["tier"] == "demo" and c["duration_seconds"] >= 29),
                      next(c for c in clips if c["duration_seconds"] >= 29))]
    if not clips:
        parser.error("No selected audio clips")
    write_json(results/"dataset_manifest.json",clips)
    # Two fixed real tracks + one synthetic track + demo, with all three lengths.
    timing_tracks = []
    for tier, count in [("real",2),("synthetic",1),("demo",1)]:
        ids = list(dict.fromkeys(c["track_id"] for c in clips if c["tier"] == tier))
        timing_tracks.extend(ids[:count])
    config = {"mode":"full" if args.full else ("smoke" if args.smoke else "quick"),"device":args.device,
              "threads":args.threads,"seed":2026,"warmups":5 if args.full else 2,
              "repetitions":args.repetitions or (20 if args.full else 5),"methods":args.methods,
              "timing_clip_ids":[c["clip_id"] for c in clips if c["track_id"] in timing_tracks],
              "results_dir":str(results),"dataset":args.dataset,"durations_seconds":[8,15,30],
              "accuracy_denominator":"all annotated selected clips, including prediction failures; MAE successful predictions only",
              "accuracy_duration_grouping":"nominal requested duration; actual sample duration retained in predictions, manifest and RTF",
              "file_timing_scope":"file read to BPM and original beat outputs, all necessary preprocessing/postprocessing; no downloads",
              "timing_selection":"first two deterministic manifest real tracks, first synthetic track, and demo; all durations",
              "date":"2026-11-21","timezone":"America/New_York"}
    write_json(results/"benchmark_config.json",config)
    write_json(results/"environment.json",environment(args.threads))
    if not args.skip_download and any(m != "OUR_EXISTING" for m in args.methods):
        subprocess.run([sys.executable,str(ROOT/"download_models.py"),"--methods",*[m for m in args.methods if m!="OUR_EXISTING"]],cwd=ROOT,check=False)
    process_starts = {}
    for name in args.methods:
        print(f"\n=== {name}: isolated {args.device} worker ===",flush=True)
        stamp = time.perf_counter_ns()
        command = [sys.executable,str(ROOT/"run_benchmark.py"),"--worker",name,"--worker-config",str(results/"benchmark_config.json")]
        with (results/f"{name}.log").open("w",encoding="utf-8") as log:
            process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",env=os.environ)
            for line in process.stdout:
                if line.strip()=="WORKER_STARTED":
                    process_starts[name] = (time.perf_counter_ns()-stamp)/1e6
                else:
                    print(line,end="",flush=True)
                log.write(line)
            code = process.wait()
        if code and not (results/f"worker_{name}.json").exists():
            from metrics import tempo_metrics
            message = f"worker exited {code}; see {name}.log"
            missing = [{"method":name, **{k:c.get(k) for k in ("clip_id","tier","track_id","duration_seconds","reference_bpm")},
                        "status":"unavailable","error":message,**tempo_metrics(None,c.get("reference_bpm"))} for c in clips]
            write_json(results/f"worker_{name}.json",{"metadata":{"method":name,"status":"unavailable","error":message},"predictions":missing,"timings":[]})
        summarize(results,process_starts)
    env = json.loads((results/"environment.json").read_text(encoding="utf-8"))
    for path in results.glob("worker_*.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        if meta.get("torch_version"):
            env["torch_version"] = meta["torch_version"]
            env["cuda_available"] = meta.get("cuda_available")
    write_json(results/"environment.json",env)
    from plot_results import generate as plot
    from report_results import generate as report
    plot(results)
    report(results)
    print(f"Results saved: {results}",flush=True)


if __name__ == "__main__":
    main()
