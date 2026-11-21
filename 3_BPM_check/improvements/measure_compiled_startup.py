"""Fresh-process first inference with an empty compiled-code cache."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
cache=ROOT/"data/cache"
cache.mkdir(parents=True,exist_ok=True)
assert cache.resolve().is_relative_to(ROOT.resolve())
with tempfile.TemporaryDirectory(prefix="numba_startup_",dir=cache) as temporary:
    assert Path(temporary).resolve().is_relative_to(cache.resolve())
    os.environ["NUMBA_CACHE_DIR"]=temporary
    stamp=time.perf_counter_ns()
    from adapters import get_estimator
    model=get_estimator("OUR_V2_FAST",device="cpu",threads=1)
    imported=time.perf_counter_ns()
    model.load()
    loaded=time.perf_counter_ns()
    clips=json.loads((ROOT/"improvements/holdout_manifest.json").read_text(encoding="utf-8"))
    clip=next(c for c in clips if c["duration_seconds"]>=29)
    start_inference=time.perf_counter_ns()
    value=model.predict(ROOT/clip["audio_path"])
    finished=time.perf_counter_ns()
    record={"method":"OUR_V2_FAST","device":"cpu","threads":1,"clip_id":clip["clip_id"],
            "adapter_import_ms":(imported-stamp)/1e6,"load_ms":(loaded-imported)/1e6,
            "first_inference_including_JIT_ms":(finished-start_inference)/1e6,
            "bpm":value["bpm"],"status":value["status"],
            "compiler_cache":"New empty temporary cache; filesystem/package cache not flushed",
            "weight_bytes":0,"source_sha256":model.metadata["sha256"]}
    (ROOT/"improvements/compiled_startup.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2))
