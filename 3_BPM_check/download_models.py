"""Download official checkpoints before initialization or inference is timed."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from adapters.model_cache import EXPECTED_SHA256

ROOT = Path(__file__).resolve().parent
MODELS = {
    "DeepRhythm": ("deeprhythm-0.7.pth", "https://raw.githubusercontent.com/bleugreen/deeprhythm/main/weights/deeprhythm-0.7.pth"),
    "TempoCNN": ("deeptemp-k4-3.pb", "https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k4-3.pb"),
    "BeatThis": ("beat_this-small0.ckpt", "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/small0.ckpt"),
}


def download_models(methods=None):
    directory = ROOT / "models"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in methods or MODELS:
        if name not in MODELS:
            continue
        filename, url = MODELS[name]
        path = directory / filename
        row = {"method": name, "filename": filename, "source_url": url}
        try:
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".partial")
                with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                temporary.replace(path)
            if path.stat().st_size < 1024:
                raise ValueError("Downloaded checkpoint is unexpectedly small")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != EXPECTED_SHA256[filename]:
                raise ValueError(f"Checkpoint SHA256 mismatch: expected {EXPECTED_SHA256[filename]}, got {digest}")
            row.update(status="cached", weight_bytes=path.stat().st_size,
                       sha256=digest)
            print(f"{name}: cached {path.stat().st_size:,} bytes", flush=True)
        except Exception as exc:
            row.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
            print(f"{name}: {row['error']}", flush=True)
        manifest.append(row)
    # Official metadata is also cached for provenance; never needed for inference.
    metadata = directory / "deeptemp-k4-3.json"
    if "TempoCNN" in (methods or MODELS) and not metadata.exists():
        try:
            with urllib.request.urlopen("https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k4-3.json", timeout=30) as response:
                metadata.write_bytes(response.read())
        except Exception as exc:
            print(f"Optional TempoCNN metadata download: {exc}")
    (directory / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(MODELS))
    manifest = download_models(parser.parse_args().methods)
    # Setup scripts must fail when a required download or hash check fails.
    # The benchmark runner may still choose to continue with other models.
    raise SystemExit(0 if all(row.get("status") == "cached" for row in manifest) else 1)
