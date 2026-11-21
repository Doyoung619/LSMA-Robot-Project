"""Fetch the pinned official Reachy source; never start a robot or a demo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "reachy_adaptation.json"
SOURCE = ROOT / "external" / "reachy_mini_dances_library"


def config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def git(*args, cwd=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", check=check)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source(source=SOURCE):
    source = Path(source).resolve()
    expected = config()["upstream"]
    if not (source / ".git").exists():
        raise FileNotFoundError("Pinned Reachy checkout missing. Run python setup_external.py first.")
    revision = git("rev-parse", "HEAD", cwd=source).stdout.strip()
    if revision != expected["revision"]:
        raise RuntimeError(f"Reachy revision mismatch: expected {expected['revision']}, found {revision}")
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=source).stdout.strip()
    if dirty:
        raise RuntimeError("Pinned Reachy tracked source is modified; refusing to call it an upstream reproduction.")
    source_files = ["LICENSE", "pyproject.toml", "examples/dance_demo.py",
                    "src/reachy_mini_dances_library/rhythmic_motion.py",
                    "src/reachy_mini_dances_library/collection/dance.py"]
    hashes = {}
    for name in source_files:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Required upstream source is missing: {name}")
        hashes[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"name": "Reachy Mini Dances Library", **expected,
            "official_source_unmodified": True, "source_files": hashes,
            "neural_checkpoint_required": False, "hardware_executed": False}


def setup(source=SOURCE):
    source = Path(source).resolve()
    expected = config()["upstream"]
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        git("clone", "--filter=blob:none", "--sparse", expected["url"], str(source))
        # Pure code/examples are sufficient; gallery GIFs need not be downloaded.
        git("sparse-checkout", "init", "--cone", cwd=source)
        git("sparse-checkout", "set", "src", "examples", cwd=source)
    remote = git("remote", "get-url", "origin", cwd=source).stdout.strip()
    if remote.removesuffix(".git").rstrip("/") != expected["url"].removesuffix(".git"):
        raise RuntimeError("Existing external directory is not the expected official repository.")
    if git("status", "--porcelain", "--untracked-files=no", cwd=source).stdout.strip():
        raise RuntimeError("External source has local modifications; setup will not discard them.")
    found = git("cat-file", "-e", expected["revision"]+"^{commit}", cwd=source, check=False)
    if found.returncode:
        git("fetch", "origin", expected["revision"], cwd=source)
    git("checkout", "--detach", expected["revision"], cwd=source)
    manifest = verify_source(source)
    destination = source.parent / "reachy_source_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_source() if args.verify_only else setup()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
