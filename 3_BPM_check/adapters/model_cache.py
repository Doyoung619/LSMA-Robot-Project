"""Local checkpoint identities. No download is performed by adapters."""
from pathlib import Path
import hashlib
import importlib.metadata

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
EXPECTED_SHA256 = {
    "deeprhythm-0.7.pth": "c7cc8cc0425929cd2bf695474d7ec1fd63ed0d0a4a68f361d4e4b57bd9b3d9c4",
    "deeptemp-k4-3.pb": "66f36ee56c5e87c98534bd5cadd1a38059096cc210712dc0557d077e6c8c8daa",
    "beat_this-small0.ckpt": "6074be2c4d490c5f6101fcc374a1ec72ae93456e23bb6019783b849f5dc7d47b",
}


def configure_torch_threads(threads):
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise


def checkpoint_metadata(filename, package, source_url, license, **extra):
    path = MODEL_DIR / filename
    if not path.is_file() or path.stat().st_size < 1024:
        raise FileNotFoundError(f"Missing checkpoint {path}; run python download_models.py first")
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256[filename]:
        raise ValueError(f"Checkpoint SHA256 mismatch for {filename}: {digest}")
    return {"checkpoint": filename, "weight_bytes": path.stat().st_size,
            "sha256": digest,
            "source_url": source_url, "license": license, "version": version,
            "pretrained_loaded": False, **extra}
