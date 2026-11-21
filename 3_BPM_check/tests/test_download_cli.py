"""A bad cached weight must fail setup before anyone tries to time inference."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DownloadCLIContractTests(unittest.TestCase):
    def test_cached_hash_failure_has_nonzero_exit_and_manifest(self):
        payload = b"fixture checkpoint bytes; not a pretrained model\n" * 64
        for correct in (False, True):
            with self.subTest(correct_hash=correct), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                shutil.copyfile(ROOT / "download_models.py", folder / "download_models.py")
                (folder / "adapters").mkdir()
                (folder / "adapters/__init__.py").write_text("", encoding="utf-8")
                expected = hashlib.sha256(payload).hexdigest() if correct else "0" * 64
                (folder / "adapters/model_cache.py").write_text(
                    "EXPECTED_SHA256 = " + repr({"deeprhythm-0.7.pth": expected}), encoding="utf-8")
                (folder / "models").mkdir()
                (folder / "models/deeprhythm-0.7.pth").write_bytes(payload)
                result = subprocess.run([sys.executable, str(folder / "download_models.py"),
                                         "--methods", "DeepRhythm"], cwd=folder,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if correct else 1)
                manifest = json.loads((folder / "models/download_manifest.json").read_text())
                self.assertEqual(manifest[0]["status"], "cached" if correct else "unavailable")
                self.assertEqual((folder / "models/deeprhythm-0.7.pth").read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
