import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import prepare
import run_benchmark


class ReproductionContracts(unittest.TestCase):
    def test_cached_artifact_must_match_size_and_hash_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'weights').mkdir()
            payload=b'known official artifact fixture'
            path=root/'weights/test.pt';path.write_bytes(payload)
            expected=(len(payload),hashlib.sha256(payload).hexdigest())
            with patch.object(prepare,'ROOT',root),patch.object(prepare,'ASSETS',{'test.pt':expected}),patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')):
                self.assertTrue(prepare.download('test.pt')['verified'])
                path.write_bytes(b'x'*len(payload))
                with self.assertRaisesRegex(ValueError,'mismatch'):prepare.download('test.pt')

    def test_cpu_port_keeps_original_source_and_limits_diff_to_device_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkout=root/'upstream';folder=checkout/'deploy/save_TRT/diffusions/diffusion';folder.mkdir(parents=True)
            original=('img = th.randn(*shape).cuda()\n'+'t = th.tensor([i] * shape[0]).cuda()\n')*2
            path=folder/'gaussian_diffusion.py';path.write_text(original,encoding='utf-8')
            results=root/'results';results.mkdir()
            with patch.object(run_benchmark,'ROOT',root):
                port,changes=run_benchmark.policy_source(checkout,True,results)
            self.assertEqual(path.read_text(encoding='utf-8'),original)
            result=(port/'diffusions/diffusion/gaussian_diffusion.py').read_text(encoding='utf-8')
            self.assertNotIn('.cuda()',result)
            self.assertIn('device=img.device',result)
            self.assertEqual(len(changes),1)
            self.assertTrue((results/'cpu_device_port.patch').exists())


if __name__=='__main__':unittest.main()
