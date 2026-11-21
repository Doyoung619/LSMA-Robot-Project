import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from run_benchmark import source_text_hash


class SourceTextPortability(unittest.TestCase):
    def test_standard_git_line_endings_preserve_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=Path(temporary)/'a.py';b=Path(temporary)/'b.py'
            a.write_bytes(b'answer = 42\nprint(answer)\n');b.write_bytes(b'answer = 42\r\nprint(answer)\r\n')
            self.assertEqual(source_text_hash(a),source_text_hash(b))
            b.write_bytes(b'answer = 43\r\nprint(answer)\r\n')
            self.assertNotEqual(source_text_hash(a),source_text_hash(b))
