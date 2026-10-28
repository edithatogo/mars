"""Exercise build-only Python ABI edges with local Cargo fixtures, without compiling."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_native_dependency_boundary.py'


class NativeBoundaryTests(unittest.TestCase):
    def test_build_only_python_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'src').mkdir()
            (root / 'src/lib.rs').write_text('')
            (root / 'python/src').mkdir(parents=True)
            (root / 'python/src/lib.rs').write_text('')
            (root / 'python/Cargo.toml').write_text('[package]\nname="pyo3"\nversion="0.0.0"\nedition="2021"\n')
            manifest = root / 'Cargo.toml'
            manifest.write_text('[package]\nname="boundary-fixture"\nversion="0.0.0"\nedition="2021"\n[build-dependencies]\npyo3={path="python"}\n')
            subprocess.run(['cargo', 'generate-lockfile', '--offline', '--manifest-path', str(manifest)], check=True, capture_output=True)
            result = subprocess.run([sys.executable, str(SCRIPT), '--manifest', str(manifest)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('pyo3 v0.0.0', result.stdout)


if __name__ == '__main__':
    unittest.main()
