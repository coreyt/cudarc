"""Fixture tests for use-baseline-lock.py.

Run with either of:

    python3 -I fathomdb/scripts/test_use_baseline_lock.py
    python3 -I -m unittest discover -s fathomdb/scripts -p 'test_*.py'

The fixture `fixtures/Cargo.lock.cudarc-fathomdb` is what cargo 1.95.0 wrote,
offline, when it updated the baseline lock for a root package named
`cudarc-fathomdb` (`cp Cargo.lock.baseline Cargo.lock`, then `cargo metadata
--offline`, on a copy of the tree at `d04ea18`). `cargo generate-lockfile
--offline` is not the oracle: it re-resolves from the local registry cache
and may pick other versions than the baseline's.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "use-baseline-lock.py"
BASELINE = HERE.parent / "qualification" / "Cargo.lock.baseline"
RENAMED = HERE / "fixtures" / "Cargo.lock.cudarc-fathomdb"
ROOT_NAME_LINE = 'name = "cudarc"'


def manifest(name):
    return f'[package]\nname = "{name}"\nversion = "0.19.10"\n'


class UseBaselineLock(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_script(self, *args, baseline=None):
        cmd = [sys.executable, "-I", str(SCRIPT)]
        if baseline is not None:
            cmd += ["--baseline", str(baseline)]
        cmd += [str(a) for a in args]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=self.dir)

    def tree(self, name):
        (self.dir / "Cargo.toml").write_text(manifest(name))
        return self.dir / "Cargo.lock"

    def test_renames_the_root_to_the_manifest_name(self):
        dest = self.tree("cudarc-fathomdb")
        r = self.run_script(dest)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(dest.read_bytes(), RENAMED.read_bytes())

    def test_default_destination_is_cargo_lock_in_the_working_directory(self):
        dest = self.tree("cudarc-fathomdb")
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(dest.read_bytes(), RENAMED.read_bytes())

    def test_is_a_byte_identical_copy_when_the_manifest_name_is_cudarc(self):
        dest = self.tree("cudarc")
        r = self.run_script(dest)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(dest.read_bytes(), BASELINE.read_bytes())

    def test_changes_exactly_the_root_name_line(self):
        dest = self.tree("cudarc-fathomdb")
        r = self.run_script(dest)
        self.assertEqual(r.returncode, 0, r.stderr)
        before = BASELINE.read_bytes().split(b"\n")
        after = dest.read_bytes().split(b"\n")
        self.assertEqual(len(before), len(after))
        changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        self.assertEqual(len(changed), 1, changed)
        i = changed[0]
        self.assertEqual(before[i], ROOT_NAME_LINE.encode())
        self.assertEqual(after[i], b'name = "cudarc-fathomdb"')
        self.assertEqual(before[i - 1], b"[[package]]")

    def test_a_registry_cudarc_is_not_the_root(self):
        text = BASELINE.read_text()
        extra = (
            '\n[[package]]\nname = "cudarc"\nversion = "0.17.8"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        )
        lock = self.dir / "baseline.lock"
        lock.write_text(text + extra)
        dest = self.tree("cudarc-fathomdb")
        r = self.run_script(dest, baseline=lock)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(
            dest.read_text(),
            RENAMED.read_text() + extra,
        )

    def test_no_root_candidate_fails_and_writes_nothing(self):
        lock = self.dir / "baseline.lock"
        lock.write_text(BASELINE.read_text().replace(ROOT_NAME_LINE, 'name = "other"', 1))
        dest = self.tree("cudarc-fathomdb")
        r = self.run_script(dest, baseline=lock)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("found 0", r.stderr)
        self.assertFalse(dest.exists())

    def test_two_root_candidates_fail_and_write_nothing(self):
        lock = self.dir / "baseline.lock"
        lock.write_text(
            BASELINE.read_text() + '\n[[package]]\nname = "cudarc"\nversion = "0.19.7"\n'
        )
        dest = self.tree("cudarc-fathomdb")
        r = self.run_script(dest, baseline=lock)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("found 2", r.stderr)
        self.assertFalse(dest.exists())

    def test_a_manifest_without_a_package_name_fails(self):
        (self.dir / "Cargo.toml").write_text("[workspace]\n")
        dest = self.dir / "Cargo.lock"
        r = self.run_script(dest)
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(dest.exists())

    def test_crlf_line_endings_are_kept(self):
        # A Windows checkout with core.autocrlf=true has a CRLF baseline.
        lock = self.dir / "baseline.lock"
        lock.write_bytes(BASELINE.read_bytes().replace(b"\n", b"\r\n"))
        dest = self.tree("cudarc-fathomdb")
        r = self.run_script(dest, baseline=lock)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(dest.read_bytes(), RENAMED.read_bytes().replace(b"\n", b"\r\n"))

    def test_the_baseline_itself_is_never_written(self):
        before = BASELINE.read_bytes()
        dest = self.tree("cudarc-fathomdb")
        self.run_script(dest)
        self.assertEqual(BASELINE.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
