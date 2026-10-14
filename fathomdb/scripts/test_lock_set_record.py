"""Fixture tests for lock-set-record.py.

Run with either of:

    python3 -I fathomdb/scripts/test_lock_set_record.py
    python3 -I -m unittest discover -s fathomdb/scripts -p 'test_*.py'

The record must be byte-for-byte what FathomDB's guard
(`scripts/check-cudarc-fathomdb-lock-set.py`, schema v2) recomputes from
FathomDB's own tree, so the expected canonical text below is written out by
hand from that guard's format, not produced by the script under test.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "lock-set-record.py"
REG = "registry+https://github.com/rust-lang/crates.io-index"
X86 = "x86_64-unknown-linux-gnu"
ARM = "aarch64-unknown-linux-gnu"

LOCK = f"""
version = 4

[[package]]
name = "cudarc-fathomdb"
version = "0.19.10"
source = "{REG}"
checksum = "aa11"
dependencies = ["libloading", "half"]

[[package]]
name = "libloading"
version = "0.9.0"
source = "{REG}"
checksum = "bb22"

[[package]]
name = "half"
version = "2.7.1"
source = "{REG}"
checksum = "cc33"

[[package]]
name = "fathomdb"
version = "0.8.28"
"""


def metadata(features=("driver", "std")):
    """A fork-rooted `cargo metadata`: the root is a path package (no source)."""
    root = "path+file:///w/cudarc#cudarc-fathomdb@0.19.10"
    lib = f"{REG}#libloading@0.9.0"
    half = f"{REG}#half@2.7.1"
    dev = f"{REG}#tempfile@3.0.0"
    build = f"{REG}#cc@1.0.0"
    return {
        "packages": [
            {"id": root, "name": "cudarc-fathomdb", "version": "0.19.10", "source": None},
            {"id": lib, "name": "libloading", "version": "0.9.0", "source": REG},
            {"id": half, "name": "half", "version": "2.7.1", "source": REG},
            {"id": dev, "name": "tempfile", "version": "3.0.0", "source": REG},
            {"id": build, "name": "cc", "version": "1.0.0", "source": REG},
        ],
        "resolve": {
            "root": root,
            "nodes": [
                {
                    "id": root,
                    "features": list(features),
                    "deps": [
                        {"pkg": lib, "dep_kinds": [{"kind": None}]},
                        {"pkg": half, "dep_kinds": [{"kind": None}]},
                        {"pkg": dev, "dep_kinds": [{"kind": "dev"}]},
                        {"pkg": build, "dep_kinds": [{"kind": "build"}]},
                    ],
                },
                {"id": lib, "deps": []},
                {"id": half, "deps": []},
                {"id": dev, "deps": []},
                {"id": build, "deps": []},
            ],
        },
    }


EXPECTED_TEXT = (
    f"# {X86}\n"
    f"cudarc-fathomdb 0.19.10 {REG} aa11\n"
    f"half 2.7.1 {REG} cc33\n"
    f"libloading 0.9.0 {REG} bb22\n"
    f"# {ARM}\n"
    f"cudarc-fathomdb 0.19.10 {REG} aa11\n"
    f"half 2.7.1 {REG} cc33\n"
    f"libloading 0.9.0 {REG} bb22\n"
    "# features driver,std\n"
)


class LockSetRecord(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        (self.dir / "Cargo.lock").write_text(LOCK)

    def tearDown(self):
        self._tmp.cleanup()

    def run_script(self, x86=None, arm=None, lock=None):
        for name, meta in (("x86.json", x86 or metadata()), ("arm.json", arm or metadata())):
            (self.dir / name).write_text(json.dumps(meta))
        out = self.dir / "record.json"
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(SCRIPT),
             "--lock", str(lock or self.dir / "Cargo.lock"),
             "--metadata", f"{X86}={self.dir / 'x86.json'}",
             "--metadata", f"{ARM}={self.dir / 'arm.json'}",
             "--lock-commit", "a" * 40, "--run-id", "123", "--head-sha", "b" * 40,
             "--out", str(out)],
            capture_output=True, text=True, check=False,
        )
        return result, out

    def test_record_matches_the_fathomdb_guard_format(self):
        result, out = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(out.read_text())
        self.assertEqual(record["schema"], "fathomdb.cudarc-fathomdb-lock-set.v2")
        self.assertEqual(record["closure"], "linux-normal")
        self.assertEqual(record["root_features"], ["driver", "std"])
        self.assertEqual(record["set_sha256"], hashlib.sha256(EXPECTED_TEXT.encode()).hexdigest())
        self.assertEqual(record["fathomdb_lock_commit"], "a" * 40)
        self.assertEqual(
            record["fathomdb_lock_sha256"],
            hashlib.sha256((self.dir / "Cargo.lock").read_bytes()).hexdigest(),
        )
        self.assertEqual(record["ci_run_id"], 123)
        self.assertEqual(record["ci_head_sha"], "b" * 40)
        self.assertIsNone(record["ci_conclusion"])

    def test_root_takes_its_source_and_checksum_from_fathomdb_lock(self):
        result, out = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        root = json.loads(out.read_text())["targets"][X86][0]
        self.assertEqual(root, {"name": "cudarc-fathomdb", "version": "0.19.10", "source": REG, "checksum": "aa11"})

    def test_dev_and_build_dependencies_are_excluded(self):
        result, out = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        names = [m["name"] for m in json.loads(out.read_text())["targets"][ARM]]
        self.assertEqual(names, ["cudarc-fathomdb", "half", "libloading"])

    def test_closure_starts_at_cudarc_fathomdb_under_a_consumer_root(self):
        def consumed():
            meta = metadata()
            consumer = "path+file:///w/consumer#fathomdb-lock-consumer@0.0.0"
            node = meta["resolve"]["nodes"][0]["id"]
            meta["packages"].append(
                {"id": consumer, "name": "fathomdb-lock-consumer", "version": "0.0.0", "source": None})
            meta["resolve"]["nodes"].append(
                {"id": consumer, "deps": [{"pkg": node, "dep_kinds": [{"kind": None}]}]})
            meta["resolve"]["root"] = consumer
            return meta

        result, out = self.run_script(x86=consumed(), arm=consumed())
        self.assertEqual(result.returncode, 0, result.stderr)
        record = json.loads(out.read_text())
        self.assertEqual(record["set_sha256"], hashlib.sha256(EXPECTED_TEXT.encode()).hexdigest())

    def test_member_missing_from_fathomdb_lock_fails(self):
        lock = self.dir / "other.lock"
        lock.write_text(LOCK.replace('version = "2.7.1"', 'version = "2.7.2"'))
        result, out = self.run_script(lock=lock)
        self.assertEqual(result.returncode, 1)
        self.assertIn("half 2.7.1", result.stderr)
        self.assertFalse(out.exists())

    def test_root_features_differing_between_targets_fail(self):
        result, out = self.run_script(arm=metadata(features=("driver",)))
        self.assertEqual(result.returncode, 1)
        self.assertIn("root features differ", result.stderr)
        self.assertFalse(out.exists())

    def test_lock_without_one_registry_root_entry_fails(self):
        lock = self.dir / "other.lock"
        lock.write_text(LOCK.replace('name = "cudarc-fathomdb"', 'name = "cudarc"'))
        result, out = self.run_script(lock=lock)
        self.assertEqual(result.returncode, 1)
        self.assertIn("cudarc-fathomdb", result.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
