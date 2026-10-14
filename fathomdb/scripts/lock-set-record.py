"""Write the lock-set record FathomDB's guard checks (hosted-test Run 2).

FathomDB's `scripts/check-cudarc-fathomdb-lock-set.py` (schema v2)
recomputes, from FathomDB's tree, the normal-dependency closure of its one
`cudarc-fathomdb` node per Linux target, with the root's resolved features,
and compares it with a copy of this record. This script computes the same
set from this fork's root after FathomDB's Cargo.lock is copied in, so the
canonical text and digest below must stay byte-identical to that guard's.

The metadata comes from `fathomdb/lock-consumer`, a crate that depends on
this checkout by path with the dependency features FathomDB enables, so
the closure starts at the `cudarc-fathomdb` node, not the metadata root.
That node is a path package, so its source and checksum come from
FathomDB's lock entry (the published crate); every other member's source
comes from `cargo metadata` and its checksum from FathomDB's lock.

Inputs: --lock (FathomDB's Cargo.lock as fetched), --metadata TARGET=FILE
for both targets (`cargo metadata --filter-platform TARGET`), the FathomDB
lock commit, and the CI run id and head sha. The conclusion is not known
inside the job, so `ci_conclusion` is null; the receipt fills it in.

Exit 0 after writing --out, 1 on any inconsistency (nothing written).
"""

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

SCHEMA = "fathomdb.cudarc-fathomdb-lock-set.v2"
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu")
ROOT_PACKAGE = "cudarc-fathomdb"


class Inconsistent(Exception):
    pass


def lock_entries(lock_path):
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    sums, roots = {}, []
    for package in lock.get("package", []):
        source = package.get("source")
        if source is None:
            continue
        sums[(package["name"], package["version"], source)] = package.get("checksum")
        if package["name"] == ROOT_PACKAGE:
            roots.append(package)
    if len(roots) != 1:
        raise Inconsistent(f"FathomDB's lock must have exactly one registry {ROOT_PACKAGE} entry, found {len(roots)}")
    return sums, roots[0]


def closure(metadata, sums, lock_root):
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    roots = [ident for ident in nodes if packages[ident]["name"] == ROOT_PACKAGE]
    if len(roots) != 1:
        raise Inconsistent(f"expected exactly one {ROOT_PACKAGE} node in the metadata, found {len(roots)}")
    root = roots[0]
    if packages[root]["version"] != lock_root["version"]:
        raise Inconsistent(
            f"metadata has {ROOT_PACKAGE} {packages[root]['version']}, "
            f"FathomDB's lock pins {lock_root['version']}"
        )
    seen, stack = {root}, [root]
    while stack:
        for dep in nodes[stack.pop()].get("deps", []):
            if not any(kind.get("kind") is None for kind in dep.get("dep_kinds", [])):
                continue
            if dep["pkg"] not in seen:
                seen.add(dep["pkg"])
                stack.append(dep["pkg"])
    members = []
    for ident in seen - {root}:
        package = packages[ident]
        if package.get("source") is None:
            raise Inconsistent(f"{package['name']} {package['version']} in the closure has no registry or git source")
        key = (package["name"], package["version"], package["source"])
        if key not in sums:
            raise Inconsistent(
                f"{package['name']} {package['version']} is not in FathomDB's lock; "
                "cargo re-resolved it, so this is not the set FathomDB compiles"
            )
        members.append({
            "name": package["name"],
            "version": package["version"],
            "source": package["source"],
            "checksum": sums[key],
        })
    members.sort(key=lambda m: (m["name"], m["version"]))
    head = {
        "name": ROOT_PACKAGE,
        "version": lock_root["version"],
        "source": lock_root["source"],
        "checksum": lock_root.get("checksum"),
    }
    return [head, *members], sorted(nodes[root].get("features", []))


def canonical_text(targets, root_features):
    text = ""
    for target in TARGETS:
        text += f"# {target}\n"
        for m in targets[target]:
            text += f"{m['name']} {m['version']} {m['source']} {m['checksum']}\n"
    text += "# features " + ",".join(sorted(root_features)) + "\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--metadata", action="append", required=True, metavar="TARGET=FILE")
    parser.add_argument("--lock-commit", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        files = dict(item.split("=", 1) for item in args.metadata)
        if set(files) != set(TARGETS):
            raise Inconsistent(f"--metadata must name exactly {', '.join(TARGETS)}")
        sums, lock_root = lock_entries(args.lock)
        targets, features = {}, {}
        for target in TARGETS:
            meta = json.loads(Path(files[target]).read_text(encoding="utf-8"))
            targets[target], features[target] = closure(meta, sums, lock_root)
        if features[TARGETS[0]] != features[TARGETS[1]]:
            raise Inconsistent(f"the root features differ between targets: {features}")
    except Inconsistent as exc:
        print(f"FAIL lock-set record: {exc}", file=sys.stderr)
        return 1
    root_features = features[TARGETS[0]]
    record = {
        "schema": SCHEMA,
        "fathomdb_lock_commit": args.lock_commit,
        "fathomdb_lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        "ci_run_id": args.run_id,
        "ci_head_sha": args.head_sha,
        "ci_conclusion": None,
        "closure": "linux-normal",
        "root_features": root_features,
        "targets": targets,
        "set_sha256": hashlib.sha256(canonical_text(targets, root_features).encode("utf-8")).hexdigest(),
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"lock-set record: {len(targets[TARGETS[0]])} packages per target, set_sha256 {record['set_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
