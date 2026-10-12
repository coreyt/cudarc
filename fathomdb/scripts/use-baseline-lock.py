#!/usr/bin/env python3
"""Put the baseline lock in place for the tree it is copied into.

Run with `python3 -I`. Standard library only (Python 3.11+ for tomllib).

Usage: use-baseline-lock.py [--baseline LOCK] [DEST]

Copies `fathomdb/qualification/Cargo.lock.baseline` (next to this script;
`--baseline` overrides) to DEST (default `Cargo.lock` in the working
directory) and sets the root package's name to the `[package] name` of the
`Cargo.toml` in DEST's directory. The baseline names the root `cudarc`, as
at `ec1cac71`; from `d04ea18` the package is `cudarc-fathomdb`, and without
the rename every `--locked` command fails. Every other byte is copied
unchanged, so the dependency set is the baseline's.

The root is the one `[[package]]` entry named `cudarc` with no `source`
line. Zero or several such entries is an error, and DEST is not written.
The baseline itself is never modified.
"""

import argparse
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE.parent / "qualification" / "Cargo.lock.baseline"
BASELINE_ROOT = "cudarc"


def package_name(manifest):
    with open(manifest, "rb") as f:
        data = tomllib.load(f)
    name = data.get("package", {}).get("name")
    if not isinstance(name, str) or not name:
        raise SystemExit(f"use-baseline-lock: no [package] name in {manifest}")
    return name


def rename_root(text, new_name):
    """`text` with the root entry's name line set to `new_name`."""
    lines = text.split("\n")
    root_name_line = f'name = "{BASELINE_ROOT}"'
    candidates = []
    i = 0
    while i < len(lines):
        if lines[i] != "[[package]]":
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("["):
            j += 1
        block = range(i + 1, j)
        names = [k for k in block if lines[k] == root_name_line]
        has_source = any(lines[k].startswith("source = ") for k in block)
        if names and not has_source:
            candidates.append(names[0])
        i = j
    if len(candidates) != 1:
        raise SystemExit(
            f"use-baseline-lock: expected exactly one root entry "
            f'(name = "{BASELINE_ROOT}", no source), found {len(candidates)}'
        )
    lines[candidates[0]] = f'name = "{new_name}"'
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("dest", nargs="?", type=Path, default=Path("Cargo.lock"))
    args = p.parse_args(argv)
    dest = args.dest
    name = package_name(dest.parent / "Cargo.toml")
    with open(args.baseline, encoding="utf-8", newline="") as f:
        text = f.read()
    out = rename_root(text, name)
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(out)
    print(f"use-baseline-lock: {args.baseline} -> {dest} (root {name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
