#!/usr/bin/env python3
"""Account for every test of a cudarc-fathomdb run by name (test plan § 1.2).

Run with `python3 -I`. Standard library only.

Subcommands:

  expect  Print the expected list for a cell: the upstream inventory
          (`fathomdb/test-inventory/upstream-<set>-<v>.txt`) plus every landed
          row of `fork-catalogue.toml` whose cfg matches, after libtest's
          filtering (substring or `--exact` filters, `--skip`, `--ignored` /
          `--include-ignored`). The list is never derived from the build.
  names   Print the landed fork tests of one catalogue cell (one per line),
          for passing to `cargo test -- --exact`.
  check   Parse `cargo test` output (libtest's pretty format, one or more
          binaries, run or `--list` output) and compare it with an expected
          list, per binary and per module: missing, extra and
          unexpected-outcome names, cargo's "filtered out" count, and the
          summary line. Exit status 1 on any difference.
  unlanded
          Read `--list` output and fail if it contains a catalogue test that
          is not yet marked `landed`, so a test cannot exist without being
          expected.

Expected-list format (`# count-tests expected list v1`):

  [<binary label>]          unittests:<crate> | test:<target> | doc:<crate>
  # filtered-out: <N|unknown>
  <full test name> <ok|ignored>
"""

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CATALOGUE = HERE.parent / "test-inventory" / "fork-catalogue.toml"
DEFAULT_INVENTORY = HERE.parent / "test-inventory"
DEFAULT_MANIFEST = HERE.parent.parent / "Cargo.toml"

UNNAMED = "?"
HEADER = "# count-tests expected list v1"
OUTCOMES = ("ok", "FAILED", "ignored")

_RUNNING = re.compile(r"^\s*Running (unittests )?(\S+) \(([^)]*)\)\s*$")
_DOCTESTS = re.compile(r"^\s*Doc-tests (\S+)\s*$")
_TEST_LINE = re.compile(r"^test (\S+) \.\.\. ?(.*)$")
_OUTCOME = re.compile(r"^(ok|FAILED|ignored)(?:, .*)?$")
_SUMMARY = re.compile(
    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out"
)
_LIST_LINE = re.compile(r"^(\S+): (test|benchmark)$")
_LIST_SUMMARY = re.compile(r"^(\d+) tests?, (\d+) benchmarks?$")


@dataclass
class BinaryResult:
    tests: dict = field(default_factory=dict)
    listed: set = field(default_factory=set)
    summary: dict | None = None
    list_total: int | None = None
    unresolved: list = field(default_factory=list)
    summaries: int = 0


def _binary_label(unittests, target, path):
    stem = Path(path).name
    crate = stem.rsplit("-", 1)[0] if "-" in stem else stem
    if unittests:
        return f"unittests:{crate}"
    return f"test:{Path(target).stem}"


def parse_output(text):
    """Group libtest output by binary. Returns {label: BinaryResult}."""
    binaries = {}
    current = None
    pending = None

    def cur():
        nonlocal current
        if current is None:
            current = UNNAMED
        return binaries.setdefault(current, BinaryResult())

    def flush_pending():
        nonlocal pending
        if pending is not None:
            cur().unresolved.append(pending)
            cur().tests[pending] = "unresolved"
            pending = None

    for raw in text.splitlines():
        line = raw.rstrip("\r\n").rstrip()
        m = _RUNNING.match(line)
        if m:
            flush_pending()
            current = _binary_label(bool(m.group(1)), m.group(2), m.group(3))
            binaries.setdefault(current, BinaryResult())
            continue
        m = _DOCTESTS.match(line)
        if m:
            flush_pending()
            current = f"doc:{m.group(1).replace('-', '_')}"
            binaries.setdefault(current, BinaryResult())
            continue
        m = _TEST_LINE.match(line)
        if m:
            flush_pending()
            name, rest = m.group(1), m.group(2).strip()
            om = _OUTCOME.match(rest)
            if om:
                cur().tests[name] = om.group(1)
            else:
                pending = name
            continue
        if pending is not None:
            om = _OUTCOME.match(line.strip())
            if om:
                cur().tests[pending] = om.group(1)
                pending = None
                continue
        m = _SUMMARY.match(line.strip())
        if m:
            flush_pending()
            b = cur()
            b.summaries += 1
            b.summary = dict(
                passed=int(m.group(1)),
                failed=int(m.group(2)),
                ignored=int(m.group(3)),
                measured=int(m.group(4)),
                filtered_out=int(m.group(5)),
            )
            continue
        m = _LIST_SUMMARY.match(line.strip())
        if m:
            cur().list_total = int(m.group(1))
            continue
        m = _LIST_LINE.match(line)
        if m and m.group(2) == "test":
            cur().listed.add(m.group(1))
    flush_pending()
    return binaries


@dataclass
class Section:
    names: dict = field(default_factory=dict)
    filtered_out: int | None = None
    filtered_out_unknown: bool = False


@dataclass
class Expected:
    sections: dict = field(default_factory=dict)


def parse_expected(text):
    exp = Expected()
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = exp.sections.setdefault(line[1:-1], Section())
            continue
        if line.startswith("#"):
            m = re.match(r"^#\s*filtered-out:\s*(\S+)", line)
            if m and section is not None:
                if m.group(1).isdigit():
                    section.filtered_out = int(m.group(1))
                else:
                    section.filtered_out_unknown = True
            continue
        if section is None:
            section = exp.sections.setdefault(UNNAMED, Section())
        parts = line.split()
        name = parts[0]
        outcome = parts[1] if len(parts) > 1 else "ok"
        if outcome not in ("ok", "ignored"):
            raise ValueError(f"expected outcome must be ok or ignored: {raw!r}")
        section.names[name] = outcome
    return exp


def _match_sections(expected, observed):
    """Pair expected sections with observed binaries.

    An unnamed observed binary (no `Running` header in the log) is paired
    with the only expected section, if there is exactly one.
    """
    obs = dict(observed)
    if UNNAMED in obs and len(expected.sections) == 1:
        (only,) = expected.sections
        if only not in obs:
            obs[only] = obs.pop(UNNAMED)
    return obs


def _observed_names(b, mode):
    return set(b.listed) if mode == "list" else set(b.tests)


def compare(expected, observed, mode="run"):
    """Return a list of problems (empty when the run matches)."""
    problems = []
    obs = _match_sections(expected, observed)
    for label, sec in expected.sections.items():
        b = obs.get(label)
        if b is None:
            if sec.names:
                problems.append(f"{label}: binary missing from the output")
            elif mode == "run" and sec.filtered_out is not None:
                problems.append(f"{label}: binary missing from the output (no summary)")
            continue
        names = _observed_names(b, mode)
        for n in sorted(set(sec.names) - names):
            problems.append(f"{label}: missing {n}")
        for n in sorted(names - set(sec.names)):
            problems.append(f"{label}: extra {n}")
        if mode == "list":
            if b.list_total is None:
                problems.append(f"{label}: no list summary line")
            elif b.list_total != len(b.listed):
                problems.append(
                    f"{label}: list summary says {b.list_total} tests, {len(b.listed)} listed"
                )
            continue
        for n in sorted(b.unresolved):
            problems.append(f"{label}: no outcome parsed for {n}")
        for n in sorted(set(sec.names) & names):
            got = b.tests[n]
            if got != sec.names[n]:
                problems.append(f"{label}: outcome of {n}: expected {sec.names[n]}, got {got}")
        if b.summary is None:
            problems.append(f"{label}: no test result summary line")
            continue
        if b.summaries > 1:
            problems.append(f"{label}: {b.summaries} summary lines for one binary")
        counts = {o: sum(1 for v in b.tests.values() if v == o) for o in OUTCOMES}
        s = b.summary
        if (s["passed"], s["failed"], s["ignored"]) != (
            counts["ok"],
            counts["FAILED"],
            counts["ignored"],
        ):
            problems.append(
                f"{label}: summary says {s['passed']} passed, {s['failed']} failed, "
                f"{s['ignored']} ignored; test lines say {counts['ok']}, "
                f"{counts['FAILED']}, {counts['ignored']}"
            )
        if sec.filtered_out is not None and s["filtered_out"] != sec.filtered_out:
            problems.append(
                f"{label}: {s['filtered_out']} filtered out, expected {sec.filtered_out}"
            )
    for label, b in obs.items():
        if label in expected.sections:
            continue
        if _observed_names(b, mode):
            problems.append(f"{label}: unexpected binary ran tests: {sorted(_observed_names(b, mode))}")
    return problems


def _module(name):
    return name.rsplit("::", 1)[0] if "::" in name else "(crate root)"


def report(expected, observed, mode="run"):
    """Human-readable per-binary, per-module accounting."""
    out = []
    obs = _match_sections(expected, observed)
    for label in sorted(set(expected.sections) | set(obs)):
        sec = expected.sections.get(label, Section())
        b = obs.get(label, BinaryResult())
        names = _observed_names(b, mode)
        out.append(f"[{label}]")
        modules = sorted({_module(n) for n in set(sec.names) | names})
        for m in modules:
            e = sum(1 for n in sec.names if _module(n) == m)
            o = sum(1 for n in names if _module(n) == m)
            out.append(f"  {m}: expected {e}, observed {o}")
        if mode == "run":
            if b.summary is not None:
                s = b.summary
                fo = (
                    "unknown (not checked: no upstream inventory for this feature set)"
                    if sec.filtered_out_unknown
                    else str(sec.filtered_out)
                )
                out.append(
                    f"  summary: {s['passed']} passed; {s['failed']} failed; "
                    f"{s['ignored']} ignored; {s['filtered_out']} filtered out "
                    f"(expected filtered out: {fo})"
                )
            for n in sorted(names):
                out.append(f"    {n} {b.tests[n]}")
        else:
            out.append(f"  listed: {len(names)} (summary {b.list_total})")
    return "\n".join(out)


# ---------------------------------------------------------------- catalogue


def load_catalogue(path):
    with open(path, "rb") as f:
        data = tomllib.load(f)
    if data.get("schema") != 1:
        raise ValueError(f"{path}: unsupported schema {data.get('schema')!r}")
    for row in data.get("test", []):
        for key in ("name", "module", "binary", "kind", "cells", "ignored", "landed"):
            if key not in row:
                raise ValueError(f"{path}: test row {row.get('name')!r} lacks {key!r}")
    return data


def _manifest_features(manifest):
    with open(manifest, "rb") as f:
        return tomllib.load(f).get("features", {})


def enabled_features(manifest, feature_set, version, extra_features=()):
    """The closed set of enabled features for a cell.

    `drv` is `--no-default-features -F std,driver,dynamic-loading,cuda-<v>`;
    `def` is the default set plus `cuda-<v>`; `none` is
    `--no-default-features` plus `extra_features` only.
    """
    table = _manifest_features(manifest)
    if feature_set == "drv":
        roots = ["std", "driver", "dynamic-loading"]
    elif feature_set == "def":
        roots = ["default"]
    elif feature_set == "none":
        roots = []
    else:
        raise ValueError(f"unknown feature set {feature_set!r}")
    roots = roots + [f"cuda-{version}"] + list(extra_features)
    enabled, todo = set(), list(roots)
    while todo:
        f = todo.pop()
        if f in enabled:
            continue
        enabled.add(f)
        for dep in table.get(f, []):
            if dep.startswith("dep:") or "/" in dep:
                continue
            todo.append(dep)
    enabled.discard("default")
    return enabled


def _cfg_matches(cfg, features, version, arch, os_name):
    if any(f not in features for f in cfg.get("features", [])):
        return False
    if cfg.get("arch") and arch not in cfg["arch"]:
        return False
    if cfg.get("os") and os_name not in cfg["os"]:
        return False
    if "min_version" in cfg and version < cfg["min_version"]:
        return False
    if "max_version" in cfg and version > cfg["max_version"]:
        return False
    return True


def _full_name(row):
    return f"{row['module']}::{row['name']}" if row["module"] else row["name"]


def _fork_rows(catalogue, features, version, arch, os_name, landed_only=True):
    for row in catalogue.get("test", []):
        if landed_only and not row["landed"]:
            continue
        if _cfg_matches(row.get("cfg", {}), features, version, arch, os_name):
            yield row


def _libtest_selects(name, filters, exact, skips):
    if filters:
        hit = any((name == f) if exact else (f in name) for f in filters)
        if not hit:
            return False
    if any((name == s) if exact else (s in name) for s in skips):
        return False
    return True


def generate_expected(
    catalogue,
    inventory_dir,
    manifest,
    feature_set,
    version,
    arch,
    os_name,
    extra_features=(),
    filters=(),
    exact=False,
    skips=(),
    ignored_mode="normal",
    binaries=(),
):
    """Expected list text for one cell; see the module docstring."""
    features = enabled_features(manifest, feature_set, version, extra_features)
    upstream_ignored = {r["name"] for r in catalogue.get("upstream_ignored", [])}
    universe = {}  # label -> {name: ignored}
    unknown = set()
    inv = Path(inventory_dir) / f"upstream-{feature_set}-{version}.txt"
    lib_label = "unittests:cudarc"
    if feature_set in ("drv", "def") and not extra_features and inv.exists():
        for n in inv.read_text().split():
            universe.setdefault(lib_label, {})[n] = n in upstream_ignored
    else:
        unknown.add(lib_label)
    for row in _fork_rows(catalogue, features, version, arch, os_name):
        universe.setdefault(row["binary"], {})[_full_name(row)] = bool(row["ignored"])
    lines = [
        HEADER,
        f"# set: {feature_set}  version: {version}  arch: {arch}  os: {os_name}",
        f"# features: {','.join(sorted(features))}",
        f"# filters: {list(filters)} exact={exact} skip={list(skips)} ignored={ignored_mode}",
    ]
    for label in sorted(set(universe) | unknown):
        names = universe.get(label, {})
        selected = {}
        for n, ign in names.items():
            if not _libtest_selects(n, filters, exact, skips):
                continue
            if ignored_mode == "only":
                if ign:
                    selected[n] = "ok"
            elif ignored_mode == "include":
                selected[n] = "ok"
            else:
                selected[n] = "ignored" if ign else "ok"
        lines.append(f"[{label}]")
        if label in unknown:
            lines.append(f"# filtered-out: unknown (no {inv.name})")
        else:
            lines.append(f"# filtered-out: {len(names) - len(selected)}")
        lines += [f"{n} {o}" for n, o in sorted(selected.items())]
    return "\n".join(lines) + "\n"


def cell_rows(
    catalogue, cell, manifest, feature_set, version, arch, os_name, extra_features=(), module=None
):
    """(binary, full name) of the landed fork tests of `cell` (all cells
    when None) that compile for this feature set, version, arch and OS."""
    features = enabled_features(manifest, feature_set, version, extra_features)
    out = []
    for row in _fork_rows(catalogue, features, version, arch, os_name):
        if cell is not None and cell not in row["cells"]:
            continue
        if module is not None and row["module"] != module:
            continue
        out.append((row["binary"], _full_name(row)))
    return sorted(out)


def cell_names(*args, **kwargs):
    """Full names of `cell_rows`."""
    return sorted(name for _, name in cell_rows(*args, **kwargs))


def unlanded_present(catalogue, observed):
    """Catalogue names marked `landed = false` that appear in a run or list."""
    unlanded = {_full_name(r) for r in catalogue.get("test", []) if not r["landed"]}
    seen = set()
    for b in observed.values():
        seen |= set(b.listed) | set(b.tests)
    return sorted(unlanded & seen)


# ---------------------------------------------------------------- CLI


def _cell_args(p):
    p.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--set", dest="feature_set", choices=("drv", "def", "none"), required=True)
    p.add_argument("--version", type=int, required=True, help="e.g. 12060")
    p.add_argument("--arch", required=True, choices=("x86_64", "aarch64"))
    p.add_argument("--os", dest="os_name", required=True, choices=("linux", "windows"))
    p.add_argument(
        "--features", default="", help="extra comma-separated features (makes the universe partial)"
    )


def _features(s):
    return tuple(f for f in s.split(",") if f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("expect", help="print the expected list for a cell")
    _cell_args(pe)
    pe.add_argument("--inventory-dir", type=Path, default=DEFAULT_INVENTORY)
    pe.add_argument("--filter", action="append", default=[], help="libtest filter (repeatable)")
    pe.add_argument("--exact", action="store_true")
    pe.add_argument("--skip", action="append", default=[])
    pe.add_argument(
        "--binary", action="append", default=[], help="only this binary label (repeatable)"
    )
    g = pe.add_mutually_exclusive_group()
    g.add_argument("--ignored", action="store_true")
    g.add_argument("--include-ignored", action="store_true")

    pn = sub.add_parser("names", help="print the landed fork tests of a catalogue cell")
    _cell_args(pn)
    pn.add_argument("--cell", required=True)
    pn.add_argument("--module")
    pn.add_argument(
        "--with-binary", action="store_true", help="print '<binary label>\\t<name>' per line"
    )

    pc = sub.add_parser("check", help="compare cargo test output with an expected list")
    pc.add_argument("--expected", type=Path, required=True)
    pc.add_argument("--mode", choices=("run", "list"), default="run")
    pc.add_argument("logs", nargs="+", type=Path)

    pu = sub.add_parser("unlanded", help="fail if a listing contains a not-yet-landed test")
    pu.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    pu.add_argument("logs", nargs="+", type=Path)

    a = ap.parse_args(argv)
    if a.cmd == "expect":
        mode = "only" if a.ignored else "include" if a.include_ignored else "normal"
        sys.stdout.write(
            generate_expected(
                load_catalogue(a.catalogue),
                a.inventory_dir,
                a.manifest,
                a.feature_set,
                a.version,
                a.arch,
                a.os_name,
                _features(a.features),
                tuple(a.filter),
                a.exact,
                tuple(a.skip),
                mode,
            )
        )
        return 0
    if a.cmd == "names":
        rows = cell_rows(
            load_catalogue(a.catalogue),
            a.cell,
            a.manifest,
            a.feature_set,
            a.version,
            a.arch,
            a.os_name,
            _features(a.features),
            a.module,
        )
        for binary, name in rows:
            print(f"{binary}\t{name}" if a.with_binary else name)
        return 0
    if a.cmd == "check":
        expected = parse_expected(a.expected.read_text())
        observed = parse_output("\n".join(p.read_text(errors="replace") for p in a.logs))
        problems = compare(expected, observed, a.mode)
        print(report(expected, observed, a.mode))
        for p in problems:
            print(f"FAIL {p}")
        print("count-tests: " + ("FAIL" if problems else "PASS"))
        return 1 if problems else 0
    if a.cmd == "unlanded":
        observed = parse_output("\n".join(p.read_text(errors="replace") for p in a.logs))
        stale = unlanded_present(load_catalogue(a.catalogue), observed)
        for n in stale:
            print(f"FAIL {n} exists but fork-catalogue.toml says landed = false")
        print("count-tests unlanded: " + ("FAIL" if stale else "PASS"))
        return 1 if stale else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
