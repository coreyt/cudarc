"""Fixture tests for count-tests.py (test plan § 1.2, § 10 FC4-0).

Run with either of:

    python3 -I -m unittest discover -s fathomdb/scripts -p 'test_*.py'
    python3 -I -m pytest fathomdb/scripts/test_count_tests.py
"""

import importlib.util
import io
import os
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("count_tests", HERE / "count-tests.py")
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

CORE = "driver::safe::core"
FB = f"{CORE}::fathomdb_alloc_fallback"
PP = f"{CORE}::fathomdb_private_pool"

# Two binaries: the lib unit tests and an integration test, as cargo prints
# them with stderr merged into stdout.
RUN_TWO_BINARIES = textwrap.dedent(
    f"""\
       Compiling cudarc v0.19.10 (/work/cudarc)
        Finished `test` profile [unoptimized + debuginfo] target(s) in 9.01s
         Running unittests src/lib.rs (target/debug/deps/cudarc-0123456789abcdef)

    running 4 tests
    test {FB}::a_failed_decision_is_not_remembered ... ok
    test {FB}::sync_context_round_trips_data ... ok
    test {PP}::private_context_in_a_fresh_process ... ignored, run by a_private_context_never_consults_or_records_the_process_decision
    test {PP}::pool_props_are_pinned_device_memory_on_the_ordinal ... ok

    test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 28 filtered out; finished in 0.21s

         Running tests/fathomdb_lib_major.rs (target/debug/deps/fathomdb_lib_major-fedcba9876543210)

    running 1 test
    test case_a_decoy ... ignored

    test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

       Doc-tests cudarc

    running 0 tests

    test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

    """
)

EXPECTED_TWO_BINARIES = textwrap.dedent(
    f"""\
    # count-tests expected list v1
    [unittests:cudarc]
    # filtered-out: 28
    {FB}::a_failed_decision_is_not_remembered ok
    {FB}::sync_context_round_trips_data ok
    {PP}::private_context_in_a_fresh_process ignored
    {PP}::pool_props_are_pinned_device_memory_on_the_ordinal ok
    [test:fathomdb_lib_major]
    # filtered-out: 0
    case_a_decoy ignored
    """
)

LIST_OUTPUT = textwrap.dedent(
    f"""\
         Running unittests src/lib.rs (target/debug/deps/cudarc-0123456789abcdef)
    {FB}::a_failed_decision_is_not_remembered: test
    {PP}::pool_props_are_pinned_device_memory_on_the_ordinal: test

    2 tests, 0 benchmarks
    """
)

FAILED_RUN = textwrap.dedent(
    f"""\
         Running unittests src/lib.rs (target/debug/deps/cudarc-0123456789abcdef)

    running 2 tests
    test {FB}::a_failed_decision_is_not_remembered ... ok
    test {FB}::sync_context_round_trips_data ... FAILED

    failures:

    ---- {FB}::sync_context_round_trips_data stdout ----
    thread 'main' panicked at src/driver/safe/core.rs:1:1:
    boom

    failures:
        {FB}::sync_context_round_trips_data

    test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 30 filtered out; finished in 0.10s

    error: test failed, to rerun pass `--lib`
    """
)


def expected_for(names_outcomes, filtered_out=None, label="unittests:cudarc"):
    lines = ["# count-tests expected list v1", f"[{label}]"]
    if filtered_out is not None:
        lines.append(f"# filtered-out: {filtered_out}")
    lines += [f"{n} {o}" for n, o in names_outcomes]
    return "\n".join(lines) + "\n"


class ParseOutput(unittest.TestCase):
    def test_run_output_is_grouped_by_binary_with_outcomes_and_summary(self):
        observed = ct.parse_output(RUN_TWO_BINARIES)
        self.assertEqual(
            sorted(observed), ["doc:cudarc", "test:fathomdb_lib_major", "unittests:cudarc"]
        )
        lib = observed["unittests:cudarc"]
        self.assertEqual(lib.tests[f"{FB}::a_failed_decision_is_not_remembered"], "ok")
        self.assertEqual(lib.tests[f"{PP}::private_context_in_a_fresh_process"], "ignored")
        self.assertEqual(len(lib.tests), 4)
        self.assertEqual(lib.summary["filtered_out"], 28)
        self.assertEqual(lib.summary["passed"], 3)
        self.assertEqual(lib.summary["ignored"], 1)
        self.assertEqual(observed["doc:cudarc"].tests, {})

    def test_list_output_records_names(self):
        observed = ct.parse_output(LIST_OUTPUT)
        lib = observed["unittests:cudarc"]
        self.assertEqual(
            lib.listed,
            {
                f"{FB}::a_failed_decision_is_not_remembered",
                f"{PP}::pool_props_are_pinned_device_memory_on_the_ordinal",
            },
        )
        self.assertEqual(lib.list_total, 2)

    def test_failed_outcome_is_parsed(self):
        lib = ct.parse_output(FAILED_RUN)["unittests:cudarc"]
        self.assertEqual(lib.tests[f"{FB}::sync_context_round_trips_data"], "FAILED")
        self.assertEqual(lib.summary["failed"], 1)

    def test_nocapture_output_between_name_and_outcome_is_tolerated(self):
        text = textwrap.dedent(
            f"""\
                 Running unittests src/lib.rs (target/debug/deps/cudarc-0123456789abcdef)

            running 1 test
            test {FB}::sync_context_round_trips_data ... SKIP fathomdb_alloc_fallback: no CUDA driver or device
            ok

            test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out; finished in 0.00s
            """
        )
        lib = ct.parse_output(text)["unittests:cudarc"]
        self.assertEqual(lib.tests, {f"{FB}::sync_context_round_trips_data": "ok"})

    def test_output_without_running_header_goes_to_an_unnamed_binary(self):
        text = "running 1 test\ntest a::b ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        observed = ct.parse_output(text)
        self.assertEqual(list(observed), ["?"])
        self.assertEqual(observed["?"].tests, {"a::b": "ok"})


class Check(unittest.TestCase):
    def check(self, expected_text, log_text, mode="run"):
        expected = ct.parse_expected(expected_text)
        observed = ct.parse_output(log_text)
        return ct.compare(expected, observed, mode)

    def test_matching_run_passes(self):
        self.assertEqual(self.check(EXPECTED_TWO_BINARIES, RUN_TWO_BINARIES), [])

    def test_missing_name_fails(self):
        exp = EXPECTED_TWO_BINARIES.replace(
            "case_a_decoy ignored\n", "case_a_decoy ignored\ncase_b_cache ignored\n"
        )
        problems = self.check(exp, RUN_TWO_BINARIES)
        self.assertTrue(any("missing" in p and "case_b_cache" in p for p in problems), problems)

    def test_extra_name_fails(self):
        exp = EXPECTED_TWO_BINARIES.replace(f"{FB}::sync_context_round_trips_data ok\n", "")
        problems = self.check(exp, RUN_TWO_BINARIES)
        self.assertTrue(
            any("extra" in p and "sync_context_round_trips_data" in p for p in problems), problems
        )

    def test_unexpected_outcome_fails(self):
        exp = EXPECTED_TWO_BINARIES.replace(
            f"{PP}::private_context_in_a_fresh_process ignored",
            f"{PP}::private_context_in_a_fresh_process ok",
        )
        problems = self.check(exp, RUN_TWO_BINARIES)
        self.assertTrue(any("outcome" in p and "fresh_process" in p for p in problems), problems)

    def test_a_failed_test_fails_the_check(self):
        exp = expected_for(
            [
                (f"{FB}::a_failed_decision_is_not_remembered", "ok"),
                (f"{FB}::sync_context_round_trips_data", "ok"),
            ],
            filtered_out=30,
        )
        problems = self.check(exp, FAILED_RUN)
        self.assertTrue(any("FAILED" in p for p in problems), problems)

    def test_filtered_out_mismatch_fails(self):
        exp = EXPECTED_TWO_BINARIES.replace("# filtered-out: 28", "# filtered-out: 27")
        problems = self.check(exp, RUN_TWO_BINARIES)
        self.assertTrue(any("filtered out" in p for p in problems), problems)

    def test_unknown_filtered_out_is_reported_but_not_a_failure(self):
        exp = EXPECTED_TWO_BINARIES.replace("# filtered-out: 28", "# filtered-out: unknown")
        self.assertEqual(self.check(exp, RUN_TWO_BINARIES), [])

    def test_missing_binary_fails(self):
        exp = EXPECTED_TWO_BINARIES + "[test:other]\n# filtered-out: 0\nx ok\n"
        problems = self.check(exp, RUN_TWO_BINARIES)
        self.assertTrue(any("test:other" in p for p in problems), problems)

    def test_unexpected_binary_with_tests_fails(self):
        exp = EXPECTED_TWO_BINARIES.split("[test:fathomdb_lib_major]")[0]
        problems = self.check(exp, RUN_TWO_BINARIES)
        self.assertTrue(any("test:fathomdb_lib_major" in p for p in problems), problems)

    def test_missing_summary_line_fails(self):
        log = RUN_TWO_BINARIES.replace(
            "test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 28 filtered out; finished in 0.21s\n",
            "",
        )
        problems = self.check(EXPECTED_TWO_BINARIES, log)
        self.assertTrue(any("summary" in p for p in problems), problems)

    def test_summary_disagreeing_with_test_lines_fails(self):
        log = RUN_TWO_BINARIES.replace("3 passed; 0 failed; 1 ignored", "4 passed; 0 failed; 1 ignored")
        problems = self.check(EXPECTED_TWO_BINARIES, log)
        self.assertTrue(any("summary" in p for p in problems), problems)

    def test_list_mode_compares_names_only(self):
        exp = expected_for(
            [
                (f"{FB}::a_failed_decision_is_not_remembered", "ok"),
                (f"{PP}::pool_props_are_pinned_device_memory_on_the_ordinal", "ignored"),
            ],
            filtered_out=99,
        )
        self.assertEqual(self.check(exp, LIST_OUTPUT, mode="list"), [])
        exp_missing = exp.replace(f"{FB}::a_failed_decision_is_not_remembered ok\n", "")
        problems = self.check(exp_missing, LIST_OUTPUT, mode="list")
        self.assertTrue(any("extra" in p for p in problems), problems)

    def test_empty_expectation_matches_a_run_of_zero_tests(self):
        log = (
            "     Running unittests src/lib.rs (target/debug/deps/cudarc-0123456789abcdef)\n\n"
            "running 0 tests\n\n"
            "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 47 filtered out; finished in 0.00s\n"
        )
        self.assertEqual(self.check(expected_for([], filtered_out=47), log), [])

    def test_unnamed_binary_matches_the_only_expected_section(self):
        text = "running 1 test\ntest a::b ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        self.assertEqual(self.check(expected_for([("a::b", "ok")], filtered_out=0), text), [])

    def test_report_gives_per_module_counts(self):
        expected = ct.parse_expected(EXPECTED_TWO_BINARIES)
        observed = ct.parse_output(RUN_TWO_BINARIES)
        report = ct.report(expected, observed, "run")
        self.assertIn(f"{FB}: expected 2, observed 2", report)
        self.assertIn(f"{PP}: expected 2, observed 2", report)


CATALOGUE = textwrap.dedent(
    """\
    schema = 1

    [[upstream_ignored]]
    name = "driver::safe::core::tests::test_post_alloc_memory"

    [[test]]
    name = "a_failed_decision_is_not_remembered"
    module = "driver::safe::core::fathomdb_alloc_fallback"
    binary = "unittests:cudarc"
    kind = "P"
    cells = ["G1", "G6", "hosted"]
    ignored = false
    landed = true
    cfg = { features = ["driver"] }

    [[test]]
    name = "sync_zero_byte_request_is_a_null_pointer_without_a_driver_call"
    module = "driver::safe::core::fathomdb_alloc_fallback"
    binary = "unittests:cudarc"
    kind = "P"
    cells = ["G6", "hosted"]
    ignored = false
    landed = true
    cfg = { features = ["driver"], arch = ["aarch64"], os = ["linux"] }

    [[test]]
    name = "private_context_in_a_fresh_process"
    module = "driver::safe::core::fathomdb_private_pool"
    binary = "unittests:cudarc"
    kind = "X"
    cells = ["G6", "hosted"]
    ignored = true
    landed = true
    cfg = { features = ["driver"], arch = ["aarch64"], os = ["linux"] }

    [[test]]
    name = "max_size_without_binding_support_is_not_supported"
    module = "driver::safe::core::fathomdb_private_pool"
    binary = "unittests:cudarc"
    kind = "P"
    cells = ["pure"]
    ignored = false
    landed = true
    cfg = { features = ["driver"], max_version = 12010 }

    [[test]]
    name = "pool_props_are_zeroed_then_written"
    module = "driver::safe::core::fathomdb_private_pool"
    binary = "unittests:cudarc"
    kind = "P"
    cells = ["pure"]
    ignored = false
    landed = false
    cfg = { features = ["driver"] }

    [[test]]
    name = "the_report_says_linked_at_build_time_when_linked"
    module = "fathomdb_lib_major"
    binary = "unittests:cudarc"
    kind = "P"
    cells = ["t-unit"]
    ignored = false
    landed = true
    cfg = { features = ["cublas"] }
    """
)

UPSTREAM_DRV = textwrap.dedent(
    """\
    driver::result::tests::peer_transfer_self
    driver::safe::core::tests::test_post_alloc_memory
    driver::safe::core::tests::test_threading
    """
)

MANIFEST = textwrap.dedent(
    """\
    [package]
    name = "cudarc"
    version = "0.19.10"

    [features]
    default = ["std", "cublas", "driver"]
    std = []
    driver = []
    cublas = ["driver"]
    dynamic-loading = []
    cuda-11040 = []
    cuda-12060 = []
    cuda-13000 = []
    """
)


class Generate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        (d / "fork-catalogue.toml").write_text(CATALOGUE)
        (d / "upstream-drv-12060.txt").write_text(UPSTREAM_DRV)
        (d / "Cargo.toml").write_text(MANIFEST)
        self.dir = d
        self.catalogue = ct.load_catalogue(d / "fork-catalogue.toml")

    def tearDown(self):
        self.tmp.cleanup()

    def gen(self, **kw):
        args = dict(
            catalogue=self.catalogue,
            inventory_dir=self.dir,
            manifest=self.dir / "Cargo.toml",
            feature_set="drv",
            version=12060,
            arch="x86_64",
            os_name="linux",
            extra_features=(),
            filters=(),
            exact=False,
            skips=(),
            ignored_mode="normal",
        )
        args.update(kw)
        return ct.parse_expected(ct.generate_expected(**args))

    def names(self, exp):
        return exp.sections["unittests:cudarc"].names

    def test_universe_is_upstream_plus_landed_fork_tests_matching_cfg(self):
        exp = self.gen()
        self.assertEqual(
            self.names(exp),
            {
                "driver::result::tests::peer_transfer_self": "ok",
                "driver::safe::core::tests::test_post_alloc_memory": "ignored",
                "driver::safe::core::tests::test_threading": "ok",
                f"{FB}::a_failed_decision_is_not_remembered": "ok",
            },
        )
        self.assertEqual(exp.sections["unittests:cudarc"].filtered_out, 0)

    def test_arch_and_os_cfg_select_aarch64_only_tests(self):
        names = self.names(self.gen(arch="aarch64", filters=("fathomdb_",)))
        self.assertEqual(
            names,
            {
                f"{FB}::a_failed_decision_is_not_remembered": "ok",
                f"{FB}::sync_zero_byte_request_is_a_null_pointer_without_a_driver_call": "ok",
                f"{PP}::private_context_in_a_fresh_process": "ignored",
            },
        )
        self.assertEqual(self.names(self.gen(arch="aarch64", os_name="windows", filters=("fathomdb_",))).keys(),
                         {f"{FB}::a_failed_decision_is_not_remembered"})

    def test_filter_counts_the_rest_as_filtered_out(self):
        exp = self.gen(filters=("fathomdb_",))
        self.assertEqual(set(self.names(exp)), {f"{FB}::a_failed_decision_is_not_remembered"})
        self.assertEqual(exp.sections["unittests:cudarc"].filtered_out, 3)

    def test_exact_filters_match_whole_names(self):
        exp = self.gen(filters=("driver::safe::core::tests::test_threading", "test_threading"), exact=True)
        self.assertEqual(set(self.names(exp)), {"driver::safe::core::tests::test_threading"})
        self.assertEqual(exp.sections["unittests:cudarc"].filtered_out, 3)

    def test_skip_removes_matches(self):
        exp = self.gen(skips=("tests::",))
        self.assertEqual(set(self.names(exp)), {f"{FB}::a_failed_decision_is_not_remembered"})

    def test_ignored_only_mode_runs_only_ignored_tests_as_ok(self):
        exp = self.gen(ignored_mode="only")
        self.assertEqual(self.names(exp), {"driver::safe::core::tests::test_post_alloc_memory": "ok"})
        self.assertEqual(exp.sections["unittests:cudarc"].filtered_out, 3)

    def test_include_ignored_mode_runs_everything_as_ok(self):
        exp = self.gen(ignored_mode="include")
        self.assertTrue(all(o == "ok" for o in self.names(exp).values()))

    def test_version_bounds_apply(self):
        self.assertNotIn(
            f"{PP}::max_size_without_binding_support_is_not_supported", self.names(self.gen())
        )
        exp = self.gen(version=11040, filters=("fathomdb_",))
        self.assertIn(f"{PP}::max_size_without_binding_support_is_not_supported", self.names(exp))

    def test_feature_implications_come_from_the_manifest(self):
        # cfg needs "cublas"; the drv set lacks it, the def set has it.
        self.assertNotIn("fathomdb_lib_major::the_report_says_linked_at_build_time_when_linked",
                         self.names(self.gen()))
        exp = self.gen(extra_features=("cublas",), filters=("fathomdb_lib_major",))
        self.assertIn("fathomdb_lib_major::the_report_says_linked_at_build_time_when_linked", self.names(exp))

    def test_missing_upstream_inventory_makes_filtered_out_unknown(self):
        exp = self.gen(version=11040, filters=("fathomdb_",))
        self.assertIsNone(exp.sections["unittests:cudarc"].filtered_out)
        self.assertTrue(exp.sections["unittests:cudarc"].filtered_out_unknown)

    def test_unlanded_tests_are_never_expected(self):
        exp = self.gen(filters=("fathomdb_",))
        self.assertNotIn(f"{PP}::pool_props_are_zeroed_then_written", self.names(exp))

    def test_cell_names_lists_landed_tests_of_a_cell(self):
        args = dict(manifest=self.dir / "Cargo.toml", feature_set="drv", arch="x86_64", os_name="linux",
                    extra_features=())
        self.assertEqual(
            ct.cell_names(self.catalogue, "pure", version=11040, **args),
            [f"{PP}::max_size_without_binding_support_is_not_supported"],
        )
        self.assertEqual(ct.cell_names(self.catalogue, "pure", version=12060, **args), [])

    def test_names_cli_can_print_the_binary_of_each_name(self):
        out = io.StringIO()
        argv = ["names", "--catalogue", str(self.dir / "fork-catalogue.toml"),
                "--manifest", str(self.dir / "Cargo.toml"), "--set", "drv", "--version", "11040",
                "--arch", "x86_64", "--os", "linux", "--cell", "pure", "--with-binary"]
        with redirect_stdout(out):
            self.assertEqual(ct.main(argv), 0)
        self.assertEqual(
            out.getvalue(),
            f"unittests:cudarc\t{PP}::max_size_without_binding_support_is_not_supported\n",
        )

    def test_unlanded_names_present_in_a_listing_are_reported(self):
        listing = (
            "     Running unittests src/lib.rs (target/debug/deps/cudarc-0123456789abcdef)\n"
            f"{PP}::pool_props_are_zeroed_then_written: test\n\n1 test, 0 benchmarks\n"
        )
        stale = ct.unlanded_present(self.catalogue, ct.parse_output(listing))
        self.assertEqual(stale, [f"{PP}::pool_props_are_zeroed_then_written"])


class Cli(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ct.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_check_exits_zero_on_match_and_nonzero_on_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d) / "exp.txt"
            log = Path(d) / "run.log"
            exp.write_text(EXPECTED_TWO_BINARIES)
            log.write_text(RUN_TWO_BINARIES)
            code, out, _ = self.run_main(["check", "--expected", str(exp), str(log)])
            self.assertEqual(code, 0, out)
            exp.write_text(EXPECTED_TWO_BINARIES.replace("# filtered-out: 28", "# filtered-out: 1"))
            code, out, _ = self.run_main(["check", "--expected", str(exp), str(log)])
            self.assertNotEqual(code, 0)
            self.assertIn("FAIL", out)

    def test_the_real_catalogue_parses_and_names_every_vendored_test(self):
        root = HERE.parent
        catalogue = ct.load_catalogue(root / "test-inventory" / "fork-catalogue.toml")
        vendored = (root / "test-inventory" / "vendored-drv-12060-x86_64.txt").read_text().split()
        landed_x86 = ct.cell_names(
            catalogue, None, manifest=root.parent / "Cargo.toml", feature_set="drv", version=12060,
            arch="x86_64", os_name="linux", extra_features=(),
        )
        self.assertEqual(sorted(landed_x86), sorted(vendored))
        landed_arm = ct.cell_names(
            catalogue, None, manifest=root.parent / "Cargo.toml", feature_set="drv", version=12060,
            arch="aarch64", os_name="linux", extra_features=(),
        )
        self.assertEqual(len(landed_arm), 24)


if __name__ == "__main__":
    unittest.main()
