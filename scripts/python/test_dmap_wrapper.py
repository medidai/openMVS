#!/usr/bin/env python3
"""Focused tests for the stable depth-map observability shell entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "tools/dmap_observability.sh"


def run_wrapper(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER), *arguments],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class DMapWrapperTests(unittest.TestCase):
    def make_fake_doctor_environment(
        self, root: Path, *, missing_modules: tuple[str, ...],
    ) -> dict[str, str]:
        fake_bin = root / "doctor-bin"
        fake_bin.mkdir()
        for command in ("git", "cmake", "ninja", "pandoc", "nvcc", "nvidia-smi"):
            path = fake_bin / command
            path.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
        python = fake_bin / "doctor-python"
        python.write_text(
            f"#!{sys.executable}\n"
            "import os, re, sys\n"
            "missing = set(filter(None, os.environ.get('DMAP_FAKE_MISSING_MODULES', '').split(',')))\n"
            "if sys.argv[1:2] == ['-c']:\n"
            "    match = re.fullmatch(r'import ([A-Za-z0-9_.]+)', sys.argv[2].strip())\n"
            "    raise SystemExit(1 if match and match.group(1) in missing else 0)\n"
            "raise SystemExit(0)\n",
            encoding="utf-8",
        )
        python.chmod(python.stat().st_mode | stat.S_IXUSR)
        production = root / "DensifyPointCloud"
        observer = root / "DensifyPointCloudDMapObserve"
        for executable in (production, observer):
            executable.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        vcpkg = root / "vcpkg"
        (vcpkg / "scripts" / "buildsystems").mkdir(parents=True)
        (vcpkg / "scripts" / "buildsystems" / "vcpkg.cmake").touch()
        env = dict(os.environ)
        env.update({
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PYTHON": str(python),
            "DMAP_PRODUCTION_BIN": str(production),
            "DMAP_OBSERVER_BIN": str(observer),
            "DMAP_FAKE_MISSING_MODULES": ",".join(missing_modules),
            "VCPKG_ROOT": str(vcpkg),
        })
        return env

    def make_fake_package_python(self, root: Path) -> tuple[Path, Path]:
        """Create a Python shim that emulates bundle package/validate calls."""

        path = root / "fake-python"
        log_path = root / "bundle_calls.jsonl"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, json, os, pathlib, sys\n"
            "if sys.argv[1:2] != ['-m']:\n"
            "    os.execv(os.environ['DMAP_REAL_PYTHON'], [os.environ['DMAP_REAL_PYTHON'], *sys.argv[1:]])\n"
            "args = sys.argv[1:]\n"
            "with open(os.environ['DMAP_BUNDLE_CALL_LOG'], 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(args) + '\\n')\n"
            "command = args[2]\n"
            "if command == 'package':\n"
            "    output = pathlib.Path(args[args.index('--output') + 1])\n"
            "    output.parent.mkdir(parents=True, exist_ok=True)\n"
            "    output.write_bytes(b'validated archive')\n"
            "    digest = hashlib.sha256(output.read_bytes()).hexdigest()\n"
            "    output.with_name(output.name + '.sha256').write_text(f'{digest}  {output.name}\\n')\n"
            "    print(json.dumps({'archive': str(output), 'checksum': str(output) + '.sha256'}))\n"
            "elif os.environ.get('DMAP_FAIL_BUNDLE_VALIDATE') == '1':\n"
            "    raise SystemExit(9)\n"
            "else:\n"
            "    print(json.dumps({'valid': True}))\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path, log_path

    def test_command_dispatch_and_build_output_have_no_duplicate_entries(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertEqual(
            source.count("  printf 'observer:   %s\\n'"),
            1,
        )
        self.assertEqual(
            source.count('  drilldown) command_drilldown "$@" ;;'),
            1,
        )

    def test_all_profiles_demo_keeps_legacy_alias_and_asserts_five_profiles(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("--all-profiles|--all-levels", source)
        self.assertIn('--pixel 120,80', source)
        self.assertIn(
            'expected = {"endpoint", "summary", "prefilter", "deep", "trace"}',
            source,
        )

        help_result = run_wrapper("demo", "--help")
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("four core profiles", help_result.stdout)
        self.assertIn("--all-profiles", help_result.stdout)
        self.assertIn("--all-levels", help_result.stdout)
        self.assertIn("five capture profiles", help_result.stdout)
        self.assertIn("does not mean CUDA or", help_result.stdout)
        self.assertIn("--multiscale", help_result.stdout)
        self.assertIn("complete release matrix", help_result.stdout)
        self.assertIn(
            'row.get("measurement_basis") == "post_pass_change_detection"',
            source,
        )

    def test_package_help_exposes_no_sanitizer_bypass(self) -> None:
        result = run_wrapper("package", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--allow-host-references", result.stdout)
        self.assertNotIn("--no-stage-report", result.stdout)
        self.assertIn("always stages and sanitizes", result.stdout)

    def test_capture_help_matches_wrapper_no_report_contract(self) -> None:
        result = run_wrapper("capture", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile", result.stdout)
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--allow-over-budget", result.stdout)
        self.assertIn("never generates a report", result.stdout)
        self.assertNotIn("--no-skip-report", result.stdout)

        rejected = run_wrapper("capture", "--no-skip-report")
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("never generates a report", rejected.stderr)

    def test_array_store_is_a_documented_stable_wrapper_command(self) -> None:
        result = run_wrapper("array-store", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("immutable Zarr v3 stores", result.stdout)
        self.assertIn("--max-uncompressed-bytes-per-store", result.stdout)
        self.assertIn("report generation does not", result.stdout)
        self.assertIn(
            'array-store) command_array_store "$@" ;;',
            WRAPPER.read_text(encoding="utf-8"),
        )
        capabilities = json.loads(
            (REPO_ROOT / "docs/dmap_observability/capabilities.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("array-store", capabilities["commands"])

    def test_capability_index_documents_multiscale_proxy_limits(self) -> None:
        capabilities = json.loads(
            (REPO_ROOT / "docs/dmap_observability/capabilities.json").read_text(
                encoding="utf-8"
            )
        )
        multiscale = capabilities["availability_notes"]["multiscale"]
        self.assertIn(
            "coarse_level_candidate_source_compatibility_proxy",
            multiscale["public_v1_available"],
        )
        self.assertEqual(
            multiscale["coarse_level_candidate_source"]["measurement_basis"],
            "post_pass_change_detection",
        )
        self.assertEqual(
            multiscale["coarse_level_cost_component_maps"]["status"],
            "unavailable",
        )
        portable = capabilities["integrity"]["portable_bundle"]
        self.assertTrue(portable["rejects_credential_bearing_urls"])
        self.assertTrue(portable["rejects_common_high_confidence_secret_formats"])

    def test_doctor_checks_the_required_pandoc_renderer(self) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertIn('for item in git cmake ninja pandoc "${python_bin}"', source)

    def test_doctor_warns_but_succeeds_when_optional_python_modules_are_missing(self) -> None:
        optional = ("pyarrow", "plotly", "jinja2", "zarr", "numcodecs")
        with tempfile.TemporaryDirectory() as temporary:
            env = self.make_fake_doctor_environment(
                Path(temporary), missing_modules=optional,
            )
            result = run_wrapper("doctor", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        for module in optional:
            self.assertIn(
                f"warning: optional Python module {module} is unavailable",
                result.stderr,
            )

    def test_doctor_fails_when_a_core_report_module_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            env = self.make_fake_doctor_environment(
                Path(temporary), missing_modules=("numpy",),
            )
            result = run_wrapper("doctor", env=env)
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing: required Python module numpy", result.stderr)

    def test_package_rejects_full_and_abbreviated_sanitizer_bypasses(self) -> None:
        for option in (
            "--allow-host-references",
            "--allow-host-ref",
            "--allow-h",
            "--no-stage-report",
            "--no-stage-rep",
            "--no-st",
        ):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temporary:
                result = run_wrapper(
                    "package",
                    "--report-dir",
                    str(Path(temporary) / "report"),
                    "--output",
                    str(Path(temporary) / "review.tar.zst"),
                    option,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("does not permit sanitizer bypasses", result.stderr)

    def test_package_propagates_custom_limits_to_staged_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python, log_path = self.make_fake_package_python(root)
            output = root / "review.tar.zst"
            env = dict(os.environ)
            env.update({
                "PYTHON": str(fake_python),
                "DMAP_REAL_PYTHON": sys.executable,
                "DMAP_BUNDLE_CALL_LOG": str(log_path),
            })
            result = run_wrapper(
                "package",
                "--report-dir", str(root / "report"),
                "--output", str(output),
                "--max-review-bytes", "2000000000",
                "--max-payload-bytes=6000000000",
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name(output.name + ".sha256").is_file())
            calls = [json.loads(line) for line in log_path.read_text().splitlines()]
            package, validate = calls
            self.assertEqual(package[2], "package")
            self.assertEqual(validate[2], "validate")
            self.assertEqual(
                validate[validate.index("--max-review-bytes") + 1], "2000000000"
            )
            self.assertEqual(
                validate[validate.index("--max-payload-bytes") + 1], "6000000000"
            )

    def test_package_validation_failure_leaves_final_pair_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python, log_path = self.make_fake_package_python(root)
            output = root / "review.tar.zst"
            checksum = output.with_name(output.name + ".sha256")
            output.write_bytes(b"previous archive")
            checksum.write_text("previous checksum\n", encoding="utf-8")
            env = dict(os.environ)
            env.update({
                "PYTHON": str(fake_python),
                "DMAP_REAL_PYTHON": sys.executable,
                "DMAP_BUNDLE_CALL_LOG": str(log_path),
                "DMAP_FAIL_BUNDLE_VALIDATE": "1",
            })
            result = run_wrapper(
                "package",
                "--report-dir", str(root / "report"),
                "--output", str(output),
                "--overwrite",
                "--max-review-bytes", "2000000000",
                env=env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_bytes(), b"previous archive")
            self.assertEqual(checksum.read_text(encoding="utf-8"), "previous checksum\n")
            self.assertFalse(any(root.glob(".review.tar.zst.staging.*")))

    def test_package_overwrite_publishes_validated_pair_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python, log_path = self.make_fake_package_python(root)
            output = root / "review.tar.zst"
            checksum = output.with_name(output.name + ".sha256")
            output.write_bytes(b"previous archive")
            checksum.write_text("previous checksum\n", encoding="utf-8")
            env = dict(os.environ)
            env.update({
                "PYTHON": str(fake_python),
                "DMAP_REAL_PYTHON": sys.executable,
                "DMAP_BUNDLE_CALL_LOG": str(log_path),
            })
            result = run_wrapper(
                "package",
                "--report-dir", str(root / "report"),
                "--output", str(output),
                "--overwrite",
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_bytes(), b"validated archive")
            self.assertIn(output.name, checksum.read_text(encoding="utf-8"))
            self.assertFalse(any(root.glob(".review.tar.zst.staging.*")))

    def test_package_without_overwrite_fails_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_python, log_path = self.make_fake_package_python(root)
            output = root / "review.tar.zst"
            output.write_bytes(b"previous archive")
            env = dict(os.environ)
            env.update({
                "PYTHON": str(fake_python),
                "DMAP_REAL_PYTHON": sys.executable,
                "DMAP_BUNDLE_CALL_LOG": str(log_path),
            })
            result = run_wrapper(
                "package",
                "--report-dir", str(root / "report"),
                "--output", str(output),
                env=env,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_bytes(), b"previous archive")
            self.assertFalse(log_path.exists())
            self.assertFalse(any(root.glob(".review.tar.zst.staging.*")))

    def test_internal_bundle_parser_does_not_expand_option_abbreviations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "report"
            report.mkdir()
            output = root / "review.tar.zst"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.python.dmap_observability.portable_bundle",
                    "package",
                    "--report-dir",
                    str(report),
                    "--output",
                    str(output),
                    "--allow-h",
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unrecognized arguments: --allow-h", result.stderr)
            self.assertFalse(output.exists())

    def test_package_rejects_archive_inside_source_tree_without_mutation(self) -> None:
        output = REPO_ROOT / f".dmap-wrapper-{uuid.uuid4().hex}.tar.zst"
        try:
            result = run_wrapper(
                "package", "--report-dir", "/tmp/missing-report", "--output", str(output)
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be outside the OpenMVS source tree", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".sha256").exists())
        finally:
            output.unlink(missing_ok=True)
            output.with_name(output.name + ".sha256").unlink(missing_ok=True)

    def test_package_rejects_external_symlink_resolving_into_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            link = Path(temporary) / "source-link"
            link.symlink_to(REPO_ROOT, target_is_directory=True)
            output = link / f".dmap-wrapper-{uuid.uuid4().hex}.tar.zst"
            result = run_wrapper(
                "package", "--report-dir", "/tmp/missing-report", "--output", str(output)
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("resolves inside the OpenMVS source tree", result.stderr)
            self.assertFalse(output.exists())

    def test_rebuild_rejects_report_inside_source_tree_before_staging(self) -> None:
        report_dir = REPO_ROOT / f".dmap-wrapper-{uuid.uuid4().hex}" / "reports"
        try:
            result = run_wrapper(
                "report", "--report-dir", str(report_dir), "--rebuild-report"
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be outside the OpenMVS source tree", result.stderr)
            self.assertFalse(report_dir.parent.exists())
        finally:
            shutil.rmtree(report_dir.parent, ignore_errors=True)

    def test_failed_staged_report_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / "reports"
            env = dict(os.environ)
            env["PYTHON"] = "/bin/false"

            result = run_wrapper(
                "report",
                "--config", str(root / "experiment.yaml"),
                "--report-dir", str(report),
                "--rebuild-report",
                env=env,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(report.exists())
            self.assertFalse(any(root.glob("reports.staging.*")))

    def test_build_rejects_same_resolved_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            build_dir = Path(temporary) / "build"
            result = run_wrapper(
                "build",
                "--production-dir",
                str(build_dir),
                "--observer-dir",
                str(build_dir / ".." / "build"),
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be distinct", result.stderr)

    def test_build_rejects_reserved_cmake_cache_overrides(self) -> None:
        reserved = (
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DOpenMVS_USE_CUDA:BOOL=OFF",
            "-UOpenMVS_DMAP_INSTRUMENTATION",
            "-D_USE_DMAP_INSTRUMENTATION=TRUE",
            "-DCMAKE_CXX_FLAGS=-D_USE_DMAP_INSTRUMENTATION",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for argument in reserved:
                with self.subTest(argument=argument):
                    result = run_wrapper(
                        "build",
                        "--production-dir",
                        str(root / "production"),
                        "--observer-dir",
                        str(root / "observer"),
                        "--cmake-arg",
                        argument,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("--cmake-arg cannot", result.stderr)

    def test_build_places_boundary_invariants_after_custom_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log_path = root / "cmake.jsonl"
            cmake = fake_bin / "cmake"
            cmake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['DMAP_FAKE_CMAKE_LOG'], 'a', encoding='utf-8') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            nvcc = fake_bin / "nvcc"
            nvcc.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
            cmake.chmod(cmake.stat().st_mode | stat.S_IXUSR)
            nvcc.chmod(nvcc.stat().st_mode | stat.S_IXUSR)
            env = dict(os.environ)
            env.pop("VCPKG_ROOT", None)
            env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
            env["DMAP_FAKE_CMAKE_LOG"] = str(log_path)

            result = run_wrapper(
                "build",
                "--production-dir",
                str(root / "production"),
                "--observer-dir",
                str(root / "observer"),
                "--jobs",
                "1",
                "--cmake-arg",
                "-DFOO=BAR",
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            invocations = [json.loads(line) for line in log_path.read_text().splitlines()]
            configure = [arguments for arguments in invocations if "-S" in arguments]
            self.assertEqual(len(configure), 2)
            for arguments, boundary in zip(
                configure,
                (
                    "-DOpenMVS_DMAP_INSTRUMENTATION=OFF",
                    "-DOpenMVS_DMAP_INSTRUMENTATION=ON",
                ),
                strict=True,
            ):
                self.assertLess(arguments.index("-DFOO=BAR"), arguments.index("-DCMAKE_BUILD_TYPE=Release"))
                self.assertLess(arguments.index("-DFOO=BAR"), arguments.index("-DOpenMVS_USE_CUDA=ON"))
                self.assertLess(arguments.index("-DFOO=BAR"), arguments.index(boundary))


if __name__ == "__main__":
    unittest.main()
