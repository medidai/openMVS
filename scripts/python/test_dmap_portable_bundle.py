#!/usr/bin/env python3
"""Tests for portable depth-map observability review bundles."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from dmap_observability import portable_bundle


class PortableBundleTests(unittest.TestCase):
	def test_shareable_content_rejects_common_secrets_and_url_userinfo(self) -> None:
		secrets = {
			"private key": "-----BEGIN " + "PRIVATE KEY-----",
			"AWS access key": "AK" + "IA" + "A" * 16,
			"GitHub token": "gh" + "p_" + "A" * 36,
			"GitLab token": "gl" + "pat-" + "A" * 24,
			"OpenAI-style key": "s" + "k-proj-" + "A" * 24,
			"Hugging Face token": "h" + "f_" + "A" * 24,
			"Slack token": "xo" + "xb-" + "A" * 24,
			"Google API key": "AI" + "za" + "A" * 35,
			"Stripe live key": "sk_" + "live_" + "A" * 24,
			"credential-bearing URL": (
				"https://reviewer:" + "private-value@example.com/report"
			),
		}
		for label, value in secrets.items():
			with self.subTest(label=label):
				with self.assertRaisesRegex(portable_bundle.BundleError, f"possible {label}"):
					portable_bundle._scan_shareable_content(
						json.dumps({"nested": {"value": value}}).encode("utf-8"),
						"report_model.json",
					)

	def test_shareable_content_rejects_secret_in_nested_serialized_json(self) -> None:
		token = "gh" + "p_" + "A" * 36
		payload = {
			"rows": [{"validation": json.dumps({"producer": {"token": token}})}],
		}
		with self.assertRaisesRegex(portable_bundle.BundleError, "possible GitHub token"):
			portable_bundle._scan_shareable_content(
				json.dumps(payload).encode("utf-8"), "instrumentation_validation.json",
			)

	def test_shareable_content_rejects_secret_split_across_read_blocks(self) -> None:
		token = ("gh" + "p_" + "A" * 36).encode("ascii")
		content = b" " * (1024 * 1024 - 10) + token
		with self.assertRaisesRegex(portable_bundle.BundleError, "possible GitHub token"):
			portable_bundle._scan_portable_stream(io.BytesIO(content), "large.json")

	def test_shareable_content_allows_hashes_schema_and_placeholders(self) -> None:
		portable_bundle._scan_shareable_content(
			json.dumps({
				"schema_name": "openmvs.dmap.report_policy",
				"sha256": "a" * 64,
				"credential": "REPLACE_WITH_API_KEY",
				"url": "https://example.org/report",
			}).encode("utf-8"),
			"report_policy.json",
		)

	def test_raw_artifact_rejects_uri_userinfo(self) -> None:
		with self.assertRaisesRegex(portable_bundle.BundleError, "URI contains userinfo"):
			portable_bundle._validate_raw_artifact({
				"artifact_id": "external-map",
				"sha256": "a" * 64,
				"size_bytes": 1,
				"uri": "https://reviewer:" + "private-value@example.com/map",
			})

	def test_portability_scanner_distinguishes_json_escape_from_windows_path(self) -> None:
		portable_bundle._scan_shareable_content(
			b'{"message":"these errors:\\n - dependency missing"}\n',
			"report_inventory.json",
		)
		with self.assertRaisesRegex(portable_bundle.BundleError, "host-specific"):
			portable_bundle._scan_shareable_content(
				b'{"path":"C:\\\\Users\\\\developer\\\\report.json"}\n',
				"report_inventory.json",
			)

	def test_float_payload_scanner_ignores_encoded_bytes_but_audits_control_plane(self) -> None:
		path = "review/interactive/maps/map-test_float32.js"
		prefix = (
			"window.__DMAP_PIXEL_PAYLOADS=window.__DMAP_PIXEL_PAYLOADS||{};"
			"window.__DMAP_PIXEL_PAYLOADS[\"map-test\"]="
		)
		payload = '{"data":"H4sI/data/==","width":1,"height":1,"channels":1};\n'
		portable_bundle._scan_shareable_content((prefix + payload).encode("ascii"), path)
		with self.assertRaisesRegex(portable_bundle.BundleError, "invalid preamble"):
			portable_bundle._scan_shareable_content(
				(prefix.replace("window.__DMAP_PIXEL_PAYLOADS[", "/home/private/[") + payload).encode("ascii"),
				path,
			)

	def test_sanitizer_covers_common_host_and_remote_path_forms(self) -> None:
		references = (
			"/opt/project/report.json",
			"/srv/results/report.json",
			"/data/scenes/scene.mvs",
			"/tmp/private/report.json",
			"/var/lib/private/report.json",
			"/etc/secret.png",
			"/root/private/report.json",
			"/workspace/private/report.json",
			r"C:\\Users\\developer\\report.json",
			r"D:/captures/scene.mvs",
			r"\\\\workstation\\share\\report.json",
			"ssh://developer@workstation/var/results",
			"developer@workstation:/var/results",
			"ssh developer@workstation",
			"ssh -p 2222 developer@workstation",
		)
		for reference in references:
			with self.subTest(reference=reference):
				sanitized = portable_bundle._sanitize_detached_text(
					f"artifact={reference}\n"
				)
				self.assertNotIn(reference, sanitized)
				with self.assertRaisesRegex(
					portable_bundle.BundleError, "host-specific reference"
				):
					portable_bundle._scan_portable_stream(
						io.BytesIO(reference.encode("utf-8")), "fixture.md"
					)

	def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
		report = root / "report"
		(report / "assets").mkdir(parents=True)
		(report / "01_report.md").write_text("# Report\n\n[UI](02_investigation.html)\n", encoding="utf-8")
		(report / "02_investigation.html").write_text("<main>portable</main>\n", encoding="utf-8")
		(report / "assets" / "values.bin").write_bytes(bytes(range(64)))
		config = root / "experiment.yaml"
		config.write_text(
			"schema_version: 2\nexperiment_id: test\ndataset_root: /mnt/private/dataset\n",
			encoding="utf-8",
		)
		provenance = root / "repro.json"
		provenance.write_text('{"git_commit":"0123456789abcdef"}\n', encoding="utf-8")
		return report, config, provenance

	def test_bundle_is_deterministic_and_self_validating(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, config, provenance = self._fixture(root)
			raw = portable_bundle.RawArtifact(
				artifact_id="deep-frame-1",
				sha256="a" * 64,
				size_bytes=1234,
				uri="raw/sha256/aa/deep-frame-1.zarr",
				media_type="application/vnd+zarr",
			)
			first = portable_bundle.create_review_bundle(
				report,
				root / "first.tar.zst",
				config_files=[config],
				provenance_files=[provenance],
				provenance={
					"experiment_id": "test", "source_clean": True,
					"source_root": "/home/person/source",
				},
				raw_artifacts=[raw],
				source_date_epoch=123,
			)
			second = portable_bundle.create_review_bundle(
				report,
				root / "second.tar.zst",
				config_files=[config],
				provenance_files=[provenance],
				provenance={
					"experiment_id": "test", "source_clean": True,
					"source_root": "/home/person/source",
				},
				raw_artifacts=[raw],
				source_date_epoch=123,
			)

			self.assertEqual(first.archive_path.read_bytes(), second.archive_path.read_bytes())
			validation = portable_bundle.validate_review_bundle(first.archive_path)
			self.assertEqual(validation.review_file_count, 5)
			self.assertEqual(validation.raw_artifact_count, 1)
			self.assertEqual(validation.payload_bytes, first.payload_bytes)
			self.assertEqual(validation.file_count, first.file_count)
			checksum = first.checksum_path.read_text(encoding="ascii").strip()
			self.assertEqual(checksum, f"{first.archive_sha256}  first.tar.zst")

	def test_package_rejects_same_size_investigation_html_tamper(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			(report / "report_policy.json").write_text(json.dumps({
				"schema_name": "openmvs.dmap.report_policy",
				"schema_version": 2,
				"integrity_contract": {
					"report_tree_closure_schema_version": 1,
				},
			}), encoding="utf-8")
			portable_bundle.integrity.write_report_tree_closure(report)
			target = report / "02_investigation.html"
			original = target.read_bytes()
			tampered = original.replace(b"portable", b"tampered")
			self.assertEqual(len(original), len(tampered))
			target.write_bytes(tampered)

			with self.assertRaisesRegex(
				portable_bundle.BundleError,
				"source report artifact closure is invalid.*02_investigation.html",
			):
				portable_bundle.create_review_bundle(
					report, root / "tampered.tar.zst",
				)

	def test_package_rejects_file_added_after_report_closure(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			(report / "report_policy.json").write_text(json.dumps({
				"schema_name": "openmvs.dmap.report_policy",
				"schema_version": 2,
				"integrity_contract": {
					"report_tree_closure_schema_version": 1,
				},
			}), encoding="utf-8")
			portable_bundle.integrity.write_report_tree_closure(report)
			(report / "unreviewed.bin").write_bytes(b"unreviewed")

			with self.assertRaisesRegex(
				portable_bundle.BundleError,
				"source report artifact closure is invalid.*unreviewed.bin",
			):
				portable_bundle.create_review_bundle(
					report, root / "unexpected.tar.zst",
				)

	def test_staging_sanitizes_host_specific_text_and_rejects_symlinks(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			(report / "host.md").write_text("artifact: /home/person/private/report\n", encoding="utf-8")
			portable = portable_bundle.create_review_bundle(report, root / "host.tar.zst")
			portable_bundle.validate_review_bundle(portable.archive_path)
			with self.assertRaisesRegex(portable_bundle.BundleError, "host-specific"):
				portable_bundle.create_review_bundle(
					report, root / "raw-host.tar.zst", stage_report=False
				)
			(report / "link").symlink_to(report / "01_report.md")
			with self.assertRaisesRegex(portable_bundle.BundleError, "symlinks"):
				portable_bundle.create_review_bundle(report, root / "link.tar.zst")

	def test_nested_serialized_validation_json_remains_valid_after_sanitization(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			logical_state_validation = {
				"depth_prior_weight_domain": {
					"parameter_source": {
						"available": True,
						"path": "/mnt/private/run/instrumentation/run_metadata.json",
					},
				},
				"warning": None,
			}
			(report / "instrumentation_validation.json").write_bytes(
				portable_bundle._canonical_json({
					"rows": [{
						"logical_state_validation": json.dumps(
							logical_state_validation, sort_keys=True,
						),
					}],
				})
			)

			stage = root / "stage"
			portable_bundle.stage_review_tree(report, stage)
			staged_validation = json.loads(
				(stage / "instrumentation_validation.json").read_text(encoding="utf-8")
			)
			staged_logical_state = json.loads(
				staged_validation["rows"][0]["logical_state_validation"]
			)
			staged_path = staged_logical_state[
				"depth_prior_weight_domain"
			]["parameter_source"]["path"]
			self.assertNotIn("/mnt/", staged_path)
			self.assertRegex(staged_path, r"^artifact:(?:omitted-)?[a-f0-9]{24}\[")
			portable_bundle.validate_staged_review_tree(stage)

			bundle = portable_bundle.create_review_bundle(
				report, root / "portable.tar.zst",
			)
			portable_bundle.validate_review_bundle(bundle.archive_path)
			with bundle.archive_path.open("rb") as compressed_stream:
				with portable_bundle.zstandard.ZstdDecompressor().stream_reader(
					compressed_stream
				) as archive_stream:
					with tarfile.open(fileobj=archive_stream, mode="r|") as archive:
						for member in archive:
							if member.name != "review/instrumentation_validation.json":
								continue
							member_stream = archive.extractfile(member)
							self.assertIsNotNone(member_stream)
							packaged_validation = json.load(member_stream)
							packaged_logical_state = json.loads(
								packaged_validation["rows"][0]["logical_state_validation"]
							)
							self.assertNotIn(
								"/mnt/", json.dumps(packaged_logical_state),
							)
							break
						else:
							self.fail("packaged instrumentation_validation.json is missing")

	def test_reference_style_markdown_images_are_sanitized_and_audited(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			(report / "reference.md").write_text(
				"![secret][asset]\n\n[asset]: /etc/secret.png\n", encoding="utf-8",
			)
			with self.assertRaisesRegex(
				portable_bundle.BundleError, "required report asset.*missing",
			):
				portable_bundle.stage_review_tree(report, root / "stage")

			staged = root / "untrusted-stage"
			staged.mkdir()
			(staged / "report.md").write_text(
				"![secret][asset]\n\n[asset]: /etc/secret.png\n", encoding="utf-8",
			)
			with self.assertRaisesRegex(portable_bundle.BundleError, "host-specific reference"):
				portable_bundle.validate_staged_review_tree(staged)

	def test_inventory_declaration_requires_receipt_at_every_portable_boundary(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			(report / "report_inventory.json").write_bytes(portable_bundle._canonical_json({
				portable_bundle.FINALIZER_INVENTORY_KEY: {
					"finalizer_receipt": portable_bundle.FINALIZER_RECEIPT_FILE,
				},
			}))

			stage = root / "stage"
			with self.assertRaisesRegex(
				portable_bundle.BundleError, "declares finalizer artifacts.*receipt is missing",
			):
				portable_bundle.stage_review_tree(report, stage)
			self.assertFalse(stage.exists())

			with self.assertRaisesRegex(
				portable_bundle.BundleError, "declares finalizer artifacts.*receipt is missing",
			):
				portable_bundle.validate_staged_review_tree(report)

			archive = root / "missing-receipt.tar.zst"
			with self.assertRaisesRegex(
				portable_bundle.BundleError, "declares finalizer artifacts.*receipt is missing",
			):
				portable_bundle.create_review_bundle(
					report, archive, stage_report=False,
				)
			self.assertFalse(archive.exists())

	def test_staging_rejects_transient_mutate_copy_revert(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			target = report / "assets" / "values.bin"
			original = target.read_bytes()
			real_copy = portable_bundle._copy_raw_report_snapshot

			def copy_transient_mutation(source: Path, destination: Path) -> None:
				target.write_bytes(b"transient alternate canonical payload")
				try:
					real_copy(source, destination)
				finally:
					target.write_bytes(original)

			stage = root / "stage"
			with mock.patch.object(
				portable_bundle, "_copy_raw_report_snapshot",
				side_effect=copy_transient_mutation,
			):
				with self.assertRaisesRegex(
					portable_bundle.BundleError,
					"changed while creating its raw portable snapshot",
				):
					portable_bundle.stage_review_tree(report, stage)
			self.assertEqual(target.read_bytes(), original)
			self.assertFalse(stage.exists())
			self.assertEqual(list(root.glob(".stage.canonical-snapshot.*")), [])

	def test_staging_remaps_absolute_canonical_references_to_snapshot(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			image = report / "assets" / "reference.png"
			image.write_bytes(b"reference-image")
			(report / "internal.parquet").write_bytes(b"omitted-raw-data")
			(report / "paths.json").write_text(
				json.dumps({"source_path": "internal.parquet"}), encoding="utf-8",
			)
			(report / "absolute.md").write_text(
				f"![reference]({image})\n", encoding="utf-8",
			)

			stage = root / "stage"
			result = portable_bundle.stage_review_tree(report, stage)
			self.assertEqual(result.copied_external_assets, 0)
			self.assertIn(
				"](assets/reference.png)",
				(stage / "absolute.md").read_text(encoding="utf-8"),
			)
			self.assertEqual(
				(stage / "assets" / "reference.png").read_bytes(), image.read_bytes()
			)
			second_stage = root / "other" / "second-stage"
			portable_bundle.stage_review_tree(report, second_stage)
			self.assertEqual(
				(stage / "portable_artifacts.json").read_bytes(),
				(second_stage / "portable_artifacts.json").read_bytes(),
			)

	def test_stages_external_assets_omits_raw_sources_and_deduplicates_previews(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report = root / "source" / "report"
			external = root / "outside"
			(report / "interactive" / "maps").mkdir(parents=True)
			external.mkdir()
			reference = external / "reference.jpg"
			preview = external / "preview.png"
			raw = external / "cost.pfm"
			reference.write_bytes(b"reference-image")
			preview.write_bytes(b"preview-image")
			raw.write_bytes(b"raw-cost-map")
			local = report / "interactive" / "maps" / "state_local.png"
			shared = report / "interactive" / "maps" / "state_shared.png"
			local.write_bytes(b"identical-map-preview")
			shared.write_bytes(local.read_bytes())
			(report / "paths.parquet").write_bytes(b"PAR1 /mnt/private/path PAR1")
			(report / "01_report.md").write_text(
				"# Report\n\n"
				f"![reference]({reference})\n\n"
				"[raw](../../outside/cost.pfm)\n",
				encoding="utf-8",
			)
			model = {
				"entrypoints": {
					"investigation_html": "02_investigation.html",
					"markdown": "01_report.md",
					"model": "report_model.json",
				},
				"experiment": {"root": "../.."},
				"inventory": {"data_artifacts": [{"parquet": "paths.parquet"}]},
				"scenes": [{
					"frames": [{
						"reference": {"available": True, "path": str(reference)},
						"maps": [
							{
								"available": True,
								"id": "external-preview",
								"preview": {"local": f"file://{preview}", "shared": str(preview)},
								"source_path": "../../outside/cost.pfm",
								"unavailable_reason": None,
							},
							{
								"available": True,
								"id": "deduplicated-preview",
								"preview": {
									"local": "interactive/maps/state_local.png",
									"shared": "interactive/maps/state_shared.png",
								},
								"source_path": "../../outside/cost.pfm",
								"unavailable_reason": None,
							},
						],
					}],
				}],
			}
			(report / "report_model.json").write_text(json.dumps(model), encoding="utf-8")
			(report / "02_investigation.html").write_text(
				"<!doctype html><img src=\"file://{}\"><a href=\"../../outside/cost.pfm\">raw</a>"
				"<script id=\"dmap-report-model\" type=\"application/json\">{}</script>\n".format(
					preview, json.dumps(model)
				),
				encoding="utf-8",
			)
			(report / "paths.csv").write_text(
				f"source_path,available\n../../outside/cost.pfm,true\n{reference},true\n",
				encoding="utf-8",
			)

			stage = root / "portable-review"
			result = portable_bundle.stage_review_tree(report, stage)
			self.assertEqual(result.deduplicated_preview_files, 1)
			self.assertEqual(result.deduplicated_preview_bytes, len(local.read_bytes()))
			self.assertTrue(shared.is_file())
			self.assertEqual(
				json.loads((report / "report_model.json").read_text(encoding="utf-8")), model
			)
			self.assertFalse((stage / "interactive" / "maps" / "state_shared.png").exists())
			staged_model = json.loads((stage / "report_model.json").read_text(encoding="utf-8"))
			frame = staged_model["scenes"][0]["frames"][0]
			external_map, deduplicated_map = frame["maps"]
			self.assertTrue(external_map["available"])
			self.assertIsNone(external_map["unavailable_reason"])
			self.assertEqual(
				external_map["source_path_artifact"]["status"], "omitted_from_review_bundle"
			)
			self.assertEqual(external_map["source_path"], "portable_artifacts.html")
			self.assertEqual(external_map["preview"]["local"], external_map["preview"]["shared"])
			self.assertTrue((stage / external_map["preview"]["local"]).is_file())
			self.assertEqual(
				deduplicated_map["preview"]["local"], deduplicated_map["preview"]["shared"]
			)
			self.assertTrue((stage / deduplicated_map["preview"]["local"]).is_file())
			self.assertFalse((stage / "paths.parquet").exists())
			self.assertIn("omitted_from_review_bundle", (stage / "portable_artifacts.md").read_text())
			markdown = (stage / "01_report.md").read_text(encoding="utf-8")
			self.assertIn("portable_assets/sha256/", markdown)
			self.assertIn("portable_artifacts.html#", markdown)
			all_text = "\n".join(
				path.read_text(encoding="utf-8")
				for path in stage.rglob("*")
				if path.is_file() and path.suffix in portable_bundle.TEXT_SUFFIXES
			)
			self.assertNotIn("/mnt/", all_text)
			self.assertNotIn("file://", all_text)
			self.assertNotIn("../../outside", all_text)

			bundle = portable_bundle.create_review_bundle(report, root / "portable.tar.zst")
			shutil_target = root / "source-removed"
			(report.parent).rename(shutil_target)
			external.rename(root / "outside-removed")
			portable_bundle.validate_staged_review_tree(stage)
			portable_bundle.validate_review_bundle(bundle.archive_path)

			(stage / "broken.md").write_text("[escape](../../outside/raw.pfm)\n", encoding="utf-8")
			with self.assertRaisesRegex(portable_bundle.BundleError, "escapes review root"):
				portable_bundle.validate_staged_review_tree(stage)

	def test_enforces_compressed_and_payload_caps_without_publishing(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			payload_output = root / "payload.tar.zst"
			with self.assertRaisesRegex(portable_bundle.BundleError, "payload requires"):
				portable_bundle.create_review_bundle(
					report, payload_output, max_payload_bytes=8
				)
			self.assertFalse(payload_output.exists())
			archive_output = root / "archive.tar.zst"
			with self.assertRaisesRegex(portable_bundle.BundleError, "compressed review"):
				portable_bundle.create_review_bundle(
					report, archive_output, max_review_bytes=8
				)
			self.assertFalse(archive_output.exists())
			self.assertFalse(archive_output.with_name("archive.tar.zst.sha256").exists())

	def test_rejects_unsafe_member_and_inventory_mismatch(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			unsafe = root / "unsafe.tar.zst"
			self._write_raw_archive(unsafe, {"../escape.txt": b"bad"})
			with self.assertRaisesRegex(portable_bundle.BundleError, "escapes"):
				portable_bundle.validate_review_bundle(unsafe, verify_checksum=False)

			bundle_metadata = portable_bundle._canonical_json({
				"bundle_type": "openmvs.dmap_observability.review",
				"file_count": 1,
				"payload_bytes": 2,
				"raw_artifact_count": 0,
				"schema_version": 1,
			})
			bad_inventory = root / "bad_inventory.tar.zst"
			self._write_raw_archive(bad_inventory, {
				"bundle.json": bundle_metadata,
				"inventory.sha256": f"{'0' * 64}  bundle.json\n".encode("ascii"),
				"review/report.md": b"ok",
			})
			with self.assertRaisesRegex(portable_bundle.BundleError, "inventory mismatch"):
				portable_bundle.validate_review_bundle(bad_inventory, verify_checksum=False)

	def test_cli_packages_and_validates(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, config, provenance = self._fixture(root)
			output = root / "review.tar.zst"
			module = "dmap_observability.portable_bundle"
			package = subprocess.run(
				[
					sys.executable, "-m", module, "package",
					"--report-dir", str(report), "--output", str(output),
					"--config", str(config), "--provenance-file", str(provenance),
				],
				cwd=SCRIPT_DIR,
				capture_output=True,
				text=True,
			)
			self.assertEqual(package.returncode, 0, package.stderr)
			self.assertEqual(json.loads(package.stdout)["archive"], str(output))
			validate = subprocess.run(
				[sys.executable, "-m", module, "validate", "--bundle", str(output)],
				cwd=SCRIPT_DIR,
				capture_output=True,
				text=True,
			)
			self.assertEqual(validate.returncode, 0, validate.stderr)
			self.assertEqual(json.loads(validate.stdout)["review_file_count"], 5)

	def test_repository_qualified_cli_packages_with_root_only_pythonpath(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			report, _, _ = self._fixture(root)
			output = root / "review.tar.zst"
			environment = dict(os.environ)
			environment["PYTHONPATH"] = str(REPO_ROOT)
			package = subprocess.run(
				[
					sys.executable, "-m",
					"scripts.python.dmap_observability.portable_bundle", "package",
					"--report-dir", str(report), "--output", str(output),
				],
				cwd=REPO_ROOT,
				env=environment,
				capture_output=True,
				text=True,
			)
			self.assertEqual(package.returncode, 0, package.stderr)
			self.assertEqual(json.loads(package.stdout)["archive"], str(output))

	@staticmethod
	def _write_raw_archive(path: Path, members: dict[str, bytes]) -> None:
		if portable_bundle.zstandard is None:
			raise unittest.SkipTest("zstandard is unavailable")
		compressor = portable_bundle.zstandard.ZstdCompressor(level=1)
		with path.open("wb") as raw:
			with compressor.stream_writer(raw, closefd=False) as compressed:
				with tarfile.open(fileobj=compressed, mode="w|") as archive:
					for name, content in members.items():
						info = tarfile.TarInfo(name)
						info.size = len(content)
						archive.addfile(info, io.BytesIO(content))


if __name__ == "__main__":
	unittest.main()
