#!/usr/bin/env python3
"""Build and validate portable depth-map observability review bundles.

The archive format deliberately has a small, stable contract.  Report files live
under ``review/``; configuration and provenance inputs live under ``metadata/``;
and every regular member except ``inventory.sha256`` is covered by that inventory.
The outer ``.sha256`` companion identifies and integrity-checks the complete
compressed archive; authenticity requires a trusted channel for that digest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import html
import io
import importlib
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import stat
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
from urllib.parse import unquote, urlsplit

from . import integrity

try:
	import zstandard
except ImportError:  # pragma: no cover - exercised through the public error path
	zstandard = None


BUNDLE_SCHEMA_VERSION = 1
DEFAULT_REVIEW_LIMIT_BYTES = 1024**3
DEFAULT_PAYLOAD_LIMIT_BYTES = 4 * 1024**3
DEFAULT_MAX_MEMBERS = 100_000
TEXT_SUFFIXES = frozenset({
	".css", ".csv", ".htm", ".html", ".ini", ".js", ".json", ".jsonl",
	".md", ".py", ".sh", ".svg", ".toml", ".tsv", ".txt", ".xml",
	".yaml", ".yml",
})
HOST_REFERENCE_MARKERS = (
	b"file://",
	b"/home/",
	b"/mnt/",
	b"/Users/",
	b"/opt/",
	b"/srv/",
	b"/data/",
	b"/tmp/",
	b"/var/",
	b"/etc/",
	b"/root/",
	b"/workspace/",
	b"ssh://",
)
HOST_REFERENCE_BYTES_RE = re.compile(
	r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[A-Za-z0-9._-]+\\|"
	r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:|"
	r"ssh(?:\s+-[A-Za-z0-9-]+(?:\s+[^\s<>\"'@]+)?)*"
	r"\s+[A-Za-z0-9._-]+@[A-Za-z0-9.-]+)".encode()
)
SENSITIVE_CONTENT_PATTERNS = (
	("private key", re.compile(rb"-----BEGIN [A-Z ]{0,64}PRIVATE KEY-----")),
	("AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
	("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,255}\b")),
	("GitLab token", re.compile(rb"\bglpat-[A-Za-z0-9_-]{20,255}\b")),
	("OpenAI-style key", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,255}\b")),
	("Hugging Face token", re.compile(rb"\bhf_[A-Za-z0-9]{20,255}\b")),
	("Slack token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,255}\b")),
	("Google API key", re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b")),
	("Stripe live key", re.compile(rb"\bsk_live_[A-Za-z0-9]{16,255}\b")),
	(
		"credential-bearing URL",
		re.compile(
			rb"\b(?:https?|ftp)://[^\s/:@]{1,256}(?::[^\s/@]{0,256})?@[^\s/]+",
			re.IGNORECASE,
		),
	),
)
SENSITIVE_SCAN_OVERLAP_BYTES = 1024


def _load_report_finalizer_module() -> Any:
	"""Import the sibling finalizer from either supported module layout."""

	module_name = (
		"scripts.python.generate_attested_dmap_report"
		if (__package__ or "").startswith("scripts.python.")
		else "generate_attested_dmap_report"
	)
	return importlib.import_module(module_name)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BASE64_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
MARKDOWN_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")
MARKDOWN_REFERENCE_DEFINITION_RE = re.compile(
	r"(?m)^(?P<prefix>[ \t]{0,3}\[(?P<label>[^\]\r\n]+)\]:[ \t]*)"
	r"(?P<target><[^>\r\n]+>|[^\s\r\n]+)(?P<suffix>[^\r\n]*)$"
)
MARKDOWN_IMAGE_REFERENCE_RE = re.compile(
	r"!\[(?P<alt>[^\]\r\n]*)\]\[(?P<label>[^\]\r\n]*)\]"
)
HTML_REFERENCE_RE = re.compile(
	r"(?P<prefix><(?P<tag>[A-Za-z][A-Za-z0-9:-]*)\b[^>]*?\b(?P<attribute>href|src)\s*=\s*)"
	r"(?P<quote>[\"'])(?P<target>.*?)(?P=quote)",
	re.IGNORECASE | re.DOTALL,
)
CSS_URL_RE = re.compile(r"url\(\s*(?P<quote>[\"']?)(?P<target>.*?)(?P=quote)\s*\)", re.IGNORECASE)
INLINE_MODEL_RE = re.compile(
	r"(?P<prefix><script\b[^>]*\bid=[\"']dmap-report-model[\"'][^>]*>)"
	r".*?(?P<suffix></script>)",
	re.IGNORECASE | re.DOTALL,
)
STYLE_BLOCK_RE = re.compile(
	r"(?P<prefix><style\b[^>]*>)(?P<content>.*?)(?P<suffix></style>)",
	re.IGNORECASE | re.DOTALL,
)
HOST_PATH_TOKEN_RE = re.compile(
	r"file://[^\s<>\"']+"
	r"|ssh://[^\s<>\"']+"
	r"|(?<![A-Za-z0-9])(?:/home/|/mnt/|/Users/|/opt/|/srv/|/data/|/tmp/|/var/|/etc/|/root/|/workspace/)[^\s<>\"']+"
	r"|(?<![A-Za-z0-9])(?:\.\./){2,}[^\s<>\"']+"
	r"|[A-Za-z]:[\\/][^\s<>\"']+"
	r"|\\\\[A-Za-z0-9._-]+\\[^\s<>\"']+"
	r"|[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:[^\s<>\"']+"
	r"|ssh(?:\s+-[A-Za-z0-9-]+(?:\s+[^\s<>\"'@]+)?)*"
	r"\s+[A-Za-z0-9._-]+@[A-Za-z0-9.-]+"
)
REFERENCE_AUDIT_SUFFIXES = frozenset({".css", ".htm", ".html", ".json", ".md", ".svg"})
PATH_KEYS = frozenset({
	"candidate_image_name", "csv", "depthmap_dir", "investigation_html", "json",
	"local", "manifest_path", "map_manifest", "markdown", "model", "parquet", "path",
	"reference_dmap", "reference_image_name", "root", "script_path", "shared",
	"source_config", "source_csv", "source_image_name", "source_json", "source_map",
	"source_path", "static_html", "timing_source", "visual_overlay_svg",
	"visual_residual_histogram_svg",
})
RAW_SOURCE_KEYS = frozenset({
	"depthmap_dir", "manifest_path", "map_manifest", "reference_dmap", "root",
	"source_csv", "source_json", "source_map", "source_path", "timing_source",
})
DISPLAY_ASSET_SUFFIXES = frozenset({
	".avif", ".css", ".gif", ".htm", ".html", ".jpeg", ".jpg", ".js", ".png",
	".svg", ".webp",
})
REMOTE_SCHEMES = frozenset({"data", "http", "https", "mailto"})
STAGING_SCHEMA_NAME = "openmvs.dmap_observability.portable_staging"
STAGING_SCHEMA_VERSION = 1
FINALIZER_RECEIPT_FILE = "pre_publish_finalizer_receipt.json"
FINALIZER_INVENTORY_KEY = "pre_publish_finalizer"
FINALIZER_INVENTORY_SCHEMA_NAME = "openmvs.dmap.pre_publish_finalizer_inventory"
PORTABLE_FINALIZER_RECEIPT_SCHEMA_NAME = (
	"openmvs.dmap.portable_pre_publish_finalizer_receipt"
)
PORTABLE_FINALIZER_RECEIPT_SCHEMA_VERSION = 1
PORTABLE_FINALIZER_RECEIPT_FIELDS = frozenset({
	"schema_name", "schema_version", "status", "source_receipt_file_sha256",
	"source_receipt_sha256", "source_finalizer_spec_sha256",
	"source_finalizer_sha256", "source_report_source_sha256",
	"source_recovery_binding", "trust_model", "approved_output_identities",
	"report_inventory_sha256", "live_binding_validation", "receipt_sha256",
})


class BundleError(RuntimeError):
	"""Raised when a bundle cannot be built or fails validation."""


@dataclass(frozen=True)
class RawArtifact:
	"""Content-addressed raw artifact referenced by a review bundle."""

	artifact_id: str
	sha256: str
	size_bytes: int
	uri: str | None = None
	media_type: str | None = None


@dataclass(frozen=True)
class BundleBuildResult:
	archive_path: Path
	checksum_path: Path
	archive_sha256: str
	archive_bytes: int
	payload_bytes: int
	file_count: int


@dataclass(frozen=True)
class BundleValidationResult:
	archive_path: Path
	archive_sha256: str
	archive_bytes: int
	payload_bytes: int
	file_count: int
	review_file_count: int
	raw_artifact_count: int


@dataclass(frozen=True)
class ReviewStageResult:
	stage_path: Path
	file_count: int
	payload_bytes: int
	copied_external_assets: int
	omitted_artifacts: int
	deduplicated_preview_files: int
	deduplicated_preview_bytes: int


@dataclass(frozen=True)
class ReviewTreeValidationResult:
	review_path: Path
	file_count: int
	payload_bytes: int
	reference_count: int
	closure_status: str


@dataclass(frozen=True)
class _SourceFile:
	source: Path | None
	archive_path: str
	role: str
	size_bytes: int
	sha256: str
	mode: int
	content: bytes | None = None


def _canonical_json(value: Any) -> bytes:
	return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
	return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _semantic_digest(value: Any) -> str:
	return hashlib.sha256(json.dumps(
		value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
		allow_nan=False,
	).encode("ascii")).hexdigest()


def _regular_identity(
	path: Path, *, include_path: bool = True, logical_path: str | None = None,
) -> dict[str, Any]:
	if path.is_symlink() or not path.is_file():
		raise BundleError(f"finalizer artifact is not a regular file: {path}")
	status = path.stat()
	identity: dict[str, Any] = {
		"sha256": _sha256_path(path),
		"bytes": status.st_size,
		"mode": stat.S_IMODE(status.st_mode),
	}
	if include_path:
		identity = {"path": logical_path or path.name, **identity}
	return identity


def _report_inventory_declares_finalizer(report_dir: Path) -> bool:
	inventory_path = report_dir / "report_inventory.json"
	if not inventory_path.exists() and not inventory_path.is_symlink():
		return False
	if inventory_path.is_symlink() or not inventory_path.is_file():
		raise BundleError("portable report inventory is missing or unsafe")
	try:
		inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise BundleError("portable report inventory is unreadable") from error
	if not isinstance(inventory, dict):
		raise BundleError("portable report inventory must be a JSON object")
	return FINALIZER_INVENTORY_KEY in inventory


def _load_canonical_finalizer_receipt(
	report_dir: Path, published_dir: Path | None = None,
) -> dict[str, Any] | None:
	"""Use the canonical validator, skipping only live executable/schedule checks."""

	path = report_dir / FINALIZER_RECEIPT_FILE
	if path.is_symlink():
		raise BundleError("canonical finalizer receipt is missing or unsafe")
	if not path.exists():
		if _report_inventory_declares_finalizer(report_dir):
			raise BundleError(
				"portable report inventory declares finalizer artifacts but the receipt is missing"
			)
		return None
	if not path.is_file():
		raise BundleError("canonical finalizer receipt is missing or unsafe")
	try:
		generate_attested_dmap_report = _load_report_finalizer_module()
		result = generate_attested_dmap_report.validate_finalizer_receipt(
			report_dir, published_dir or report_dir, require_receipt=True,
			verify_live_bindings=False,
		)
	except (
		OSError, RuntimeError, TypeError, ValueError, UnicodeError,
		json.JSONDecodeError,
	) as error:
		raise BundleError(f"canonical finalizer receipt is invalid: {error}") from error
	return result["receipt"]


def _portable_receipt_payload(
	report_dir: Path, source_receipt_path: Path, source_receipt_file_sha256: str,
	source_receipt: Mapping[str, Any],
) -> dict[str, Any]:
	if _sha256_path(source_receipt_path) != source_receipt_file_sha256:
		raise BundleError("canonical finalizer receipt changed while packaging")
	outputs = []
	for source_output in source_receipt["approved_output_identities"]:
		relative = _validate_archive_path(str(source_output["path"]))
		outputs.append({"path": relative, **_regular_identity(
			report_dir / relative, include_path=False,
		)})
	inventory_path = report_dir / "report_inventory.json"
	try:
		inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise BundleError("portable report inventory is unreadable") from error
	inventory[FINALIZER_INVENTORY_KEY] = {
		"schema_name": FINALIZER_INVENTORY_SCHEMA_NAME,
		"schema_version": 1,
		"finalizer_receipt": FINALIZER_RECEIPT_FILE,
		"approved_outputs": outputs,
	}
	inventory_path.write_bytes(_canonical_json(inventory))
	finalizer = source_receipt["finalizer"]
	recovery = source_receipt.get("recovery_binding")
	recovery_summary = None if recovery is None else {
		"sha256": recovery.get("sha256"),
		"schedule_identity_sha256": recovery.get("schedule_identity_sha256"),
		"evidence_digest": recovery.get("evidence_digest"),
	}
	value: dict[str, Any] = {
		"schema_name": PORTABLE_FINALIZER_RECEIPT_SCHEMA_NAME,
		"schema_version": PORTABLE_FINALIZER_RECEIPT_SCHEMA_VERSION,
		"status": "complete",
		"source_receipt_file_sha256": source_receipt_file_sha256,
		"source_receipt_sha256": source_receipt["receipt_sha256"],
		"source_finalizer_spec_sha256": finalizer["spec_sha256"],
		"source_finalizer_sha256": finalizer["sha256"],
		"source_report_source_sha256": source_receipt["report_source_sha256"],
		"source_recovery_binding": recovery_summary,
		"trust_model": {
			"explicitly_trusted": True,
			"arbitrary_code_execution": True,
			"isolation": "none",
		},
		"approved_output_identities": outputs,
		"report_inventory_sha256": _sha256_path(inventory_path),
		"live_binding_validation": "not_applicable_to_sanitized_portable_copy",
	}
	value["receipt_sha256"] = _semantic_digest(value)
	return value


def _validate_portable_receipt_value(value: Any) -> list[dict[str, Any]]:
	if (
		not isinstance(value, dict)
		or set(value) != PORTABLE_FINALIZER_RECEIPT_FIELDS
		or value.get("schema_name") != PORTABLE_FINALIZER_RECEIPT_SCHEMA_NAME
		or value.get("schema_version") != PORTABLE_FINALIZER_RECEIPT_SCHEMA_VERSION
		or value.get("status") != "complete"
	):
		raise BundleError("portable finalizer receipt has an unsupported schema")
	payload = dict(value)
	payload.pop("receipt_sha256", None)
	if value.get("receipt_sha256") != _semantic_digest(payload):
		raise BundleError("portable finalizer receipt self-digest is invalid")
	for key in (
		"source_receipt_file_sha256", "source_receipt_sha256",
		"source_finalizer_spec_sha256", "source_finalizer_sha256",
		"source_report_source_sha256", "report_inventory_sha256",
		"receipt_sha256",
	):
		if not isinstance(value.get(key), str) or not SHA256_RE.fullmatch(value[key]):
			raise BundleError(f"portable finalizer receipt has invalid {key}")
	recovery = value.get("source_recovery_binding")
	if recovery is not None and (
		not isinstance(recovery, dict)
		or set(recovery) != {"sha256", "schedule_identity_sha256", "evidence_digest"}
		or any(
			not isinstance(item, str) or not SHA256_RE.fullmatch(item)
			for item in recovery.values()
		)
	):
		raise BundleError("portable finalizer recovery digest binding is invalid")
	if value.get("trust_model") != {
		"explicitly_trusted": True,
		"arbitrary_code_execution": True,
		"isolation": "none",
	} or value.get("live_binding_validation") != (
		"not_applicable_to_sanitized_portable_copy"
	):
		raise BundleError("portable finalizer receipt trust model is invalid")
	outputs = value.get("approved_output_identities")
	if not isinstance(outputs, list) or not outputs:
		raise BundleError("portable finalizer receipt has no outputs")
	for output in outputs:
		if not isinstance(output, dict) or set(output) != {
			"path", "sha256", "bytes", "mode",
		}:
			raise BundleError("portable finalizer output identity is malformed")
		_validate_archive_path(str(output["path"]))
		if not (
			isinstance(output["sha256"], str)
			and SHA256_RE.fullmatch(output["sha256"])
			and isinstance(output["bytes"], int) and output["bytes"] >= 0
			and isinstance(output["mode"], int) and 0 <= output["mode"] <= 0o7777
		):
			raise BundleError("portable finalizer output identity values are invalid")
	return outputs


def _validate_staged_portable_receipt(report_dir: Path) -> None:
	receipt_path = report_dir / FINALIZER_RECEIPT_FILE
	if receipt_path.is_symlink():
		raise BundleError("portable finalizer receipt is missing or unsafe")
	if not receipt_path.exists():
		if _report_inventory_declares_finalizer(report_dir):
			raise BundleError(
				"portable report inventory declares finalizer artifacts but the receipt is missing"
			)
		return
	if not receipt_path.is_file():
		raise BundleError("portable finalizer receipt is missing or unsafe")
	try:
		value = json.loads(receipt_path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise BundleError("portable finalizer receipt is unreadable") from error
	outputs = _validate_portable_receipt_value(value)
	for output in outputs:
		if _regular_identity(
			report_dir / output["path"], logical_path=output["path"],
		) != output:
			raise BundleError(f"portable finalizer output drifted: {output['path']}")
	inventory_path = report_dir / "report_inventory.json"
	if _sha256_path(inventory_path) != value["report_inventory_sha256"]:
		raise BundleError("portable finalizer inventory digest is invalid")
	try:
		inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise BundleError("portable report inventory is unreadable") from error
	if not isinstance(inventory, dict):
		raise BundleError("portable report inventory must be a JSON object")
	expected = {
		"schema_name": FINALIZER_INVENTORY_SCHEMA_NAME,
		"schema_version": 1,
		"finalizer_receipt": FINALIZER_RECEIPT_FILE,
		"approved_outputs": outputs,
	}
	if inventory.get(FINALIZER_INVENTORY_KEY) != expected:
		raise BundleError("portable report inventory omits finalizer artifacts")


def _validate_archive_path(raw_path: str) -> str:
	if not raw_path or "\\" in raw_path or "\x00" in raw_path:
		raise BundleError(f"invalid archive path: {raw_path!r}")
	if any(ord(character) < 32 for character in raw_path):
		raise BundleError(f"archive path contains a control character: {raw_path!r}")
	path = PurePosixPath(raw_path)
	if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
		raise BundleError(f"archive path is absolute or escapes its root: {raw_path!r}")
	if path.as_posix() != raw_path:
		raise BundleError(f"archive path is not normalized: {raw_path!r}")
	return raw_path


def _archive_is_within(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve())
		return True
	except ValueError:
		return False


def _safe_suffix(path: Path) -> str:
	suffix = path.suffix.lower()
	return suffix if re.fullmatch(r"\.[a-z0-9]{1,12}", suffix) else ".bin"


def _reference_path(raw_value: str) -> str:
	value = html.unescape(raw_value.strip())
	if value.startswith("<") and value.endswith(">"):
		value = value[1:-1].strip()
	return value


def _is_remote_or_inline(raw_value: str) -> bool:
	value = _reference_path(raw_value)
	if not value or value.startswith("#"):
		return True
	return urlsplit(value).scheme.lower() in REMOTE_SCHEMES


def _is_dynamic_reference(raw_value: str) -> bool:
	return any(marker in raw_value for marker in ("${", "{{", "<%"))


class _StageContext:
	def __init__(
		self,
		source_root: Path,
		stage_root: Path,
		omitted_paths: set[Path],
		aliases: Mapping[Path, Path] | None = None,
		reference_root: Path | None = None,
	) -> None:
		self.source_root = source_root.resolve()
		self.reference_root = (
			Path(reference_root).resolve() if reference_root is not None
			else self.source_root
		)
		self.stage_root = stage_root.resolve()
		self.omitted_paths = {path.resolve() for path in omitted_paths}
		self.aliases = {
			source.resolve(): canonical.resolve() for source, canonical in (aliases or {}).items()
		}
		self.artifacts: dict[str, dict[str, Any]] = {}
		self._artifact_by_identity: dict[str, str] = {}
		self._asset_by_digest: dict[tuple[str, str], str] = {}

	def _resolve_local(self, raw_value: str, source_file: Path) -> tuple[Path | None, str, str]:
		value = _reference_path(raw_value)
		parsed = urlsplit(value)
		if parsed.scheme.lower() in REMOTE_SCHEMES or value.startswith("#"):
			return None, parsed.query, parsed.fragment
		if parsed.scheme.lower() == "file":
			path_text = unquote(parsed.path)
		elif parsed.scheme and not re.match(r"^[A-Za-z]:[\\/]", value):
			return None, parsed.query, parsed.fragment
		else:
			path_text = unquote(parsed.path or value)
		if re.match(r"^[A-Za-z]:[\\/]", path_text):
			return None, parsed.query, parsed.fragment
		target = Path(path_text)
		if not target.is_absolute():
			snapshot_target = Path(os.path.abspath(source_file.parent / target))
			if self.reference_root != self.source_root:
				try:
					snapshot_target.relative_to(self.source_root)
				except ValueError:
					try:
						source_relative = source_file.relative_to(self.source_root)
					except ValueError:
						pass
					else:
						target = self.reference_root / source_relative.parent / target
				else:
					target = snapshot_target
			else:
				target = snapshot_target
		elif self.reference_root != self.source_root:
			# Canonical absolute paths must resolve against the immutable raw snapshot,
			# never against the live report after its stable-copy window has closed.
			normalized = Path(os.path.abspath(target))
			try:
				target = self.source_root / normalized.relative_to(self.reference_root)
			except ValueError:
				pass
		target = target.resolve(strict=False)
		if self.reference_root != self.source_root and _archive_is_within(
			target, self.reference_root
		):
			target = self.source_root / target.relative_to(self.reference_root)
		return target.resolve(strict=False), parsed.query, parsed.fragment

	def _relative_stage_reference(
		self, target: Path, stage_file: Path, query: str = "", fragment: str = ""
	) -> str:
		relative = os.path.relpath(target, start=stage_file.parent).replace(os.sep, "/")
		if query:
			relative += f"?{query}"
		if fragment:
			relative += f"#{fragment}"
		return relative

	def _artifact_descriptor(self, raw_value: str, source_file: Path, *, role: str) -> dict[str, Any]:
		target, _query, _fragment = self._resolve_local(raw_value, source_file)
		if target is not None and _archive_is_within(target, self.source_root):
			identity = f"report:{target.relative_to(self.source_root).as_posix()}"
		else:
			identity = str(target) if target is not None else _reference_path(raw_value)
		identity_digest = hashlib.sha256(identity.encode("utf-8", errors="surrogateescape")).hexdigest()
		artifact_id = self._artifact_by_identity.get(identity)
		if artifact_id is None:
			artifact_id = f"omitted-{identity_digest[:24]}"
			self._artifact_by_identity[identity] = artifact_id
			is_file = bool(target is not None and target.is_file())
			is_dir = bool(target is not None and target.is_dir())
			status = (
				"omitted_from_review_bundle" if is_file
				else "external_directory_omitted" if is_dir
				else "source_not_present_at_package_time"
			)
			name = target.name if target is not None and target.name else "external source"
			media_type = mimetypes.guess_type(name)[0]
			descriptor = {
				"artifact_id": artifact_id,
				"identity_basis": "source_reference_sha256",
				"media_type": media_type,
				"name": name,
				"role": role,
				"size_bytes": target.stat().st_size if is_file else None,
				"status": status,
			}
			self.artifacts[artifact_id] = descriptor
		return dict(self.artifacts[artifact_id])

	def _copy_external_asset(self, target: Path) -> Path:
		if target.is_symlink() or not target.is_file():
			raise BundleError(f"required portable report asset is not a regular file: {target}")
		digest = _sha256_path(target)
		suffix = _safe_suffix(target)
		key = (digest, suffix)
		relative = self._asset_by_digest.get(key)
		if relative is None:
			relative = f"portable_assets/sha256/{digest[:2]}/{digest}{suffix}"
			destination = self.stage_root / relative
			destination.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(target, destination)
			self._asset_by_digest[key] = relative
		return self.stage_root / relative

	def sanitize_reference(
		self,
		raw_value: str,
		*,
		source_file: Path,
		stage_file: Path,
		role: str,
		copy_external: bool,
		required: bool = False,
	) -> tuple[str, dict[str, Any] | None]:
		value = _reference_path(raw_value)
		if not value or value.startswith("#"):
			return value, None
		parsed = urlsplit(value)
		if parsed.scheme.lower() in REMOTE_SCHEMES:
			if required and parsed.scheme.lower() in {"http", "https"}:
				raise BundleError(
					f"required report asset for {role} is remote and cannot be bundled: {value}"
				)
			return value, None
		target, query, fragment = self._resolve_local(value, source_file)
		if target is None:
			descriptor = self._artifact_descriptor(value, source_file, role=role)
			return "portable_artifacts.html", descriptor
		if _archive_is_within(target, self.source_root):
			relative_source = target.relative_to(self.source_root)
			canonical_source = self.aliases.get(target, target)
			stage_target = self.stage_root / canonical_source.relative_to(self.source_root)
			if target in self.omitted_paths:
				descriptor = self._artifact_descriptor(value, source_file, role=role)
				return "portable_artifacts.html", descriptor
			if target.is_file() and stage_target.is_file():
				return self._relative_stage_reference(stage_target, stage_file, query, fragment), None
			if required:
				raise BundleError(
					f"required report asset for {role} is missing: {target} (from {source_file})"
				)
			descriptor = self._artifact_descriptor(value, source_file, role=role)
			return "portable_artifacts.html", descriptor
		if copy_external and target.is_file():
			stage_target = self._copy_external_asset(target)
			return self._relative_stage_reference(stage_target, stage_file, query, fragment), None
		if required:
			raise BundleError(
				f"required report asset for {role} is missing: {target} (from {source_file})"
			)
		descriptor = self._artifact_descriptor(value, source_file, role=role)
		return "portable_artifacts.html", descriptor

	def requires_reference_rewrite(self, raw_value: str, source_file: Path) -> bool:
		if _looks_like_unsafe_local_reference(raw_value):
			return True
		target, _query, _fragment = self._resolve_local(raw_value, source_file)
		return bool(
			target is not None
			and (target in self.aliases or target in self.omitted_paths)
		)

	@property
	def copied_external_assets(self) -> int:
		return len(self._asset_by_digest)


def _path_copy_policy(key: str, raw_value: str, parent_key: str | None) -> tuple[bool, bool]:
	suffix = Path(urlsplit(_reference_path(raw_value)).path).suffix.lower()
	strict_required = (
		key in {"local", "shared", "script_path", "investigation_html", "markdown", "model", "static_html"}
		or (key == "path" and parent_key == "reference")
	)
	copy_external = (
		strict_required
		or key in {"candidate_image_name", "reference_image_name", "source_image_name"}
		or (key == "path" and suffix in DISPLAY_ASSET_SUFFIXES)
	)
	return copy_external, strict_required


def _looks_like_unsafe_local_reference(value: str) -> bool:
	parsed = urlsplit(_reference_path(value))
	return (
		parsed.scheme.lower() in {"file", "ssh"}
		or Path(parsed.path or value).is_absolute()
		or bool(re.match(r"^[A-Za-z]:[\\/]", value))
		or value.startswith("\\\\")
		or bool(re.match(r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:", value))
		or any(part == ".." for part in PurePosixPath(parsed.path or value).parts)
	)


def _is_path_field(key: str, value: str, parent_key: str | None) -> bool:
	if key in {"investigation_html", "markdown", "model", "static_html"}:
		return parent_key == "entrypoints"
	if key == "root":
		return parent_key == "experiment" or _looks_like_unsafe_local_reference(value)
	if key == "path":
		parsed_path = urlsplit(_reference_path(value)).path
		return (
			parent_key == "reference"
			or _looks_like_unsafe_local_reference(value)
			or "/" in parsed_path
			or bool(Path(parsed_path).suffix)
		)
	return key in PATH_KEYS


def _sanitize_json_value(
	value: Any,
	*,
	context: _StageContext,
	source_file: Path,
	stage_file: Path,
	parent_key: str | None = None,
) -> Any:
	if isinstance(value, dict):
		result: dict[str, Any] = {}
		for key, item in value.items():
			if isinstance(item, str) and (
				_is_path_field(key, item, parent_key) or _looks_like_unsafe_local_reference(item)
			):
				copy_external, required = _path_copy_policy(key, item, parent_key)
				if key in RAW_SOURCE_KEYS:
					copy_external = required = False
				rewritten, descriptor = context.sanitize_reference(
					item,
					source_file=source_file,
					stage_file=stage_file,
					role=key,
					copy_external=copy_external,
					required=required,
				)
				result[key] = _scrub_host_path_tokens(
					rewritten, context=context, source_file=source_file,
				)
				if descriptor is not None:
					result[f"{key}_artifact"] = descriptor
			else:
				result[key] = _sanitize_json_value(
					item,
					context=context,
					source_file=source_file,
					stage_file=stage_file,
					parent_key=key,
				)
		return result
	if isinstance(value, list):
		return [
			_sanitize_json_value(
				item,
				context=context,
				source_file=source_file,
				stage_file=stage_file,
				parent_key=parent_key,
			)
			for item in value
		]
	if isinstance(value, str) and context.requires_reference_rewrite(value, source_file):
		suffix = Path(urlsplit(_reference_path(value)).path).suffix.lower()
		rewritten, _descriptor = context.sanitize_reference(
			value,
			source_file=source_file,
			stage_file=stage_file,
			role=parent_key or "list_reference",
			copy_external=suffix in DISPLAY_ASSET_SUFFIXES,
			required=False,
		)
		value = rewritten
	if isinstance(value, str):
		return _scrub_host_path_tokens(
			value, context=context, source_file=source_file,
		)
	return value


def _preview_aliases(source_root: Path) -> tuple[dict[Path, Path], int]:
	aliases: dict[Path, Path] = {}
	reclaimed_bytes = 0
	preview_root = source_root / "interactive" / "maps"
	if not preview_root.is_dir():
		return aliases, reclaimed_bytes
	groups: dict[tuple[int, str], list[Path]] = {}
	for preview in sorted(preview_root.rglob("*")):
		if preview.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
			continue
		if preview.is_symlink() or not preview.is_file():
			continue
		key = (preview.stat().st_size, _sha256_path(preview))
		groups.setdefault(key, []).append(preview)
	for previews in groups.values():
		if len(previews) < 2:
			continue
		canonical = min(
			previews,
			key=lambda path: (
				0 if path.name.endswith("_local.png") else 1,
				path.relative_to(preview_root).as_posix(),
			),
		)
		for duplicate in previews:
			if duplicate == canonical:
				continue
			aliases[duplicate.resolve()] = canonical.resolve()
			reclaimed_bytes += duplicate.stat().st_size
	return aliases, reclaimed_bytes


def _artifact_link(descriptor: Mapping[str, Any]) -> str:
	return f"portable_artifacts.html#{descriptor['artifact_id']}"


def _sanitize_markdown(
	text_value: str,
	*,
	context: _StageContext,
	source_file: Path,
	stage_file: Path,
) -> str:
	image_labels = {
		" ".join((match.group("label") or match.group("alt")).split()).casefold()
		for match in MARKDOWN_IMAGE_REFERENCE_RE.finditer(text_value)
	}

	def replace(match: re.Match[str]) -> str:
		is_image, label, target = match.groups()
		if _is_remote_or_inline(target) or _is_dynamic_reference(target):
			return match.group(0)
		rewritten, descriptor = context.sanitize_reference(
			target,
			source_file=source_file,
			stage_file=stage_file,
			role="markdown_image" if is_image else "markdown_link",
			copy_external=bool(is_image),
			required=bool(is_image),
		)
		if descriptor is not None:
			rewritten = _artifact_link(descriptor)
		return f"{is_image}[{label}]({rewritten})"

	text_value = MARKDOWN_LINK_RE.sub(replace, text_value)

	def replace_definition(match: re.Match[str]) -> str:
		target = match.group("target")
		if _is_remote_or_inline(target) or _is_dynamic_reference(target):
			return match.group(0)
		label = " ".join(match.group("label").split()).casefold()
		is_image = label in image_labels
		rewritten, descriptor = context.sanitize_reference(
			target,
			source_file=source_file,
			stage_file=stage_file,
			role="markdown_image_reference" if is_image else "markdown_link_reference",
			copy_external=is_image,
			required=is_image,
		)
		if descriptor is not None:
			rewritten = _artifact_link(descriptor)
		return f"{match.group('prefix')}{rewritten}{match.group('suffix')}"

	return MARKDOWN_REFERENCE_DEFINITION_RE.sub(replace_definition, text_value)


def _sanitize_html_references(
	text_value: str,
	*,
	context: _StageContext,
	source_file: Path,
	stage_file: Path,
) -> str:
	def replace(match: re.Match[str]) -> str:
		tag = match.group("tag").lower()
		attribute = match.group("attribute").lower()
		target = match.group("target")
		if _is_remote_or_inline(target) or _is_dynamic_reference(target):
			return match.group(0)
		required = attribute == "src" or tag in {"img", "link", "script", "source"}
		rewritten, descriptor = context.sanitize_reference(
			target,
			source_file=source_file,
			stage_file=stage_file,
			role=f"html_{tag}_{attribute}",
			copy_external=required,
			required=required,
		)
		if descriptor is not None:
			rewritten = _artifact_link(descriptor)
		return f"{match.group('prefix')}{match.group('quote')}{html.escape(rewritten, quote=True)}{match.group('quote')}"

	return HTML_REFERENCE_RE.sub(replace, text_value)


def _sanitize_css_references(
	text_value: str,
	*,
	context: _StageContext,
	source_file: Path,
	stage_file: Path,
) -> str:
	def replace(match: re.Match[str]) -> str:
		target = match.group("target")
		if _is_remote_or_inline(target) or _is_dynamic_reference(target):
			return match.group(0)
		rewritten, _descriptor = context.sanitize_reference(
			target,
			source_file=source_file,
			stage_file=stage_file,
			role="css_asset",
			copy_external=True,
			required=True,
		)
		quote = match.group("quote")
		return f"url({quote}{rewritten}{quote})"

	return CSS_URL_RE.sub(replace, text_value)


def _scrub_host_path_tokens(
	text_value: str,
	*,
	context: _StageContext,
	source_file: Path,
) -> str:
	def replace(match: re.Match[str]) -> str:
		token = match.group(0)
		trimmed = token.rstrip(".,;:)]}")
		trailing = token[len(trimmed):]
		descriptor = context._artifact_descriptor(trimmed, source_file, role="text_reference")
		return f"artifact:{descriptor['artifact_id']}[{descriptor['status']}]{trailing}"

	return HOST_PATH_TOKEN_RE.sub(replace, text_value)


def _sanitize_detached_text(text_value: str) -> str:
	"""Sanitize host paths in config/provenance files that live outside the review tree."""
	def replace(match: re.Match[str]) -> str:
		token = match.group(0)
		trimmed = token.rstrip(".,;:)]}")
		trailing = token[len(trimmed):]
		identifier = hashlib.sha256(trimmed.encode("utf-8", errors="surrogateescape")).hexdigest()
		return f"artifact:omitted-{identifier[:24]}[omitted_from_review_bundle]{trailing}"

	return HOST_PATH_TOKEN_RE.sub(replace, text_value)


def _sanitize_delimited_file(
	path: Path,
	*,
	source_file: Path,
	context: _StageContext,
	delimiter: str,
) -> None:
	try:
		with path.open("r", encoding="utf-8", newline="") as stream:
			reader = csv.reader(stream, delimiter=delimiter)
			rows = list(reader)
	except (OSError, UnicodeError, csv.Error) as error:
		raise BundleError(f"cannot parse shareable table {source_file}: {error}") from error
	if not rows:
		return
	headers = rows[0]
	output: list[list[str]] = [headers]
	for row in rows[1:]:
		rewritten_row: list[str] = []
		for index, value in enumerate(row):
			key = headers[index] if index < len(headers) else ""
			if value and (
				_is_path_field(key, value, None) or _looks_like_unsafe_local_reference(value)
			):
				copy_external, required = _path_copy_policy(key, value, None)
				if key in RAW_SOURCE_KEYS:
					copy_external = required = False
				rewritten, descriptor = context.sanitize_reference(
					value,
					source_file=source_file,
					stage_file=path,
					role=key or "table_reference",
					copy_external=copy_external,
					required=required,
				)
				if descriptor is not None:
					rewritten = f"artifact:{descriptor['artifact_id']}[{descriptor['status']}]"
				value = rewritten
			value = _scrub_host_path_tokens(value, context=context, source_file=source_file)
			rewritten_row.append(value)
		output.append(rewritten_row)
	buffer = io.StringIO(newline="")
	writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
	writer.writerows(output)
	path.write_text(buffer.getvalue(), encoding="utf-8", newline="")


def _artifact_pages(context: _StageContext) -> tuple[bytes, bytes, bytes]:
	rows = [context.artifacts[key] for key in sorted(context.artifacts)]
	json_content = _canonical_json({
		"artifacts": rows,
		"schema_name": "openmvs.dmap_observability.portable_artifact_index",
		"schema_version": 1,
	})
	markdown = [
		"# Portable Artifact Index", "",
		"Raw sources are intentionally omitted from the review payload. Their signals and metrics remain available in the report.", "",
		"| Artifact ID | Status | Role | Name | Size |",
		"|---|---|---|---|---:|",
	]
	for row in rows:
		markdown.append(
			f"| <a id=\"{row['artifact_id']}\"></a>`{row['artifact_id']}` | "
			f"{row['status']} | {row['role']} | `{row['name']}` | {row['size_bytes'] or ''} |"
		)
	markdown_content = ("\n".join(markdown) + "\n").encode("utf-8")
	html_rows = "".join(
		"<tr id=\"{id}\"><td><code>{id}</code></td><td>{status}</td><td>{role}</td>"
		"<td><code>{name}</code></td><td>{size}</td></tr>".format(
			id=html.escape(str(row["artifact_id"])),
			status=html.escape(str(row["status"])),
			role=html.escape(str(row["role"])),
			name=html.escape(str(row["name"])),
			size=html.escape(str(row["size_bytes"] or "")),
		)
		for row in rows
	)
	html_content = (
		"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
		"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
		"<title>Portable Artifact Index</title><style>body{font:14px system-ui;margin:2rem;"
		"color:#172126}table{border-collapse:collapse;width:100%}th,td{border:1px solid #cad4d8;"
		"padding:.45rem;text-align:left}th{background:#edf3f2}code{overflow-wrap:anywhere}</style>"
		"</head><body><h1>Portable Artifact Index</h1><p>Raw sources are intentionally omitted "
		"from the review payload. Their signals and metrics remain available in the report.</p>"
		f"<table><thead><tr><th>Artifact ID</th><th>Status</th><th>Role</th><th>Name</th>"
		f"<th>Size</th></tr></thead><tbody>{html_rows}</tbody></table></body></html>\n"
	).encode("utf-8")
	return json_content, markdown_content, html_content


def _write_sanitized_json(
	source_file: Path,
	stage_file: Path,
	context: _StageContext,
) -> Any:
	try:
		value = json.loads(source_file.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise BundleError(f"cannot parse shareable JSON {source_file}: {error}") from error
	rewritten = _sanitize_json_value(
		value,
		context=context,
		source_file=source_file,
		stage_file=stage_file,
	)
	stage_file.write_bytes(_canonical_json(rewritten))
	return rewritten


def _write_sanitized_jsonl(source_file: Path, stage_file: Path, context: _StageContext) -> None:
	lines: list[bytes] = []
	try:
		for line_number, line in enumerate(source_file.read_text(encoding="utf-8").splitlines(), start=1):
			if not line.strip():
				continue
			value = json.loads(line)
			rewritten = _sanitize_json_value(
				value,
				context=context,
				source_file=source_file,
				stage_file=stage_file,
			)
			lines.append(json.dumps(rewritten, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n")
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise BundleError(f"cannot parse shareable JSONL {source_file} at line {line_number}: {error}") from error
	stage_file.write_bytes(b"".join(lines))


def _copy_report_tree(
	source_root: Path,
	stage_root: Path,
	aliases: Mapping[Path, Path],
	omitted_paths: set[Path],
) -> list[Path]:
	source_files: list[Path] = []
	alias_paths = {path.resolve() for path in aliases}
	for source in sorted(source_root.rglob("*"), key=lambda path: path.relative_to(source_root).as_posix()):
		if source.is_symlink():
			raise BundleError(f"symlinks are not portable bundle inputs: {source}")
		status = source.stat()
		if stat.S_ISDIR(status.st_mode):
			continue
		if not stat.S_ISREG(status.st_mode):
			raise BundleError(f"special files are not portable bundle inputs: {source}")
		source_files.append(source)
		if source.resolve() in alias_paths or source.resolve() in omitted_paths:
			continue
		destination = stage_root / source.relative_to(source_root)
		destination.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(source, destination)
		destination.chmod(0o755 if status.st_mode & 0o111 else 0o644)
	return source_files


def _copy_raw_report_snapshot(source_root: Path, snapshot_root: Path) -> None:
	"""Copy the unsanitized canonical tree before any portable transformation."""

	shutil.copytree(
		source_root, snapshot_root, symlinks=True, copy_function=shutil.copy2,
	)


def _create_stable_report_snapshot(source_root: Path, snapshot_root: Path) -> None:
	try:
		integrity.require_report_tree_closure(source_root, allow_legacy=True)
		generate_attested_dmap_report = _load_report_finalizer_module()
		before = generate_attested_dmap_report.recursive_tree_identity(source_root)
		_copy_raw_report_snapshot(source_root, snapshot_root)
		integrity.require_report_tree_closure(snapshot_root, allow_legacy=True)
		snapshot = generate_attested_dmap_report.recursive_tree_identity(snapshot_root)
		integrity.require_report_tree_closure(source_root, allow_legacy=True)
		after = generate_attested_dmap_report.recursive_tree_identity(source_root)
	except (OSError, RuntimeError, integrity.IntegrityError) as error:
		if "symlink" in str(error).lower():
			raise BundleError(f"symlinks are not portable bundle inputs: {error}") from error
		raise BundleError(f"cannot create stable canonical report snapshot: {error}") from error
	if before != snapshot or snapshot != after:
		raise BundleError(
			"canonical report changed while creating its raw portable snapshot"
		)


def stage_review_tree(
	report_dir: Path,
	stage_dir: Path,
	*,
	overwrite: bool = False,
) -> ReviewStageResult:
	"""Create a sanitized, self-contained review tree without modifying the source report."""
	report_dir = Path(report_dir).resolve()
	stage_dir = Path(stage_dir).resolve(strict=False)
	if not report_dir.is_dir():
		raise BundleError(f"report directory does not exist: {report_dir}")
	try:
		source_closure = integrity.require_report_tree_closure(
			report_dir, allow_legacy=True,
		)
	except integrity.IntegrityError as error:
		raise BundleError(f"source report artifact closure is invalid: {error}") from error
	if _archive_is_within(stage_dir, report_dir) or _archive_is_within(report_dir, stage_dir):
		raise BundleError("portable staging directory must be separate from the source report")
	if stage_dir.exists():
		if not overwrite:
			raise BundleError(f"refusing to overwrite an existing staging directory: {stage_dir}")
		if stage_dir.is_symlink() or not stage_dir.is_dir():
			raise BundleError(f"staging output is not a regular directory: {stage_dir}")
		shutil.rmtree(stage_dir)
	stage_dir.mkdir(parents=True)
	snapshot_container: Path | None = None
	try:
		snapshot_container = Path(tempfile.mkdtemp(
			prefix=f".{stage_dir.name}.canonical-snapshot.", dir=stage_dir.parent,
		))
		snapshot_dir = snapshot_container / "report"
		_create_stable_report_snapshot(report_dir, snapshot_dir)
		canonical_receipt_path = snapshot_dir / FINALIZER_RECEIPT_FILE
		canonical_receipt = _load_canonical_finalizer_receipt(
			snapshot_dir, published_dir=report_dir,
		)
		canonical_receipt_file_sha256 = (
			_sha256_path(canonical_receipt_path)
			if canonical_receipt is not None else None
		)
		aliases, deduplicated_preview_bytes = _preview_aliases(snapshot_dir)
		omitted_paths = {
			path.resolve() for path in snapshot_dir.rglob("*.parquet") if path.is_file()
		}
		context = _StageContext(
			snapshot_dir, stage_dir, omitted_paths, aliases,
			reference_root=report_dir,
		)
		source_files = _copy_report_tree(snapshot_dir, stage_dir, aliases, omitted_paths)
		if canonical_receipt is not None:
			(stage_dir / FINALIZER_RECEIPT_FILE).unlink()
			source_files = [
				source for source in source_files
				if source != canonical_receipt_path
			]

		report_model: Any = None
		model_source = snapshot_dir / "report_model.json"
		model_stage = stage_dir / "report_model.json"
		if model_source.is_file() and model_stage.is_file():
			report_model = _write_sanitized_json(model_source, model_stage, context)

		for source in source_files:
			if source.resolve() in aliases or source.resolve() in omitted_paths:
				continue
			relative = source.relative_to(snapshot_dir)
			stage_file = stage_dir / relative
			suffix = source.suffix.lower()
			if source == model_source:
				continue
			if suffix == ".json":
				_write_sanitized_json(source, stage_file, context)
			elif suffix == ".jsonl":
				_write_sanitized_jsonl(source, stage_file, context)

		for source in source_files:
			if source.resolve() in aliases or source.resolve() in omitted_paths:
				continue
			relative = source.relative_to(snapshot_dir)
			stage_file = stage_dir / relative
			suffix = source.suffix.lower()
			if suffix in {".csv", ".tsv"}:
				_sanitize_delimited_file(
					stage_file,
					source_file=source,
					context=context,
					delimiter="," if suffix == ".csv" else "\t",
				)

		processed_text: set[Path] = set()
		for source in source_files:
			if source.resolve() in aliases or source.resolve() in omitted_paths:
				continue
			relative = source.relative_to(snapshot_dir)
			stage_file = stage_dir / relative
			suffix = source.suffix.lower()
			if suffix not in {".md", ".htm", ".html", ".css", ".svg"}:
				continue
			try:
				text_value = stage_file.read_text(encoding="utf-8")
			except (OSError, UnicodeError) as error:
				raise BundleError(f"cannot sanitize shareable text {source}: {error}") from error
			if suffix == ".md":
				text_value = _sanitize_markdown(
					text_value, context=context, source_file=source, stage_file=stage_file
				)
			if suffix in {".htm", ".html", ".svg"}:
				if report_model is not None and suffix in {".htm", ".html"}:
					model_json = json.dumps(
						report_model, sort_keys=True, ensure_ascii=True, separators=(",", ":")
					).replace("</", "<\\/")
					text_value = INLINE_MODEL_RE.sub(
						lambda match: f"{match.group('prefix')}{model_json}{match.group('suffix')}",
						text_value,
					)
				text_value = _sanitize_html_references(
					text_value, context=context, source_file=source, stage_file=stage_file
				)
			if suffix in {".htm", ".html"}:
				text_value = STYLE_BLOCK_RE.sub(
					lambda match: (
						f"{match.group('prefix')}"
						f"{_sanitize_css_references(match.group('content'), context=context, source_file=source, stage_file=stage_file)}"
						f"{match.group('suffix')}"
					),
					text_value,
				)
			elif suffix in {".css", ".svg"}:
				text_value = _sanitize_css_references(
					text_value, context=context, source_file=source, stage_file=stage_file
				)
			text_value = _scrub_host_path_tokens(text_value, context=context, source_file=source)
			stage_file.write_text(text_value, encoding="utf-8")
			processed_text.add(stage_file)

		for source in source_files:
			if source.resolve() in aliases or source.resolve() in omitted_paths:
				continue
			stage_file = stage_dir / source.relative_to(snapshot_dir)
			if stage_file in processed_text or source.suffix.lower() not in TEXT_SUFFIXES:
				continue
			if source.suffix.lower() in {".csv", ".json", ".jsonl", ".tsv"}:
				# Structured sanitizers scrub parsed values before their writers escape and
				# serialize them. A raw-text pass could remove JSON/CSV escape characters.
				continue
			try:
				text_value = stage_file.read_text(encoding="utf-8")
			except (OSError, UnicodeError) as error:
				raise BundleError(f"cannot sanitize shareable text {source}: {error}") from error
			stage_file.write_text(
				_scrub_host_path_tokens(text_value, context=context, source_file=source),
				encoding="utf-8",
			)

		if context.artifacts:
			json_content, markdown_content, html_content = _artifact_pages(context)
			(stage_dir / "portable_artifacts.json").write_bytes(json_content)
			(stage_dir / "portable_artifacts.md").write_bytes(markdown_content)
			(stage_dir / "portable_artifacts.html").write_bytes(html_content)

		if canonical_receipt is not None:
			assert canonical_receipt_file_sha256 is not None
			portable_receipt = _portable_receipt_payload(
				stage_dir, canonical_receipt_path,
				canonical_receipt_file_sha256, canonical_receipt,
			)
			(stage_dir / FINALIZER_RECEIPT_FILE).write_bytes(
				_canonical_json(portable_receipt)
			)

		staging_metadata = {
			"copied_external_assets": context.copied_external_assets,
			"deduplicated_preview_bytes": deduplicated_preview_bytes,
			"deduplicated_preview_files": len(aliases),
			"omitted_artifacts": len(context.artifacts),
			"omitted_parquet_files": len(omitted_paths),
			"schema_name": STAGING_SCHEMA_NAME,
			"schema_version": STAGING_SCHEMA_VERSION,
			"source_closure_status": source_closure.status,
			"source_closure_files_sha256": source_closure.files_sha256,
			"source_file_count": len(source_files),
		}
		(stage_dir / "portable_staging.json").write_bytes(_canonical_json(staging_metadata))
		integrity.write_report_tree_closure(stage_dir)
		validation = validate_staged_review_tree(stage_dir)
		return ReviewStageResult(
			stage_path=stage_dir,
			file_count=validation.file_count,
			payload_bytes=validation.payload_bytes,
			copied_external_assets=context.copied_external_assets,
			omitted_artifacts=len(context.artifacts),
			deduplicated_preview_files=len(aliases),
			deduplicated_preview_bytes=deduplicated_preview_bytes,
		)
	except Exception:
		shutil.rmtree(stage_dir, ignore_errors=True)
		raise
	finally:
		if snapshot_container is not None:
			shutil.rmtree(snapshot_container, ignore_errors=True)


def _iter_tree_files(root: Path, archive_root: str, role: str) -> Iterable[_SourceFile]:
	if not root.is_dir():
		raise BundleError(f"report directory does not exist: {root}")
	for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
		if path.is_symlink():
			raise BundleError(f"symlinks are not portable bundle inputs: {path}")
		status = path.stat()
		if stat.S_ISDIR(status.st_mode):
			continue
		if not stat.S_ISREG(status.st_mode):
			raise BundleError(f"special files are not portable bundle inputs: {path}")
		relative = path.relative_to(root).as_posix()
		archive_path = _validate_archive_path(f"{archive_root}/{relative}")
		yield _SourceFile(
			source=path,
			archive_path=archive_path,
			role=role,
			size_bytes=status.st_size,
			sha256=_sha256_path(path),
			mode=0o755 if status.st_mode & 0o111 else 0o644,
		)


def _supplemental_file(path: Path, archive_root: str, role: str) -> _SourceFile:
	if path.is_symlink():
		raise BundleError(f"symlinks are not portable bundle inputs: {path}")
	if not path.is_file():
		raise BundleError(f"metadata input is not a regular file: {path}")
	status = path.stat()
	content: bytes | None = None
	if path.suffix.lower() in TEXT_SUFFIXES:
		try:
			original = path.read_text(encoding="utf-8")
		except (OSError, UnicodeError) as error:
			raise BundleError(f"cannot read shareable metadata input {path}: {error}") from error
		sanitized = _sanitize_detached_text(original)
		if sanitized != original:
			content = sanitized.encode("utf-8")
	return _SourceFile(
		source=None if content is not None else path,
		archive_path=_validate_archive_path(f"{archive_root}/{path.name}"),
		role=role,
		size_bytes=len(content) if content is not None else status.st_size,
		sha256=_sha256_bytes(content) if content is not None else _sha256_path(path),
		mode=0o644,
		content=content,
	)


def _generated_file(archive_path: str, role: str, content: bytes) -> _SourceFile:
	return _SourceFile(
		source=None,
		archive_path=_validate_archive_path(archive_path),
		role=role,
		size_bytes=len(content),
		sha256=_sha256_bytes(content),
		mode=0o644,
		content=content,
	)


def _validate_raw_artifact(value: RawArtifact | Mapping[str, Any]) -> RawArtifact:
	artifact = value if isinstance(value, RawArtifact) else RawArtifact(
		artifact_id=str(value.get("artifact_id", "")),
		sha256=str(value.get("sha256", "")).lower(),
		size_bytes=int(value.get("size_bytes", -1)),
		uri=str(value["uri"]) if value.get("uri") is not None else None,
		media_type=str(value["media_type"]) if value.get("media_type") is not None else None,
	)
	if not ARTIFACT_ID_RE.fullmatch(artifact.artifact_id):
		raise BundleError(f"invalid raw artifact id: {artifact.artifact_id!r}")
	if not SHA256_RE.fullmatch(artifact.sha256):
		raise BundleError(f"invalid SHA-256 for raw artifact {artifact.artifact_id!r}")
	if artifact.size_bytes < 0:
		raise BundleError(f"negative size for raw artifact {artifact.artifact_id!r}")
	if artifact.uri:
		try:
			parsed = urlsplit(artifact.uri)
		except ValueError as error:
			raise BundleError(
				f"raw artifact {artifact.artifact_id!r} has an invalid URI"
			) from error
		if parsed.scheme.lower() == "file":
			raise BundleError(f"raw artifact {artifact.artifact_id!r} uses a local file URI")
		if parsed.username is not None or parsed.password is not None:
			raise BundleError(f"raw artifact {artifact.artifact_id!r} URI contains userinfo")
		if not parsed.scheme:
			_validate_archive_path(artifact.uri)
	return artifact


def _raw_artifacts_json(values: Sequence[RawArtifact | Mapping[str, Any]]) -> tuple[bytes | None, int]:
	artifacts = [_validate_raw_artifact(value) for value in values]
	identifiers = [artifact.artifact_id for artifact in artifacts]
	if len(identifiers) != len(set(identifiers)):
		raise BundleError("raw artifact ids must be unique")
	if not artifacts:
		return None, 0
	rows = [
		{
			"artifact_id": artifact.artifact_id,
			"media_type": artifact.media_type,
			"sha256": artifact.sha256,
			"size_bytes": artifact.size_bytes,
			"uri": artifact.uri,
		}
		for artifact in sorted(artifacts, key=lambda item: item.artifact_id)
	]
	return _canonical_json({"artifacts": rows, "schema_version": 1}), len(rows)


def _scan_sensitive_window(window: bytes, path: str) -> None:
	for marker in HOST_REFERENCE_MARKERS:
		if marker in window:
			raise BundleError(
				f"host-specific reference {marker.decode('utf-8', errors='replace')!r} "
				f"in shareable text member {path!r}"
			)
	match = HOST_REFERENCE_BYTES_RE.search(window)
	if match is not None:
		raise BundleError(
			f"host-specific reference {match.group(0).decode('utf-8', errors='replace')!r} "
			f"in shareable text member {path!r}"
		)
	for label, pattern in SENSITIVE_CONTENT_PATTERNS:
		if pattern.search(window) is not None:
			# Never echo the matched credential into a validation log.
			raise BundleError(f"possible {label} in shareable text member {path!r}")


def _scan_portable_stream(stream: BinaryIO, path: str) -> None:
	overlap = b""
	while True:
		block = stream.read(1024 * 1024)
		if not block:
			break
		window = overlap + block
		_scan_sensitive_window(window, path)
		overlap = window[-SENSITIVE_SCAN_OVERLAP_BYTES:]


def _is_float_payload_script(path: str) -> bool:
	parts = PurePosixPath(path).parts
	return (
		len(parts) >= 3
		and parts[-3:-1] == ("interactive", "maps")
		and parts[-1].endswith("_float32.js")
	)


def _float_payload_control_bytes(content: bytes, path: str) -> bytes:
	"""Validate generated pixel data and return only its auditable JS control plane."""

	prefix = (
		"window.__DMAP_PIXEL_PAYLOADS=window.__DMAP_PIXEL_PAYLOADS||{};"
		"window.__DMAP_PIXEL_PAYLOADS["
	)
	try:
		text_value = content.decode("ascii")
	except UnicodeDecodeError as error:
		raise BundleError(f"float payload script is not ASCII: {path}") from error
	if not text_value.startswith(prefix):
		raise BundleError(f"float payload script has an invalid preamble: {path}")
	assignment = text_value.find("]=", len(prefix))
	if assignment < 0 or not text_value.rstrip().endswith(";"):
		raise BundleError(f"float payload script has an invalid assignment: {path}")
	try:
		payload_id = json.loads(text_value[len(prefix):assignment])
		payload = json.loads(text_value[assignment + 2:].rstrip()[:-1])
	except (TypeError, ValueError, json.JSONDecodeError) as error:
		raise BundleError(f"float payload script is malformed: {path}") from error
	expected_id = PurePosixPath(path).name.removesuffix("_float32.js")
	if payload_id != expected_id or not isinstance(payload, dict):
		raise BundleError(f"float payload script identity is invalid: {path}")
	if set(payload) != {"data", "width", "height", "channels"}:
		raise BundleError(f"float payload script fields are invalid: {path}")
	encoded = payload["data"]
	if (
		not isinstance(encoded, str)
		or not encoded.startswith("H4sI")
		or len(encoded) % 4
		or BASE64_RE.fullmatch(encoded) is None
	):
		raise BundleError(f"float payload script data is not canonical gzip/base64: {path}")
	dimensions = [payload["width"], payload["height"], payload["channels"]]
	if (
		any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions)
		or dimensions[2] > 16
		or dimensions[0] * dimensions[1] * dimensions[2] > 2**31
	):
		raise BundleError(f"float payload script dimensions are invalid: {path}")
	return json.dumps(
		{
			"payload_id": payload_id,
			"encoding": "opaque_gzip_base64_float32",
			"width": dimensions[0],
			"height": dimensions[1],
			"channels": dimensions[2],
		},
		sort_keys=True,
		separators=(",", ":"),
	).encode("ascii")


def _scan_shareable_content(content: bytes, path: str) -> None:
	if _is_float_payload_script(path):
		content = _float_payload_control_bytes(content, path)
	_scan_portable_stream(io.BytesIO(content), path)


def _scan_source_portability(source: _SourceFile) -> None:
	if PurePosixPath(source.archive_path).suffix.lower() not in TEXT_SUFFIXES:
		return
	if source.content is not None:
		_scan_shareable_content(source.content, source.archive_path)
	elif source.source is not None:
		if _is_float_payload_script(source.archive_path):
			_scan_shareable_content(source.source.read_bytes(), source.archive_path)
		else:
			with source.source.open("rb") as stream:
				_scan_portable_stream(stream, source.archive_path)


def _json_path_references(
	value: Any, location: str = "json", parent_key: str | None = None
) -> Iterable[tuple[str, bool, str]]:
	if isinstance(value, dict):
		for key, item in value.items():
			child = f"{location}.{key}"
			if (
				isinstance(item, str)
				and item
				and _is_path_field(key, item, parent_key)
			):
				yield item, key in {
					"investigation_html", "local", "markdown", "model", "path", "script_path",
					"shared", "static_html",
				}, child
			yield from _json_path_references(item, child, key)
	elif isinstance(value, list):
		for index, item in enumerate(value):
			yield from _json_path_references(item, f"{location}[{index}]", parent_key)


def _text_references(path: str, content: str) -> Iterable[tuple[str, bool, str]]:
	suffix = PurePosixPath(path).suffix.lower()
	if suffix == ".md":
		for index, match in enumerate(MARKDOWN_LINK_RE.finditer(content)):
			yield match.group(3), bool(match.group(1)), f"markdown link {index}"
		image_labels = {
			" ".join((match.group("label") or match.group("alt")).split()).casefold()
			for match in MARKDOWN_IMAGE_REFERENCE_RE.finditer(content)
		}
		for index, match in enumerate(MARKDOWN_REFERENCE_DEFINITION_RE.finditer(content)):
			label = " ".join(match.group("label").split()).casefold()
			yield (
				match.group("target"), label in image_labels,
				f"markdown reference definition {index}",
			)
	if suffix in {".htm", ".html", ".svg"}:
		for index, match in enumerate(HTML_REFERENCE_RE.finditer(content)):
			tag = match.group("tag").lower()
			attribute = match.group("attribute").lower()
			required = attribute == "src" or tag in {"img", "link", "script", "source"}
			yield match.group("target"), required, f"HTML reference {index}"
	css_fragments = [content] if suffix in {".css", ".svg"} else [
		match.group("content") for match in STYLE_BLOCK_RE.finditer(content)
	] if suffix in {".htm", ".html"} else []
	for fragment_index, fragment in enumerate(css_fragments):
		for index, match in enumerate(CSS_URL_RE.finditer(fragment)):
			yield match.group("target"), True, f"CSS URL {fragment_index}:{index}"
	if suffix == ".json":
		try:
			value = json.loads(content)
		except json.JSONDecodeError as error:
			raise BundleError(f"invalid JSON in staged review member {path}: {error}") from error
		yield from _json_path_references(value)


def _validate_reference_value(
	raw_value: str,
	*,
	current_path: PurePosixPath,
	members: set[str],
	required_asset: bool,
	location: str,
) -> None:
	value = _reference_path(raw_value)
	if not value or value.startswith("#") or _is_dynamic_reference(value):
		return
	parsed = urlsplit(value)
	scheme = parsed.scheme.lower()
	if scheme in REMOTE_SCHEMES:
		if required_asset and scheme in {"http", "https"}:
			raise BundleError(f"remote asset is not self-contained in {current_path}: {value}")
		return
	if scheme or value.startswith("/") or "\\" in value:
		raise BundleError(f"non-portable reference in {current_path} ({location}): {value}")
	decoded = unquote(parsed.path)
	if not decoded:
		return
	joined = posixpath.normpath(posixpath.join(current_path.parent.as_posix(), decoded))
	if joined == ".." or joined.startswith("../") or joined.startswith("/"):
		raise BundleError(f"reference escapes review root in {current_path} ({location}): {value}")
	if joined not in members:
		raise BundleError(f"broken portable reference in {current_path} ({location}): {value}")


def _validate_virtual_review(
	text_members: Mapping[str, bytes],
	members: set[str],
) -> int:
	reference_count = 0
	for raw_path, content in text_members.items():
		try:
			text_value = content.decode("utf-8")
		except UnicodeDecodeError as error:
			raise BundleError(f"shareable text member is not UTF-8: {raw_path}") from error
		for target, required, location in _text_references(raw_path, text_value):
			_validate_reference_value(
				target,
				current_path=PurePosixPath(raw_path),
				members=members,
				required_asset=required,
				location=location,
			)
			reference_count += 1
	return reference_count


def validate_staged_review_tree(review_dir: Path) -> ReviewTreeValidationResult:
	"""Validate a staged tree using only files contained by that tree."""
	review_dir = Path(review_dir).resolve()
	if not review_dir.is_dir():
		raise BundleError(f"staged review directory does not exist: {review_dir}")
	members: set[str] = set()
	text_members: dict[str, bytes] = {}
	payload_bytes = 0
	for path in sorted(review_dir.rglob("*"), key=lambda item: item.relative_to(review_dir).as_posix()):
		if path.is_symlink():
			raise BundleError(f"symlinks are not allowed in staged reviews: {path}")
		status = path.stat()
		if stat.S_ISDIR(status.st_mode):
			continue
		if not stat.S_ISREG(status.st_mode):
			raise BundleError(f"special files are not allowed in staged reviews: {path}")
		relative = path.relative_to(review_dir).as_posix()
		_validate_archive_path(f"review/{relative}")
		members.add(relative)
		payload_bytes += status.st_size
		if path.suffix.lower() in TEXT_SUFFIXES:
			if _is_float_payload_script(relative):
				_scan_shareable_content(path.read_bytes(), relative)
			else:
				with path.open("rb") as stream:
					_scan_portable_stream(stream, relative)
		if (
			relative != integrity.REPORT_CLOSURE_FILE
			and path.suffix.lower() in REFERENCE_AUDIT_SUFFIXES
		):
			if status.st_size > 64 * 1024 * 1024:
				raise BundleError(f"reference-bearing review member is too large to audit: {relative}")
			text_members[relative] = path.read_bytes()
	if not members:
		raise BundleError(f"staged review directory is empty: {review_dir}")
	metadata_path = review_dir / "portable_staging.json"
	if metadata_path.is_file():
		try:
			metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
		except (OSError, UnicodeError, json.JSONDecodeError) as error:
			raise BundleError("portable_staging.json is invalid") from error
		if (
			metadata.get("schema_name") != STAGING_SCHEMA_NAME
			or metadata.get("schema_version") != STAGING_SCHEMA_VERSION
		):
			raise BundleError("portable_staging.json has an unsupported schema")
	_validate_staged_portable_receipt(review_dir)
	reference_count = _validate_virtual_review(text_members, members)
	try:
		closure = integrity.require_report_tree_closure(
			review_dir, allow_legacy=True,
		)
	except integrity.IntegrityError as error:
		raise BundleError(f"staged report artifact closure is invalid: {error}") from error
	return ReviewTreeValidationResult(
		review_path=review_dir,
		file_count=len(members),
		payload_bytes=payload_bytes,
		reference_count=reference_count,
		closure_status=closure.status,
	)


def _tar_info(source: _SourceFile, source_date_epoch: int) -> tarfile.TarInfo:
	info = tarfile.TarInfo(source.archive_path)
	info.size = source.size_bytes
	info.mode = source.mode
	info.mtime = source_date_epoch
	info.uid = 0
	info.gid = 0
	info.uname = ""
	info.gname = ""
	return info


def _write_archive(path: Path, sources: Sequence[_SourceFile], source_date_epoch: int) -> None:
	if zstandard is None:
		raise BundleError(
			"zstandard is required; install scripts/python/requirements-depth-benchmark.txt"
		)
	compressor = zstandard.ZstdCompressor(
		level=19,
		threads=0,
		write_checksum=True,
		write_content_size=False,
	)
	with path.open("wb") as raw_stream:
		with compressor.stream_writer(raw_stream, closefd=False) as compressed_stream:
			with tarfile.open(fileobj=compressed_stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
				for source in sources:
					info = _tar_info(source, source_date_epoch)
					if source.content is not None:
						archive.addfile(info, io.BytesIO(source.content))
					elif source.source is not None:
						with source.source.open("rb") as input_stream:
							archive.addfile(info, input_stream)


def _inventory_content(sources: Sequence[_SourceFile]) -> bytes:
	lines = [f"{source.sha256}  {source.archive_path}\n" for source in sources]
	return "".join(lines).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
	fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
	temp = Path(raw_temp)
	try:
		with os.fdopen(fd, "wb") as stream:
			stream.write(content)
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temp, path)
	finally:
		if temp.exists():
			temp.unlink()


def create_review_bundle(
	report_dir: Path,
	output_path: Path,
	*,
	config_files: Sequence[Path] = (),
	provenance_files: Sequence[Path] = (),
	provenance: Mapping[str, Any] | None = None,
	raw_artifacts: Sequence[RawArtifact | Mapping[str, Any]] = (),
	max_review_bytes: int = DEFAULT_REVIEW_LIMIT_BYTES,
	max_payload_bytes: int = DEFAULT_PAYLOAD_LIMIT_BYTES,
	source_date_epoch: int = 0,
	overwrite: bool = False,
	check_portable_references: bool = True,
	stage_report: bool = True,
) -> BundleBuildResult:
	"""Stage, validate, and atomically publish a deterministic ``tar.zst`` bundle."""
	report_dir = Path(report_dir)
	output_path = Path(output_path)
	if not stage_report:
		return _create_review_bundle_from_tree(
			report_dir,
			output_path,
			config_files=config_files,
			provenance_files=provenance_files,
			provenance=provenance,
			raw_artifacts=raw_artifacts,
			max_review_bytes=max_review_bytes,
			max_payload_bytes=max_payload_bytes,
			source_date_epoch=source_date_epoch,
			overwrite=overwrite,
			check_portable_references=check_portable_references,
		)
	checksum_path = output_path.with_name(output_path.name + ".sha256")
	if _archive_is_within(output_path, report_dir):
		raise BundleError("bundle output must be outside the report directory")
	if not overwrite and (output_path.exists() or checksum_path.exists()):
		raise BundleError(f"refusing to overwrite an existing bundle: {output_path}")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.TemporaryDirectory(
		prefix=f".{output_path.name}.stage.", dir=output_path.parent
	) as temp_directory:
		staged = stage_review_tree(report_dir, Path(temp_directory) / "review")
		return _create_review_bundle_from_tree(
			staged.stage_path,
			output_path,
			config_files=config_files,
			provenance_files=provenance_files,
			provenance=provenance,
			raw_artifacts=raw_artifacts,
			max_review_bytes=max_review_bytes,
			max_payload_bytes=max_payload_bytes,
			source_date_epoch=source_date_epoch,
			overwrite=overwrite,
			check_portable_references=check_portable_references,
		)


def _create_review_bundle_from_tree(
	report_dir: Path,
	output_path: Path,
	*,
	config_files: Sequence[Path] = (),
	provenance_files: Sequence[Path] = (),
	provenance: Mapping[str, Any] | None = None,
	raw_artifacts: Sequence[RawArtifact | Mapping[str, Any]] = (),
	max_review_bytes: int = DEFAULT_REVIEW_LIMIT_BYTES,
	max_payload_bytes: int = DEFAULT_PAYLOAD_LIMIT_BYTES,
	source_date_epoch: int = 0,
	overwrite: bool = False,
	check_portable_references: bool = True,
) -> BundleBuildResult:
	"""Create a bundle from an already portable report tree."""
	report_dir = Path(report_dir)
	output_path = Path(output_path)
	checksum_path = output_path.with_name(output_path.name + ".sha256")
	if max_review_bytes <= 0 or max_payload_bytes <= 0:
		raise BundleError("bundle and payload limits must be positive")
	if source_date_epoch < 0:
		raise BundleError("source_date_epoch must be non-negative")
	if _archive_is_within(output_path, report_dir):
		raise BundleError("bundle output must be outside the report directory")
	if not overwrite and (output_path.exists() or checksum_path.exists()):
		raise BundleError(f"refusing to overwrite an existing bundle: {output_path}")
	validate_staged_review_tree(report_dir)

	sources = list(_iter_tree_files(report_dir, "review", "review"))
	if not sources:
		raise BundleError(f"report directory is empty: {report_dir}")
	sources.extend(_supplemental_file(Path(path), "metadata/config", "config") for path in config_files)
	sources.extend(
		_supplemental_file(Path(path), "metadata/provenance", "provenance")
		for path in provenance_files
	)
	if provenance is not None:
		provenance_content = _canonical_json(dict(provenance))
		provenance_content = _sanitize_detached_text(
			provenance_content.decode("utf-8")
		).encode("utf-8")
		sources.append(_generated_file(
			"metadata/provenance.json", "provenance", provenance_content
		))
	raw_content, raw_artifact_count = _raw_artifacts_json(raw_artifacts)
	if raw_content is not None:
		sources.append(_generated_file("raw_artifacts.json", "raw_manifest", raw_content))

	archive_paths = [source.archive_path for source in sources]
	if len(archive_paths) != len(set(archive_paths)):
		duplicates = sorted(path for path in set(archive_paths) if archive_paths.count(path) > 1)
		raise BundleError(f"duplicate bundle paths: {', '.join(duplicates)}")
	payload_bytes = sum(source.size_bytes for source in sources)
	if payload_bytes > max_payload_bytes:
		raise BundleError(
			f"bundle payload requires {payload_bytes} bytes, exceeding {max_payload_bytes} bytes"
		)
	if check_portable_references:
		for source in sources:
			_scan_source_portability(source)
	validate_staged_review_tree(report_dir)

	bundle_metadata = {
		"archive_format": "tar+zstd",
		"bundle_type": "openmvs.dmap_observability.review",
		"file_count": len(sources),
		"max_review_bytes": max_review_bytes,
		"payload_bytes": payload_bytes,
		"payload_root": "review",
		"raw_artifact_count": raw_artifact_count,
		"schema_version": BUNDLE_SCHEMA_VERSION,
		"source_date_epoch": source_date_epoch,
	}
	sources.append(_generated_file("bundle.json", "bundle_metadata", _canonical_json(bundle_metadata)))
	sources.sort(key=lambda source: source.archive_path)
	inventory = _generated_file("inventory.sha256", "inventory", _inventory_content(sources))
	archive_sources = [*sources, inventory]
	archive_sources.sort(key=lambda source: source.archive_path)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fd, raw_temp = tempfile.mkstemp(
		prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
	)
	os.close(fd)
	temp_path = Path(raw_temp)
	try:
		_write_archive(temp_path, archive_sources, source_date_epoch)
		archive_bytes = temp_path.stat().st_size
		if archive_bytes > max_review_bytes:
			raise BundleError(
				f"compressed review bundle requires {archive_bytes} bytes, "
				f"exceeding {max_review_bytes} bytes"
			)
		validation = validate_review_bundle(
			temp_path,
			verify_checksum=False,
			max_review_bytes=max_review_bytes,
			max_payload_bytes=max_payload_bytes,
			check_portable_references=check_portable_references,
		)
		archive_sha256 = validation.archive_sha256
		checksum_content = f"{archive_sha256}  {output_path.name}\n".encode("ascii")
		os.replace(temp_path, output_path)
		_atomic_write(checksum_path, checksum_content)
		return BundleBuildResult(
			archive_path=output_path,
			checksum_path=checksum_path,
			archive_sha256=archive_sha256,
			archive_bytes=archive_bytes,
			payload_bytes=payload_bytes,
			file_count=int(bundle_metadata["file_count"]),
		)
	finally:
		if temp_path.exists():
			temp_path.unlink()


def _parse_inventory(content: bytes) -> dict[str, str]:
	try:
		text = content.decode("utf-8")
	except UnicodeDecodeError as error:
		raise BundleError("inventory.sha256 is not UTF-8") from error
	entries: dict[str, str] = {}
	for line_number, line in enumerate(text.splitlines(), start=1):
		if not line:
			continue
		if len(line) < 67 or line[64:66] != "  ":
			raise BundleError(f"malformed inventory line {line_number}")
		digest, path = line[:64], line[66:]
		if not SHA256_RE.fullmatch(digest):
			raise BundleError(f"invalid inventory digest on line {line_number}")
		_validate_archive_path(path)
		if path in entries:
			raise BundleError(f"duplicate inventory path: {path}")
		entries[path] = digest
	if list(entries) != sorted(entries):
		raise BundleError("inventory paths are not sorted")
	return entries


def _read_checksum(path: Path, archive_name: str) -> str:
	try:
		line = path.read_text(encoding="ascii").strip()
	except (OSError, UnicodeError) as error:
		raise BundleError(f"cannot read bundle checksum: {path}") from error
	if len(line) < 67 or line[64:66] != "  ":
		raise BundleError(f"malformed bundle checksum: {path}")
	digest, name = line[:64], line[66:]
	if not SHA256_RE.fullmatch(digest) or name != archive_name:
		raise BundleError(f"bundle checksum does not identify {archive_name!r}")
	return digest


def _stream_member(
	stream: BinaryIO,
	*,
	path: str,
	check_portable_references: bool,
) -> tuple[str, bytes | None]:
	digest = hashlib.sha256()
	capture_reference_member = (
		path.startswith("review/")
		and PurePosixPath(path).suffix.lower() in REFERENCE_AUDIT_SUFFIXES
	)
	float_payload_member = _is_float_payload_script(path)
	captured = bytearray() if (
		path in {"bundle.json", "inventory.sha256", "raw_artifacts.json"}
		or capture_reference_member
		or float_payload_member
	) else None
	overlap = b""
	text_member = PurePosixPath(path).suffix.lower() in TEXT_SUFFIXES
	for block in iter(lambda: stream.read(1024 * 1024), b""):
		digest.update(block)
		if captured is not None:
			captured.extend(block)
			limit = (
				512 * 1024 * 1024 if float_payload_member
				else 64 * 1024 * 1024 if capture_reference_member
				else 16 * 1024 * 1024
			)
			if len(captured) > limit:
				raise BundleError(f"bundle reference/control member is too large: {path}")
			if check_portable_references and text_member and not float_payload_member:
				window = overlap + block
				_scan_sensitive_window(window, path)
				overlap = window[-SENSITIVE_SCAN_OVERLAP_BYTES:]
	if check_portable_references and float_payload_member:
		assert captured is not None
		_scan_shareable_content(bytes(captured), path)
	if float_payload_member and not capture_reference_member:
		return digest.hexdigest(), None
	return digest.hexdigest(), bytes(captured) if captured is not None else None


def _parse_raw_artifacts(content: bytes | None) -> int:
	if content is None:
		return 0
	try:
		value = json.loads(content)
	except (UnicodeDecodeError, json.JSONDecodeError) as error:
		raise BundleError("raw_artifacts.json is invalid") from error
	if value.get("schema_version") != 1 or not isinstance(value.get("artifacts"), list):
		raise BundleError("raw_artifacts.json has an unsupported schema")
	artifacts = [_validate_raw_artifact(item) for item in value["artifacts"]]
	identifiers = [artifact.artifact_id for artifact in artifacts]
	if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
		raise BundleError("raw artifacts must be uniquely sorted by artifact_id")
	return len(artifacts)


def _validate_virtual_portable_finalizer_receipt(
	controls: Mapping[str, bytes], observed: Mapping[str, str],
	member_sizes: Mapping[str, int], member_modes: Mapping[str, int],
) -> None:
	inventory_member = "review/report_inventory.json"
	inventory_content = controls.get(inventory_member)
	inventory: dict[str, Any] | None = None
	if inventory_content is not None:
		try:
			parsed_inventory = json.loads(inventory_content)
		except (UnicodeDecodeError, json.JSONDecodeError) as error:
			raise BundleError("portable report inventory in bundle is unreadable") from error
		if not isinstance(parsed_inventory, dict):
			raise BundleError("portable report inventory in bundle must be a JSON object")
		inventory = parsed_inventory
	receipt_member = f"review/{FINALIZER_RECEIPT_FILE}"
	content = controls.get(receipt_member)
	if content is None:
		if inventory is not None and FINALIZER_INVENTORY_KEY in inventory:
			raise BundleError(
				"portable report inventory declares finalizer artifacts but the receipt is missing"
			)
		return
	try:
		value = json.loads(content)
	except (UnicodeDecodeError, json.JSONDecodeError) as error:
		raise BundleError("portable finalizer receipt in bundle is unreadable") from error
	outputs = _validate_portable_receipt_value(value)
	for output in outputs:
		member = f"review/{output['path']}"
		actual = {
			"sha256": observed.get(member), "bytes": member_sizes.get(member),
			"mode": member_modes.get(member),
		}
		expected = {key: output[key] for key in ("sha256", "bytes", "mode")}
		if actual != expected:
			raise BundleError(
				f"portable finalizer output drifted in bundle: {output['path']}: "
				f"expected={expected}, actual={actual}"
			)
	if observed.get(inventory_member) != value["report_inventory_sha256"]:
		raise BundleError("portable finalizer inventory digest differs in bundle")
	if inventory is None:
		raise BundleError("portable finalizer inventory is unavailable in bundle")
	expected = {
		"schema_name": FINALIZER_INVENTORY_SCHEMA_NAME,
		"schema_version": 1,
		"finalizer_receipt": FINALIZER_RECEIPT_FILE,
		"approved_outputs": outputs,
	}
	if inventory.get(FINALIZER_INVENTORY_KEY) != expected:
		raise BundleError("portable report inventory omits finalizer artifacts")


def validate_review_bundle(
	archive_path: Path,
	*,
	verify_checksum: bool = True,
	max_review_bytes: int = DEFAULT_REVIEW_LIMIT_BYTES,
	max_payload_bytes: int = DEFAULT_PAYLOAD_LIMIT_BYTES,
	max_members: int = DEFAULT_MAX_MEMBERS,
	check_portable_references: bool = True,
) -> BundleValidationResult:
	"""Validate archive safety, inventory closure, metadata, size, and checksum."""
	archive_path = Path(archive_path)
	if zstandard is None:
		raise BundleError(
			"zstandard is required; install scripts/python/requirements-depth-benchmark.txt"
		)
	if not archive_path.is_file():
		raise BundleError(f"bundle does not exist: {archive_path}")
	archive_bytes = archive_path.stat().st_size
	if archive_bytes > max_review_bytes:
		raise BundleError(
			f"compressed review bundle requires {archive_bytes} bytes, exceeding {max_review_bytes} bytes"
		)
	archive_sha256 = _sha256_path(archive_path)
	if verify_checksum:
		checksum_path = archive_path.with_name(archive_path.name + ".sha256")
		expected = _read_checksum(checksum_path, archive_path.name)
		if expected != archive_sha256:
			raise BundleError("compressed bundle SHA-256 does not match its checksum file")

	observed: dict[str, str] = {}
	member_sizes: dict[str, int] = {}
	member_modes: dict[str, int] = {}
	controls: dict[str, bytes] = {}
	total_bytes = 0
	member_count = 0
	decompressor = zstandard.ZstdDecompressor()
	try:
		with archive_path.open("rb") as raw_stream:
			with decompressor.stream_reader(raw_stream, closefd=False) as decompressed_stream:
				with tarfile.open(fileobj=decompressed_stream, mode="r|") as archive:
					for member in archive:
						member_count += 1
						if member_count > max_members:
							raise BundleError(f"bundle has more than {max_members} members")
						path = _validate_archive_path(member.name)
						if path in observed or path in member_sizes:
							raise BundleError(f"duplicate archive member: {path}")
						if not member.isreg():
							raise BundleError(f"only regular files are allowed in review bundles: {path}")
						if not (
							path.startswith("review/")
							or path.startswith("metadata/")
							or path in {"bundle.json", "inventory.sha256", "raw_artifacts.json"}
						):
							raise BundleError(f"unexpected top-level bundle member: {path}")
						total_bytes += member.size
						if total_bytes > max_payload_bytes + 32 * 1024 * 1024:
							raise BundleError(
								f"bundle payload exceeds the {max_payload_bytes}-byte validation limit"
							)
						member_stream = archive.extractfile(member)
						if member_stream is None:
							raise BundleError(f"cannot read archive member: {path}")
						digest, captured = _stream_member(
							member_stream,
							path=path,
							check_portable_references=check_portable_references,
						)
						observed[path] = digest
						member_sizes[path] = member.size
						member_modes[path] = stat.S_IMODE(member.mode)
						if captured is not None:
							controls[path] = captured
	except (zstandard.ZstdError, tarfile.TarError, EOFError) as error:
		raise BundleError(f"cannot decode review bundle: {error}") from error

	if "inventory.sha256" not in controls or "bundle.json" not in controls:
		raise BundleError("bundle is missing inventory.sha256 or bundle.json")
	inventory = _parse_inventory(controls["inventory.sha256"])
	observed_payload = {path: digest for path, digest in observed.items() if path != "inventory.sha256"}
	if inventory != observed_payload:
		missing = sorted(set(inventory) - set(observed_payload))
		unexpected = sorted(set(observed_payload) - set(inventory))
		mismatched = sorted(
			path for path in set(inventory) & set(observed_payload)
			if inventory[path] != observed_payload[path]
		)
		raise BundleError(
			"bundle inventory mismatch: "
			f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
		)
	try:
		metadata = json.loads(controls["bundle.json"])
	except (UnicodeDecodeError, json.JSONDecodeError) as error:
		raise BundleError("bundle.json is invalid") from error
	if metadata.get("schema_version") != BUNDLE_SCHEMA_VERSION:
		raise BundleError("bundle.json has an unsupported schema")
	if metadata.get("bundle_type") != "openmvs.dmap_observability.review":
		raise BundleError("bundle.json has an unsupported bundle type")
	payload_file_count = len(observed_payload) - 1  # bundle.json is generated metadata
	payload_bytes = sum(
		size for path, size in member_sizes.items()
		if path not in {"bundle.json", "inventory.sha256"}
	)
	if metadata.get("file_count") != payload_file_count:
		raise BundleError("bundle.json file_count does not match archive contents")
	if metadata.get("payload_bytes") != payload_bytes:
		raise BundleError("bundle.json payload_bytes does not match archive contents")
	if payload_bytes > max_payload_bytes:
		raise BundleError(
			f"bundle payload requires {payload_bytes} bytes, exceeding {max_payload_bytes} bytes"
		)
	if not any(path.startswith("review/") for path in observed_payload):
		raise BundleError("bundle contains no review files")
	raw_artifact_count = _parse_raw_artifacts(controls.get("raw_artifacts.json"))
	if metadata.get("raw_artifact_count") != raw_artifact_count:
		raise BundleError("bundle.json raw_artifact_count does not match raw_artifacts.json")
	_validate_virtual_portable_finalizer_receipt(
		controls, observed, member_sizes, member_modes,
	)
	if check_portable_references:
		review_text = {
			path: content for path, content in controls.items()
			if (
				path.startswith("review/")
				and path != f"review/{integrity.REPORT_CLOSURE_FILE}"
			)
		}
		_validate_virtual_review(review_text, set(observed_payload))
	return BundleValidationResult(
		archive_path=archive_path,
		archive_sha256=archive_sha256,
		archive_bytes=archive_bytes,
		payload_bytes=payload_bytes,
		file_count=payload_file_count,
		review_file_count=sum(path.startswith("review/") for path in observed_payload),
		raw_artifact_count=raw_artifact_count,
	)


def _load_raw_manifest(path: Path | None) -> list[Mapping[str, Any]]:
	if path is None:
		return []
	try:
		value = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, UnicodeError, json.JSONDecodeError) as error:
		raise BundleError(f"cannot read raw artifact manifest: {path}") from error
	rows = value.get("artifacts") if isinstance(value, dict) else value
	if not isinstance(rows, list):
		raise BundleError("raw artifact manifest must be a list or an object with an artifacts list")
	return rows


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
	subparsers = parser.add_subparsers(dest="command", required=True)
	stage = subparsers.add_parser(
		"stage", help="create a sanitized self-contained review tree", allow_abbrev=False
	)
	stage.add_argument("--report-dir", type=Path, required=True)
	stage.add_argument("--output-dir", type=Path, required=True)
	stage.add_argument("--overwrite", action="store_true")
	package = subparsers.add_parser(
		"package", help="create a deterministic portable review bundle", allow_abbrev=False
	)
	package.add_argument("--report-dir", type=Path, required=True)
	package.add_argument("--output", type=Path, required=True)
	package.add_argument("--config", type=Path, action="append", default=[])
	package.add_argument("--provenance-file", type=Path, action="append", default=[])
	package.add_argument("--raw-manifest", type=Path)
	package.add_argument("--max-review-bytes", type=int, default=DEFAULT_REVIEW_LIMIT_BYTES)
	package.add_argument("--max-payload-bytes", type=int, default=DEFAULT_PAYLOAD_LIMIT_BYTES)
	package.add_argument("--source-date-epoch", type=int, default=0)
	package.add_argument("--overwrite", action="store_true")
	package.add_argument("--allow-host-references", action="store_true")
	package.add_argument("--no-stage-report", action="store_true")
	validate = subparsers.add_parser(
		"validate", help="validate a portable review bundle", allow_abbrev=False
	)
	validate.add_argument("--bundle", type=Path, required=True)
	validate.add_argument("--max-review-bytes", type=int, default=DEFAULT_REVIEW_LIMIT_BYTES)
	validate.add_argument("--max-payload-bytes", type=int, default=DEFAULT_PAYLOAD_LIMIT_BYTES)
	validate.add_argument("--no-checksum", action="store_true")
	validate.add_argument("--allow-host-references", action="store_true")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)
	try:
		if args.command == "stage":
			result = stage_review_tree(args.report_dir, args.output_dir, overwrite=args.overwrite)
			print(json.dumps({
				"copied_external_assets": result.copied_external_assets,
				"deduplicated_preview_bytes": result.deduplicated_preview_bytes,
				"deduplicated_preview_files": result.deduplicated_preview_files,
				"file_count": result.file_count,
				"omitted_artifacts": result.omitted_artifacts,
				"payload_bytes": result.payload_bytes,
				"stage": str(result.stage_path),
			}, sort_keys=True))
		elif args.command == "package":
			result = create_review_bundle(
				args.report_dir,
				args.output,
				config_files=args.config,
				provenance_files=args.provenance_file,
				raw_artifacts=_load_raw_manifest(args.raw_manifest),
				max_review_bytes=args.max_review_bytes,
				max_payload_bytes=args.max_payload_bytes,
				source_date_epoch=args.source_date_epoch,
				overwrite=args.overwrite,
				check_portable_references=not args.allow_host_references,
				stage_report=not args.no_stage_report,
			)
			print(json.dumps({
				"archive": str(result.archive_path),
				"archive_bytes": result.archive_bytes,
				"archive_sha256": result.archive_sha256,
				"checksum": str(result.checksum_path),
				"file_count": result.file_count,
				"payload_bytes": result.payload_bytes,
			}, sort_keys=True))
		else:
			result = validate_review_bundle(
				args.bundle,
				verify_checksum=not args.no_checksum,
				max_review_bytes=args.max_review_bytes,
				max_payload_bytes=args.max_payload_bytes,
				check_portable_references=not args.allow_host_references,
			)
			print(json.dumps({
				"archive": str(result.archive_path),
				"archive_bytes": result.archive_bytes,
				"archive_sha256": result.archive_sha256,
				"file_count": result.file_count,
				"payload_bytes": result.payload_bytes,
				"raw_artifact_count": result.raw_artifact_count,
				"review_file_count": result.review_file_count,
			}, sort_keys=True))
	except BundleError as error:
		print(f"error: {error}", file=os.sys.stderr)
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
