"""Scoped predecessor validation reuse for attested campaign finalizers."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import os
from pathlib import Path
import stat
from types import ModuleType
from typing import Any, Callable, Iterator, Sequence


class FinalizationContextError(RuntimeError):
	"""Raised when a finalization context cannot fail closed."""


@dataclass(frozen=True)
class LockRequest:
	"""One lock in the global campaign finalization order."""

	order: int
	label: str
	path: Path
	shared: bool
	create: bool = False


@dataclass
class OrderedFileLocks:
	"""Acquire non-blocking flock locks without changing existing lock contents."""

	requests: Sequence[LockRequest]
	_handles: list[Any] = field(default_factory=list, init=False, repr=False)
	acquired_labels: list[str] = field(default_factory=list, init=False)
	_held: bool = field(default=False, init=False, repr=False)

	def __enter__(self) -> "OrderedFileLocks":
		if self._held or self._handles:
			raise FinalizationContextError("finalization locks cannot be re-entered")
		self.acquired_labels.clear()
		ordered = sorted(self.requests, key=lambda request: request.order)
		if len({request.order for request in ordered}) != len(ordered):
			raise FinalizationContextError("finalization lock orders must be unique")
		resolved_paths: set[Path] = set()
		try:
			for request in ordered:
				path = request.path.expanduser()
				if request.create:
					path.parent.mkdir(parents=True, exist_ok=True)
					if path.is_symlink():
						raise FinalizationContextError(
							f"{request.label} lock is a symlink: {path}"
						)
					mode = "a+b"
				else:
					if not path.is_file() or path.is_symlink():
						raise FinalizationContextError(
							f"{request.label} lock is missing or not a regular file: {path}"
						)
					mode = "rb"
				handle = path.open(mode)
				opened = os.fstat(handle.fileno())
				current = path.stat()
				if (
					not stat.S_ISREG(opened.st_mode)
					or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
				):
					handle.close()
					raise FinalizationContextError(
						f"{request.label} lock changed while it was opened: {path}"
					)
				resolved = path.resolve(strict=True)
				if resolved in resolved_paths:
					handle.close()
					raise FinalizationContextError(
						f"duplicate finalization lock path: {resolved}"
					)
				resolved_paths.add(resolved)
				operation = fcntl.LOCK_SH if request.shared else fcntl.LOCK_EX
				try:
					fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
				except BlockingIOError as exc:
					handle.close()
					raise FinalizationContextError(
						f"{request.label} lock is already held: {path}"
					) from exc
				self._handles.append(handle)
				self.acquired_labels.append(request.label)
		except BaseException:
			self._release()
			raise
		self._held = True
		return self

	def _release(self) -> None:
		for handle in reversed(self._handles):
			try:
				fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
			finally:
				handle.close()
		self._handles.clear()
		self._held = False

	def __exit__(self, *_args: Any) -> None:
		self._release()

	def assert_held(self) -> None:
		if not self._held or not self._handles or any(handle.closed for handle in self._handles):
			raise FinalizationContextError("required finalization locks are not held")


def _regular_path(path: Path, label: str) -> Path:
	expanded = path.expanduser()
	if not expanded.is_file() or expanded.is_symlink():
		raise FinalizationContextError(f"{label} is missing or not a regular file: {path}")
	return expanded.resolve(strict=True)


@dataclass(frozen=True)
class ValidatedPredecessor:
	"""One exact canonical predecessor result, valid only for a locked action."""

	template_path: Path
	gate_path: Path
	digest_field: str
	digest: str
	_canonical_gate: dict[str, Any] = field(repr=False)
	_assert_locks: Callable[[], None] = field(repr=False, compare=False)
	_original_builder: Callable[[Path], dict[str, Any]] = field(repr=False, compare=False)

	@classmethod
	def create(
		cls,
		validator: ModuleType,
		*,
		template_path: Path,
		gate_path: Path,
		assert_locks: Callable[[], None],
		digest_field: str = "gate_sha256",
	) -> "ValidatedPredecessor":
		"""Call the original canonical builder exactly once and bind its result."""
		assert_locks()
		template = _regular_path(template_path, "predecessor template")
		gate_file = _regular_path(gate_path, "stored predecessor gate")
		builder = getattr(validator, "build_predecessor_gate", None)
		load_json = getattr(validator, "load_json", None)
		digest_valid = getattr(validator, "_self_digest_valid", None)
		if not callable(builder) or not callable(load_json) or not callable(digest_valid):
			raise FinalizationContextError("validator lacks canonical predecessor helpers")
		stored = load_json(gate_file)
		canonical = builder(template)
		if not isinstance(stored, dict) or not isinstance(canonical, dict):
			raise FinalizationContextError("canonical predecessor result is not a JSON object")
		if (
			stored.get("valid") is not True
			or canonical.get("valid") is not True
			or not digest_valid(stored, digest_field)
			or not digest_valid(canonical, digest_field)
		):
			raise FinalizationContextError(
				"stored and canonical predecessor gates must be valid and self-digested"
			)
		if stored != canonical:
			raise FinalizationContextError("stored predecessor gate is stale")
		digest = stored.get(digest_field)
		if not isinstance(digest, str) or len(digest) != 64:
			raise FinalizationContextError("predecessor gate digest is malformed")
		assert_locks()
		return cls(
			template_path=template,
			gate_path=gate_file,
			digest_field=digest_field,
			digest=digest,
			_canonical_gate=copy.deepcopy(canonical),
			_assert_locks=assert_locks,
			_original_builder=builder,
		)

	def gate_copy(self, requested_template: Path) -> dict[str, Any]:
		self._assert_locks()
		absolute = requested_template.expanduser().absolute()
		requested = _regular_path(requested_template, "requested predecessor template")
		if absolute != self.template_path or requested != self.template_path:
			raise FinalizationContextError(
				f"validated predecessor is bound to {self.template_path}, not {requested}"
			)
		return copy.deepcopy(self._canonical_gate)


_MISSING = object()
_CONTEXT_ATTRIBUTE = "_validated_predecessor_context"


@contextmanager
def scoped_predecessor_cache(
	validator: ModuleType,
	context: ValidatedPredecessor,
) -> Iterator[ValidatedPredecessor]:
	"""Temporarily route only canonical predecessor builds through ``context``."""
	targets: list[Any] = []
	for target in (validator, getattr(validator, "CORE", None)):
		if target is not None and all(target is not existing for existing in targets):
			targets.append(target)
	prior: list[tuple[Any, Any, Any]] = []

	def cached_builder(path: Path) -> dict[str, Any]:
		return context.gate_copy(Path(path))

	try:
		context._assert_locks()
		for target in targets:
			builder = getattr(target, "build_predecessor_gate", _MISSING)
			marker = getattr(target, _CONTEXT_ATTRIBUTE, _MISSING)
			if builder is _MISSING or not callable(builder):
				raise FinalizationContextError(
					"validator cache target lacks build_predecessor_gate"
				)
			if builder is not context._original_builder:
				raise FinalizationContextError(
					"validator cache targets do not share the validated original builder"
				)
			if marker is not _MISSING:
				raise FinalizationContextError("a validated predecessor context is already active")
			prior.append((target, builder, marker))
			setattr(target, "build_predecessor_gate", cached_builder)
			setattr(target, _CONTEXT_ATTRIBUTE, context)
		yield context
	finally:
		for target, builder, marker in reversed(prior):
			setattr(target, "build_predecessor_gate", builder)
			if marker is _MISSING:
				delattr(target, _CONTEXT_ATTRIBUTE)
			else:
				setattr(target, _CONTEXT_ATTRIBUTE, marker)
