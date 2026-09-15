#!/usr/bin/env python3
"""Focused tests for the unattended depth-map sweep scheduler."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_sweep


def arguments(root: Path, **values: object) -> dmap_sweep.Arguments:
    defaults: dict[str, object] = {
        "config": root / "config.yaml",
        "generate_report": False,
        "free_space_floor_gb": 0.0,
        "finalization_reserve_gb": 0.0,
        "finalization_reserve_minutes": 0.0,
    }
    defaults.update(values)
    return dmap_sweep.Arguments(**defaults)


def scene(scan_id: str, root: Path) -> dict:
    work = root / scan_id / "input"
    mvs = work / "mvs" / "scene.mvs"
    mvs.parent.mkdir(parents=True, exist_ok=True)
    mvs.write_bytes(b"mvs")
    (mvs.parent / "Densify.ini").write_text("[Densify]\nIters = 10\n", encoding="utf-8")
    return {
        "scan_id": scan_id,
        "name": scan_id,
        "working_folder": str(work),
        "mvs_file": str(mvs),
    }


def run(label: str, *, family: str, baseline: bool = False) -> dict:
    return {
        "label": label,
        "role": "baseline" if baseline else "variant",
        "repeats": 3,
        "sweep": {
            "family": family,
            "always_run": baseline,
            "priority": -10 if baseline else 0,
        },
        "densify_args": ["--iters", "10", "--geometric-iters", "4"],
    }


def write_fake_densify_binary(path: Path, *, observer: bool) -> None:
    observer_help = "" if not observer else """
printf '%s\n' '--dmap-instrumentation-dir arg'
printf '%s\n' '--dmap-instrumentation-level arg'
printf '%s\n' '--dmap-instrumentation-write-maps arg'
"""
    path.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' 'DensifyPointCloud options'\n"
        + observer_help
        + "exit 1\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def sweep_job(root: Path, *, job_id: str = "job-test", stage: str = "screen") -> dmap_sweep.SweepJob:
    run_spec = run("candidate-a", family="family-a")
    scene_spec = scene("scene-a", root)
    return dmap_sweep.SweepJob(
        job_id=job_id,
        stage=stage,
        stage_index=0,
        priority=0,
        run_label="candidate-a",
        run_role="variant",
        family="family-a",
        always_run=False,
        repeat=0,
        scene_id="scene-a",
        profile="endpoint",
        mode="endpoint",
        run_dir=root / "run",
        timeout_seconds=60.0,
        estimated_output_bytes=1024,
        argument_overrides={},
        run_spec=run_spec,
        scene_spec=scene_spec,
    )


def lossy_summary_job(
    root: Path, *, requested: str, evidenced_ids: tuple[int, ...]
) -> dmap_sweep.SweepJob:
    job = sweep_job(root)
    object.__setattr__(job, "profile", "summary")
    object.__setattr__(job, "mode", "timing")
    object.__setattr__(job, "retention_policy", {
        "schema_name": dmap_sweep.RETENTION_POLICY_SCHEMA_NAME,
        "schema_version": 1,
        "profile": "summary",
        "dmap_policy": "instrumented_image_ids_only",
        "lossy": True,
    })
    depth_dir = job.run_dir / "depth_maps"
    work_dir = job.run_dir / "work" / "mvs"
    depth_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)
    for image_id in (1, 2, 3):
        retained = depth_dir / f"depth{image_id:04d}.dmap"
        retained.write_bytes(f"depth-map-{image_id}".encode())
        os.link(retained, work_dir / retained.name)
    for image_id in evidenced_ids:
        frame = (
            job.run_dir / "dmap_instrumentation" / "depthmaps"
            / f"image{image_id:04d}"
        )
        frame.mkdir(parents=True)
        (frame / "summary.json").write_text(
            json.dumps({"image_id": image_id}) + "\n", encoding="utf-8"
        )
    (job.run_dir / "command.sh").write_text("true\n", encoding="utf-8")
    (job.run_dir / "repro.json").write_text(json.dumps({
        "command": [
            "DensifyPointCloudDMapObserve",
            "--dmap-instrumentation-image-list", requested,
        ],
        "return_code": 0,
        "dry_run": False,
    }) + "\n", encoding="utf-8")
    return job


def confirmation_fixture(root: Path) -> tuple[dict, list[dmap_sweep.SweepJob]]:
    sentinel_scenes = ("scene-a", "scene-b")
    all_scenes = (*sentinel_scenes, "scene-c")
    jobs: list[dmap_sweep.SweepJob] = []
    for label in ("control", "selected"):
        for scene_id in all_scenes:
            job = sweep_job(
                root / f"{label}-{scene_id}",
                job_id=f"job-{label}-{scene_id}",
                stage="confirmation",
            )
            object.__setattr__(job, "run_label", label)
            object.__setattr__(job, "scene_id", scene_id)
            object.__setattr__(
                job,
                "run_dir",
                root / "runs" / label / "repeat_00" / scene_id / "timing",
            )
            jobs.append(job)
    config = {
        "execution_contract": {"expected_reused_jobs": 4},
        "manual_confirmation": {
            "baseline_run": "control",
            "sentinel_scenes": list(sentinel_scenes),
            "resolution": {"selected_run": "selected"},
        },
    }
    return config, jobs


class Sha256FileCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        dmap_sweep._clear_sha256_file_cache()

    def tearDown(self) -> None:
        dmap_sweep._clear_sha256_file_cache()

    def test_repeat_hash_reuses_cached_digest_without_open_or_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(b"repeatable-artifact" * 1024)
            original_open = dmap_sweep._sha256_open_stable_descriptor
            counts = {"opens": 0, "reads": 0}

            class CountingHandle:
                def __init__(self, handle: object) -> None:
                    self.handle = handle

                def __enter__(self) -> "CountingHandle":
                    self.handle.__enter__()
                    return self

                def __exit__(self, *args: object) -> object:
                    return self.handle.__exit__(*args)

                def fileno(self) -> int:
                    return self.handle.fileno()

                def read(self, size: int = -1) -> bytes:
                    counts["reads"] += 1
                    return self.handle.read(size)

            def counting_open(path: Path) -> CountingHandle:
                counts["opens"] += 1
                return CountingHandle(original_open(path))

            with mock.patch.object(
                dmap_sweep, "_sha256_open_stable_descriptor", side_effect=counting_open
            ):
                first = dmap_sweep.sha256_file(source)
                first_counts = dict(counts)
                second = dmap_sweep.sha256_file(source)

            self.assertEqual(first, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(second, first)
            self.assertEqual(first_counts, counts)
            self.assertEqual(counts["opens"], 1)
            self.assertGreater(counts["reads"], 0)

    def test_content_change_with_same_size_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(b"first")
            first_stat = source.stat()
            first = dmap_sweep.sha256_file(source)

            source.write_bytes(b"other")
            second_stat = source.stat()
            second = dmap_sweep.sha256_file(source)

            self.assertEqual(first_stat.st_size, second_stat.st_size)
            self.assertNotEqual(first_stat.st_ctime_ns, second_stat.st_ctime_ns)
            self.assertNotEqual(second, first)
            self.assertEqual(second, hashlib.sha256(b"other").hexdigest())

    def test_atomic_replacement_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            replacement = Path(directory) / "replacement.bin"
            source.write_bytes(b"old-content")
            first_inode = source.stat().st_ino
            first = dmap_sweep.sha256_file(source)

            replacement.write_bytes(b"new-content")
            os.replace(replacement, source)
            second_inode = source.stat().st_ino
            second = dmap_sweep.sha256_file(source)

            self.assertNotEqual(first_inode, second_inode)
            self.assertNotEqual(second, first)
            self.assertEqual(second, hashlib.sha256(b"new-content").hexdigest())

    def test_atomic_replacement_after_eof_discards_completed_descriptor_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            replacement = Path(directory) / "replacement.bin"
            source.write_bytes(b"old-content")
            replacement.write_bytes(b"new-content")
            original_snapshot = dmap_sweep._sha256_path_snapshot
            snapshots = 0
            reads = 0
            original_read = dmap_sweep._sha256_read_stable_file

            def replacing_snapshot(
                *args: object, **kwargs: object
            ) -> tuple[dmap_sweep._Sha256FileCacheKey, os.stat_result]:
                nonlocal snapshots
                snapshots += 1
                if snapshots == 2:
                    os.replace(replacement, source)
                return original_snapshot(*args, **kwargs)

            def counting_read(
                *args: object, **kwargs: object
            ) -> dmap_sweep._StableSha256Read:
                nonlocal reads
                reads += 1
                return original_read(*args, **kwargs)

            with mock.patch.object(
                dmap_sweep, "_sha256_path_snapshot", side_effect=replacing_snapshot
            ), mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=counting_read
            ):
                digest = dmap_sweep.sha256_file(source)

            self.assertEqual(digest, hashlib.sha256(b"new-content").hexdigest())
            self.assertEqual(reads, 2)
            self.assertEqual(snapshots, 4)

    def test_timestamp_and_mode_changes_force_revalidation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(b"same-content")
            original_read = dmap_sweep._sha256_read_stable_file
            calls = 0

            def counting_read(
                *args: object, **kwargs: object
            ) -> dmap_sweep._StableSha256Read:
                nonlocal calls
                calls += 1
                return original_read(*args, **kwargs)

            with mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=counting_read
            ):
                first = dmap_sweep.sha256_file(source)
                source.chmod(source.stat().st_mode ^ 0o100)
                after_mode = dmap_sweep.sha256_file(source)
                stat = source.stat()
                os.utime(
                    source,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
                )
                after_timestamp = dmap_sweep.sha256_file(source)

            self.assertEqual(first, after_mode)
            self.assertEqual(first, after_timestamp)
            self.assertEqual(calls, 3)

    def test_concurrent_calls_share_one_stable_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(os.urandom(2 * dmap_sweep.SHA256_FILE_BLOCK_BYTES))
            original_read = dmap_sweep._sha256_read_stable_file
            owner_started = threading.Event()
            calls = 0
            calls_lock = threading.Lock()
            results: list[str] = []
            errors: list[BaseException] = []

            def slow_read(
                *args: object, **kwargs: object
            ) -> dmap_sweep._StableSha256Read:
                nonlocal calls
                with calls_lock:
                    calls += 1
                owner_started.set()
                time.sleep(0.05)
                return original_read(*args, **kwargs)

            def worker() -> None:
                try:
                    results.append(dmap_sweep.sha256_file(source))
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            with mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=slow_read
            ):
                threads = [threading.Thread(target=worker) for _ in range(8)]
                for thread in threads:
                    thread.start()
                self.assertTrue(owner_started.wait(timeout=1.0))
                for thread in threads:
                    thread.join(timeout=2.0)

            self.assertFalse(errors)
            self.assertEqual(len(results), 8)
            self.assertEqual(len(set(results)), 1)
            self.assertEqual(calls, 1)
            self.assertFalse(any(thread.is_alive() for thread in threads))

    def test_waiters_retry_after_atomic_replacement_before_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            replacement = Path(directory) / "replacement.bin"
            source.write_bytes(b"old-content")
            replacement.write_bytes(b"new-content")
            original_read = dmap_sweep._sha256_read_stable_file
            owner_entered = threading.Event()
            replace_now = threading.Event()
            call_lock = threading.Lock()
            calls = 0
            results: list[str] = []
            errors: list[BaseException] = []

            def replacing_read(
                *args: object, **kwargs: object
            ) -> dmap_sweep._StableSha256Read:
                nonlocal calls
                with call_lock:
                    calls += 1
                    call_index = calls
                if call_index == 1:
                    owner_entered.set()
                    self.assertTrue(replace_now.wait(timeout=1.0))
                    os.replace(replacement, source)
                return original_read(*args, **kwargs)

            def worker() -> None:
                try:
                    results.append(dmap_sweep.sha256_file(source))
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            with mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=replacing_read
            ):
                owner = threading.Thread(target=worker)
                owner.start()
                self.assertTrue(owner_entered.wait(timeout=1.0))
                waiters = [threading.Thread(target=worker) for _ in range(7)]
                for waiter in waiters:
                    waiter.start()
                time.sleep(0.05)
                replace_now.set()
                for thread in [owner, *waiters]:
                    thread.join(timeout=2.0)

            self.assertFalse(errors)
            self.assertEqual(len(results), 8)
            self.assertEqual(
                set(results), {hashlib.sha256(b"new-content").hexdigest()}
            )
            self.assertEqual(calls, 2)
            self.assertFalse(any(thread.is_alive() for thread in [owner, *waiters]))

    def test_read_exception_releases_waiters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(b"stable-content")
            original_read = dmap_sweep._sha256_read_stable_file
            owner_entered = threading.Event()
            fail_now = threading.Event()
            call_lock = threading.Lock()
            calls = 0
            results: list[str] = []
            errors: list[BaseException] = []

            def failing_read(
                *args: object, **kwargs: object
            ) -> dmap_sweep._StableSha256Read:
                nonlocal calls
                with call_lock:
                    calls += 1
                    call_index = calls
                if call_index == 1:
                    owner_entered.set()
                    self.assertTrue(fail_now.wait(timeout=1.0))
                    raise OSError("synthetic read failure")
                return original_read(*args, **kwargs)

            def worker() -> None:
                try:
                    results.append(dmap_sweep.sha256_file(source))
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=failing_read
            ):
                owner = threading.Thread(target=worker)
                owner.start()
                self.assertTrue(owner_entered.wait(timeout=1.0))
                waiter = threading.Thread(target=worker)
                waiter.start()
                time.sleep(0.02)
                fail_now.set()
                owner.join(timeout=2.0)
                waiter.join(timeout=2.0)

            self.assertEqual(calls, 2)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], OSError)
            self.assertEqual(
                results, [hashlib.sha256(b"stable-content").hexdigest()]
            )
            self.assertFalse(owner.is_alive())
            self.assertFalse(waiter.is_alive())
            self.assertFalse(dmap_sweep._SHA256_FILE_INFLIGHT)

    def test_mutation_during_read_is_discarded_and_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            original = b"a" * (2 * dmap_sweep.SHA256_FILE_BLOCK_BYTES)
            replacement = b"b" * len(original)
            source.write_bytes(original)
            original_open = dmap_sweep._sha256_open_stable_descriptor
            mutated = False
            opens = 0

            class MutatingHandle:
                def __init__(self, handle: object) -> None:
                    self.handle = handle

                def __enter__(self) -> "MutatingHandle":
                    self.handle.__enter__()
                    return self

                def __exit__(self, *args: object) -> object:
                    return self.handle.__exit__(*args)

                def fileno(self) -> int:
                    return self.handle.fileno()

                def read(self, size: int = -1) -> bytes:
                    nonlocal mutated
                    block = self.handle.read(size)
                    if block and not mutated:
                        mutated = True
                        source.write_bytes(replacement)
                    return block

            def mutating_open(path: Path) -> MutatingHandle:
                nonlocal opens
                opens += 1
                return MutatingHandle(original_open(path))

            with mock.patch.object(
                dmap_sweep, "_sha256_open_stable_descriptor", side_effect=mutating_open
            ):
                digest = dmap_sweep.sha256_file(source)

            self.assertTrue(mutated)
            self.assertEqual(opens, 2)
            self.assertEqual(digest, hashlib.sha256(replacement).hexdigest())

    def test_repeated_instability_fails_after_bounded_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(b"unstable")
            error = dmap_sweep._FileChangedDuringHash("test mutation")
            with mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=error
            ) as read:
                with self.assertRaisesRegex(
                    RuntimeError, "file changed while hashing after 3 attempts"
                ):
                    dmap_sweep.sha256_file(source)

            self.assertEqual(read.call_count, dmap_sweep.SHA256_FILE_MAX_ATTEMPTS)

    def test_symlinks_follow_targets_and_retargeting_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.bin"
            second = root / "second.bin"
            link = root / "artifact.bin"
            first.write_bytes(b"first-target")
            second.write_bytes(b"second-target")
            link.symlink_to(first)
            original_read = dmap_sweep._sha256_read_stable_file
            reads = 0

            def counting_read(
                *args: object, **kwargs: object
            ) -> dmap_sweep._StableSha256Read:
                nonlocal reads
                reads += 1
                return original_read(*args, **kwargs)

            with mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=counting_read
            ):
                direct_digest = dmap_sweep.sha256_file(first)
                linked_digest = dmap_sweep.sha256_file(link)
                link.unlink()
                link.symlink_to(second)
                retargeted_digest = dmap_sweep.sha256_file(link)

            self.assertEqual(linked_digest, direct_digest)
            self.assertEqual(direct_digest, hashlib.sha256(b"first-target").hexdigest())
            self.assertEqual(
                retargeted_digest, hashlib.sha256(b"second-target").hexdigest()
            )
            self.assertEqual(reads, 2)
            link.unlink()
            link.symlink_to(root / "missing.bin")
            with self.assertRaises(FileNotFoundError):
                dmap_sweep.sha256_file(link)

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires procfs file descriptors")
    def test_open_unlinked_regular_magic_link_preserves_follow_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            payload = b"open but unlinked artifact"
            source.write_bytes(payload)
            descriptor = os.open(source, os.O_RDONLY)
            source.unlink()
            magic_link = Path(f"/proc/self/fd/{descriptor}")
            try:
                first = dmap_sweep.sha256_file(magic_link)
                second = dmap_sweep.sha256_file(magic_link)
            finally:
                os.close(descriptor)

            self.assertEqual(first, hashlib.sha256(payload).hexdigest())
            self.assertEqual(second, first)
            self.assertFalse(dmap_sweep._SHA256_FILE_CACHE)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFO support")
    def test_fifo_without_writer_is_rejected_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fifo = Path(directory) / "artifact.fifo"
            os.mkfifo(fifo)
            started = time.monotonic()
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                dmap_sweep.sha256_file(fifo)

            self.assertLess(time.monotonic() - started, 0.5)
            self.assertFalse(dmap_sweep._SHA256_FILE_INFLIGHT)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_forked_child_discards_inherited_cache_and_inflight_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(b"child-must-read-this")
            key, _ = dmap_sweep._sha256_path_snapshot(source)
            with dmap_sweep._SHA256_FILE_CACHE_LOCK:
                dmap_sweep._SHA256_FILE_CACHE[key] = "0" * 64
                dmap_sweep._SHA256_FILE_INFLIGHT[key] = threading.Event()

            read_fd, write_fd = os.pipe()
            child = os.fork()
            if child == 0:  # pragma: no cover - assertions run in the parent
                os.close(read_fd)
                try:
                    payload = dmap_sweep.sha256_file(source).encode("ascii")
                except BaseException as exc:
                    payload = f"ERROR:{exc!r}".encode("utf-8")
                try:
                    os.write(write_fd, payload)
                finally:
                    os.close(write_fd)
                    os._exit(0)

            os.close(write_fd)
            timed_out = False
            try:
                ready, _, _ = select.select([read_fd], [], [], 2.0)
                if not ready:
                    timed_out = True
                    os.kill(child, signal.SIGKILL)
                    payload = b""
                else:
                    payload = os.read(read_fd, 4096)
            finally:
                os.close(read_fd)
                os.waitpid(child, 0)
                with dmap_sweep._SHA256_FILE_CACHE_LOCK:
                    dmap_sweep._SHA256_FILE_INFLIGHT.pop(key, None)

            self.assertFalse(timed_out, "forked child blocked on an inherited Event")
            self.assertEqual(
                payload.decode("ascii"),
                hashlib.sha256(b"child-must-read-this").hexdigest(),
            )

    def test_nonregular_is_rejected_and_zero_size_pseudo_file_is_not_cached(self) -> None:
        if Path("/dev/null").exists():
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                dmap_sweep.sha256_file(Path("/dev/null"))
        uptime = Path("/proc/uptime")
        if not uptime.exists():
            self.skipTest("/proc/uptime is not available")
        original_read = dmap_sweep._sha256_read_uncached
        calls = 0

        def counting_read(
            *args: object, **kwargs: object
        ) -> str:
            nonlocal calls
            calls += 1
            return original_read(*args, **kwargs)

        with mock.patch.object(
            dmap_sweep, "_sha256_read_uncached", side_effect=counting_read
        ):
            dmap_sweep.sha256_file(uptime)
            dmap_sweep.sha256_file(uptime)

        self.assertEqual(calls, 2)

    def test_cache_has_bounded_lru_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            dmap_sweep, "SHA256_FILE_CACHE_MAX_ENTRIES", 2
        ):
            root = Path(directory)
            files = [root / f"artifact-{index}.bin" for index in range(3)]
            original_read = dmap_sweep._sha256_read_stable_file
            reads = 0

            def counting_read(
                *args: object, **kwargs: object
            ) -> dmap_sweep._StableSha256Read:
                nonlocal reads
                reads += 1
                return original_read(*args, **kwargs)

            with mock.patch.object(
                dmap_sweep, "_sha256_read_stable_file", side_effect=counting_read
            ):
                for index, source in enumerate(files):
                    source.write_bytes(bytes([index]))
                dmap_sweep.sha256_file(files[0])
                dmap_sweep.sha256_file(files[1])
                dmap_sweep.sha256_file(files[0])
                dmap_sweep.sha256_file(files[2])

                self.assertEqual(len(dmap_sweep._SHA256_FILE_CACHE), 2)
                self.assertIn(
                    dmap_sweep._sha256_path_snapshot(files[0])[0],
                    dmap_sweep._SHA256_FILE_CACHE,
                )
                self.assertNotIn(
                    dmap_sweep._sha256_path_snapshot(files[1])[0],
                    dmap_sweep._SHA256_FILE_CACHE,
                )
                dmap_sweep.sha256_file(files[1])

            self.assertEqual(len(dmap_sweep._SHA256_FILE_CACHE), 2)
            self.assertEqual(reads, 4)


class DMapSweepTests(unittest.TestCase):
    def test_confirmation_reuse_contract_resolves_exact_sentinel_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, jobs = confirmation_fixture(Path(directory))
            definition = dmap_sweep.configured_reuse_contract(config, jobs)

        self.assertIsNotNone(definition)
        assert definition is not None
        self.assertEqual(definition["expected_reused_jobs"], 4)
        self.assertEqual(definition["expected_new_jobs"], 2)
        self.assertEqual(
            definition["expected_reused_job_ids"],
            [
                "job-control-scene-a",
                "job-control-scene-b",
                "job-selected-scene-a",
                "job-selected-scene-b",
            ],
        )

    def test_confirmation_reuse_preflight_requires_expected_and_rejects_extra(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, jobs = confirmation_fixture(root)
            definition = dmap_sweep.configured_reuse_contract(config, jobs)
            assert definition is not None
            by_id = {job.job_id: job for job in jobs}
            for job_id in definition["expected_reused_job_ids"]:
                by_id[job_id].run_dir.mkdir(parents=True)
                (by_id[job_id].run_dir / "complete").write_text("ok", encoding="utf-8")
            with mock.patch.object(
                dmap_sweep,
                "validate_completed_sweep_run",
                return_value=(True, "validated"),
            ):
                valid = dmap_sweep.reuse_contract_preflight(jobs, definition)
                by_id["job-control-scene-a"].run_dir.joinpath("complete").unlink()
                by_id["job-control-scene-a"].run_dir.rmdir()
                missing = dmap_sweep.reuse_contract_preflight(jobs, definition)
                extra_job = by_id["job-control-scene-c"]
                extra_job.run_dir.mkdir(parents=True)
                (extra_job.run_dir / "complete").write_text("ok", encoding="utf-8")
                extra = dmap_sweep.reuse_contract_preflight(jobs, definition)

        self.assertTrue(valid["valid"], valid["failed_checks"])
        self.assertFalse(missing["valid"])
        self.assertTrue(any("expected reusable job" in row for row in missing["failed_checks"]))
        self.assertFalse(extra["valid"])
        self.assertTrue(any("unexpectedly reusable" in row for row in extra["failed_checks"]))

    def test_confirmation_reuse_preflight_allows_new_output_only_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, jobs = confirmation_fixture(root)
            definition = dmap_sweep.configured_reuse_contract(config, jobs)
            assert definition is not None
            by_id = {job.job_id: job for job in jobs}
            for job_id in [
                *definition["expected_reused_job_ids"],
                "job-control-scene-c",
            ]:
                by_id[job_id].run_dir.mkdir(parents=True)
                (by_id[job_id].run_dir / "complete").write_text("ok", encoding="utf-8")
            prior = {
                "jobs": {
                    "job-control-scene-c": {
                        "status": "complete",
                        "reused": False,
                    }
                }
            }
            with mock.patch.object(
                dmap_sweep,
                "validate_completed_sweep_run",
                return_value=(True, "validated"),
            ):
                initial = dmap_sweep.reuse_contract_preflight(jobs, definition)
                resumed = dmap_sweep.reuse_contract_preflight(jobs, definition, prior)

        self.assertFalse(initial["valid"])
        self.assertTrue(resumed["valid"], resumed["failed_checks"])
        self.assertEqual(
            resumed["valid_completed_new_job_ids"], ["job-control-scene-c"]
        )

    def test_confirmation_reuse_postcheck_requires_exact_reuse_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, jobs = confirmation_fixture(Path(directory))
            definition = dmap_sweep.configured_reuse_contract(config, jobs)
            assert definition is not None
        expected_reused = set(definition["expected_reused_job_ids"])
        ledger = {
            "jobs": {
                job.job_id: {
                    "status": "complete",
                    "reused": job.job_id in expected_reused,
                }
                for job in jobs
            }
        }

        valid = dmap_sweep.reuse_contract_postcheck(ledger, definition)
        ledger["jobs"]["job-control-scene-c"]["reused"] = True
        invalid = dmap_sweep.reuse_contract_postcheck(ledger, definition)

        self.assertTrue(valid["valid"], valid["failed_checks"])
        self.assertEqual(valid["actual_reused_jobs"], 4)
        self.assertEqual(valid["actual_new_jobs"], 2)
        self.assertFalse(invalid["valid"])
        self.assertIn(
            "job-control-scene-c", " ".join(invalid["failed_checks"])
        )

    def test_stage_templates_define_non_cartesian_priority_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = [scene("scene-a", root), scene("scene-b", root)]
            config = {
                "capture_profiles": ["summary"],
                "suite": {"name": "development", "scan_ids": ["scene-a", "scene-b"]},
                "scenes": scenes,
                "runs": [
                    run("control", family="control", baseline=True),
                    run("candidate-a", family="family-a"),
                    run("candidate-b", family="family-b"),
                ],
                "sweep": {
                    "stages": [
                        {
                            "name": "control",
                            "runs": ["control"],
                            "scenes": ["scene-a", "scene-b"],
                            "profiles": ["summary"],
                            "repeat_indices": [0, 1, 2],
                        },
                        {
                            "name": "screen",
                            "runs": ["candidate-a", "candidate-b"],
                            "scenes": ["scene-a"],
                            "profiles": ["summary"],
                            "repeat_indices": [0],
                        },
                        {
                            "name": "confirm",
                            "runs": ["candidate-a", "candidate-b"],
                            "scenes": ["scene-b"],
                            "profiles": ["endpoint"],
                            "repeat_indices": [1],
                            "argument_overrides": {
                                "--resolution-level": 4,
                                "--max-resolution": 640,
                            },
                        },
                    ],
                },
            }

            jobs = dmap_sweep.build_jobs(arguments(root), config, root / "experiment")

            self.assertEqual(len(jobs), 10)
            self.assertEqual([job.stage for job in jobs[:6]], ["control"] * 6)
            self.assertTrue(all(job.priority == -10 for job in jobs[:6]))
            screen = [job for job in jobs if job.stage == "screen"]
            self.assertEqual({job.scene_id for job in screen}, {"scene-a"})
            self.assertEqual({job.repeat for job in screen}, {0})
            confirm = [job for job in jobs if job.stage == "confirm"]
            self.assertEqual({job.profile for job in confirm}, {"endpoint"})
            self.assertEqual({job.repeat for job in confirm}, {1})
            self.assertEqual(confirm[0].argument_overrides["--resolution-level"], "4")

    def test_unselected_stage_never_generates_or_reuses_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config_path = root / "config.yaml"
            output_root = root / "output"
            config_path.write_text(json.dumps({
                "schema_version": 2,
                "experiment_id": "skip_empty_stage_test",
                "output_root": str(output_root),
                "dataset_root": str(root / "dataset"),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "capture_profiles": ["endpoint"],
                "suite": {"name": "development", "scan_ids": ["scene-a"]},
                "scenes": [scene("scene-a", root)],
                "instrumentation": {
                    "expected_width": 1,
                    "expected_height": 1,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 1,
                },
                "runs": [
                    {**run("control", family="control", baseline=True), "repeats": 1},
                    {**run("candidate", family="candidate"), "repeats": 1},
                ],
                "sweep": {"stages": [
                    {
                        "name": "selected",
                        "runs": ["control"],
                        "scenes": ["scene-a"],
                        "profiles": ["endpoint"],
                        "repeat_indices": [0],
                        "report_after": False,
                    },
                    {
                        "name": "unselected",
                        "runs": ["candidate"],
                        "scenes": ["scene-a"],
                        "profiles": ["endpoint"],
                        "repeat_indices": [0],
                        "report_after": True,
                    },
                ]},
            }), encoding="utf-8")
            provenance = {
                "archive": str(root / "source.tar.zst"),
                "sha256": "c" * 64,
                "commit": "d" * 40,
                "dirty": True,
            }

            def complete_job(
                _arguments, _config, _root, job, ledger, ledger_path,
                _heartbeat_path, _stop_event, deadline_monotonic=None,
            ) -> bool:
                ledger["jobs"][job.job_id].update(status="complete", reused=False)
                dmap_sweep.save_ledger(ledger_path, ledger)
                return True

            args = arguments(
                root,
                config=config_path,
                run=["control"],
                generate_report=False,
                compact=False,
            )
            with (
                mock.patch.object(
                    dmap_sweep, "ensure_source_provenance", return_value=provenance
                ),
                mock.patch.object(dmap_sweep, "admission_reason", return_value=None),
                mock.patch.object(dmap_sweep, "run_job", side_effect=complete_job),
                mock.patch.object(dmap_sweep, "generate_stage_report") as stage_report,
            ):
                return_code = dmap_sweep.run_sweep(args)

            ledger = json.loads(
                (
                    output_root / "skip_empty_stage_test" / "sweep" / "schedule.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(return_code, 0)
        stage_report.assert_not_called()
        self.assertEqual(ledger["stages"]["unselected"]["status"], "not_selected")
        self.assertFalse(ledger["stages"]["unselected"]["report_after"])
        self.assertNotIn("unselected", ledger["stage_reports"])

    def test_failure_report_generation_is_charged_to_wall_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config_path = root / "config.yaml"
            output_root = root / "output"
            config_path.write_text(json.dumps({
                "schema_version": 2,
                "experiment_id": "failure_report_budget_test",
                "output_root": str(output_root),
                "dataset_root": str(root / "dataset"),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "capture_profiles": ["endpoint"],
                "suite": {"name": "development", "scan_ids": ["scene-a"]},
                "scenes": [scene("scene-a", root)],
                "instrumentation": {
                    "expected_width": 1,
                    "expected_height": 1,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 1,
                },
                "runs": [{
                    **run("control", family="control", baseline=True),
                    "repeats": 1,
                }],
                "sweep": {"stages": [{
                    "name": "control",
                    "runs": ["control"],
                    "scenes": ["scene-a"],
                    "profiles": ["endpoint"],
                    "repeat_indices": [0],
                }]},
            }), encoding="utf-8")
            provenance = {
                "archive": str(root / "source.tar.zst"),
                "sha256": "c" * 64,
                "commit": "d" * 40,
                "dirty": True,
            }

            def fail_after_completion(
                _arguments, _config, _root, job, ledger, ledger_path,
                _heartbeat_path, _stop_event, deadline_monotonic=None,
            ) -> bool:
                ledger["jobs"][job.job_id].update(status="complete", reused=False)
                dmap_sweep.save_ledger(ledger_path, ledger)
                raise RuntimeError("capture bookkeeping failed")

            def slow_failure_report(
                _arguments, _config, _root, _report_dir, ledger, ledger_path,
                _heartbeat_path, _stop_event,
            ) -> bool:
                time.sleep(0.05)
                ledger["report"].update(status="complete", valid=True)
                dmap_sweep.save_ledger(ledger_path, ledger)
                return True

            args = arguments(
                root, config=config_path, generate_report=True, compact=False,
            )
            with (
                mock.patch.object(
                    dmap_sweep, "ensure_source_provenance", return_value=provenance
                ),
                mock.patch.object(dmap_sweep, "admission_reason", return_value=None),
                mock.patch.object(
                    dmap_sweep, "run_job", side_effect=fail_after_completion
                ),
                mock.patch.object(
                    dmap_sweep, "generate_final_report", side_effect=slow_failure_report
                ) as report,
                self.assertRaisesRegex(RuntimeError, "capture bookkeeping failed"),
            ):
                dmap_sweep.run_sweep(args)

            ledger = json.loads((
                output_root / "failure_report_budget_test" / "sweep" / "schedule.json"
            ).read_text(encoding="utf-8"))
            report.assert_called_once()
            self.assertGreaterEqual(ledger["sessions"][-1]["elapsed_seconds"], 0.05)
            session_identity = ledger["sessions"][-1]["process_identity"]
            current_identity = dmap_sweep.linux_process_identity(os.getpid())
            self.assertIsNotNone(current_identity)
            for field in (
                "pid", "process_group_id", "start_ticks", "boot_id", "cmdline",
                "cmdline_sha256",
            ):
                self.assertEqual(session_identity[field], current_identity[field])
            self.assertEqual(
                ledger["budget"]["elapsed_seconds"],
                sum(row["elapsed_seconds"] for row in ledger["sessions"]),
            )

    def test_scene_argument_overrides_merge_over_stage_and_enter_identity_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = scene("scene-a", root)
            first["argument_overrides"] = {
                "--dmap-instrumentation-image-list": "1,3",
            }
            second = scene("scene-b", root)
            config_path = root / "config.yaml"
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            config_path.write_text("experiment_id: test\n", encoding="utf-8")
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config = {
                "_config_path": str(config_path),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "capture_profiles": ["summary"],
                "suite": {"name": "development", "scan_ids": ["scene-a", "scene-b"]},
                "scenes": [first, second],
                "runs": [run("candidate-a", family="family-a")],
                "default_densify_args": [
                    "--dmap-instrumentation-image-list", "9",
                    "--resolution-level", "0",
                ],
                "sweep": {
                    "retention": {
                        "schema_version": 1,
                        "summary_dmap_policy": "instrumented_image_ids_only",
                    },
                    "stages": [{
                        "name": "screen",
                        "runs": ["candidate-a"],
                        "profiles": ["summary"],
                        "repeat_indices": [0],
                        "argument_overrides": {
                            "--dmap-instrumentation-image-list": "2",
                            "--resolution-level": "4",
                        },
                    }],
                },
            }

            jobs = dmap_sweep.build_jobs(arguments(root), config, root / "experiment")
            by_scene = {job.scene_id: job for job in jobs}
            self.assertEqual(
                by_scene["scene-a"].argument_overrides,
                {
                    "--dmap-instrumentation-image-list": "1,3",
                    "--resolution-level": "4",
                },
            )
            self.assertEqual(
                by_scene["scene-b"].argument_overrides["--dmap-instrumentation-image-list"],
                "2",
            )
            self.assertTrue(by_scene["scene-a"].retention_policy["lossy"])
            command, _work, _binary, _ini = dmap_sweep.build_capture_command(
                arguments(root), config, by_scene["scene-a"]
            )
            self.assertEqual(
                dmap_sweep.argument_value(
                    command, "--dmap-instrumentation-image-list", "missing"
                ),
                "1,3",
            )
            ledger = dmap_sweep.initialize_ledger(
                root / "schedule.json", {"sha256": "a" * 64}, jobs, arguments(root)
            )
            self.assertEqual(
                ledger["jobs"][by_scene["scene-a"].job_id]["argument_overrides"]
                ["--dmap-instrumentation-image-list"],
                "1,3",
            )
            self.assertTrue(
                ledger["jobs"][by_scene["scene-a"].job_id]["retention_policy"]["lossy"]
            )
            identity = dmap_sweep.identity_record(
                arguments(root), config, jobs, {"sha256": "b" * 64}
            )
            matrix = {
                row["scene"]: row for row in identity["selection"]["job_matrix"]
            }
            self.assertEqual(
                matrix["scene-a"]["argument_overrides"]
                ["--dmap-instrumentation-image-list"],
                "1,3",
            )
            self.assertTrue(matrix["scene-a"]["retention_policy"]["lossy"])
            self.assertEqual(
                matrix["scene-a"]["timeout_seconds"],
                by_scene["scene-a"].timeout_seconds,
            )
            self.assertEqual(
                matrix["scene-a"]["estimated_output_bytes"],
                by_scene["scene-a"].estimated_output_bytes,
            )
            timeout_changed = [
                replace(job, timeout_seconds=job.timeout_seconds + 1.0)
                if job.scene_id == "scene-a"
                else job
                for job in jobs
            ]
            estimate_changed = [
                replace(
                    job,
                    estimated_output_bytes=job.estimated_output_bytes + 1,
                )
                if job.scene_id == "scene-a"
                else job
                for job in jobs
            ]
            self.assertNotEqual(
                identity["sha256"],
                dmap_sweep.identity_record(
                    arguments(root), config, timeout_changed, {"sha256": "b" * 64}
                )["sha256"],
            )
            self.assertNotEqual(
                identity["sha256"],
                dmap_sweep.identity_record(
                    arguments(root), config, estimate_changed, {"sha256": "b" * 64}
                )["sha256"],
            )

    def test_schedule_identity_binds_generated_phase_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text("schema_version: 2\n", encoding="utf-8")
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config = {
                "_config_path": str(config_path),
                "experiment_id": "phase",
                "output_root": str(root / "output"),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "experiment_phase": {
                    "phase_id": "confirmation",
                    "lineage_sha256": "a" * 64,
                    "selection_manifest": {
                        "path": str(root / "selection.json"),
                        "sha256": "b" * 64,
                    },
                },
            }
            experiment_root = dmap_sweep.dmap_dev.experiment_root(config)
            phase_root = dmap_sweep.dmap_dev.experiment_phase_evidence_dir(
                experiment_root, "confirmation"
            )
            phase_root.mkdir(parents=True)
            (phase_root / "00_phase_lock.json").write_text("{}\n", encoding="utf-8")
            (phase_root / "01_resolved_phase.yaml").write_text(
                "phase: confirmation\n", encoding="utf-8"
            )
            experiment_root.joinpath("00_experiment_lock.json").write_text(
                "{}\n", encoding="utf-8"
            )
            job = sweep_job(root)

            first = dmap_sweep.identity_record(
                arguments(root), config, [job], {"sha256": "c" * 64}
            )
            (phase_root / "00_phase_lock.json").write_text(
                '{"changed":true}\n', encoding="utf-8"
            )
            changed = dmap_sweep.identity_record(
                arguments(root), config, [job], {"sha256": "c" * 64}
            )

        self.assertEqual(
            first["experiment_phase"]["lineage_sha256"], "a" * 64
        )
        self.assertTrue(first["experiment_phase"]["phase_lock"]["exists"])
        self.assertNotEqual(first["sha256"], changed["sha256"])

    def test_scene_argument_overrides_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_scene = scene("scene-a", root)
            invalid_scene["argument_overrides"] = ["--not-a-mapping"]
            config = {
                "capture_profiles": ["summary"],
                "suite": {"name": "development", "scan_ids": ["scene-a"]},
                "scenes": [invalid_scene],
                "runs": [run("candidate-a", family="family-a")],
            }
            with self.assertRaisesRegex(ValueError, "scene scene-a must be a mapping"):
                dmap_sweep.build_jobs(arguments(root), config, root / "experiment")

            invalid_scene["argument_overrides"] = {"--bad option": "1"}
            with self.assertRaisesRegex(ValueError, "invalid argument override for scene"):
                dmap_sweep.build_jobs(arguments(root), config, root / "experiment")

    def test_summary_retention_policy_is_opt_in_and_never_applies_to_endpoint_or_deep(self) -> None:
        default = dmap_sweep.retention_policy_for_profile({}, "summary")
        self.assertEqual(default["dmap_policy"], dmap_sweep.DMAP_POLICY_COMPLETE)
        self.assertFalse(default["lossy"])
        config = {"sweep": {"retention": {
            "schema_version": 1,
            "summary_dmap_policy": "instrumented_image_ids_only",
        }}}
        summary = dmap_sweep.retention_policy_for_profile(config, "summary")
        endpoint = dmap_sweep.retention_policy_for_profile(config, "endpoint")
        deep = dmap_sweep.retention_policy_for_profile(config, "deep")
        self.assertTrue(summary["lossy"])
        self.assertEqual(
            summary["dmap_policy"], dmap_sweep.DMAP_POLICY_INSTRUMENTED_IMAGE_IDS_ONLY
        )
        self.assertEqual(endpoint["dmap_policy"], dmap_sweep.DMAP_POLICY_COMPLETE)
        self.assertEqual(deep["dmap_policy"], dmap_sweep.DMAP_POLICY_COMPLETE)
        with self.assertRaisesRegex(ValueError, "unsupported.*summary_dmap_policy"):
            dmap_sweep.retention_policy_for_profile(
                {"sweep": {"retention": {"summary_dmap_policy": "typo"}}}, "summary"
            )

    def test_accuracy_promotion_uses_rank_and_family_without_hiding_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for label, family in (("a1", "a"), ("a2", "a"), ("b1", "b")):
                job = sweep_job(root, job_id=f"job-{label}", stage="confirm")
                object.__setattr__(job, "run_label", label)
                object.__setattr__(job, "family", family)
                jobs.append(job)
            ledger = {
                "jobs": {
                    f"screen-{label}": {
                        "run": label, "stage": "screen", "status": "complete",
                    }
                    for label in ("a1", "a2", "b1")
                },
                "promotions": {},
            }
            accuracy = {
                "available": True,
                "sha256": "f" * 64,
                "rows": [
                    {"candidate": "a2", "accuracy_rank": 1, "availability_biased": "True"},
                    {"candidate": "a1", "accuracy_rank": 2, "availability_biased": "False"},
                    {"candidate": "b1", "accuracy_rank": 3, "availability_biased": "False"},
                ],
            }

            result = dmap_sweep.select_promoted_runs(
                target_stage="confirm",
                source_stage="screen",
                policy={"family_winners": True, "top_n": 2, "required": True},
                target_jobs=jobs,
                accuracy=accuracy,
                ledger=ledger,
            )

            self.assertEqual(result["selected_runs"], ["a2", "b1"])
            self.assertEqual(result["not_promoted"], ["a1"])
            self.assertFalse(result["candidates_hidden"])
            self.assertTrue(result["ranked_candidates"][0]["availability_biased"])

    def test_finalist_and_endpoint_eligibility_requires_completed_prior_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            finalist_jobs = []
            for label in ("external-good", "external-screen-only", "internal-good"):
                job = sweep_job(root, job_id=label, stage="finalists")
                object.__setattr__(job, "run_label", label)
                finalist_jobs.append(job)
            ledger = {
                "jobs": {
                    "external-confirm": {"run": "external-good", "stage": "external_confirm", "status": "complete"},
                    "external-screen": {"run": "external-screen-only", "stage": "external_screen", "status": "complete"},
                    "internal-confirm": {"run": "internal-good", "stage": "internal_confirm", "status": "complete"},
                },
                "promotions": {
                    "external_confirm": {"target_stage": "external_confirm", "selected_runs": ["external-good"]},
                    "internal_confirm": {"target_stage": "internal_confirm", "selected_runs": ["internal-good"]},
                },
            }
            accuracy = {"available": True, "sha256": "a", "rows": [
                {"candidate": "external-screen-only", "accuracy_rank": 1},
                {"candidate": "external-good", "accuracy_rank": 2},
                {"candidate": "internal-good", "accuracy_rank": 3},
            ]}

            finalists = dmap_sweep.select_promoted_runs(
                target_stage="finalists", source_stage="internal_confirm",
                policy={
                    "top_n": 2,
                    "required": True,
                    "eligible_stages": ["external_confirm", "internal_confirm"],
                },
                target_jobs=finalist_jobs,
                accuracy=accuracy, ledger=ledger,
            )

            self.assertEqual(finalists["selected_runs"], ["external-good", "internal-good"])
            self.assertIn("external-screen-only", finalists["ineligible_candidates"])
            ledger["jobs"]["finalist-complete"] = {
                "run": "internal-good", "stage": "finalists", "status": "complete",
            }
            endpoint = dmap_sweep.select_promoted_runs(
                target_stage="endpoint", source_stage="finalists",
                policy={"top_n": 1, "required": True}, target_jobs=finalist_jobs,
                accuracy=accuracy, ledger=ledger,
            )
            self.assertEqual(endpoint["selected_runs"], ["internal-good"])

    def test_deep_promotion_adds_control_to_endpoint_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deep_jobs = []
            for label in ("control", "candidate-a", "candidate-b"):
                job = sweep_job(root, job_id=f"deep-{label}", stage="deep")
                object.__setattr__(job, "run_label", label)
                if label == "control":
                    object.__setattr__(job, "always_run", True)
                    object.__setattr__(job, "family", "control")
                deep_jobs.append(job)
            ledger = {"jobs": {
                "endpoint-a": {
                    "run": "candidate-a", "stage": "endpoint", "status": "complete",
                },
            }, "promotions": {}}
            accuracy = {"available": True, "sha256": "a", "rows": [
                {"candidate": "candidate-a", "accuracy_rank": 1},
                {"candidate": "candidate-b", "accuracy_rank": 2},
            ]}

            promoted = dmap_sweep.select_promoted_runs(
                target_stage="deep", source_stage="endpoint",
                policy={"top_n": 1, "required": True}, target_jobs=deep_jobs,
                accuracy=accuracy, ledger=ledger,
            )

            self.assertEqual(promoted["selected_runs"], ["candidate-a", "control"])
            self.assertEqual(promoted["always_run"], ["control"])
            self.assertIn("candidate-b", promoted["ineligible_candidates"])

    def test_scene_failed_candidate_is_ineligible_for_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_a = sweep_job(root, job_id="target-a", stage="confirm")
            object.__setattr__(candidate_a, "run_label", "candidate-a")
            candidate_b = sweep_job(root, job_id="target-b", stage="confirm")
            object.__setattr__(candidate_b, "run_label", "candidate-b")
            scene_failure = {
                "kind": dmap_sweep.SCENE_FAILURE_ZERO_NEIGHBOR_VIEWS,
                "scope": "scene",
                "terminal": True,
                "continuable": True,
                "promotion_eligible": False,
            }
            ledger = {"jobs": {
                "a-complete": {
                    "run": "candidate-a", "stage": "screen", "status": "complete",
                },
                "a-scene-failed": {
                    "run": "candidate-a", "stage": "screen", "status": "scene_failed",
                    "scene_failure": scene_failure,
                },
                "b-complete": {
                    "run": "candidate-b", "stage": "screen", "status": "complete",
                },
            }, "promotions": {}}
            accuracy = {
                "available": True,
                "rows": [
                    {"candidate": "candidate-a", "accuracy_rank": 1},
                    {"candidate": "candidate-b", "accuracy_rank": 2},
                ],
            }

            promoted = dmap_sweep.select_promoted_runs(
                target_stage="confirm",
                source_stage="screen",
                policy={"top_n": 2, "required": True},
                target_jobs=[candidate_a, candidate_b],
                accuracy=accuracy,
                ledger=ledger,
            )

            self.assertEqual(promoted["selected_runs"], ["candidate-b"])
            self.assertEqual(promoted["scene_failed_candidates"], ["candidate-a"])
            self.assertIn("candidate-a", promoted["ineligible_candidates"])

    def test_ini_override_is_generated_without_modifying_frozen_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "frozen" / "Densify.ini"
            source.parent.mkdir()
            source.write_text("[Densify]\nIters = 10\nViews = 15\n", encoding="utf-8")
            before = dmap_sweep.sha256_file(source)
            destination = root / "run" / "generated" / "Densify.sweep.ini"

            metadata = dmap_sweep.render_ini_override(
                source, destination, {"Iters": "12", "New Value": "3"}
            )

            self.assertEqual(dmap_sweep.sha256_file(source), before)
            self.assertEqual(metadata["source_sha256"], before)
            rendered = destination.read_text(encoding="utf-8")
            self.assertIn("Iters = 12", rendered)
            self.assertIn("New Value = 3", rendered)

    def test_stage_argument_overrides_replace_run_arguments_and_enter_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            object.__setattr__(job, "argument_overrides", {
                "--resolution-level": "4",
                "--max-resolution": "640",
            })
            config = {
                "_config_path": str(root / "config.yaml"),
                "densify_bin": str(root / "DensifyPointCloud"),
                "densify_observe_bin": str(root / "DensifyPointCloudDMapObserve"),
                "default_densify_args": [
                    "--resolution-level", "0", "--max-resolution", "2560",
                    "--dense-config-file", "Densify.ini",
                ],
            }

            command, _work, _binary, _ini = dmap_sweep.build_capture_command(
                arguments(root), config, job
            )

            self.assertEqual(
                dmap_sweep.argument_value(command, "--resolution-level", "missing"), "4"
            )
            self.assertEqual(
                dmap_sweep.argument_value(command, "--max-resolution", "missing"), "640"
            )

    def test_prefilter_profile_uses_bounded_observer_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            object.__setattr__(job, "profile", "prefilter")
            object.__setattr__(job, "mode", "prefilter")
            config = {
                "_config_path": str(root / "config.yaml"),
                "densify_bin": str(root / "DensifyPointCloud"),
                "densify_observe_bin": str(root / "DensifyPointCloudDMapObserve"),
                "default_densify_args": ["--fusion-mode", "1"],
            }

            command, _work, binary, _ini = dmap_sweep.build_capture_command(
                arguments(root), config, job
            )

            self.assertEqual(binary, Path(config["densify_observe_bin"]).resolve())
            self.assertEqual(
                dmap_sweep.argument_value(
                    command, "--dmap-instrumentation-level", "missing"
                ),
                "prefilter",
            )
            self.assertEqual(
                dmap_sweep.argument_value(
                    command, "--dmap-instrumentation-write-maps", "missing"
                ),
                "0",
            )

    def test_timeout_terminates_and_kills_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
            ]

            result = dmap_sweep.run_process_group(
                command,
                cwd=root,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                timeout_seconds=0.15,
                term_grace_seconds=0.05,
                heartbeat_seconds=0.05,
                heartbeat_path=root / "heartbeat.json",
                job_id="timeout-test",
                stop_event=threading.Event(),
            )

            self.assertTrue(result.timed_out)
            self.assertTrue(result.terminated)
            self.assertTrue(result.killed)
            self.assertLess(result.elapsed_seconds, 2.0)

    def test_flock_prevents_two_schedulers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / ".sweep.lock"
            with dmap_sweep.SweepLock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "another sweep"):
                    with dmap_sweep.SweepLock(lock_path):
                        pass

    def test_campaign_report_lock_serializes_writers_and_allows_readers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "reports" / ".campaign_report.lock"
            with dmap_sweep.CampaignReportLock(lock_path, shared=True):
                with dmap_sweep.CampaignReportLock(lock_path, shared=True):
                    pass
                with self.assertRaisesRegex(RuntimeError, "campaign report"):
                    with dmap_sweep.CampaignReportLock(lock_path):
                        pass
            with dmap_sweep.CampaignReportLock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "campaign report"):
                    with dmap_sweep.CampaignReportLock(lock_path, shared=True):
                        pass

    def test_stage_and_final_reports_use_the_same_campaign_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = dmap_sweep.campaign_report_lock_path(root)
            args = arguments(root)
            common = (
                args,
                {"_config_path": str(root / "config.yaml")},
                root,
            )
            with (
                mock.patch.object(
                    dmap_sweep,
                    "_generate_final_report_unlocked",
                    return_value=True,
                ) as final_unlocked,
                mock.patch.object(
                    dmap_sweep,
                    "_generate_stage_report_unlocked",
                    return_value={"status": "complete"},
                ) as stage_unlocked,
            ):
                dmap_sweep.generate_final_report(
                    *common,
                    root / "report",
                    {},
                    root / "schedule.json",
                    root / "heartbeat.json",
                    threading.Event(),
                )
                dmap_sweep.generate_stage_report(
                    *common,
                    {"name": "stage"},
                    0,
                    {},
                    root / "schedule.json",
                    root / "heartbeat.json",
                    threading.Event(),
                )

            self.assertTrue(lock_path.is_file())
            final_unlocked.assert_called_once()
            stage_unlocked.assert_called_once()

    def test_resume_rejects_changed_immutable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "schedule.json"
            job = sweep_job(root)
            first = {"sha256": "a" * 64}
            dmap_sweep.initialize_ledger(path, first, [job], arguments(root))

            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                dmap_sweep.initialize_ledger(
                    path, {"sha256": "b" * 64}, [job], arguments(root)
                )

    def test_plan_only_can_execute_but_scheduler_policy_cannot_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "schedule.json"
            report_dir = root / "reports" / "canonical"
            job = sweep_job(root)
            identity = {"sha256": "a" * 64}
            planned = arguments(root, plan_only=True)
            dmap_sweep.initialize_ledger(
                path, identity, [job], planned, default_report_dir=report_dir
            )

            resumed = dmap_sweep.initialize_ledger(
                path,
                identity,
                [job],
                arguments(root, plan_only=False, report_dir=report_dir),
                default_report_dir=report_dir,
            )
            self.assertEqual(resumed["status"], "planned")

            changes = {
                "budget_hours": 13.0,
                "capture_budget_hours": 1.0,
                "free_space_floor_gb": 1.0,
                "finalization_reserve_minutes": 1.0,
                "finalization_reserve_gb": 1.0,
                "report_timeout_minutes": 2.0,
                "retry_transient_failures": 0,
                "compact": False,
                "generate_report": True,
                "require_accuracy_ledger": True,
                "report_dir": root / "reports" / "different",
            }
            for field, value in changes.items():
                with self.subTest(field=field):
                    with self.assertRaisesRegex(RuntimeError, "sweep policy changed"):
                        dmap_sweep.initialize_ledger(
                            path,
                            identity,
                            [job],
                            arguments(root, **{field: value}),
                            default_report_dir=report_dir,
                        )

            operational = dmap_sweep.initialize_ledger(
                path,
                identity,
                [job],
                arguments(root, heartbeat_seconds=2.0, term_grace_seconds=3.0),
                default_report_dir=report_dir,
            )
            self.assertEqual(operational["policy"]["report_dir"], str(report_dir.resolve()))

    def test_uncapped_scheduler_policy_matches_legacy_hash_fixture(self) -> None:
        policy = dmap_sweep.scheduler_policy(
            arguments(Path("/tmp/legacy-policy-fixture"))
        )

        self.assertEqual(policy, {
            "schema_version": 1,
            "budget_hours": 12.0,
            "job_timeout_minutes": None,
            "endpoint_timeout_minutes": None,
            "summary_timeout_minutes": None,
            "deep_timeout_minutes": None,
            "report_timeout_minutes": 60.0,
            "free_space_floor_gb": 0.0,
            "finalization_reserve_minutes": 0.0,
            "finalization_reserve_gb": 0.0,
            "default_job_output_gb": 24.0,
            "retry_transient_failures": 1,
            "report_dir": None,
            "generate_report": False,
            "compact": True,
            "require_accuracy_ledger": False,
            "sha256": "e0b619ebb78909434561debe07bd2be2c2d87ae0405d68b897e5cde16cb22fbb",
        })
        self.assertNotIn("capture_budget_hours", policy)

    def test_campaign_budget_is_cumulative_across_sessions(self) -> None:
        ledger = {
            "sessions": [
                {"status": "incomplete", "elapsed_seconds": 1800.0},
                {"status": "failed", "elapsed_seconds": 900.0},
            ],
        }

        remaining = dmap_sweep.remaining_campaign_budget_seconds(
            arguments(Path("/tmp"), budget_hours=1.0), ledger
        )

        self.assertEqual(remaining, 900.0)

    def test_persisted_wall_elapsed_mutations_fail_closed(self) -> None:
        invalid = (True, math.nan, math.inf, -1.0, "900")
        for elapsed in invalid:
            with self.subTest(elapsed=elapsed):
                ledger = {"sessions": [{
                    "status": "failed", "elapsed_seconds": elapsed,
                }]}
                with self.assertRaises(ValueError):
                    dmap_sweep.remaining_campaign_budget_seconds(
                        arguments(Path("/tmp"), budget_hours=1.0), ledger
                    )
        with self.assertRaisesRegex(ValueError, "missing elapsed"):
            dmap_sweep.cumulative_elapsed_seconds({
                "sessions": [{"status": "complete"}],
            })
        self.assertEqual(dmap_sweep.cumulative_elapsed_seconds({
            "sessions": [{"status": "running"}],
        }), 0.0)

    def test_capture_supervision_numeric_arguments_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = (
                ("budget_hours", math.nan),
                ("budget_hours", True),
                ("capture_budget_hours", math.nan),
                ("capture_budget_hours", math.inf),
                ("capture_budget_hours", True),
                ("term_grace_seconds", math.nan),
                ("term_grace_seconds", -1.0),
                ("term_grace_seconds", True),
                ("heartbeat_seconds", math.nan),
                ("heartbeat_seconds", 0.0),
                ("finalization_reserve_minutes", math.nan),
                ("finalization_reserve_minutes", -1.0),
                ("job_timeout_minutes", math.nan),
                ("endpoint_timeout_minutes", True),
                ("report_timeout_minutes", math.inf),
                ("free_space_floor_gb", math.nan),
                ("finalization_reserve_gb", True),
                ("default_job_output_gb", -1.0),
                ("retry_transient_failures", True),
            )
            for field, value in invalid:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        dmap_sweep.run_sweep(arguments(root, **{field: value}))

    def test_configured_timeouts_and_output_estimates_fail_closed(self) -> None:
        root = Path("/tmp/configured-numeric-validation")
        args = arguments(root)
        for value in (math.nan, math.inf, -1.0, 0.0, True, "15"):
            with self.subTest(timeout=value):
                with self.assertRaisesRegex(ValueError, "timeout"):
                    dmap_sweep.profile_timeout_seconds(
                        args,
                        {"sweep": {"profile_timeout_minutes": {"deep": value}}},
                        "deep",
                    )
        for value in (math.nan, math.inf, -1.0, True, "24"):
            with self.subTest(output=value):
                with self.assertRaisesRegex(ValueError, "output estimate"):
                    dmap_sweep.initial_output_estimate_bytes(
                        args,
                        {"sweep": {"profile_estimated_output_gb": {"deep": value}}},
                        {},
                        "deep",
                    )

    def test_capture_budget_is_separate_persisted_attempt_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "schedule.json"
            job = sweep_job(root)
            args = arguments(root, budget_hours=4.25, capture_budget_hours=2.5)
            ledger = dmap_sweep.initialize_ledger(
                path, {"sha256": "a" * 64}, [job], args
            )
            self.assertEqual(ledger["policy"]["budget_hours"], 4.25)
            self.assertEqual(ledger["policy"]["capture_budget_hours"], 2.5)
            self.assertEqual(ledger["capture_budget"], {
                "enforced": True,
                "limit_seconds": 9000.0,
                "elapsed_seconds": 0.0,
                "remaining_seconds": 9000.0,
                "valid": True,
            })
            ledger["jobs"][job.job_id]["attempts"] = [{
                "status": "complete", "elapsed_seconds": 125.0,
            }]
            dmap_sweep.save_ledger(path, ledger)
            self.assertEqual(ledger["capture_budget"]["elapsed_seconds"], 125.0)
            self.assertEqual(ledger["capture_budget"]["remaining_seconds"], 8875.0)

    def test_capture_attempt_without_elapsed_fails_closed(self) -> None:
        ledger = {"jobs": {"job-a": {"attempts": [{"status": "running"}]}}}
        with self.assertRaisesRegex(ValueError, "missing elapsed time"):
            dmap_sweep.cumulative_capture_elapsed_seconds(ledger)

    def test_capture_admission_blocks_only_when_cumulative_cap_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            args = arguments(root, capture_budget_hours=2.5)
            ledger = {"jobs": {"prior": {"attempts": [{"elapsed_seconds": 8990.0}]}}}
            with mock.patch.object(
                dmap_sweep.shutil, "disk_usage",
                return_value=mock.Mock(total=10**9, used=0, free=10**9),
            ):
                allowed = dmap_sweep.admission_reason(
                    args, ledger, job, time.monotonic() + job.timeout_seconds + 1.0, root
                )
            self.assertIsNone(allowed)
            ledger["jobs"]["prior"]["attempts"][0]["elapsed_seconds"] = 8995.0
            blocked = dmap_sweep.admission_reason(
                args, ledger, job, time.monotonic() + 10000.0, root
            )
            self.assertIn("capture admission failed", str(blocked))

    def test_capture_limited_attempt_uses_hard_remaining_time_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "schedule.json"
            heartbeat = root / "heartbeat.json"
            job = sweep_job(root)
            args = arguments(
                root, capture_budget_hours=2.5, compact=False,
                term_grace_seconds=120.0,
            )
            ledger = dmap_sweep.initialize_ledger(
                path, {"sha256": "a" * 64}, [job], args
            )
            ledger["jobs"]["prior"] = {
                "attempts": [{"status": "complete", "elapsed_seconds": 8900.0}]
            }

            def successful_attempt(*_args, **_kwargs):
                job.run_dir.mkdir(parents=True, exist_ok=True)
                return dmap_sweep.ProcessResult(0, 90.0, False, False, False, False), True, "valid"

            with mock.patch.object(
                dmap_sweep, "execute_capture_attempt", side_effect=successful_attempt
            ) as execute:
                completed = dmap_sweep.run_job(
                    args, {}, root, job, ledger, path, heartbeat, threading.Event()
                )

            self.assertTrue(completed)
            self.assertEqual(execute.call_args.kwargs["timeout_seconds"], 95.0)
            self.assertEqual(execute.call_args.kwargs["term_grace_seconds"], 0.0)
            self.assertTrue(execute.call_args.kwargs["capture_budget_limited"])
            attempt = ledger["jobs"][job.job_id]["attempts"][0]
            self.assertEqual(attempt["capture_budget_remaining_before_seconds"], 100.0)
            self.assertEqual(attempt["effective_term_grace_seconds"], 0.0)
            self.assertEqual(ledger["capture_budget"]["elapsed_seconds"], 8990.0)
            self.assertTrue(ledger["capture_budget"]["valid"])

    def test_capture_overrun_is_terminal_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "schedule.json"
            job = sweep_job(root)
            args = arguments(
                root, capture_budget_hours=2.5, compact=False,
                retry_transient_failures=0,
            )
            ledger = dmap_sweep.initialize_ledger(
                path, {"sha256": "a" * 64}, [job], args
            )
            ledger["jobs"]["prior"] = {
                "attempts": [{"status": "complete", "elapsed_seconds": 8900.0}]
            }

            def overrunning_attempt(*_args, **_kwargs):
                job.run_dir.mkdir(parents=True, exist_ok=True)
                return dmap_sweep.ProcessResult(0, 101.0, False, False, False, False), True, "valid"

            with mock.patch.object(
                dmap_sweep, "execute_capture_attempt", side_effect=overrunning_attempt
            ):
                completed = dmap_sweep.run_job(
                    args, {}, root, job, ledger, path, root / "heartbeat.json",
                    threading.Event(),
                )

            self.assertFalse(completed)
            self.assertEqual(ledger["jobs"][job.job_id]["status"], "failed")
            self.assertFalse(ledger["jobs"][job.job_id]["attempts"][0]["transient"])
            self.assertFalse(ledger["capture_budget"]["valid"])
            self.assertEqual(ledger["capture_budget"]["elapsed_seconds"], 9001.0)

    def test_running_budget_snapshot_uses_monotonic_session_elapsed(self) -> None:
        ledger = {
            "sessions": [
                {"status": "complete", "elapsed_seconds": 100.0},
                {
                    "status": "running",
                    "pid": os.getpid(),
                    "started_epoch": 1_000.0,
                    "started_monotonic": 50.0,
                },
            ],
            "budget": {
                "limit_seconds": 1_000.0,
                "elapsed_seconds": 100.0,
                "remaining_seconds": 900.0,
            },
        }

        updated = dmap_sweep.refresh_running_budget(
            ledger, now_epoch=2_000.0, now_monotonic=75.0
        )

        self.assertTrue(updated)
        self.assertEqual(ledger["sessions"][1]["elapsed_seconds"], 25.0)
        self.assertEqual(ledger["budget"]["elapsed_seconds"], 125.0)
        self.assertEqual(ledger["budget"]["remaining_seconds"], 875.0)

    def test_process_heartbeat_persists_live_budget_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule_path = root / "schedule.json"
            started_monotonic = time.monotonic()
            dmap_sweep.atomic_write_json(schedule_path, {
                "sessions": [{
                    "status": "running",
                    "pid": os.getpid(),
                    "started_epoch": time.time(),
                    "started_monotonic": started_monotonic,
                }],
                "budget": {
                    "limit_seconds": 60.0,
                    "elapsed_seconds": 0.0,
                    "remaining_seconds": 60.0,
                },
            })

            result = dmap_sweep.run_process_group(
                [sys.executable, "-c", "import time; time.sleep(0.15)"],
                cwd=root,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                timeout_seconds=2.0,
                term_grace_seconds=0.1,
                heartbeat_seconds=0.05,
                heartbeat_path=root / "heartbeat.json",
                job_id="budget-heartbeat-test",
                stop_event=threading.Event(),
                schedule_path=schedule_path,
            )

            persisted = json.loads(schedule_path.read_text(encoding="utf-8"))
            elapsed = persisted["budget"]["elapsed_seconds"]
            self.assertEqual(result.return_code, 0)
            self.assertGreater(elapsed, 0.1)
            self.assertAlmostEqual(
                persisted["budget"]["remaining_seconds"], 60.0 - elapsed
            )

    def test_process_heartbeat_persists_live_capture_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule_path = root / "schedule.json"
            job_id = "capture-heartbeat-test"
            dmap_sweep.atomic_write_json(schedule_path, {
                "policy": {"capture_budget_hours": 1.0},
                "sessions": [{
                    "status": "running",
                    "pid": os.getpid(),
                    "started_epoch": time.time(),
                    "started_monotonic": time.monotonic(),
                }],
                "budget": {
                    "limit_seconds": 7200.0,
                    "elapsed_seconds": 0.0,
                    "remaining_seconds": 7200.0,
                },
                "jobs": {job_id: {"attempts": [{
                    "attempt": 1,
                    "status": "running",
                    "elapsed_seconds": 0.0,
                    "configured_timeout_seconds": 60.0,
                    "effective_timeout_seconds": 60.0,
                }]}},
            })

            result = dmap_sweep.run_process_group(
                [sys.executable, "-c", "import time; time.sleep(0.15)"],
                cwd=root,
                stdout_path=root / "stdout.log",
                stderr_path=root / "stderr.log",
                timeout_seconds=2.0,
                term_grace_seconds=0.1,
                heartbeat_seconds=0.05,
                heartbeat_path=root / "heartbeat.json",
                job_id=job_id,
                stop_event=threading.Event(),
                schedule_path=schedule_path,
            )

            persisted = json.loads(schedule_path.read_text(encoding="utf-8"))
            attempt_elapsed = persisted["jobs"][job_id]["attempts"][0][
                "elapsed_seconds"
            ]
            self.assertEqual(result.return_code, 0)
            self.assertGreater(attempt_elapsed, 0.1)
            self.assertAlmostEqual(
                persisted["capture_budget"]["elapsed_seconds"], attempt_elapsed
            )
            self.assertAlmostEqual(
                persisted["capture_budget"]["remaining_seconds"],
                3600.0 - attempt_elapsed,
            )
            attempt = persisted["jobs"][job_id]["attempts"][0]
            self.assertTrue(attempt["supervisor_prepared"])
            self.assertEqual(attempt["supervisor_kind"], "coreutils_timeout")
            self.assertGreater(attempt["supervisor_not_after_epoch"], time.time())
            self.assertEqual(attempt["process_identity"]["pid"], result.process_identity["pid"])

    def test_heartbeat_failure_terminates_spawned_process_group(self) -> None:
        class FakeProcess:
            pid = 424242
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -signal.SIGTERM
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = FakeProcess()
            with (
                mock.patch.object(subprocess, "Popen", return_value=process),
                mock.patch.object(
                    dmap_sweep, "linux_process_identity",
                    return_value={"pid": process.pid},
                ),
                mock.patch.object(
                    dmap_sweep, "write_heartbeat", side_effect=OSError("disk full")
                ) as heartbeat,
                mock.patch.object(os, "killpg") as killpg,
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    dmap_sweep.run_process_group(
                        ["fake-capture"],
                        cwd=root,
                        stdout_path=root / "stdout.log",
                        stderr_path=root / "stderr.log",
                        timeout_seconds=60.0,
                        term_grace_seconds=1.0,
                        heartbeat_seconds=1.0,
                        heartbeat_path=root / "heartbeat.json",
                        job_id="fault-injection",
                        stop_event=threading.Event(),
                    )

            killpg.assert_called_once_with(process.pid, signal.SIGTERM)
            self.assertEqual(process.returncode, -signal.SIGTERM)
            self.assertEqual(heartbeat.call_count, 2)

    def test_schedule_heartbeat_failure_terminates_spawned_process_group(self) -> None:
        class FakeProcess:
            pid = 434343
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = -signal.SIGTERM
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = FakeProcess()
            with (
                mock.patch.object(subprocess, "Popen", return_value=process),
                mock.patch.object(
                    dmap_sweep, "linux_process_identity",
                    return_value={"pid": process.pid},
                ),
                mock.patch.object(dmap_sweep, "write_heartbeat"),
                mock.patch.object(
                    dmap_sweep, "refresh_schedule_budget",
                    side_effect=[True, OSError("schedule write failed"), True],
                ) as refresh,
                mock.patch.object(os, "killpg") as killpg,
            ):
                with self.assertRaisesRegex(OSError, "schedule write failed"):
                    dmap_sweep.run_process_group(
                        ["fake-capture"],
                        cwd=root,
                        stdout_path=root / "stdout.log",
                        stderr_path=root / "stderr.log",
                        timeout_seconds=60.0,
                        term_grace_seconds=1.0,
                        heartbeat_seconds=1.0,
                        heartbeat_path=root / "heartbeat.json",
                        job_id="fault-injection",
                        stop_event=threading.Event(),
                        schedule_path=root / "schedule.json",
                    )

            killpg.assert_called_once_with(process.pid, signal.SIGTERM)
            self.assertEqual(process.returncode, -signal.SIGTERM)
            self.assertEqual(refresh.call_count, 3)

    def test_orphaned_capture_attempt_charges_effective_timeout_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heartbeat = root / "heartbeat.json"
            heartbeat.write_text(json.dumps({
                "job_id": "job-a",
                "elapsed_seconds": 45.0,
            }), encoding="utf-8")
            ledger = {
                "policy": {"capture_budget_hours": 1.0},
                "jobs": {"job-a": {
                    "status": "running",
                    "timeout_seconds": 900.0,
                    "attempts": [{
                        "attempt": 1,
                        "status": "running",
                        "elapsed_seconds": 30.0,
                        "configured_timeout_seconds": 900.0,
                        "effective_timeout_seconds": 120.0,
                        "effective_term_grace_seconds": 10.0,
                        "process_identity": {"pid": 1234},
                    }],
                }},
            }

            with mock.patch.object(
                dmap_sweep, "terminate_verified_orphan_process_group",
                return_value={"status": "killed", "pid": 1234},
            ):
                recovered = dmap_sweep.recover_interrupted_capture_attempts(
                    ledger, heartbeat
                )

            attempt = ledger["jobs"]["job-a"]["attempts"][0]
            self.assertEqual(recovered[0]["charged_seconds"], 130.0)
            self.assertEqual(attempt["status"], "interrupted")
            self.assertEqual(attempt["observed_elapsed_seconds"], 45.0)
            self.assertEqual(attempt["elapsed_seconds"], 130.0)
            self.assertEqual(ledger["jobs"]["job-a"]["status"], "failed")
            self.assertEqual(ledger["capture_budget"]["elapsed_seconds"], 130.0)
            self.assertTrue(ledger["capture_budget"]["valid"])
            job = sweep_job(root, job_id="job-a")
            with mock.patch.object(dmap_sweep, "execute_capture_attempt") as execute:
                self.assertFalse(dmap_sweep.run_job(
                    arguments(root, capture_budget_hours=1.0), {}, root, job,
                    ledger, root / "schedule.json", heartbeat,
                    threading.Event(),
                ))
            execute.assert_not_called()

    def test_orphaned_capture_without_bound_consumes_cap_fail_closed(self) -> None:
        ledger = {
            "policy": {"capture_budget_hours": 1.0},
            "jobs": {"job-a": {
                "status": "running",
                "attempts": [{
                    "attempt": 1,
                    "status": "running",
                    "elapsed_seconds": 10.0,
                    "process_identity": {"pid": 1234},
                }],
            }},
        }

        with mock.patch.object(
            dmap_sweep, "terminate_verified_orphan_process_group",
            return_value={"status": "killed", "pid": 1234},
        ):
            dmap_sweep.recover_interrupted_capture_attempts(
                ledger, Path("/definitely/missing/heartbeat.json")
            )

        attempt = ledger["jobs"]["job-a"]["attempts"][0]
        self.assertEqual(attempt["elapsed_seconds"], 3600.0)
        self.assertEqual(ledger["capture_budget"]["remaining_seconds"], 0.0)
        self.assertTrue(ledger["capture_budget"]["valid"])

    def test_legacy_uncapped_orphan_remains_recoverable_with_signed_time(self) -> None:
        ledger = {
            "policy": {"retry_transient_failures": 1},
            "jobs": {"job-a": {
                "status": "running",
                "timeout_seconds": 60.0,
                "attempts": [{
                    "attempt": 1,
                    "status": "running",
                    "elapsed_seconds": 10.0,
                    "effective_timeout_seconds": 60.0,
                    "effective_term_grace_seconds": 5.0,
                    "process_identity": {"pid": 1234},
                }],
            }},
        }

        with mock.patch.object(
            dmap_sweep, "terminate_verified_orphan_process_group",
            return_value={"status": "killed", "pid": 1234},
        ):
            recovered = dmap_sweep.recover_interrupted_capture_attempts(
                ledger, Path("/definitely/missing/heartbeat.json")
            )

        attempt = ledger["jobs"]["job-a"]["attempts"][0]
        self.assertEqual(recovered[0]["charged_seconds"], 65.0)
        self.assertEqual(attempt["elapsed_seconds"], 65.0)
        self.assertTrue(attempt["transient"])
        self.assertEqual(ledger["jobs"]["job-a"]["status"], "retry_pending")
        self.assertFalse(ledger["capture_budget"]["enforced"])
        self.assertTrue(ledger["capture_budget"]["valid"])

    def test_uncapped_orphan_with_zero_retries_is_terminal_failed(self) -> None:
        ledger = {
            "policy": {"retry_transient_failures": 0},
            "jobs": {"job-a": {
                "status": "running",
                "timeout_seconds": 60.0,
                "attempts": [{
                    "attempt": 1,
                    "status": "running",
                    "elapsed_seconds": 10.0,
                    "effective_timeout_seconds": 60.0,
                    "effective_term_grace_seconds": 5.0,
                    "process_identity": {"pid": 1234},
                }],
            }},
        }
        with mock.patch.object(
            dmap_sweep, "terminate_verified_orphan_process_group",
            return_value={"status": "killed", "pid": 1234},
        ):
            dmap_sweep.recover_interrupted_capture_attempts(
                ledger, Path("/definitely/missing/heartbeat.json")
            )

        attempt = ledger["jobs"]["job-a"]["attempts"][0]
        self.assertEqual(ledger["jobs"]["job-a"]["status"], "failed")
        self.assertFalse(attempt["transient"])
        self.assertFalse(attempt["recovery_retry_available"])

    def test_legacy_orphan_without_identity_or_supervisor_refuses_resume(self) -> None:
        ledger = {
            "policy": {},
            "jobs": {"job-a": {
                "status": "running",
                "timeout_seconds": 60.0,
                "attempts": [{
                    "attempt": 1,
                    "status": "running",
                    "elapsed_seconds": 10.0,
                    "effective_timeout_seconds": 60.0,
                    "effective_term_grace_seconds": 5.0,
                }],
            }},
        }

        with self.assertRaisesRegex(RuntimeError, "no verifiable process identity"):
            dmap_sweep.recover_interrupted_capture_attempts(
                ledger, Path("/definitely/missing/heartbeat.json")
            )
        self.assertEqual(ledger["jobs"]["job-a"]["status"], "running")

    def test_prepared_supervisor_wait_uses_boot_clock_not_wall_time(self) -> None:
        ledger = {
            "policy": {"capture_budget_hours": 1.0},
            "jobs": {"job-a": {
                "status": "running",
                "timeout_seconds": 60.0,
                "attempts": [{
                    "attempt": 1,
                    "status": "running",
                    "elapsed_seconds": 1.0,
                    "effective_timeout_seconds": 60.0,
                    "effective_term_grace_seconds": 5.0,
                    "supervisor_kind": "coreutils_timeout",
                    "supervisor_prepared": True,
                    "supervisor_not_after_epoch": 200.0,
                }],
            }},
        }
        with self.assertRaisesRegex(RuntimeError, "supervision wait started"):
            dmap_sweep.recover_interrupted_capture_attempts(
                ledger, Path("/definitely/missing/heartbeat.json"),
                current_boot_id="boot-a", current_boottime_seconds=100.0,
            )
        wait = ledger["jobs"]["job-a"]["attempts"][0]["orphan_recovery_wait"]
        self.assertEqual(wait["safe_after_boottime_seconds"], 170.0)
        with (
            mock.patch.object(dmap_sweep.time, "time", return_value=10**12),
            self.assertRaisesRegex(RuntimeError, "same-boot resume"),
        ):
            dmap_sweep.recover_interrupted_capture_attempts(
                ledger, Path("/definitely/missing/heartbeat.json"),
                current_boot_id="boot-a", current_boottime_seconds=150.0,
            )
        self.assertEqual(ledger["jobs"]["job-a"]["status"], "running")

        recovered = dmap_sweep.recover_interrupted_capture_attempts(
            ledger, Path("/definitely/missing/heartbeat.json"),
            current_boot_id="boot-b", current_boottime_seconds=1.0,
        )
        self.assertEqual(len(recovered), 1)
        self.assertEqual(
            recovered[0]["charged_seconds"], 65.0
        )
        self.assertEqual(
            ledger["jobs"]["job-a"]["attempts"][0]["orphan_cleanup"]["status"],
            "prior_boot_process_gone",
        )

    def test_no_identity_wait_persists_until_real_supervisor_expires(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger_path = root / "schedule.json"
            heartbeat_path = root / "heartbeat.json"
            ledger = {
                "policy": {"retry_transient_failures": 0},
                "jobs": {"job-a": {
                    "status": "running",
                    "timeout_seconds": 0.15,
                    "attempts": [{
                        "attempt": 1,
                        "status": "running",
                        "elapsed_seconds": 0.0,
                        "effective_timeout_seconds": 0.15,
                        "effective_term_grace_seconds": 0.0,
                        "supervisor_kind": "coreutils_timeout",
                        "supervisor_prepared": True,
                    }],
                }},
            }
            dmap_sweep.atomic_write_json(ledger_path, ledger)
            command = dmap_sweep.timeout_supervisor_command(
                [sys.executable, "-c", "import time; time.sleep(10)"], 0.15, 0.0
            )
            process = subprocess.Popen(command, start_new_session=True)
            try:
                with (
                    mock.patch.object(
                        dmap_sweep, "ORPHAN_RECOVERY_MARGIN_SECONDS", 0.05
                    ),
                    self.assertRaisesRegex(RuntimeError, "supervision wait started"),
                ):
                    dmap_sweep.recover_interrupted_capture_attempts(
                        ledger, heartbeat_path, ledger_path=ledger_path,
                    )
                persisted = json.loads(ledger_path.read_text(encoding="utf-8"))
                self.assertIn(
                    "orphan_recovery_wait",
                    persisted["jobs"]["job-a"]["attempts"][0],
                )
                process.wait(timeout=2.0)
                time.sleep(0.1)
                persisted = json.loads(ledger_path.read_text(encoding="utf-8"))
                with mock.patch.object(
                    dmap_sweep, "ORPHAN_RECOVERY_MARGIN_SECONDS", 0.05
                ):
                    recovered = dmap_sweep.recover_interrupted_capture_attempts(
                        persisted, heartbeat_path, ledger_path=ledger_path,
                    )
                self.assertEqual(len(recovered), 1)
                self.assertEqual(persisted["jobs"]["job-a"]["status"], "failed")
                self.assertFalse(
                    persisted["jobs"]["job-a"]["attempts"][0]["transient"]
                )
                self.assertFalse(dmap_sweep.live_process_group_members(process.pid))
            finally:
                if dmap_sweep.live_process_group_members(process.pid):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2.0)

    def test_timeout_supervisor_command_executes_logical_command(self) -> None:
        for grace in (0.0, 0.1):
            with self.subTest(grace=grace):
                command = dmap_sweep.timeout_supervisor_command(
                    [sys.executable, "-c", "print('supervised')"], 2.0, grace
                )
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "supervised")
                self.assertEqual(command[3 if grace else 2], "--")

    def test_orphan_identity_mismatch_refuses_pid_reuse_kill(self) -> None:
        expected = {
            "pid": 1234,
            "process_group_id": 1234,
            "state": "S",
            "start_ticks": 100,
            "boot_id": "boot",
            "cmdline": ["/usr/bin/timeout", "60s", "capture"],
            "cmdline_sha256": "a" * 64,
        }
        current = dict(expected, start_ticks=101)
        with (
            mock.patch.object(
                dmap_sweep, "linux_process_identity", return_value=current
            ),
            mock.patch.object(os, "killpg") as killpg,
            self.assertRaisesRegex(RuntimeError, "PID-reuse"),
        ):
            dmap_sweep.terminate_verified_orphan_process_group(
                expected, term_grace_seconds=0.0
            )
        killpg.assert_not_called()

    def test_orphan_cleanup_refuses_live_group_after_sigkill(self) -> None:
        expected = {
            "pid": 1234,
            "process_group_id": 1234,
            "state": "S",
            "start_ticks": 100,
            "boot_id": "boot",
            "cmdline": ["/usr/bin/timeout", "60s", "capture"],
            "cmdline_sha256": "a" * 64,
        }
        with (
            mock.patch.object(
                dmap_sweep, "linux_process_identity", return_value=expected
            ),
            mock.patch.object(
                dmap_sweep, "live_process_group_members", return_value=[1234]
            ),
            mock.patch.object(
                dmap_sweep.time, "monotonic", side_effect=[0.0, 1.0, 10.0, 20.0]
            ),
            mock.patch.object(os, "killpg") as killpg,
            self.assertRaisesRegex(RuntimeError, "remains live"),
        ):
            dmap_sweep.terminate_verified_orphan_process_group(
                expected, term_grace_seconds=0.0
            )
        self.assertGreaterEqual(killpg.call_count, 1)

    def test_zombie_supervisor_with_empty_group_is_already_exited(self) -> None:
        expected = {
            "pid": 1234,
            "process_group_id": 1234,
            "state": "S",
            "start_ticks": 100,
            "boot_id": "boot",
            "cmdline": ["/usr/bin/timeout", "60s", "capture"],
            "cmdline_sha256": "a" * 64,
        }
        zombie = dict(expected, state="Z", cmdline=[], cmdline_sha256="b" * 64)
        with (
            mock.patch.object(
                dmap_sweep, "linux_process_identity", return_value=zombie
            ),
            mock.patch.object(
                dmap_sweep, "live_process_group_members", return_value=[]
            ),
            mock.patch.object(os, "killpg") as killpg,
        ):
            result = dmap_sweep.terminate_verified_orphan_process_group(
                expected, term_grace_seconds=0.0
            )
        self.assertEqual(result["status"], "already_exited")
        killpg.assert_not_called()

    def test_zombie_supervisor_with_live_descendant_kills_verified_group(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import os,time; child=os.fork(); "
                "time.sleep(30) if child == 0 else (time.sleep(0.2), os._exit(0))"
            ),
        ]
        process = subprocess.Popen(command, start_new_session=True)
        expected = dmap_sweep.linux_process_identity(process.pid)
        self.assertIsNotNone(expected)
        try:
            deadline = time.monotonic() + 2.0
            zombie = None
            while time.monotonic() < deadline:
                zombie = dmap_sweep.linux_process_identity(process.pid)
                if zombie is not None and zombie.get("state") == "Z":
                    break
                time.sleep(0.02)
            self.assertIsNotNone(zombie)
            self.assertEqual(zombie["state"], "Z")
            self.assertTrue(dmap_sweep.live_process_group_members(process.pid))

            result = dmap_sweep.terminate_verified_orphan_process_group(
                expected, term_grace_seconds=0.2
            )

            self.assertIn(result["status"], {"terminated", "killed"})
            self.assertFalse(dmap_sweep.live_process_group_members(process.pid))
        finally:
            if dmap_sweep.live_process_group_members(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)

    def test_orphaned_session_uses_matching_heartbeat_for_budget_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = 1_700_000_000.0
            ledger = {
                "updated_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started + 10.0)
                ),
                "sessions": [{
                    "status": "running",
                    "started_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)
                    ),
                    "started_epoch": started,
                    "pid": 1234,
                    "heartbeat_seconds": 15.0,
                    "term_grace_seconds": 20.0,
                }],
            }
            heartbeat = root / "heartbeat.json"
            heartbeat.write_text(json.dumps({
                "at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started + 50.0)
                ),
                "scheduler_pid": 1234,
            }), encoding="utf-8")

            recovered = dmap_sweep.recover_interrupted_sessions(
                ledger, heartbeat, now_epoch=started + 100.0
            )

            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["status"], "interrupted")
            self.assertEqual(recovered[0]["elapsed_seconds"], 85.0)
            self.assertEqual(dmap_sweep.cumulative_elapsed_seconds(ledger), 85.0)

    def test_plan_only_binds_source_snapshot_into_immutable_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_spec = scene("scene-a", root)
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config_path = root / "config.yaml"
            output_root = root / "output"
            config_path.write_text(json.dumps({
                "schema_version": 2,
                "experiment_id": "plan_test",
                "output_root": str(output_root),
                "dataset_root": str(root / "dataset"),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "capture_profiles": ["summary"],
                "suite": {"name": "development", "scan_ids": ["scene-a"]},
                "scenes": [scene_spec],
                "instrumentation": {
                    "expected_width": 1,
                    "expected_height": 1,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 1,
                },
                "runs": [{
                    **run("control", family="control", baseline=True),
                    "repeats": 1,
                }],
                "sweep": {"stages": [{
                    "name": "control", "runs": ["control"],
                    "scenes": ["scene-a"], "profiles": ["summary"],
                    "repeat_indices": [0],
                }]},
            }), encoding="utf-8")
            provenance = {
                "archive": str(root / "source.tar.zst"),
                "sha256": "c" * 64,
                "commit": "d" * 40,
                "dirty": True,
            }
            args = arguments(root, config=config_path, plan_only=True, budget_hours=1.0)

            with mock.patch.object(
                dmap_sweep, "ensure_source_provenance", return_value=provenance
            ):
                return_code = dmap_sweep.run_sweep(args)

            self.assertEqual(return_code, 0)
            ledger = json.loads(
                (output_root / "plan_test" / "sweep" / "schedule.json").read_text()
            )
            self.assertEqual(ledger["identity"]["source_provenance"]["sha256"], "c" * 64)
            self.assertEqual(ledger["status"], "planned")

            ledger["sessions"] = [{
                "status": "incomplete",
                "elapsed_seconds": 45.0 * 60.0,
            }]
            dmap_sweep.atomic_write_json(
                output_root / "plan_test" / "sweep" / "schedule.json", ledger
            )
            with (
                mock.patch.object(
                    dmap_sweep, "ensure_source_provenance", return_value=provenance
                ),
                mock.patch.object(dmap_sweep, "run_job") as run_job,
            ):
                return_code = dmap_sweep.run_sweep(arguments(
                    root, config=config_path, plan_only=False, budget_hours=1.0
                ))

            resumed = json.loads(
                (output_root / "plan_test" / "sweep" / "schedule.json").read_text()
            )
            self.assertEqual(return_code, 1)
            run_job.assert_not_called()
            self.assertEqual(resumed["jobs"][next(iter(resumed["jobs"]))]["status"], "deferred")
            self.assertEqual(
                resumed["sessions"][-1]["budget_elapsed_before_seconds"], 45.0 * 60.0
            )

    def test_unchanged_plan_source_snapshot_is_reusable_for_execution(self) -> None:
        if dmap_sweep.source_snapshot.zstandard is None:
            raise unittest.SkipTest("zstandard is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            (repository / "guide.md").write_text("operator guide\n", encoding="utf-8")
            subprocess.run(["git", "add", "guide.md"], cwd=repository, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-q", "-m", "initial",
            ], cwd=repository, check=True)
            schedule = root / "schedule"

            with mock.patch.object(dmap_sweep, "REPO_ROOT", repository):
                planned = dmap_sweep.ensure_source_provenance(schedule)
                executing = dmap_sweep.ensure_source_provenance(schedule)

            self.assertEqual(executing["sha256"], planned["sha256"])
            self.assertTrue((schedule / "source_snapshot.tar.zst").is_file())

    def test_budget_deferral_still_generates_partial_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = [scene("scene-a", root), scene("scene-b", root)]
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config_path = root / "config.yaml"
            output_root = root / "output"
            config_path.write_text(json.dumps({
                "schema_version": 2,
                "experiment_id": "partial_report_test",
                "output_root": str(output_root),
                "dataset_root": str(root / "dataset"),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "capture_profiles": ["summary"],
                "suite": {"name": "development", "scan_ids": ["scene-a", "scene-b"]},
                "scenes": scenes,
                "instrumentation": {
                    "expected_width": 1,
                    "expected_height": 1,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 1,
                },
                "runs": [{
                    **run("control", family="control", baseline=True),
                    "repeats": 1,
                }],
                "sweep": {"stages": [{
                    "name": "control",
                    "runs": ["control"],
                    "scenes": ["scene-a", "scene-b"],
                    "profiles": ["summary"],
                    "repeat_indices": [0],
                }]},
            }), encoding="utf-8")
            provenance = {
                "archive": str(root / "source.tar.zst"),
                "sha256": "c" * 64,
                "commit": "d" * 40,
                "dirty": True,
            }

            def complete_job(
                _arguments, _config, _root, job, ledger, ledger_path,
                _heartbeat_path, _stop_event, deadline_monotonic=None,
            ) -> bool:
                ledger["jobs"][job.job_id]["status"] = "complete"
                dmap_sweep.save_ledger(ledger_path, ledger)
                return True

            def complete_report(
                _arguments, _config, _root, _report_dir, ledger, ledger_path,
                _heartbeat_path, _stop_event,
            ) -> bool:
                ledger["report"].update(status="complete", valid=True)
                dmap_sweep.save_ledger(ledger_path, ledger)
                return True

            args = arguments(
                root,
                config=config_path,
                generate_report=True,
                compact=False,
            )
            with (
                mock.patch.object(
                    dmap_sweep, "ensure_source_provenance", return_value=provenance
                ),
                mock.patch.object(
                    dmap_sweep, "admission_reason",
                    side_effect=[None, "time admission failed: bounded campaign"],
                ),
                mock.patch.object(dmap_sweep, "run_job", side_effect=complete_job),
                mock.patch.object(
                    dmap_sweep, "generate_final_report", side_effect=complete_report
                ) as generate_report,
            ):
                return_code = dmap_sweep.run_sweep(args)

            ledger = json.loads(
                (output_root / "partial_report_test" / "sweep" / "schedule.json").read_text()
            )
            self.assertEqual(return_code, 1)
            self.assertEqual(generate_report.call_count, 1)
            self.assertEqual(ledger["status"], "incomplete")
            self.assertEqual(ledger["capture"]["status"], "incomplete")
            self.assertIn("bounded campaign", ledger["capture"]["reason"])
            self.assertEqual(ledger["report"]["status"], "complete")

    def test_terminal_scene_failure_continues_independent_jobs_and_is_resume_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = sweep_job(root, job_id="job-scene-a", stage="baseline")
            second = sweep_job(root, job_id="job-scene-b", stage="baseline")
            object.__setattr__(second, "scene_id", "scene-b")
            object.__setattr__(second, "run_dir", root / "runs" / "scene-b")
            config = {
                "_config_path": str(root / "config.yaml"),
                "runs": [first.run_spec],
                "sweep": {"stages": [{
                    "name": "baseline",
                    "runs": [first.run_label],
                    "scenes": ["scene-a", "scene-b"],
                    "profiles": ["endpoint"],
                }]},
            }
            provenance = {
                "archive": str(root / "source.tar.zst"),
                "sha256": "c" * 64,
                "commit": "d" * 40,
                "dirty": True,
            }
            args = arguments(
                root,
                config=root / "config.yaml",
                generate_report=False,
                compact=False,
                budget_hours=1.0,
            )
            failure = {
                "kind": dmap_sweep.SCENE_FAILURE_ZERO_NEIGHBOR_VIEWS,
                "scope": "scene",
                "terminal": True,
                "continuable": True,
                "promotion_eligible": False,
            }
            executed: list[str] = []

            def capture(
                _arguments, _config, _root, job, ledger, ledger_path,
                _heartbeat_path, _stop_event, deadline_monotonic=None,
            ) -> bool:
                executed.append(job.scene_id)
                row = ledger["jobs"][job.job_id]
                if job.scene_id == "scene-a":
                    row.update(
                        status="scene_failed",
                        validation="run_metadata.json is missing or malformed",
                        scene_failure=failure,
                    )
                    dmap_sweep.save_ledger(ledger_path, ledger)
                    return False
                row.update(status="complete", validation="validated independent scene")
                dmap_sweep.save_ledger(ledger_path, ledger)
                return True

            with (
                mock.patch.object(dmap_sweep.dmap_dev, "load_config", return_value=config),
                mock.patch.object(dmap_sweep.dmap_dev, "experiment_root", return_value=root),
                mock.patch.object(
                    dmap_sweep.dmap_dev, "prepare_experiment", return_value=(config, root)
                ),
                mock.patch.object(dmap_sweep, "build_jobs", return_value=[first, second]),
                mock.patch.object(
                    dmap_sweep, "ensure_source_provenance", return_value=provenance
                ),
                mock.patch.object(
                    dmap_sweep, "identity_record", return_value={"sha256": "a" * 64}
                ),
                mock.patch.object(dmap_sweep, "admission_reason", return_value=None),
                mock.patch.object(dmap_sweep, "run_job", side_effect=capture),
            ):
                return_code = dmap_sweep.run_sweep(args)

            ledger_path = root / "sweep" / "schedule.json"
            ledger = json.loads(ledger_path.read_text())
            self.assertEqual(return_code, 1)
            self.assertEqual(executed, ["scene-a", "scene-b"])
            self.assertEqual(ledger["jobs"][first.job_id]["status"], "scene_failed")
            self.assertEqual(ledger["jobs"][second.job_id]["status"], "complete")
            self.assertEqual(
                ledger["stages"]["baseline"]["status"],
                "complete_with_scene_failures",
            )
            self.assertEqual(ledger["capture"]["status"], "incomplete")
            self.assertEqual(ledger["capture"]["jobs_scene_failed"], 1)

            executed.clear()
            with (
                mock.patch.object(dmap_sweep.dmap_dev, "load_config", return_value=config),
                mock.patch.object(dmap_sweep.dmap_dev, "experiment_root", return_value=root),
                mock.patch.object(
                    dmap_sweep.dmap_dev, "prepare_experiment", return_value=(config, root)
                ),
                mock.patch.object(dmap_sweep, "build_jobs", return_value=[first, second]),
                mock.patch.object(
                    dmap_sweep, "ensure_source_provenance", return_value=provenance
                ),
                mock.patch.object(
                    dmap_sweep, "identity_record", return_value={"sha256": "a" * 64}
                ),
                mock.patch.object(
                    dmap_sweep, "validate_completed_sweep_run",
                    return_value=(True, "validated resumed capture"),
                ) as validate,
                mock.patch.object(dmap_sweep, "run_job") as run_job,
            ):
                resumed_code = dmap_sweep.run_sweep(args)

            self.assertEqual(resumed_code, 1)
            validate.assert_called_once_with(second)
            run_job.assert_not_called()

    def test_accuracy_ledger_csv_is_ranked_without_composite_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            (report / "accuracy_ledger.csv").write_text(
                "candidate,accuracy_rank,scene_count,primary_scene_metric_rows,"
                "availability_biased,worst_normalized_noise_loss\n"
                "candidate-b,2,2,6,False,0.2\n"
                "candidate-a,1,1,3,True,0.4\n",
                encoding="utf-8",
            )

            status = dmap_sweep.accuracy_ledger_status(report)

            self.assertTrue(status["available"])
            self.assertEqual(
                [row["candidate"] for row in status["rows"]],
                ["candidate-a", "candidate-b"],
            )
            self.assertNotIn("composite_score", status["rows"][0])

    def test_accuracy_ledger_requires_scene_and_primary_metric_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            (report / "accuracy_ledger.csv").write_text(
                "candidate,accuracy_rank,scene_count,primary_scene_metric_rows\n"
                "never-run,1,0,0\n"
                "no-noise-evidence,2,1,0\n",
                encoding="utf-8",
            )

            status = dmap_sweep.accuracy_ledger_status(report)

            self.assertFalse(status["available"])
            self.assertIn("primary metric evidence", status["reason"])

    def test_accuracy_ledger_excludes_rows_without_primary_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            (report / "accuracy_ledger.csv").write_text(
                "candidate,accuracy_rank,scene_count,primary_scene_metric_rows\n"
                "valid,2,1,3\n"
                "never-run,1,0,0\n",
                encoding="utf-8",
            )

            status = dmap_sweep.accuracy_ledger_status(report)

            self.assertTrue(status["available"])
            self.assertEqual(
                [row["candidate"] for row in status["rows"]],
                ["valid"],
            )

    def test_final_report_reuse_is_bound_to_completed_job_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            ledger = {
                "identity": {"sha256": "a" * 64},
                "jobs": {
                    "job-a": {"status": "complete"},
                    "job-b": {"status": "pending"},
                },
                "report": {},
            }
            digest = dmap_sweep.report_evidence_digest(ledger)
            ledger["report"]["evidence_digest"] = digest
            report_source = {"sha256": "b" * 64}
            ledger["report"]["report_source_sha256"] = report_source["sha256"]
            dmap_sweep.write_report_binding(report, ledger, digest, report_source)

            with (
                mock.patch.object(
                    dmap_sweep, "valid_report", return_value=(True, "validated")
                ),
                mock.patch.object(
                    dmap_sweep,
                    "validate_report_source_provenance",
                    return_value=(True, "validated source", report_source),
                ),
            ):
                reusable, _reason, current = dmap_sweep.final_report_reuse_status(
                    report, ledger
                )
                self.assertTrue(reusable)
                self.assertEqual(current, digest)

                ledger["jobs"]["job-b"]["status"] = "complete"
                reusable, reason, current = dmap_sweep.final_report_reuse_status(
                    report, ledger
                )

            self.assertFalse(reusable)
            self.assertIn("stale", reason)
            self.assertNotEqual(current, digest)

    def test_final_report_reuse_requires_bound_report_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            ledger = {
                "identity": {"sha256": "a" * 64},
                "jobs": {"job-a": {"status": "complete"}},
                "report": {},
            }
            digest = dmap_sweep.report_evidence_digest(ledger)
            ledger["report"]["evidence_digest"] = digest
            ledger["report"]["report_source_sha256"] = "b" * 64
            dmap_sweep.write_report_binding(
                report, ledger, digest, {"sha256": "b" * 64}
            )

            with (
                mock.patch.object(
                    dmap_sweep, "valid_report", return_value=(True, "validated")
                ),
                mock.patch.object(
                    dmap_sweep,
                    "validate_report_source_provenance",
                    return_value=(True, "validated source", {"sha256": "c" * 64}),
                ),
            ):
                reusable, reason, _digest = dmap_sweep.final_report_reuse_status(
                    report, ledger
                )

            self.assertFalse(reusable)
            self.assertIn("report-generator source", reason)

    def test_stage_report_reuse_is_bound_to_its_current_job_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            ledger = {
                "identity": {"sha256": "a" * 64},
                "jobs": {
                    "job-screen": {"stage": "screen", "status": "complete"},
                    "job-other": {"stage": "other", "status": "complete"},
                },
            }
            source = {"sha256": "b" * 64}
            evidence = dmap_sweep.stage_report_evidence(ledger, "screen", 0)
            dmap_sweep.write_stage_report_binding(report, evidence, source)

            with (
                mock.patch.object(
                    dmap_sweep, "valid_report", return_value=(True, "validated")
                ),
                mock.patch.object(
                    dmap_sweep,
                    "validate_report_source_provenance",
                    return_value=(True, "validated source", source),
                ),
            ):
                reusable, _reason, current, _source = (
                    dmap_sweep.stage_report_reuse_status(
                        report, ledger, "screen", 0
                    )
                )
                ledger["jobs"]["job-other"]["status"] = "failed"
                unrelated, _reason, unchanged, _source = (
                    dmap_sweep.stage_report_reuse_status(
                        report, ledger, "screen", 0
                    )
                )
                ledger["jobs"]["job-screen"]["status"] = "failed"
                stale, reason, changed, _source = (
                    dmap_sweep.stage_report_reuse_status(
                        report, ledger, "screen", 0
                    )
                )

            self.assertTrue(reusable)
            self.assertTrue(unrelated)
            self.assertEqual(current["evidence_digest"], unchanged["evidence_digest"])
            self.assertFalse(stale)
            self.assertIn("stale", reason)
            self.assertNotEqual(current["evidence_digest"], changed["evidence_digest"])

    def test_stage_report_without_binding_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            ledger = {
                "identity": {"sha256": "a" * 64},
                "jobs": {"job-a": {"stage": "screen", "status": "complete"}},
            }
            with mock.patch.object(
                dmap_sweep, "valid_report", return_value=(True, "validated")
            ):
                reusable, reason, _evidence, _source = (
                    dmap_sweep.stage_report_reuse_status(
                        report, ledger, "screen", 0
                    )
                )

            self.assertFalse(reusable)
            self.assertIn("evidence binding", reason)

    def test_report_source_provenance_is_stable_and_tamper_evident(self) -> None:
        if dmap_sweep.source_snapshot.zstandard is None:
            raise unittest.SkipTest("zstandard is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            guide = repository / "guide.md"
            guide.write_text("version one\n", encoding="utf-8")
            subprocess.run(["git", "add", "guide.md"], cwd=repository, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-q", "-m", "initial",
            ], cwd=repository, check=True)
            snapshots = root / "snapshots"
            snapshots.mkdir()

            with mock.patch.object(dmap_sweep, "REPO_ROOT", repository):
                before = dmap_sweep.create_report_source_snapshot(
                    snapshots, "before.tar.zst"
                )
                after = dmap_sweep.create_report_source_snapshot(
                    snapshots, "after.tar.zst"
                )
                report = root / "report"
                record = dmap_sweep.publish_report_source_provenance(
                    report, before, after
                )
                valid, _reason, validated = (
                    dmap_sweep.validate_report_source_provenance(report)
                )
                self.assertTrue(valid)
                self.assertEqual(validated, record)

                guide.write_text("version two\n", encoding="utf-8")
                changed = dmap_sweep.create_report_source_snapshot(
                    snapshots, "changed.tar.zst"
                )
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    dmap_sweep.publish_report_source_provenance(
                        root / "changed-report", before, changed
                    )

            with (report / dmap_sweep.REPORT_SOURCE_SNAPSHOT_NAME).open("ab") as handle:
                handle.write(b"tamper")
            valid, reason, _record = dmap_sweep.validate_report_source_provenance(
                report
            )
            self.assertFalse(valid)
            self.assertIn("failed validation", reason)

    def test_quarantine_preserves_incomplete_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "incomplete"
            run_dir.mkdir(parents=True)
            (run_dir / "stderr.log").write_text("failed", encoding="utf-8")

            destination = dmap_sweep.quarantine_path(
                run_dir, root, "job-failed", "validator rejected output"
            )

            self.assertIsNotNone(destination)
            self.assertFalse(run_dir.exists())
            self.assertEqual((destination / "stderr.log").read_text(), "failed")
            record = json.loads((destination.parent / "quarantine.json").read_text())
            self.assertEqual(record["reason"], "validator rejected output")

    def test_capture_exception_is_persisted_as_a_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            ledger_path = root / "schedule.json"
            ledger = dmap_sweep.initialize_ledger(
                ledger_path, {"sha256": "a" * 64}, [job], arguments(root)
            )

            with mock.patch.object(
                dmap_sweep, "execute_capture_attempt", side_effect=OSError("spawn failed")
            ):
                complete = dmap_sweep.run_job(
                    arguments(root), {}, root, job, ledger, ledger_path,
                    root / "heartbeat.json", threading.Event(),
                )

            self.assertFalse(complete)
            persisted = json.loads(ledger_path.read_text())
            row = persisted["jobs"][job.job_id]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["attempts"][0]["status"], "failed")
            self.assertFalse(row["attempts"][0]["transient"])
            self.assertIn("spawn failed", row["attempts"][0]["validation"])
            self.assertEqual(
                row["attempts"][0]["elapsed_seconds"],
                job.timeout_seconds + arguments(root).term_grace_seconds,
            )
            self.assertEqual(
                row["attempts"][0]["capture_accounting"],
                "conservative_supervision_bound",
            )
            self.assertEqual(
                persisted["capture_budget"]["elapsed_seconds"],
                job.timeout_seconds + arguments(root).term_grace_seconds,
            )

    def test_capture_exception_cannot_erase_durable_live_elapsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            ledger_path = root / "schedule.json"
            args = arguments(root, capture_budget_hours=1.0)
            ledger = dmap_sweep.initialize_ledger(
                ledger_path, {"sha256": "a" * 64}, [job], args
            )

            def fail_after_heartbeat(*_args, **_kwargs):
                durable = json.loads(ledger_path.read_text(encoding="utf-8"))
                durable["jobs"][job.job_id]["attempts"][0][
                    "elapsed_seconds"
                ] = 20.0
                dmap_sweep.refresh_capture_budget(durable)
                dmap_sweep.atomic_write_json(ledger_path, durable)
                raise OSError("post-process failed")

            with mock.patch.object(
                dmap_sweep, "execute_capture_attempt", side_effect=fail_after_heartbeat
            ):
                complete = dmap_sweep.run_job(
                    args, {}, root, job, ledger, ledger_path,
                    root / "heartbeat.json", threading.Event(),
                )

            self.assertFalse(complete)
            persisted = json.loads(ledger_path.read_text(encoding="utf-8"))
            attempt = persisted["jobs"][job.job_id]["attempts"][0]
            self.assertEqual(attempt["observed_elapsed_seconds"], 20.0)
            self.assertEqual(
                attempt["elapsed_seconds"],
                job.timeout_seconds + args.term_grace_seconds,
            )
            self.assertEqual(
                persisted["capture_budget"]["elapsed_seconds"],
                job.timeout_seconds + args.term_grace_seconds,
            )

    def test_transient_retry_is_readmitted_against_campaign_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            ledger_path = root / "schedule.json"
            args = arguments(
                root,
                retry_transient_failures=1,
                finalization_reserve_minutes=0.0,
            )
            ledger = dmap_sweep.initialize_ledger(
                ledger_path, {"sha256": "a" * 64}, [job], args
            )
            timed_out = dmap_sweep.ProcessResult(
                return_code=124,
                elapsed_seconds=job.timeout_seconds,
                timed_out=True,
                terminated=True,
                killed=False,
                stop_requested=False,
            )

            with mock.patch.object(
                dmap_sweep,
                "execute_capture_attempt",
                return_value=(timed_out, False, "process timed out"),
            ) as execute:
                complete = dmap_sweep.run_job(
                    args,
                    {},
                    root,
                    job,
                    ledger,
                    ledger_path,
                    root / "heartbeat.json",
                    threading.Event(),
                    deadline_monotonic=time.monotonic() + job.timeout_seconds / 2.0,
                )

            self.assertFalse(complete)
            execute.assert_called_once()
            row = json.loads(ledger_path.read_text())["jobs"][job.job_id]
            self.assertEqual(row["status"], "deferred")
            self.assertTrue(row["attempts"][0]["transient"])
            self.assertIn("retry time admission failed", row["admission_reason"])

    def test_zero_neighbor_return_zero_is_recorded_as_terminal_scene_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            ledger_path = root / "schedule.json"
            ledger = dmap_sweep.initialize_ledger(
                ledger_path, {"sha256": "a" * 64}, [job], arguments(root)
            )
            result = dmap_sweep.ProcessResult(
                return_code=0,
                elapsed_seconds=6.0,
                timed_out=False,
                terminated=False,
                killed=False,
                stop_requested=False,
            )

            def zero_neighbor_attempt(*_args, **_kwargs):
                log_path = (
                    job.run_dir / "work" / "mvs"
                    / "DensifyPointCloudDMapObserve-test.log"
                )
                log_path.parent.mkdir(parents=True)
                log_path.write_text(
                    "Reference image   0 paired with 0 views\n"
                    "Reference image   1 paired with 0 views\n",
                    encoding="utf-8",
                )
                return result, False, "run_metadata.json is missing or malformed"

            with mock.patch.object(
                dmap_sweep, "execute_capture_attempt", side_effect=zero_neighbor_attempt
            ):
                complete = dmap_sweep.run_job(
                    arguments(root), {}, root, job, ledger, ledger_path,
                    root / "heartbeat.json", threading.Event(),
                )

            self.assertFalse(complete)
            persisted = json.loads(ledger_path.read_text())
            row = persisted["jobs"][job.job_id]
            self.assertEqual(row["status"], "scene_failed")
            self.assertTrue(dmap_sweep.recorded_scene_failure(row))
            self.assertEqual(
                row["scene_failure"]["kind"],
                dmap_sweep.SCENE_FAILURE_ZERO_NEIGHBOR_VIEWS,
            )
            self.assertEqual(
                row["scene_failure"]["evidence"]["reference_image_ids"], [0, 1]
            )
            self.assertEqual(row["attempts"][0]["status"], "scene_failed")
            self.assertFalse(row["attempts"][0]["transient"])
            self.assertTrue(Path(row["scene_failure"]["quarantine"]).is_dir())
            with mock.patch.object(dmap_sweep, "execute_capture_attempt") as execute:
                complete = dmap_sweep.run_job(
                    arguments(root), {}, root, job, ledger, ledger_path,
                    root / "heartbeat.json", threading.Event(),
                )
            self.assertFalse(complete)
            execute.assert_not_called()

    def test_zero_neighbor_classifier_rejects_any_positive_neighbor_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            log_path = (
                job.run_dir / "work" / "mvs"
                / "DensifyPointCloudDMapObserve-test.log"
            )
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                "Reference image   0 paired with 0 views\n"
                "Reference image   1 paired with 2 views\n",
                encoding="utf-8",
            )
            result = dmap_sweep.ProcessResult(
                return_code=0,
                elapsed_seconds=6.0,
                timed_out=False,
                terminated=False,
                killed=False,
                stop_requested=False,
            )

            classification = dmap_sweep.classify_scene_capture_failure(
                job, result, "run_metadata.json is missing or malformed"
            )

            self.assertIsNone(classification)

    def test_validated_compaction_removes_only_equivalent_working_dmaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            working = job.run_dir / "work" / "mvs" / "depth0001.dmap"
            retained = job.run_dir / "depth_maps" / "depth0001.dmap"
            working.parent.mkdir(parents=True)
            retained.parent.mkdir(parents=True)
            working.write_bytes(b"depth-map")
            os.link(working, retained)
            (job.run_dir / "command.sh").write_text("true\n")
            (job.run_dir / "repro.json").write_text("{}\n")

            with mock.patch.object(
                dmap_sweep.dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "validated"),
            ) as validate:
                completion = dmap_sweep.compact_validated_run(job)

            self.assertFalse(working.exists())
            self.assertTrue(retained.exists())
            self.assertTrue((job.run_dir / "retention_manifest.json").is_file())
            self.assertTrue((job.run_dir / "compacted_completion.json").is_file())
            self.assertEqual(completion["retained_dmap_count"], 1)
            manifest = json.loads((job.run_dir / "retention_manifest.json").read_text())
            self.assertFalse(manifest["lossy"])
            self.assertTrue(manifest["full_dmap_set_retained"])
            self.assertEqual(
                manifest["complete_pre_compaction_dmap_set"],
                manifest["retained_dmap_set"],
            )
            self.assertEqual(validate.call_count, 2)

    def test_opt_in_summary_compaction_retains_only_instrumented_dmaps_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            object.__setattr__(job, "profile", "summary")
            object.__setattr__(job, "mode", "timing")
            object.__setattr__(job, "retention_policy", {
                "schema_name": dmap_sweep.RETENTION_POLICY_SCHEMA_NAME,
                "schema_version": 1,
                "profile": "summary",
                "dmap_policy": "instrumented_image_ids_only",
                "lossy": True,
            })
            depth_dir = job.run_dir / "depth_maps"
            work_dir = job.run_dir / "work" / "mvs"
            depth_dir.mkdir(parents=True)
            work_dir.mkdir(parents=True)
            for image_id in (1, 2, 3):
                retained = depth_dir / f"depth{image_id:04d}.dmap"
                retained.write_bytes(f"depth-map-{image_id}".encode())
                os.link(retained, work_dir / retained.name)
            for image_id in (1, 3):
                frame = (
                    job.run_dir / "dmap_instrumentation" / "depthmaps"
                    / f"image{image_id:04d}"
                )
                frame.mkdir(parents=True)
                (frame / "summary.json").write_text(
                    json.dumps({"image_id": image_id}) + "\n", encoding="utf-8"
                )
            (job.run_dir / "command.sh").write_text("true\n", encoding="utf-8")
            (job.run_dir / "repro.json").write_text(json.dumps({
                "command": [
                    "DensifyPointCloudDMapObserve",
                    "--dmap-instrumentation-image-list", "3,1,3",
                ],
                "return_code": 0,
                "dry_run": False,
            }) + "\n", encoding="utf-8")

            with mock.patch.object(
                dmap_sweep.dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "validated timing capture"),
            ) as validate:
                completion = dmap_sweep.compact_validated_run(job)
                resumed, resume_reason = dmap_sweep.validate_completed_sweep_run(job)

            self.assertTrue(resumed, resume_reason)
            self.assertEqual(validate.call_count, 3)
            self.assertTrue(completion["lossy"])
            self.assertEqual(completion["complete_pre_compaction_dmap_count"], 3)
            self.assertEqual(completion["retained_dmap_count"], 2)
            self.assertEqual(
                sorted(path.name for path in depth_dir.glob("depth*.dmap")),
                ["depth0001.dmap", "depth0003.dmap"],
            )
            self.assertFalse(list(work_dir.glob("depth*.dmap")))
            manifest = json.loads((job.run_dir / "retention_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 2)
            self.assertTrue(manifest["lossy"])
            self.assertEqual(manifest["requested_instrumentation_image_list"], "3,1,3")
            self.assertEqual(manifest["requested_instrumentation_image_ids"], [1, 3])
            self.assertEqual(manifest["requested_instrumentation_image_scope"], "explicit")
            self.assertEqual(manifest["selected_image_ids"], [1, 3])
            self.assertEqual(manifest["complete_pre_compaction_dmap_set"]["count"], 3)
            self.assertEqual(manifest["retained_dmap_set"]["count"], 2)
            removed = [
                row for row in manifest["removed"]
                if row["reason"] == "not_instrumented_image_id"
            ]
            self.assertEqual(len(removed), 1)
            self.assertEqual(removed[0]["path"], "depth_maps/depth0002.dmap")
            self.assertEqual(removed[0]["bytes"], len(b"depth-map-2"))
            self.assertEqual(len(removed[0]["sha256"]), 64)

            repro_path = job.run_dir / "repro.json"
            original_repro = json.loads(repro_path.read_text())
            tampered_repro = json.loads(json.dumps(original_repro))
            tampered_repro["command"][-1] = "1"
            dmap_sweep.atomic_write_json(repro_path, tampered_repro)
            with mock.patch.object(
                dmap_sweep.dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "superficially valid timing capture"),
            ):
                resumed, resume_reason = dmap_sweep.validate_completed_sweep_run(job)
            self.assertFalse(resumed)
            self.assertIn("image list changed after compaction", resume_reason)
            dmap_sweep.atomic_write_json(repro_path, original_repro)

            manifest_path = job.run_dir / "retention_manifest.json"
            completion_path = job.run_dir / "compacted_completion.json"
            original_manifest = json.loads(manifest_path.read_text())
            tampered_manifest = json.loads(json.dumps(original_manifest))
            tampered_manifest["requested_instrumentation_image_ids"] = [1]
            dmap_sweep.atomic_write_json(manifest_path, tampered_manifest)
            rebound_completion = json.loads(completion_path.read_text())
            rebound_completion["retention_manifest_sha256"] = dmap_sweep.sha256_file(manifest_path)
            dmap_sweep.atomic_write_json(completion_path, rebound_completion)
            with mock.patch.object(
                dmap_sweep.dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "superficially valid timing capture"),
            ):
                resumed, resume_reason = dmap_sweep.validate_completed_sweep_run(job)
            self.assertFalse(resumed)
            self.assertIn("inconsistent normalized requested image IDs", resume_reason)
            dmap_sweep.atomic_write_json(manifest_path, original_manifest)
            rebound_completion["retention_manifest_sha256"] = dmap_sweep.sha256_file(manifest_path)
            dmap_sweep.atomic_write_json(completion_path, rebound_completion)

            (depth_dir / "depth0001.dmap").write_bytes(b"tampered")
            with mock.patch.object(
                dmap_sweep.dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "superficially valid timing capture"),
            ):
                resumed, resume_reason = dmap_sweep.validate_completed_sweep_run(job)
            self.assertFalse(resumed)
            self.assertIn("does not match compacted completion", resume_reason)

    def test_requested_instrumentation_image_list_parser_is_canonical_and_strict(self) -> None:
        self.assertEqual(
            dmap_sweep.parse_requested_instrumentation_image_ids(" 3, 1,3 "), [1, 3]
        )
        self.assertIsNone(dmap_sweep.parse_requested_instrumentation_image_ids(" ALL "))
        for invalid in ("", "1,,3", "image0001", "-1,3"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                RuntimeError, "comma-separated numeric image list"
            ):
                dmap_sweep.parse_requested_instrumentation_image_ids(invalid)

    def test_explicit_requested_ids_must_exactly_match_evidence_before_removal(self) -> None:
        cases = (
            ("missing", "1,3", (1,), "missing_evidence_for_requested_ids=[3]"),
            ("extra", "1", (1, 3), "unexpected_evidenced_ids=[3]"),
        )
        for label, requested, evidenced, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                job = lossy_summary_job(
                    Path(directory), requested=requested, evidenced_ids=evidenced
                )
                depth_dir = job.run_dir / "depth_maps"
                work_dir = job.run_dir / "work" / "mvs"
                with mock.patch.object(
                    dmap_sweep.dmap_dev,
                    "validate_completed_run_mode",
                    return_value=(True, "validated timing capture"),
                ) as validate, self.assertRaisesRegex(RuntimeError, re.escape(expected)):
                    dmap_sweep.compact_validated_run(job)

                self.assertEqual(validate.call_count, 1)
                self.assertEqual(len(list(depth_dir.glob("depth*.dmap"))), 3)
                self.assertEqual(len(list(work_dir.glob("depth*.dmap"))), 3)
                self.assertFalse((job.run_dir / "retention_manifest.json").exists())
                self.assertFalse((job.run_dir / "compacted_completion.json").exists())

    def test_all_requested_scope_uses_evidenced_ids_without_finite_set_assertion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = lossy_summary_job(
                Path(directory), requested="all", evidenced_ids=(1, 3)
            )
            with mock.patch.object(
                dmap_sweep.dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "validated timing capture"),
            ):
                dmap_sweep.compact_validated_run(job)
                valid, reason = dmap_sweep.validate_completed_sweep_run(job)

            self.assertTrue(valid, reason)
            manifest = json.loads((job.run_dir / "retention_manifest.json").read_text())
            self.assertIsNone(manifest["requested_instrumentation_image_ids"])
            self.assertEqual(manifest["requested_instrumentation_image_scope"], "all")
            self.assertEqual(manifest["selected_image_ids"], [1, 3])

    def test_instrumented_only_retention_rejects_non_summary_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = sweep_job(Path(directory))
            object.__setattr__(job, "retention_policy", {
                "schema_name": dmap_sweep.RETENTION_POLICY_SCHEMA_NAME,
                "schema_version": 1,
                "profile": "endpoint",
                "dmap_policy": "instrumented_image_ids_only",
                "lossy": True,
            })
            with self.assertRaisesRegex(ValueError, "restricted to summary/timing"):
                dmap_sweep.resolved_retention_policy(job)

    def test_resume_accepts_legacy_v1_lossless_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            depth_dir = job.run_dir / "depth_maps"
            depth_dir.mkdir(parents=True)
            (depth_dir / "depth0001.dmap").write_bytes(b"depth-map")
            identity = dmap_sweep.dmap_set_identity(depth_dir)
            manifest_path = job.run_dir / "retention_manifest.json"
            dmap_sweep.atomic_write_json(manifest_path, {
                "schema_name": dmap_sweep.RETENTION_SCHEMA_NAME,
                "schema_version": 1,
                "complete": True,
                "retained_dmap_set": identity,
            })
            dmap_sweep.atomic_write_json(job.run_dir / "compacted_completion.json", {
                "schema_name": dmap_sweep.COMPACTED_SCHEMA_NAME,
                "schema_version": 1,
                "complete": True,
                "profile": job.profile,
                "mode": job.mode,
                "retained_dmap_count": identity["count"],
                "retained_dmap_set_sha256": identity["sha256"],
                "retention_manifest": manifest_path.name,
                "retention_manifest_sha256": dmap_sweep.sha256_file(manifest_path),
            })

            with mock.patch.object(
                dmap_sweep.dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "validated endpoint"),
            ):
                valid, reason = dmap_sweep.validate_completed_sweep_run(job)

            self.assertTrue(valid, reason)
            self.assertIn("legacy lossless compaction", reason)

    def test_admission_reserves_time_and_dynamic_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            args = arguments(
                root,
                finalization_reserve_minutes=1.0,
                free_space_floor_gb=1.0,
                finalization_reserve_gb=1.0,
            )
            ledger = {"jobs": {}}
            disk = os.statvfs(root)
            fake_usage = shutil_disk_usage = mock.Mock(
                total=10 * 1024**3,
                used=0,
                free=10 * 1024**3,
            )

            with mock.patch.object(dmap_sweep.shutil, "disk_usage", return_value=fake_usage):
                reason = dmap_sweep.admission_reason(
                    args, ledger, job, time.monotonic() + 30.0, root
                )
            self.assertIn("time admission failed", str(reason))
            self.assertGreater(disk.f_bsize, 0)

    def test_valid_existing_artifact_is_reused_before_time_or_storage_admission(self) -> None:
        cases = (
            (
                "time",
                {"budget_hours": 1e-9},
                "time admission failed",
                lambda: time.monotonic() - 1.0,
            ),
            (
                "storage",
                {"free_space_floor_gb": 1e9},
                "storage admission failed",
                lambda: time.monotonic() + 3600.0,
            ),
        )
        for condition, argument_values, expected_failure, deadline in cases:
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                job = sweep_job(root)
                job.run_dir.mkdir(parents=True)
                (job.run_dir / "existing-artifact").write_text("complete\n", encoding="utf-8")
                config = {
                    "_config_path": str(root / "config.yaml"),
                    "runs": [job.run_spec],
                    "sweep": {"stages": [{"name": job.stage, "runs": [job.run_label]}]},
                }
                provenance = {
                    "archive": str(root / "source.tar.zst"),
                    "sha256": "c" * 64,
                    "commit": "d" * 40,
                    "dirty": True,
                }
                args = arguments(root, compact=False, **argument_values)
                blocked_reason = dmap_sweep.admission_reason(
                    args, {"jobs": {}}, job, deadline(), root.parent
                )
                self.assertIn(expected_failure, str(blocked_reason))

                with (
                    mock.patch.object(dmap_sweep.dmap_dev, "load_config", return_value=config),
                    mock.patch.object(dmap_sweep.dmap_dev, "experiment_root", return_value=root),
                    mock.patch.object(
                        dmap_sweep.dmap_dev, "prepare_experiment", return_value=(config, root)
                    ),
                    mock.patch.object(dmap_sweep, "build_jobs", return_value=[job]),
                    mock.patch.object(
                        dmap_sweep, "ensure_source_provenance", return_value=provenance
                    ),
                    mock.patch.object(
                        dmap_sweep, "identity_record", return_value={"sha256": "a" * 64}
                    ),
                    mock.patch.object(
                        dmap_sweep.dmap_dev,
                        "validate_completed_run_mode",
                        return_value=(True, "validated existing capture"),
                    ) as validate,
                    mock.patch.object(
                        dmap_sweep,
                        "admission_reason",
                        wraps=dmap_sweep.admission_reason,
                    ) as admission,
                    mock.patch.object(dmap_sweep, "run_job") as run_job,
                ):
                    return_code = dmap_sweep.run_sweep(args)

                ledger = json.loads((root / "sweep" / "schedule.json").read_text())
                row = ledger["jobs"][job.job_id]
                self.assertEqual(return_code, 0)
                self.assertEqual(ledger["status"], "complete")
                self.assertEqual(row["status"], "complete")
                self.assertTrue(row["reused"])
                self.assertEqual(row["validation"], "validated existing capture")
                validate.assert_called_once_with(job.run_dir, job.mode)
                admission.assert_not_called()
                run_job.assert_not_called()

    def test_invalid_existing_artifact_remains_admission_gated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = sweep_job(root)
            job.run_dir.mkdir(parents=True)
            (job.run_dir / "invalid-artifact").write_text("incomplete\n", encoding="utf-8")
            config = {
                "_config_path": str(root / "config.yaml"),
                "runs": [job.run_spec],
                "sweep": {"stages": [{"name": job.stage, "runs": [job.run_label]}]},
            }
            provenance = {
                "archive": str(root / "source.tar.zst"),
                "sha256": "c" * 64,
                "commit": "d" * 40,
                "dirty": True,
            }
            args = arguments(root, compact=False)

            with (
                mock.patch.object(dmap_sweep.dmap_dev, "load_config", return_value=config),
                mock.patch.object(dmap_sweep.dmap_dev, "experiment_root", return_value=root),
                mock.patch.object(
                    dmap_sweep.dmap_dev, "prepare_experiment", return_value=(config, root)
                ),
                mock.patch.object(dmap_sweep, "build_jobs", return_value=[job]),
                mock.patch.object(
                    dmap_sweep, "ensure_source_provenance", return_value=provenance
                ),
                mock.patch.object(
                    dmap_sweep, "identity_record", return_value={"sha256": "a" * 64}
                ),
                mock.patch.object(
                    dmap_sweep.dmap_dev,
                    "validate_completed_run_mode",
                    return_value=(False, "missing completion metadata"),
                ) as validate,
                mock.patch.object(
                    dmap_sweep,
                    "admission_reason",
                    return_value="storage admission failed: bounded recovery",
                ) as admission,
                mock.patch.object(dmap_sweep, "run_job") as run_job,
            ):
                return_code = dmap_sweep.run_sweep(args)

            ledger = json.loads((root / "sweep" / "schedule.json").read_text())
            row = ledger["jobs"][job.job_id]
            self.assertEqual(return_code, 1)
            self.assertEqual(ledger["status"], "incomplete")
            self.assertEqual(row["status"], "deferred")
            self.assertEqual(row["reuse_validation_error"], "missing completion metadata")
            self.assertIn("bounded recovery", row["admission_reason"])
            validate.assert_called_once_with(job.run_dir, job.mode)
            admission.assert_called_once()
            run_job.assert_not_called()


if __name__ == "__main__":
    unittest.main()
