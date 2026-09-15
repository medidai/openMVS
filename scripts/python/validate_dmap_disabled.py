#!/usr/bin/env python3
"""Validate that disabled depth-map instrumentation is inert."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import time
from typing import Any

import numpy as np
import tyro

from report_dmap_annotation_fit import load_dmap


@dataclass
class Arguments:
    """Run an uninstrumented depth-map smoke test and compare production DMAPs."""

    densify_bin: Path
    source_work: Path
    mvs_file: Path
    reference_dmap_dir: Path
    output_dir: Path
    image_ids: list[int] = field(default_factory=list)
    gpu_device: int = 0
    resolution_level: int = 4
    sub_resolution_levels: int = 0
    number_views: int = 3
    number_views_fuse: int = 2
    patch_match_cuda_instances: int = 1
    iters: int = 1
    geometric_iters: int = 0


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    finite_first = np.isfinite(first)
    finite_second = np.isfinite(second)
    if not np.array_equal(finite_first, finite_second):
        return float("inf")
    if not finite_first.any():
        return 0.0
    return float(np.max(np.abs(first[finite_first] - second[finite_first])))


def dmap_image_id(path: Path) -> int | None:
    match = re.fullmatch(r"depth(\d+)\.dmap", path.name)
    return int(match.group(1)) if match else None


def validate(arguments: Arguments) -> dict[str, Any]:
    output_dir = arguments.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite disabled-contract output: {output_dir}")
    output_dir.mkdir(parents=True)
    source_work = arguments.source_work.expanduser().resolve()
    source_mvs = arguments.mvs_file.expanduser().resolve()
    work_dir = output_dir / "work"
    shutil.copytree(
        source_work,
        work_dir,
        copy_function=hardlink_or_copy,
        ignore=shutil.ignore_patterns("depth*.dmap", "*.log", "*_dense.mvs"),
    )
    for directory in [work_dir, *(path for path in work_dir.rglob("*") if path.is_dir())]:
        directory.chmod(directory.stat().st_mode | stat.S_IWUSR)
    try:
        relative_mvs = source_mvs.relative_to(source_work)
    except ValueError:
        relative_mvs = Path(source_mvs.name)
    local_mvs = work_dir / relative_mvs
    if not local_mvs.is_file():
        local_mvs.parent.mkdir(parents=True, exist_ok=True)
        hardlink_or_copy(str(source_mvs), str(local_mvs))
    command_work_dir = local_mvs.parent
    command = [
        str(arguments.densify_bin.expanduser().resolve()),
        "--working-folder", str(command_work_dir),
        "--input-file", str(local_mvs),
        "--output-file", str(output_dir / "disabled_dense.mvs"),
        "--fusion-mode", "1",
        "--gpu-device", str(arguments.gpu_device),
        "--resolution-level", str(arguments.resolution_level),
        "--sub-resolution-levels", str(arguments.sub_resolution_levels),
        "--number-views", str(arguments.number_views),
        "--number-views-fuse", str(arguments.number_views_fuse),
        "--patch-match-cuda-instances", str(arguments.patch_match_cuda_instances),
        "--iters", str(arguments.iters),
        "--geometric-iters", str(arguments.geometric_iters),
    ]
    (output_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True)
    elapsed_seconds = time.perf_counter() - started

    generated = {}
    for path in sorted(command_work_dir.glob("depth*.dmap")):
        image_id = dmap_image_id(path)
        if image_id is not None:
            generated[image_id] = path
    reference_dir = arguments.reference_dmap_dir.expanduser().resolve()
    requested_ids = sorted(arguments.image_ids or generated.keys())
    comparisons: dict[str, Any] = {}
    missing_outputs = []
    missing_references = []
    for image_id in requested_ids:
        output_path = generated.get(image_id)
        reference_path = reference_dir / f"depth{image_id:04d}.dmap"
        if output_path is None:
            missing_outputs.append(image_id)
            continue
        if not reference_path.is_file():
            missing_references.append(image_id)
            continue
        output = load_dmap(output_path)
        reference = load_dmap(reference_path)
        comparisons[str(image_id)] = {
            key: max_abs_difference(output[key], reference[key])
            for key in ("depth_map", "normal_map", "confidence_map")
            if key in output and key in reference
        }
    forbidden = []
    forbidden_names = {
        "dmap_instrumentation",
        "run_metadata.json",
        "scene_summary.json",
        "map_manifest.json",
        "counters.csv",
        "traces.jsonl",
        "timings.csv",
    }
    for path in output_dir.rglob("*"):
        if path.name in forbidden_names:
            forbidden.append(str(path))
    parity_exact = bool(comparisons) and all(
        len(values) == 3 and all(value == 0.0 for value in values.values())
        for values in comparisons.values()
    )
    checks = {
        "command_succeeded": completed.returncode == 0,
        "no_instrumentation_artifacts": not forbidden,
        "requested_outputs_present": not missing_outputs,
        "references_present": not missing_references,
        "production_dmaps_bit_exact": parity_exact,
    }
    result = {
        "schema_version": 1,
        "valid": all(checks.values()),
        "checks": checks,
        "return_code": completed.returncode,
        "elapsed_seconds": elapsed_seconds,
        "command": command,
        "generated_dmaps": {str(key): str(value) for key, value in generated.items()},
        "comparisons_max_abs": comparisons,
        "missing_outputs": missing_outputs,
        "missing_references": missing_references,
        "forbidden_instrumentation_artifacts": forbidden,
    }
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    result = validate(tyro.cli(Arguments))
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
