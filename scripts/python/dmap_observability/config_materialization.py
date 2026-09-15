"""Materialize immutable Densify.ini variants for observability runs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ini_overrides(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"INI override must use KEY=VALUE: {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key or "\n" in key or "\r" in key or "\n" in value or "\r" in value:
            raise ValueError(f"invalid INI override: {raw!r}")
        result[key] = value.strip()
    return result


def merge_ini_overrides(
    config: dict[str, Any],
    run: dict[str, Any],
    scene: dict[str, Any],
    cli_values: Iterable[str] = (),
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in (
        (config.get("sweep") or {}).get("ini_overrides") or {},
        run.get("ini_overrides") or {},
        scene.get("ini_overrides") or {},
    ):
        if not isinstance(source, dict):
            raise ValueError("ini_overrides must be a mapping")
        merged.update({str(key): str(value) for key, value in source.items()})
    merged.update(parse_ini_overrides(cli_values))
    return merged


def render_ini_override(
    source: Path, destination: Path, overrides: dict[str, str]
) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8") if source.is_file() else "[Densify]\n"
    lines = text.splitlines()
    matched: set[str] = set()
    lookup = {key.casefold(): (key, value) for key, value in overrides.items()}
    pattern = re.compile(r"^(\s*)([^#;\[=][^=]*?)(\s*)=(.*)$")
    output: list[str] = []
    for line in lines:
        match = pattern.match(line)
        if match is None:
            output.append(line)
            continue
        key = match.group(2).strip()
        selected = lookup.get(key.casefold())
        if selected is None:
            output.append(line)
            continue
        requested_key, value = selected
        output.append(f"{match.group(1)}{key}{match.group(3)}= {value}")
        matched.add(requested_key.casefold())
    missing = [key for key in overrides if key.casefold() not in matched]
    if missing:
        if output and output[-1] != "":
            output.append("")
        output.append("; Generated observability overrides")
        output.extend(f"{key} = {overrides[key]}" for key in missing)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source) if source.is_file() else None,
        "generated": str(destination.resolve()),
        "generated_sha256": sha256_file(destination),
        "overrides": overrides,
    }
