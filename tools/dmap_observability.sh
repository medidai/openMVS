#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
production_build="${DMAP_PRODUCTION_BUILD:-${repo_root}/build-dmap-production}"
observer_build="${DMAP_OBSERVER_BUILD:-${repo_root}/build-dmap-observer}"
production_bin="${DMAP_PRODUCTION_BIN:-${production_build}/bin/DensifyPointCloud}"
observer_bin="${DMAP_OBSERVER_BIN:-${observer_build}/bin/DensifyPointCloudDMapObserve}"
dmap_dev="${repo_root}/scripts/python/dmap_dev.py"
bundle_module="scripts.python.dmap_observability.portable_bundle"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage: tools/dmap_observability.sh COMMAND [OPTIONS]

Commands:
  doctor     Check public-tree, build, CUDA, and Python prerequisites.
  build      Build separate production and observer Release binaries.
  init       Resolve and lock an experiment; estimate storage without capture.
  capture    Run selected endpoint/summary/prefilter/deep profiles; do not report.
  report     Generate the canonical Markdown/model/HTML report.
  validate   Validate a generated report and its report-owned evidence.
  serve      Validate and serve an investigation report over HTTP.
  drilldown  Create or execute a paired frame/pixel/ROI trace request.
  array-store Convert selected validated frame maps to immutable Zarr v3 stores.
  package    Create and validate a sanitized portable review archive.
  demo       Generate an end-to-end ephemeral example under /tmp.

Common environment:
  PYTHON                 Python interpreter (default: python3).
  DMAP_PRODUCTION_BUILD  Production build directory.
  DMAP_OBSERVER_BUILD    Observer build directory.
  DMAP_PRODUCTION_BIN    Production DensifyPointCloud path.
  DMAP_OBSERVER_BIN      DensifyPointCloudDMapObserve path.
  DMAP_PUBLIC_BASE       Git ref used by the publication guard.

Run `tools/dmap_observability.sh COMMAND --help` for command details.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

need_value() {
  local option="$1"
  local count="$2"
  ((count >= 2)) || fail "${option} requires a value"
}

require_dmap_dev() {
  [[ -f "${dmap_dev}" ]] || fail \
    "public observability Python tooling is unavailable at ${dmap_dev}"
}

require_regular_directory() {
  local path="$1"
  [[ -d "${path}" && ! -L "${path}" ]] || fail \
    "expected a regular directory: ${path}"
}

external_output_path() {
  local path="$1"
  local label="$2"
  local lexical
  local resolved
  lexical="$(realpath -ms -- "${path}")" || fail "cannot resolve ${label}: ${path}"
  resolved="$(realpath -m -- "${path}")" || fail "cannot resolve ${label}: ${path}"
  case "${lexical}" in
    "${repo_root}"|"${repo_root}"/*)
      fail "${label} must be outside the OpenMVS source tree: ${lexical}"
      ;;
  esac
  case "${resolved}" in
    "${repo_root}"|"${repo_root}"/*)
      fail "${label} resolves inside the OpenMVS source tree: ${resolved}"
      ;;
  esac
  printf '%s\n' "${resolved}"
}

validate_build_cmake_arg() {
  local argument="$1"
  case "${argument}" in
    *"_USE_DMAP_INSTRUMENTATION"*)
      fail "--cmake-arg cannot inject the private instrumentation source guard: ${argument}"
      ;;
    -DCMAKE_BUILD_TYPE=*|-DCMAKE_BUILD_TYPE:*=*|-UCMAKE_BUILD_TYPE*)
      fail "--cmake-arg cannot override the required Release build type: ${argument}"
      ;;
    -DOpenMVS_USE_CUDA=*|-DOpenMVS_USE_CUDA:*=*|-UOpenMVS_USE_CUDA*)
      fail "--cmake-arg cannot override the required CUDA build: ${argument}"
      ;;
    -DOpenMVS_DMAP_INSTRUMENTATION=*|-DOpenMVS_DMAP_INSTRUMENTATION:*=*|-UOpenMVS_DMAP_INSTRUMENTATION*)
      fail "--cmake-arg cannot override the production/observer boundary: ${argument}"
      ;;
  esac
}

canonical_report() {
  local report_dir="$1"
  local report="${report_dir}/01_development_report.md"
  [[ -f "${report}" ]] || fail "canonical report is missing: ${report}"
  printf '%s\n' "${report}"
}

command_doctor() {
  local public_base="${DMAP_PUBLIC_BASE:-origin/develop}"
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh doctor [OPTIONS]

Options:
  --python PATH          Python interpreter.
  --production-bin PATH Production DensifyPointCloud binary.
  --observer-bin PATH   Observer DensifyPointCloudDMapObserve binary.
  --public-base REF     Git base for publication and disabled-source checks.
EOF
    return 0
  fi
  while (($#)); do
    case "$1" in
      --python) need_value "$1" "$#"; python_bin="$2"; shift 2 ;;
      --production-bin) need_value "$1" "$#"; production_bin="$2"; shift 2 ;;
      --observer-bin) need_value "$1" "$#"; observer_bin="$2"; shift 2 ;;
      --public-base) need_value "$1" "$#"; public_base="$2"; shift 2 ;;
      *) fail "unknown doctor option: $1" ;;
    esac
  done

  local status=0
  local item
  for item in git cmake ninja pandoc "${python_bin}"; do
    if command -v "${item}" >/dev/null 2>&1; then
      printf 'ok: command %s\n' "${item}"
    else
      printf 'missing: command %s\n' "${item}" >&2
      status=1
    fi
  done
  if command -v nvcc >/dev/null 2>&1; then
    printf 'ok: CUDA compiler %s\n' "$(command -v nvcc)"
  else
    printf 'missing: nvcc is not on PATH; add the CUDA bin directory before building\n' >&2
    status=1
  fi
  if [[ -n "${VCPKG_ROOT:-}" && -f "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" ]]; then
    printf 'ok: vcpkg toolchain %s\n' "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
  elif [[ -n "${VCPKG_ROOT:-}" ]]; then
    printf 'invalid: VCPKG_ROOT does not contain scripts/buildsystems/vcpkg.cmake: %s\n' \
      "${VCPKG_ROOT}" >&2
    status=1
  else
    printf 'warning: VCPKG_ROOT is unset or does not contain the vcpkg toolchain; pass an explicit CMake toolchain if dependencies are not installed system-wide\n' >&2
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'ok: NVIDIA tooling available\n'
  else
    printf 'warning: nvidia-smi is unavailable; CUDA capture cannot be qualified here\n' >&2
  fi
  if [[ -x "${production_bin}" ]]; then
    printf 'ok: production binary %s\n' "${production_bin}"
  else
    printf 'not built: production binary %s\n' "${production_bin}" >&2
  fi
  if [[ -x "${observer_bin}" ]]; then
    printf 'ok: observer binary %s\n' "${observer_bin}"
  else
    printf 'not built: observer binary %s\n' "${observer_bin}" >&2
  fi
  if [[ -f "${dmap_dev}" ]]; then
    if "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
      printf 'ok: Python 3.11 or newer\n'
    else
      printf 'unsupported: array-store tooling requires Python 3.11 or newer\n' >&2
      status=1
    fi
    local module
    for module in numpy pandas yaml tyro matplotlib PIL rich zstandard; do
      if "${python_bin}" -c "import ${module}" >/dev/null 2>&1; then
        printf 'ok: required Python module %s\n' "${module}"
      else
        printf 'missing: required Python module %s for %s\n' "${module}" "${python_bin}" >&2
        status=1
      fi
    done
    for module in pyarrow plotly jinja2 zarr numcodecs; do
      if "${python_bin}" -c "import ${module}" >/dev/null 2>&1; then
        printf 'ok: optional Python module %s\n' "${module}"
      else
        printf 'warning: optional Python module %s is unavailable for %s; related archival or plotting extras are disabled\n' \
          "${module}" "${python_bin}" >&2
      fi
    done
    if "${python_bin}" -c \
      'import scripts.python.dmap_observability.component_registry' >/dev/null 2>&1; then
      printf 'ok: observability Python package\n'
    else
      printf 'missing: observability Python package import\n' >&2
      status=1
    fi
  else
    printf 'missing: public observability Python entry point %s\n' "${dmap_dev}" >&2
    status=1
  fi
  if [[ -f "${repo_root}/tools/check_dmap_observability_public_tree.py" ]]; then
    if ! git -C "${repo_root}" rev-parse --verify --quiet "${public_base}^{commit}" >/dev/null; then
      printf 'missing: publication base ref %s (set DMAP_PUBLIC_BASE or --public-base)\n' \
        "${public_base}" >&2
      status=1
    elif "${python_bin}" "${repo_root}/tools/check_dmap_observability_public_tree.py" \
      --repo "${repo_root}" --base "${public_base}"; then
      printf 'ok: public-tree publication guard against %s\n' "${public_base}"
    else
      printf 'failed: public-tree publication guard\n' >&2
      status=1
    fi
  else
    printf 'warning: public-tree publication guard is not installed\n' >&2
  fi
  if [[ ! -x "${production_bin}" || ! -x "${observer_bin}" ]]; then
    printf 'next: tools/dmap_observability.sh build\n'
  fi
  return "${status}"
}

command_build() {
  local jobs
  jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '4')"
  local generator="Ninja"
  local -a cmake_args=()
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh build [OPTIONS]

Options:
  --production-dir PATH Production build directory.
  --observer-dir PATH   Observer build directory.
  --jobs N              Parallel build jobs.
  --generator NAME      CMake generator (default: Ninja).
  --cmake-arg VALUE     Additional configure argument; repeat as needed.
EOF
    return 0
  fi
  while (($#)); do
    case "$1" in
      --production-dir) need_value "$1" "$#"; production_build="$2"; shift 2 ;;
      --observer-dir) need_value "$1" "$#"; observer_build="$2"; shift 2 ;;
      --jobs) need_value "$1" "$#"; jobs="$2"; shift 2 ;;
      --generator) need_value "$1" "$#"; generator="$2"; shift 2 ;;
      --cmake-arg) need_value "$1" "$#"; cmake_args+=("$2"); shift 2 ;;
      --cmake-arg=*) cmake_args+=("${1#--cmake-arg=}"); shift ;;
      *) fail "unknown build option: $1" ;;
    esac
  done
  [[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer"

  local production_build_resolved
  local observer_build_resolved
  production_build_resolved="$(realpath -m -- "${production_build}")"
  observer_build_resolved="$(realpath -m -- "${observer_build}")"
  [[ "${production_build_resolved}" != "${observer_build_resolved}" ]] || fail \
    "production and observer build directories must be distinct"
  local cmake_argument
  for cmake_argument in "${cmake_args[@]}"; do
    validate_build_cmake_arg "${cmake_argument}"
  done
  command -v nvcc >/dev/null 2>&1 || fail \
    "nvcc is not on PATH; add the CUDA bin directory before building"
  if [[ -n "${VCPKG_ROOT:-}" && ! -f "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" ]]; then
    fail "VCPKG_ROOT does not contain scripts/buildsystems/vcpkg.cmake: ${VCPKG_ROOT}"
  fi

  cmake "${cmake_args[@]}" \
    -S "${repo_root}" -B "${production_build_resolved}" -G "${generator}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenMVS_USE_CUDA=ON \
    -DOpenMVS_DMAP_INSTRUMENTATION=OFF
  cmake --build "${production_build_resolved}" --target DensifyPointCloud --parallel "${jobs}"

  cmake "${cmake_args[@]}" \
    -S "${repo_root}" -B "${observer_build_resolved}" -G "${generator}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenMVS_USE_CUDA=ON \
    -DOpenMVS_DMAP_INSTRUMENTATION=ON
  cmake --build "${observer_build_resolved}" --target DensifyPointCloudDMapObserve --parallel "${jobs}"

  printf 'production: %s\n' "${production_build_resolved}/bin/DensifyPointCloud"
  printf 'observer:   %s\n' "${observer_build_resolved}/bin/DensifyPointCloudDMapObserve"
}

command_init() {
  if (($#)) && [[ "$1" == "--help" ]]; then
    require_dmap_dev
    exec "${python_bin}" "${dmap_dev}" prepare --help
  fi
  require_dmap_dev
  exec "${python_bin}" "${dmap_dev}" prepare "$@"
}

command_capture() {
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh capture --config PATH [OPTIONS]

Options:
  --profile PROFILE [PROFILE ...]
                         Capture endpoint, summary, prefilter, and/or deep.
  --dry-run              Write command plans without launching CUDA work.
  --allow-over-budget    Explicitly override storage admission failure.

Capture never generates a report. Run `tools/dmap_observability.sh report`
after the requested capture profiles complete.
EOF
    return 0
  fi
  require_dmap_dev
  local has_skip=false
  local argument
  for argument in "$@"; do
    [[ "${argument}" != "--no-skip-report" ]] || fail \
      "capture never generates a report; use the report command separately"
    [[ "${argument}" == "--skip-report" ]] && has_skip=true
  done
  if [[ "${has_skip}" == true ]]; then
    exec "${python_bin}" "${dmap_dev}" run "$@"
  fi
  exec "${python_bin}" "${dmap_dev}" run "$@" --skip-report
}

command_array_store() {
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh array-store --config PATH [OPTIONS]

Convert selected validated frame maps to immutable Zarr v3 stores.

Options:
  --run LABEL [...]     Restrict conversion to run labels.
  --scene ID [...]      Restrict conversion to scene IDs.
  --frame ID [...]      Restrict conversion to frame IDs.
  --chunk-size N        Zarr chunk edge length.
  --shard-size N        Zarr shard edge length.
  --zstd-level N        Compression level.
  --allow-incomplete    Include otherwise incomplete selected inputs.
  --max-uncompressed-bytes-per-store N
                        Fail a store that exceeds this decoded-size bound.
  --no-verify-data      Skip full payload verification after materialization.

Stores are experiment-owned archival derivatives; report generation does not
require conversion.
EOF
    return 0
  fi
  require_dmap_dev
  exec "${python_bin}" "${dmap_dev}" array-store "$@"
}

command_report() {
  local report_dir=""
  local rebuild=false
  local -a forwarded=()
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh report --config PATH [OPTIONS]

Options:
  --report-dir PATH      Canonical report directory.
  --rebuild-report       Build in a validated sibling stage, then promote.
  --evidence-context PATH
                         Optional versioned external evidence context.
  --skip-diagnostics     Omit expensive diagnostic visual synthesis.
  --python PATH          Python interpreter.

Other supported dmap_dev report options are forwarded unchanged.
EOF
    return 0
  fi
  while (($#)); do
    case "$1" in
      --report-dir)
        need_value "$1" "$#"
        report_dir="$2"
        shift 2
        ;;
      --rebuild-report)
        rebuild=true
        shift
        ;;
      --python)
        need_value "$1" "$#"
        python_bin="$2"
        shift 2
        ;;
      *)
        forwarded+=("$1")
        shift
        ;;
    esac
  done
  require_dmap_dev
  if [[ "${rebuild}" == false ]]; then
    if [[ -n "${report_dir}" ]]; then
      exec "${python_bin}" "${dmap_dev}" report \
        "${forwarded[@]}" --output-dir "${report_dir}"
    fi
    exec "${python_bin}" "${dmap_dev}" report "${forwarded[@]}"
  fi

  [[ -n "${report_dir}" ]] || fail "--rebuild-report requires --report-dir"
  [[ ! -L "${report_dir}" ]] || fail "report directory must not be a symlink"
  report_dir="$(external_output_path "${report_dir}" "report directory")"
  local parent
  parent="$(dirname "${report_dir}")"
  mkdir -p "${parent}"
  local stage
  stage="$(mktemp -d "${report_dir}.staging.XXXXXX")"
  local cleanup_command
  printf -v cleanup_command 'rm -rf -- %q' "${stage}"
  trap "${cleanup_command}" EXIT
  printf 'staging report: %s\n' "${stage}"
  "${python_bin}" "${dmap_dev}" report "${forwarded[@]}" \
    --output-dir "${stage}" --published-output-dir "${report_dir}"
  "${python_bin}" "${dmap_dev}" validate \
    --report "${stage}/01_development_report.md"

  local backup=""
  if [[ -e "${report_dir}" ]]; then
    require_regular_directory "${report_dir}"
    backup="${report_dir}.previous.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv "${report_dir}" "${backup}"
  fi
  if ! mv "${stage}" "${report_dir}"; then
    if [[ -n "${backup}" && ! -e "${report_dir}" ]]; then
      mv "${backup}" "${report_dir}"
    fi
    fail "failed to promote staged report; prior report was restored"
  fi
  trap - EXIT
  printf 'report: %s\n' "${report_dir}/01_development_report.md"
  if [[ -n "${backup}" ]]; then
    printf 'previous report preserved: %s\n' "${backup}"
  fi
}

command_validate() {
  local report=""
  local report_dir=""
  local config=""
  local evidence_context=""
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh validate [OPTIONS]

Choose one report location:
  --report PATH          Canonical Markdown report path.
  --report-dir PATH      Directory containing 01_development_report.md.

Optional identity checks:
  --config PATH          Require the report's bound source-config identity.
  --evidence-context PATH
                         Require the report's bound external-context identity.
  --python PATH          Python interpreter.
EOF
    return 0
  fi
  while (($#)); do
    case "$1" in
      --report) need_value "$1" "$#"; report="$2"; shift 2 ;;
      --report-dir) need_value "$1" "$#"; report_dir="$2"; shift 2 ;;
      --config) need_value "$1" "$#"; config="$2"; shift 2 ;;
      --evidence-context) need_value "$1" "$#"; evidence_context="$2"; shift 2 ;;
      --python) need_value "$1" "$#"; python_bin="$2"; shift 2 ;;
      *) fail "unknown validate option: $1" ;;
    esac
  done
  require_dmap_dev
  [[ -z "${report}" || -z "${report_dir}" ]] || fail \
    "validate accepts either --report or --report-dir, not both"
  if [[ -n "${report_dir}" ]]; then
    require_regular_directory "${report_dir}"
    report="$(canonical_report "${report_dir}")"
  fi
  [[ -n "${report}" ]] || fail "validate requires --report or --report-dir"
  report="$(realpath -m "${report}")"
  [[ -f "${report}" ]] || fail "canonical report is missing: ${report}"
  report_dir="$(dirname "${report}")"
  "${python_bin}" "${dmap_dev}" validate --report "${report}"

  if [[ -n "${config}" || -n "${evidence_context}" ]]; then
    local policy="${report_dir}/report_policy.json"
    [[ -f "${policy}" ]] || fail "report policy is missing: ${policy}"
    "${python_bin}" -c '
import json
import sys
from pathlib import Path
from scripts.python import dmap_dev

policy_path = Path(sys.argv[1])
config_path = sys.argv[2]
context_path = sys.argv[3]
value = json.loads(policy_path.read_text(encoding="utf-8"))
checks = (("source_config", config_path), ("evidence_context", context_path))
for key, raw_path in checks:
    if not raw_path:
        continue
    expected = value.get(key)
    actual = dmap_dev.file_identity(Path(raw_path))
    if expected != actual:
        raise SystemExit(
            f"{key} identity does not match report policy: expected={expected!r}, actual={actual!r}"
        )
' "${policy}" "${config}" "${evidence_context}"
  fi
}

command_serve() {
  local report_dir=""
  local bind="127.0.0.1"
  local port="8765"
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh serve --report-dir PATH [OPTIONS]

Options:
  --bind ADDRESS  HTTP bind address (default: 127.0.0.1).
  --port PORT     HTTP port (default: 8765).
  --python PATH   Python interpreter.
EOF
    return 0
  fi
  while (($#)); do
    case "$1" in
      --report-dir) need_value "$1" "$#"; report_dir="$2"; shift 2 ;;
      --bind) need_value "$1" "$#"; bind="$2"; shift 2 ;;
      --port) need_value "$1" "$#"; port="$2"; shift 2 ;;
      --python) need_value "$1" "$#"; python_bin="$2"; shift 2 ;;
      *) fail "unknown serve option: $1" ;;
    esac
  done
  [[ -n "${report_dir}" ]] || fail "serve requires --report-dir"
  [[ "${port}" =~ ^[1-9][0-9]*$ ]] || fail "--port must be a positive integer"
  require_regular_directory "${report_dir}"
  require_dmap_dev
  local report
  report="$(canonical_report "${report_dir}")"
  [[ -f "${report_dir}/02_investigation.html" ]] || fail \
    "investigation UI is missing: ${report_dir}/02_investigation.html"
  "${python_bin}" "${dmap_dev}" validate --report "${report}"
  printf 'URL: http://%s:%s/02_investigation.html\n' "${bind}" "${port}"
  exec "${python_bin}" -m http.server "${port}" \
    --bind "${bind}" --directory "${report_dir}"
}

command_drilldown() {
  if (($#)) && [[ "$1" == "--help" ]]; then
    require_dmap_dev
    exec "${python_bin}" "${dmap_dev}" drilldown --help
  fi
  require_dmap_dev
  exec "${python_bin}" "${dmap_dev}" drilldown "$@"
}

command_package() {
  local output=""
  local max_review_bytes=""
  local max_payload_bytes=""
  local overwrite=false
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh package --report-dir PATH --output PATH [OPTIONS]

Options:
  --report-dir PATH          Canonical report directory to sanitize and package.
  --output PATH              External .tar.zst archive path.
  --config PATH              Config metadata to include; repeat as needed.
  --provenance-file PATH     Provenance metadata to include; repeat as needed.
  --raw-manifest PATH        Optional bounded raw-artifact manifest.
  --max-review-bytes N       Maximum compressed archive size.
  --max-payload-bytes N      Maximum uncompressed payload size.
  --source-date-epoch N      Reproducible archive timestamp (default: 0).
  --overwrite                Replace an existing archive and checksum.

The public wrapper always stages and sanitizes the report, rejects host
references, validates the resulting archive, and keeps output outside the source
tree. Sanitizer bypass options exposed by the internal module are intentionally
unavailable here.
EOF
    return 0
  fi
  local -a forwarded=()
  while (($#)); do
    case "$1" in
      --output) need_value "$1" "$#"; output="$2"; shift 2 ;;
      --output=*) output="${1#--output=}"; shift ;;
      --max-review-bytes)
        need_value "$1" "$#"
        max_review_bytes="$2"
        forwarded+=("$1" "$2")
        shift 2
        ;;
      --max-review-bytes=*)
        max_review_bytes="${1#--max-review-bytes=}"
        forwarded+=("$1")
        shift
        ;;
      --max-payload-bytes)
        need_value "$1" "$#"
        max_payload_bytes="$2"
        forwarded+=("$1" "$2")
        shift 2
        ;;
      --max-payload-bytes=*)
        max_payload_bytes="${1#--max-payload-bytes=}"
        forwarded+=("$1")
        shift
        ;;
      --overwrite)
        overwrite=true
        shift
        ;;
      --allow*|--no*)
        fail "the public wrapper does not permit sanitizer bypasses in shared packages: $1"
        ;;
      *) forwarded+=("$1"); shift ;;
    esac
  done
  [[ -n "${output}" ]] || fail "package requires --output"
  output="$(external_output_path "${output}" "bundle output")"
  local output_checksum="${output}.sha256"
  if [[ "${overwrite}" == false ]] && (
    [[ -e "${output}" || -L "${output}" ]] ||
    [[ -e "${output_checksum}" || -L "${output_checksum}" ]]
  ); then
    fail "refusing to overwrite an existing bundle or checksum: ${output}"
  fi
  local output_parent
  output_parent="$(dirname "${output}")"
  mkdir -p "${output_parent}"
  local stage
  stage="$(mktemp -d "${output_parent}/.$(basename "${output}").staging.XXXXXX")"
  local cleanup_command
  printf -v cleanup_command 'rm -rf -- %q' "${stage}"
  trap "${cleanup_command}" EXIT
  local staged_output="${stage}/$(basename "${output}")"
  local package_result
  if ! package_result="$("${python_bin}" -m "${bundle_module}" package \
    "${forwarded[@]}" --output "${staged_output}")"; then
    return 1
  fi
  local -a validation_limits=()
  if [[ -n "${max_review_bytes}" ]]; then
    validation_limits+=(--max-review-bytes "${max_review_bytes}")
  fi
  if [[ -n "${max_payload_bytes}" ]]; then
    validation_limits+=(--max-payload-bytes "${max_payload_bytes}")
  fi
  if ! "${python_bin}" -m "${bundle_module}" validate \
    --bundle "${staged_output}" "${validation_limits[@]}" >/dev/null; then
    return 1
  fi

  # Package and validate away from the final names. Publishing both files is not
  # power-loss atomic, but ordinary failures roll back the pair and --overwrite
  # keeps the previous valid pair until both staged files have been validated.
  if ! "${python_bin}" - "${staged_output}" "${output}" "${overwrite}" "${stage}" <<'PY'
import os
from pathlib import Path
import stat
import sys

source_archive = Path(sys.argv[1])
destination_archive = Path(sys.argv[2])
overwrite = sys.argv[3] == "true"
stage = Path(sys.argv[4])
sources = [source_archive, source_archive.with_name(source_archive.name + ".sha256")]
destinations = [
    destination_archive,
    destination_archive.with_name(destination_archive.name + ".sha256"),
]
for source in sources:
    metadata = source.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"staged package output is not a regular file: {source}")

if not overwrite:
    linked: list[Path] = []
    try:
        for source, destination in zip(sources, destinations, strict=True):
            os.link(source, destination, follow_symlinks=False)
            linked.append(destination)
    except BaseException:
        for destination in reversed(linked):
            destination.unlink(missing_ok=True)
        raise
else:
    for destination in destinations:
        if not os.path.lexists(destination):
            continue
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"refusing to overwrite a non-regular package output: {destination}"
            )
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for index, destination in enumerate(destinations):
            if not os.path.lexists(destination):
                continue
            backup = stage / f"previous_{index}"
            os.replace(destination, backup)
            backups.append((backup, destination))
        for source, destination in zip(sources, destinations, strict=True):
            os.replace(source, destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for backup, destination in reversed(backups):
            os.replace(backup, destination)
        raise
PY
  then
    return 1
  fi
  rm -rf -- "${stage}"
  trap - EXIT
  "${python_bin}" -c '
import json
import sys

value = json.loads(sys.argv[1])
value["archive"] = sys.argv[2]
value["checksum"] = sys.argv[2] + ".sha256"
print(json.dumps(value, sort_keys=True))
' "${package_result}" "${output}"
}

write_demo_inputs() {
  local output="$1"
  local config="$2"
  local sub_resolution_levels="$3"
  local geometric_iters="$4"
  local annotations="${output}/inputs/annotation_sidecar.json"
  mkdir -p "${output}/inputs/annotation-dataset" "${output}/inputs/cache"
  cat >"${annotations}" <<'EOF'
{
  "schema_name": "openmvs.dmap.annotation_sidecar",
  "schema_version": 1,
  "scene_id": "public-test-scene",
  "image_mapping": {"demo-frame-0": "images/00000.jpg"},
  "frames": [
    {"id": "demo-frame-0", "imageResolution": {"width": 640, "height": 479}}
  ],
  "annotations": {
    "controlEdges": [
      {
        "id": "demo-edge",
        "chunks": [
          {
            "id": "demo-edge-0",
            "frameId": "demo-frame-0",
            "start": {"x": 80.0, "y": 240.0},
            "end": {"x": 560.0, "y": 240.0}
          }
        ]
      }
    ],
    "controlPlanes": [
      {
        "id": "demo-plane",
        "chunks": [
          {
            "id": "demo-plane-0",
            "frameId": "demo-frame-0",
            "points": [
              {"x": 160.0, "y": 120.0},
              {"x": 480.0, "y": 120.0},
              {"x": 480.0, "y": 360.0},
              {"x": 160.0, "y": 360.0}
            ]
          }
        ]
      }
    ]
  },
  "camera_final": {"width": 640, "height": 479}
}
EOF

  cat >"${config}" <<EOF
schema_version: 2
experiment_id: 01_ephemeral_demo
hypothesis: >-
  An additional logical PatchMatch iteration changes cost convergence and update
  mechanics on the public test scene; this demo exercises investigation rather
  than asserting that either configuration is better.
output_root: ${output}/experiments
dataset_root: ${output}/inputs/annotation-dataset
cache_root: ${output}/inputs/cache
densify_bin: ${production_bin}
densify_observe_bin: ${observer_bin}
capture_profiles: [endpoint, summary, prefilter, deep]
suite:
  name: smoke
  scan_ids: [public-test-scene]
instrumentation:
  expected_width: 640
  expected_height: 479
  expected_frames_per_scene: 1
  estimated_compression_ratio: 1.0
  max_artifact_gb: 2
  maps_capability: true
  allow_process_specialization_divergence_for_diagnostics: true
evaluation:
  annotation_space: final
  ransac_threshold_m: 0.02
  ransac_thresholds_m: [0.005, 0.01, 0.02, 0.05]
  edge_ribbon_source_px: 5.0
  max_ransac_points: 50000
  ransac_trials: 2000
  seed: 0
  plane_grid: 32
  line_bins: 100
  max_visual_points: 6000
scenes:
  - scan_id: public-test-scene
    name: public_test_scene
    working_folder: ${repo_root}/apps/Tests/data
    mvs_file: ${repo_root}/apps/Tests/data/scene.mvs
    annotation_sidecar: ${annotations}
default_densify_args:
  - --gpu-device
  - "0"
  - --fusion-mode
  - "1"
  - --resolution-level
  - "1"
  - --sub-resolution-levels
  - "${sub_resolution_levels}"
  - --number-views
  - "3"
  - --number-views-fuse
  - "2"
  - --patch-match-cuda-instances
  - "1"
  - --dmap-instrumentation-image-list
  - "0"
  - --dmap-instrumentation-sample-seed
  - "0"
  - --dmap-instrumentation-max-device-mb
  - "1024"
  - --dmap-instrumentation-max-host-mb
  - "2048"
  - --dmap-instrumentation-max-frame-storage-mb
  - "2048"
  - --dmap-instrumentation-budget-policy
  - error
runs:
  - label: baseline
    role: baseline
    repeats: 1
    densify_args: [--iters, "1", --geometric-iters, "${geometric_iters}"]
  - label: candidate
    role: variant
    repeats: 1
    densify_args: [--iters, "2", --geometric-iters, "${geometric_iters}"]
EOF
}

command_demo() {
  local output="/tmp/openmvs-dmap-observability-demo"
  local all_profiles=false
  local sub_resolution_levels=0
  local geometric_iters=0
  if (($#)) && [[ "$1" == "--help" ]]; then
    cat <<'EOF'
Usage: tools/dmap_observability.sh demo [OPTIONS]

Options:
  --output PATH          Empty external output directory (default under /tmp).
  --production-bin PATH Production DensifyPointCloud binary.
  --observer-bin PATH   Observer DensifyPointCloudDMapObserve binary.
  --python PATH          Python interpreter.
  --all-profiles         Add a paired exact trace and require all five capture profiles.
  --all-levels           Backward-compatible alias for --all-profiles.
  --multiscale           Capture and require pyramid levels 0 and 1.
  --geometric-iters N    Capture and require N geometric-consistency stages.

The default demo captures the four core profiles: endpoint, summary, prefilter,
and deep. --all-profiles adds trace as the fifth capture profile by running a
separate full-frame Process<true> rerun for one fixed trace pixel. "Profiles"
means endpoint, summary, prefilter, deep, and trace; it does not mean CUDA or
PatchMatch pyramid levels. This mode uses more GPU time and storage. The demo
creates its synthetic annotations at runtime, never writes evidence into the
repository, and refuses nonempty output.

For the complete release matrix across profiles, pyramid levels, and estimation
stages, combine --all-profiles, --multiscale, and --geometric-iters 1.
EOF
    return 0
  fi
  while (($#)); do
    case "$1" in
      --output) need_value "$1" "$#"; output="$2"; shift 2 ;;
      --production-bin) need_value "$1" "$#"; production_bin="$2"; shift 2 ;;
      --observer-bin) need_value "$1" "$#"; observer_bin="$2"; shift 2 ;;
      --python) need_value "$1" "$#"; python_bin="$2"; shift 2 ;;
      --all-profiles|--all-levels) all_profiles=true; shift ;;
      --multiscale) sub_resolution_levels=1; shift ;;
      --geometric-iters)
        need_value "$1" "$#"
        [[ "$2" =~ ^[0-9]+$ ]] || fail "--geometric-iters must be a non-negative integer"
        geometric_iters="$2"
        shift 2
        ;;
      *) fail "unknown demo option: $1" ;;
    esac
  done
  output="$(realpath -m "${output}")"
  case "${output}/" in
    "${repo_root}/"*) fail "demo output must be outside the repository" ;;
  esac
  if [[ -e "${output}" ]]; then
    [[ -d "${output}" && ! -L "${output}" ]] || fail \
      "demo output exists and is not a regular directory: ${output}"
    if find "${output}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      fail "demo output is nonempty; choose a new --output directory"
    fi
  else
    mkdir -p "${output}"
  fi
  [[ -x "${production_bin}" ]] || fail \
    "production binary is missing; run tools/dmap_observability.sh build"
  [[ -x "${observer_bin}" ]] || fail \
    "observer binary is missing; run tools/dmap_observability.sh build"
  require_dmap_dev

  local config="${output}/experiment.yaml"
  write_demo_inputs "${output}" "${config}" "${sub_resolution_levels}" "${geometric_iters}"
  "${python_bin}" "${dmap_dev}" prepare --config "${config}"
  "${python_bin}" "${dmap_dev}" run --config "${config}" \
    --profile endpoint summary prefilter deep --skip-report
  if [[ "${all_profiles}" == true ]]; then
    "${python_bin}" "${dmap_dev}" drilldown \
      --config "${config}" \
      --scene public-test-scene \
      --frame 0 \
      --pixel 120,80 \
      --variant candidate \
      --execute
  fi
  local report_dir="${output}/experiments/01_ephemeral_demo/reports"
  "${python_bin}" "${dmap_dev}" report --config "${config}" \
    --output-dir "${report_dir}" \
    --allow-process-specialization-divergence-for-diagnostics
  "${python_bin}" "${dmap_dev}" validate \
    --report "${report_dir}/01_development_report.md"
  if [[ "${all_profiles}" == true ]]; then
    "${python_bin}" -c '
import json
import sys
from pathlib import Path

coverage = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {"endpoint", "summary", "prefilter", "deep", "trace"}
profiles = {row.get("profile"): row for row in coverage.get("profiles", [])}
failed = {
    profile: profiles.get(profile)
    for profile in expected
    if profiles.get(profile, {}).get("status") != "complete"
    or int(profiles.get(profile, {}).get("complete_units", 0)) < 1
}
if failed:
    raise SystemExit(f"all-profile demo capture assertion failed: {failed}")
' "${report_dir}/capture_profile_coverage.json"
  fi
  if ((sub_resolution_levels > 0)); then
    "${python_bin}" -c '
import json
import sys
from pathlib import Path

model = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
frames = [
    frame
    for scene in model.get("scenes", [])
    for frame in scene.get("frames", [])
]
if not frames or not any({0, 1}.issubset(set(frame.get("pyramid_levels", []))) for frame in frames):
    raise SystemExit("multiscale demo report does not expose pyramid levels 0 and 1")
coarse_maps = [
    row
    for frame in frames
    for row in frame.get("maps", [])
    if row.get("pyramid_level") == 1
]
if not any(
    row.get("signal") == "candidate_source"
    and row.get("measurement_quality") == "proxy"
    and row.get("measurement_basis") == "post_pass_change_detection"
    for row in coarse_maps
):
    raise SystemExit("multiscale demo report has no validated coarse candidate_source proxy")
coarse_availability = (
    model.get("mechanics", {}).get("coarse_compatibility_map_availability", [])
)
if not any(
    row.get("pyramid_level") == 1
    and row.get("cost_map_expected") is False
    and row.get("cost_map_available") is False
    and row.get("cost_map_unavailable_reason")
        == "production confidence maps are retained at pyramid level 0 only"
    for row in coarse_availability
):
    raise SystemExit(
        "multiscale demo report does not preserve the producer-declared "
        "coarse cost-map unavailability reason"
    )
' "${report_dir}/report_model.json"
  fi
  if [[ "${all_profiles}" == true ]]; then
    "${python_bin}" -c '
import json
import sys
from pathlib import Path

model = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_levels = set(range(int(sys.argv[2]) + 1))
expected_stages = {"photometric"} | {
    f"geometric_consistency:{index}" for index in range(int(sys.argv[3]))
}
entries = [
    entry
    for entry in model.get("drilldowns", {}).get("entries", [])
    if entry.get("status") == "complete"
]
if not entries:
    raise SystemExit("release demo has no completed exact trace entry")
for entry in entries:
    rows = entry.get("trace_data", {}).get("rows", [])
    observed_levels = {
        int(row["pyramid_level"])
        for row in rows
        if row.get("pyramid_level") is not None
    }
    observed_stages = {
        (
            f"geometric_consistency:{row.get('"'"'geometric_iteration'"'"')}"
            if row.get("estimation_stage") == "geometric_consistency"
            else "photometric"
        )
        for row in rows
    }
    if not expected_levels.issubset(observed_levels):
        raise SystemExit(
            f"release demo trace misses pyramid levels: "
            f"expected={sorted(expected_levels)} observed={sorted(observed_levels)}"
        )
    if not expected_stages.issubset(observed_stages):
        raise SystemExit(
            f"release demo trace misses estimation stages: "
            f"expected={sorted(expected_stages)} observed={sorted(observed_stages)}"
        )
' "${report_dir}/report_model.json" "${sub_resolution_levels}" "${geometric_iters}"
  fi

  printf '\nDemo complete. Generated evidence is external and ephemeral.\n'
  printf 'Coverage: %s\n' "$([[ "${all_profiles}" == true ]] && printf 'all five capture profiles' || printf 'four core profiles')"
  printf 'Pyramids: %s\n' "$([[ "${sub_resolution_levels}" -gt 0 ]] && printf 'levels 0 and 1' || printf 'level 0')"
  printf 'Stages:   photometric + %s geometric-consistency iteration(s)\n' "${geometric_iters}"
  printf 'Markdown: %s\n' "${report_dir}/01_development_report.md"
  printf 'UI:       %s\n' "${report_dir}/02_investigation.html"
  printf 'Serve:    tools/dmap_observability.sh serve --report-dir %q\n' "${report_dir}"
}

if (($# == 0)); then
  usage
  exit 0
fi

command="$1"
shift
case "${command}" in
  doctor) command_doctor "$@" ;;
  build) command_build "$@" ;;
  init) command_init "$@" ;;
  capture) command_capture "$@" ;;
  report) command_report "$@" ;;
  validate) command_validate "$@" ;;
  serve) command_serve "$@" ;;
  drilldown) command_drilldown "$@" ;;
  array-store) command_array_store "$@" ;;
  package) command_package "$@" ;;
  demo) command_demo "$@" ;;
  -h|--help|help) usage ;;
  *) fail "unknown command: ${command}" ;;
esac
