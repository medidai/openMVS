#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "${DEBUG:-0}" == "1" ]]; then
  BUILD_TYPE="Debug"
else
  BUILD_TYPE="Release"
fi

cmake -B build -S . \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1

cmake --build build -j
