---
description: "Use when configuring, building, or testing OpenMVS with CMake. Prefer the make/ build directory workflow and incremental builds."
name: "OpenMVS CMake Build Workflow"
---
# OpenMVS CMake Build Workflow

- Prefer `make/` as the CMake build directory.
- Run configure/build commands from `make/`: `cmake ..` then `cmake --build .`.
- Prefer incremental builds in the existing `make/` directory instead of creating new build folders, unless explicitly requested otherwise.
- For CMake/CTest-based test execution, use the configured build tree rooted at `make/`.
