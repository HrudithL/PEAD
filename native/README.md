# native — PEAD options compute engine (C++ / CUDA)

Reads the derived event panel (`data/derived/event_panel.parquet`) via Arrow,
computes per-event ATM implied-vol drift, and writes `event_results.parquet`.
This is the compute stage that consumes DuckDB's reduced output; it produces the
**same schema** as the Python fallback (`pead/options/engine.py`), so the two are
interchangeable.

## Layout

| Path | Purpose |
|------|---------|
| `include/pead/engine.hpp` | `Panel` / `Results` structs + backend declarations |
| `include/pead/arrow_io.hpp` | parquet read/write |
| `src/arrow_io.cpp` | Arrow/Parquet IO |
| `src/compute_cpu.cpp` | CPU backend (hash-grouped reduction) |
| `src/compute_cuda.cu` | CUDA backend (atomic scatter into per-group accumulators) |
| `src/compute.cpp` | backend dispatch (`PEAD_USE_CUDA`) |
| `src/main.cpp` | CLI: `--panel <in> --out <out>` |
| `bindings/pybind.cpp` | optional in-process Python module |

## Dependencies

- CMake ≥ 3.18, a C++17 compiler
- Apache Arrow + Parquet C++ (`libarrow-dev libparquet-dev`, or vcpkg/conda)
- Optional: CUDA Toolkit (for `PEAD_USE_CUDA`), pybind11 (for the Python module)

## Build

CPU only:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
```

With CUDA and the Python module:

```bash
cmake -S native -B native/build -DPEAD_USE_CUDA=ON -DPEAD_BUILD_PYBIND=ON
cmake --build native/build --config Release
```

The build produces `native/build/pead_engine`. `pead/options/engine.py`
auto-detects this binary; until it is built, the pipeline transparently uses the
pandas fallback, so `run_pead_options.py` works with or without a compiler.

## Run

```bash
native/build/pead_engine --panel data/derived/event_panel.parquet \
                         --out   data/derived/event_results.parquet
```
