#pragma once

#include <string>

#include "pead/engine.hpp"

namespace pead {

// Load the derived event panel (parquet) into a columnar Panel.
Panel read_panel(const std::string& path);

// Write per-event Results to a parquet file matching the pandas-fallback schema.
void write_results(const Results& results, const std::string& path);

}  // namespace pead
