#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "pead/arrow_io.hpp"
#include "pead/engine.hpp"

namespace py = pybind11;

// Thin binding: read a panel parquet, compute features, write results parquet.
// Lets Python call the same native engine in-process instead of via subprocess.
static std::size_t compute_to_parquet(const std::string& panel_path,
                                      const std::string& out_path) {
    const pead::Panel panel = pead::read_panel(panel_path);
    const pead::Results results = pead::compute_features(panel);
    pead::write_results(results, out_path);
    return results.size();
}

PYBIND11_MODULE(pead_native, m) {
    m.doc() = "Native PEAD options compute engine (C++/CUDA).";
    m.def("compute_to_parquet", &compute_to_parquet,
          py::arg("panel_path"), py::arg("out_path"),
          "Compute per-event features from a derived panel; returns #events.");
}
