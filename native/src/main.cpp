#include <iostream>
#include <string>

#include "pead/arrow_io.hpp"
#include "pead/engine.hpp"

namespace {

void usage() {
    std::cerr << "usage: pead_engine --panel <in.parquet> --out <out.parquet>\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string panel_path, out_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--panel" && i + 1 < argc) panel_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "-h" || a == "--help") { usage(); return 0; }
    }
    if (panel_path.empty() || out_path.empty()) { usage(); return 2; }

    try {
        const pead::Panel panel = pead::read_panel(panel_path);
        const pead::Results results = pead::compute_features(panel);
        pead::write_results(results, out_path);
        std::cout << "pead_engine: " << panel.size() << " rows -> "
                  << results.size() << " events -> " << out_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "pead_engine error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
