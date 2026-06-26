#pragma once

#include <cstdint>
#include <vector>

namespace pead {

// Columnar view of the derived event panel: one entry per option-day row.
// Mirrors data/derived/event_panel.parquet (see pead/options/panel.py).
struct Panel {
    std::vector<int64_t> secid;
    std::vector<int32_t> ann_date;   // days since epoch (Arrow date32)
    std::vector<int32_t> rel_day;    // calendar days relative to announcement
    std::vector<uint8_t> is_call;    // 1 if cp_flag == 'C'
    std::vector<double>  impl_vol;
    std::vector<double>  delta;
    std::vector<double>  volume;

    std::size_t size() const { return secid.size(); }
};

// Per-event output features (one entry per (secid, ann_date) group).
struct Results {
    std::vector<int64_t> secid;
    std::vector<int32_t> ann_date;
    std::vector<double>  atm_iv_pre;
    std::vector<double>  atm_iv_post;
    std::vector<double>  iv_drift;
    std::vector<int64_t> n_pre;
    std::vector<int64_t> n_post;
    std::vector<double>  total_volume;

    std::size_t size() const { return secid.size(); }
};

// At-the-money band: a call with |abs(delta) - kAtmDelta| <= kAtmBand.
constexpr double kAtmDelta = 0.5;
constexpr double kAtmBand  = 0.1;

// Dispatch: uses the CUDA backend when built with PEAD_USE_CUDA, else CPU.
Results compute_features(const Panel& panel);

// Backends (selected by compute_features).
Results compute_features_cpu(const Panel& panel);
#ifdef PEAD_USE_CUDA
Results compute_features_cuda(const Panel& panel);
#endif

}  // namespace pead
