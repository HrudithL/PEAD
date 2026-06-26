#include "pead/engine.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>

namespace pead {
namespace {

// Pack (secid, ann_date) into a single 96-bit-ish key via a 64-bit hash combine.
struct Key {
    int64_t secid;
    int32_t ann_date;
    bool operator==(const Key& o) const {
        return secid == o.secid && ann_date == o.ann_date;
    }
};
struct KeyHash {
    std::size_t operator()(const Key& k) const {
        std::size_t h = std::hash<int64_t>{}(k.secid);
        h ^= std::hash<int32_t>{}(k.ann_date) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct Acc {
    double pre_sum = 0.0, post_sum = 0.0, vol_sum = 0.0;
    int64_t pre_cnt = 0, post_cnt = 0;
    int order = 0;  // first-seen order for stable output
};

inline bool is_atm(uint8_t is_call, double delta) {
    return is_call && std::fabs(std::fabs(delta) - kAtmDelta) <= kAtmBand;
}

}  // namespace

Results compute_features_cpu(const Panel& p) {
    std::unordered_map<Key, Acc, KeyHash> groups;
    groups.reserve(p.size() / 64 + 16);
    int next_order = 0;

    for (std::size_t i = 0; i < p.size(); ++i) {
        Key k{p.secid[i], p.ann_date[i]};
        auto it = groups.find(k);
        if (it == groups.end()) {
            Acc a;
            a.order = next_order++;
            it = groups.emplace(k, a).first;
        }
        Acc& a = it->second;
        a.vol_sum += p.volume[i];
        if (is_atm(p.is_call[i], p.delta[i])) {
            if (p.rel_day[i] < 0) { a.pre_sum += p.impl_vol[i]; ++a.pre_cnt; }
            else if (p.rel_day[i] > 0) { a.post_sum += p.impl_vol[i]; ++a.post_cnt; }
        }
    }

    Results r;
    r.secid.resize(groups.size());
    r.ann_date.resize(groups.size());
    r.atm_iv_pre.resize(groups.size());
    r.atm_iv_post.resize(groups.size());
    r.iv_drift.resize(groups.size());
    r.n_pre.resize(groups.size());
    r.n_post.resize(groups.size());
    r.total_volume.resize(groups.size());

    for (const auto& [key, a] : groups) {
        const int idx = a.order;
        const double pre = a.pre_cnt ? a.pre_sum / a.pre_cnt : std::nan("");
        const double post = a.post_cnt ? a.post_sum / a.post_cnt : std::nan("");
        r.secid[idx] = key.secid;
        r.ann_date[idx] = key.ann_date;
        r.atm_iv_pre[idx] = pre;
        r.atm_iv_post[idx] = post;
        r.iv_drift[idx] = post - pre;
        r.n_pre[idx] = a.pre_cnt;
        r.n_post[idx] = a.post_cnt;
        r.total_volume[idx] = a.vol_sum;
    }
    return r;
}

}  // namespace pead
