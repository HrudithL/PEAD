#include "pead/engine.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

// GPU compute backend. Grouping (secid, ann_date) is dense-indexed on the host;
// the kernel scatters per-row contributions into per-group accumulators with
// atomics. This keeps the heavy per-row work (millions of option-days) on the
// GPU while the tiny final reduction stays on the host.

namespace pead {
namespace {

#define CUDA_OK(call)                                                          \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA: ") +                   \
                                     cudaGetErrorString(_e));                  \
    } while (0)

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

__global__ void accumulate(int n,
                           const int* __restrict__ group,
                           const unsigned char* __restrict__ is_call,
                           const int* __restrict__ rel_day,
                           const double* __restrict__ impl_vol,
                           const double* __restrict__ delta,
                           const double* __restrict__ volume,
                           double* pre_sum, unsigned long long* pre_cnt,
                           double* post_sum, unsigned long long* post_cnt,
                           double* vol_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int g = group[i];
    atomicAdd(&vol_sum[g], volume[i]);
    bool atm = is_call[i] && fabs(fabs(delta[i]) - kAtmDelta) <= kAtmBand;
    if (!atm) return;
    if (rel_day[i] < 0) {
        atomicAdd(&pre_sum[g], impl_vol[i]);
        atomicAdd(&pre_cnt[g], 1ULL);
    } else if (rel_day[i] > 0) {
        atomicAdd(&post_sum[g], impl_vol[i]);
        atomicAdd(&post_cnt[g], 1ULL);
    }
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
    T* d = nullptr;
    CUDA_OK(cudaMalloc(&d, host.size() * sizeof(T)));
    CUDA_OK(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

}  // namespace

Results compute_features_cuda(const Panel& p) {
    const int n = static_cast<int>(p.size());

    // Dense group ids on the host.
    std::unordered_map<Key, int, KeyHash> ids;
    std::vector<int> group(n);
    std::vector<Key> keys;
    for (int i = 0; i < n; ++i) {
        Key k{p.secid[i], p.ann_date[i]};
        auto it = ids.find(k);
        if (it == ids.end()) {
            int id = static_cast<int>(keys.size());
            ids.emplace(k, id);
            keys.push_back(k);
            group[i] = id;
        } else {
            group[i] = it->second;
        }
    }
    const int ng = static_cast<int>(keys.size());

    // Device inputs.
    int* d_group = device_copy(group);
    unsigned char* d_call = device_copy(p.is_call);
    int* d_rel = device_copy(p.rel_day);
    double* d_iv = device_copy(p.impl_vol);
    double* d_delta = device_copy(p.delta);
    double* d_vol = device_copy(p.volume);

    // Device accumulators (zeroed).
    double *d_pre_sum, *d_post_sum, *d_vol_sum;
    unsigned long long *d_pre_cnt, *d_post_cnt;
    CUDA_OK(cudaMalloc(&d_pre_sum, ng * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_post_sum, ng * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_vol_sum, ng * sizeof(double)));
    CUDA_OK(cudaMalloc(&d_pre_cnt, ng * sizeof(unsigned long long)));
    CUDA_OK(cudaMalloc(&d_post_cnt, ng * sizeof(unsigned long long)));
    CUDA_OK(cudaMemset(d_pre_sum, 0, ng * sizeof(double)));
    CUDA_OK(cudaMemset(d_post_sum, 0, ng * sizeof(double)));
    CUDA_OK(cudaMemset(d_vol_sum, 0, ng * sizeof(double)));
    CUDA_OK(cudaMemset(d_pre_cnt, 0, ng * sizeof(unsigned long long)));
    CUDA_OK(cudaMemset(d_post_cnt, 0, ng * sizeof(unsigned long long)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    accumulate<<<blocks, threads>>>(n, d_group, d_call, d_rel, d_iv, d_delta, d_vol,
                                    d_pre_sum, d_pre_cnt, d_post_sum, d_post_cnt, d_vol_sum);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<double> pre_sum(ng), post_sum(ng), vol_sum(ng);
    std::vector<unsigned long long> pre_cnt(ng), post_cnt(ng);
    CUDA_OK(cudaMemcpy(pre_sum.data(), d_pre_sum, ng * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(post_sum.data(), d_post_sum, ng * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(vol_sum.data(), d_vol_sum, ng * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(pre_cnt.data(), d_pre_cnt, ng * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(post_cnt.data(), d_post_cnt, ng * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    for (void* ptr : {(void*)d_group, (void*)d_call, (void*)d_rel, (void*)d_iv,
                      (void*)d_delta, (void*)d_vol, (void*)d_pre_sum, (void*)d_post_sum,
                      (void*)d_vol_sum, (void*)d_pre_cnt, (void*)d_post_cnt}) {
        cudaFree(ptr);
    }

    Results r;
    r.secid.resize(ng);
    r.ann_date.resize(ng);
    r.atm_iv_pre.resize(ng);
    r.atm_iv_post.resize(ng);
    r.iv_drift.resize(ng);
    r.n_pre.resize(ng);
    r.n_post.resize(ng);
    r.total_volume.resize(ng);

    for (int g = 0; g < ng; ++g) {
        const double pre = pre_cnt[g] ? pre_sum[g] / pre_cnt[g] : std::nan("");
        const double post = post_cnt[g] ? post_sum[g] / post_cnt[g] : std::nan("");
        r.secid[g] = keys[g].secid;
        r.ann_date[g] = keys[g].ann_date;
        r.atm_iv_pre[g] = pre;
        r.atm_iv_post[g] = post;
        r.iv_drift[g] = post - pre;
        r.n_pre[g] = static_cast<int64_t>(pre_cnt[g]);
        r.n_post[g] = static_cast<int64_t>(post_cnt[g]);
        r.total_volume[g] = vol_sum[g];
    }
    return r;
}

}  // namespace pead
