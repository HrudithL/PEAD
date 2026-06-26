#include "pead/engine.hpp"

namespace pead {

Results compute_features(const Panel& panel) {
#ifdef PEAD_USE_CUDA
    return compute_features_cuda(panel);
#else
    return compute_features_cpu(panel);
#endif
}

}  // namespace pead
