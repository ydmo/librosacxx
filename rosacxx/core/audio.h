#ifndef ROSACXX_CORE_AUDIO_H
#define ROSACXX_CORE_AUDIO_H

#include <rosacxx/core/audio.h>
#include <rosacxx/resamcxx/core.h>
#include <rosacxx/util/utils.h>
#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

constexpr double BW_BEST = 0.94759372;
constexpr double BW_FASTEST = 0.85;

// nc::NDArrayF32Ptr tone(const float& __freq, const float& __sr, const int * __length = NULL, const float * __duration = NULL, const float * __phi = NULL);

template<typename DType>
inline nc::NDArrayPtr<DType> tone(const DType __freq, const DType __sr, const int * __length = NULL, const DType * __duration = NULL, const DType * __phi = NULL) {
    if (__length == NULL && __duration == NULL) throw std::invalid_argument("__length == NULL && __duration == NULL");
    int length = __length == NULL? int(__sr * __duration[0]) : *__length;
    DType phi = __phi == NULL? -M_PI * .5 : *__phi;
    nc::NDArrayPtr<DType> ret = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({length}));
    DType *p_ret = ret.data();
    for (int i = 0; i < length; i++) {
        p_ret[i] = std::cos(2 * M_PI * __freq * i / __sr + phi);
    }
    return ret;
}

// nc::NDArrayF32Ptr resample(const nc::NDArrayF32Ptr& y, const float& origin_sr, const float& target_sr, const char * res_type, const bool& fix = true, const bool& scale = false);

template<typename DType>
inline nc::NDArrayPtr<DType> resample(const nc::NDArrayPtr<DType>& y, const double origin_sr, const double target_sr, const char * res_type, const bool fix, const bool scale) {
    double ratio = target_sr / origin_sr;
    int n_samples = int(std::ceil(y.shape().back() * ratio));
    nc::NDArrayPtr<DType> y_hat = nullptr;
    if (strcmp(res_type, "kaiser_fast") == 0 || strcmp(res_type, "kaiser_best") == 0) {
        y_hat = resam::resample(y, origin_sr, target_sr, -1, res_type);
    }
    else {
        throw std::runtime_error("Not implemented.");
    }
    if (fix) {
        y_hat = util::fix_length(y_hat, n_samples);
    }
    if (scale) {
        y_hat /= std::sqrt(ratio);
    }
    return y_hat;
}

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_AUDIO_H
