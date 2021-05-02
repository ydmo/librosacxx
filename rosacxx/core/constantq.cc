#include <rosacxx/core/constantq.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/pitch.h>

#include <cmath>

namespace rosacxx {
namespace core {

nc::NDArrayF32Ptr vqt(
        const nc::NDArrayF32Ptr&        __y,
        const float                     __sr,
        const int                       __hop_length,
        const float                     __fmin,
        const int                       __bins,
        const float                     __gamma,
        const int                       __bins_per_octave,
        const float                     __tuning,
        const float                     __filter_scale,
        const float                     __norm,
        const float                     __sparsity,
        const filters::STFTWindowType&  __window, //  = filters::STFTWindowType::Hanning,
        const bool                      __scale,
        const char *                    __pad_mode,
        const char *                    __res_type,
        const char *                    __dtype
        ) {
    int bins_per_octave = __bins_per_octave;

    int n_octaves = int(std::ceil(float(__bins) / __bins_per_octave));
    int n_filters = std::min(__bins_per_octave, __bins);

    int len_orig = __y.shape()[0];
    float alpha = std::pow(2.0, (1.0 / __bins_per_octave)) - 1;

    float fmin = __fmin;
    if (fmin == INFINITY) {
        fmin = note_to_hz("C1");
    }

    nc::NDArrayF32Ptr S = nullptr;
    float tuning = __tuning;
    if (tuning == INFINITY) {
        tuning = estimate_tuning(__y, __sr, S, 2048, 0.01f, __bins_per_octave);
    }

    float gamma = __gamma;
    if (gamma == INFINITY) {
        gamma = 24.7 * alpha / 0.108;
    }

    fmin = fmin * std::pow(2.0, (tuning / bins_per_octave));

    return nullptr;
}

nc::NDArrayF32Ptr cqt(
        const nc::NDArrayF32Ptr& y,
        const float sr,
        const int hop_length,
        const float fmin,
        const int n_bins,
        const int bins_per_octave,
        const float tuning,
        const float filter_scale,
        const float norm,
        const float sparsity,
        const char * window,
        const bool scale,
        const char * pad_mode,
        const char * res_type,
        const char * dtype
        ) {
    // CQT is the special case of VQT with gamma=0
    return vqt(y, sr, hop_length, fmin, n_bins, 0.f, bins_per_octave, tuning, filter_scale, norm, sparsity, window, scale, pad_mode, res_type, dtype);
}



} // namespace core
} // namespace rosacxx
