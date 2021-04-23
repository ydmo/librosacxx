#include <rosacxx/core/constantq.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/pitch.h>

#include <cmath>

namespace rosacxx {
namespace core {

nc::NDArrayF32Ptr vqt(
        const nc::NDArrayF32Ptr& i_y,
        const float i_sr,
        const int hop_length,
        const float i_fmin,
        const int n_bins,
        const float gamma,
        const int bins_per_octave,
        const float i_tuning,
        const float filter_scale,
        const float norm,
        const float sparsity,
        const char * window,
        const bool scale,
        const char * pad_mode,
        const char * res_type,
        const char * dtype
        ) {

    int n_octaves = int(std::ceil(float(n_bins) / bins_per_octave));
    int n_filters = std::min(bins_per_octave, n_bins);

    int len_orig = i_y.shape()[0];
    float alpha = std::pow(2.0, (1.0 / bins_per_octave)) - 1;

    float fmin = i_fmin;
    if (fmin == INFINITY) {
        fmin = note_to_hz("C1");
    }

    float tuning = i_tuning;
    if (tuning == INFINITY) {
        tuning = estimate_tuning(i_y, i_sr, nullptr, 2048, 0.01f, bins_per_octave);
    }


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
