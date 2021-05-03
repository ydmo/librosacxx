#include <rosacxx/core/constantq.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/audio.h>

#include <cmath>

namespace rosacxx {
namespace core {

/// __num_two_factors
/// Return how many times integer x can be evenly divided by 2.
/// Returns 0 for non-positive integers.
inline int __num_two_factors(const int& __x) {
    if (__x <= 0) return 0;
    int num_twos = 0;
    int x = __x;
    while (x % 2 == 0) {
        num_twos += 1;
        x = x / 2;
    }
    return num_twos;
}

/// __early_downsample_count
/// Compute the number of early downsampling operations
inline int __early_downsample_count(const float& nyquist, const float& filter_cutoff, const int& hop_length, const int& n_octaves) {
    int downsample_count1 = std::max(0, int(std::ceil(std::log2(BW_FASTEST * nyquist / filter_cutoff)) - 1) - 1);
    int num_twos = __num_two_factors(hop_length);
    int downsample_count2 = std::max(0, num_twos - n_octaves + 1);
    return std::min(downsample_count1, downsample_count2);
}

/// __early_downsample
/// Perform early downsampling on an audio signal, if it applies.
inline void __early_downsample(nc::NDArrayF32Ptr& y, int& sr, int& hop_length, const char * res_type, const int& n_octaves, const float& nyquist, const float& filter_cutoff, const float& scale) {
    int downsample_count = __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves);
    if (downsample_count > 0 && strcmp(res_type, "kaiser_fast") == 0) {
        auto downsample_factor = std::pow(2, downsample_count);
    }
}

nc::NDArrayF32Ptr vqt(
        const nc::NDArrayF32Ptr&        __y,
        const float                     __sr,
        const int                       __hop_length,
        const float                     __fmin,
        const int                       __n_bins,
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
    nc::NDArrayF32Ptr y = __y;
    int hop_length = __hop_length;
    float sr = __sr;
    int bins_per_octave = __bins_per_octave;
    int n_bins = __n_bins;
    float filter_scale = __filter_scale;
    filters::STFTWindowType window = __window;
    std::string res_type = __res_type;

    int n_octaves = int(std::ceil(float(n_bins) / bins_per_octave));
    int n_filters = std::min(bins_per_octave, n_bins);

    int len_orig = __y.shape()[0];
    float alpha = std::pow(2.0, (1.0 / bins_per_octave)) - 1;

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

    auto freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);

    float fmin_t = freqs.min();
    float fmax_t = freqs.max();

    float Q = filter_scale / alpha;
    float filter_cutoff = fmax_t * (1 + 0.5 * filters::window_bandwidth(window) / Q) + 0.5 * gamma;
    float nyquist = sr / 2.0;

    bool auto_resample = false;
    if (strlen(__res_type) > 0) {
        auto_resample = true;
        if (filter_cutoff < 0.85 * nyquist) {
            res_type = "kaiser_fast";
        }
        else {
            res_type = "kaiser_best";
        }
    }

    // y, sr, hop_length = __early_downsample(y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale);

    return nullptr;
}

nc::NDArrayF32Ptr cqt(
        const nc::NDArrayF32Ptr&        __y,
        const float&                    __sr,
        const int&                      __hop_length,
        const float&                    __fmin,
        const int&                      __n_bins,
        const int&                      __bins_per_octave,
        const float&                    __tuning,
        const float&                    __filter_scale,
        const float&                    __norm,
        const float&                    __sparsity,
        const filters::STFTWindowType&  __window, //  = filters::STFTWindowType::Hanning,
        const bool                      __scale,
        const char *                    __pad_mode,
        const char *                    __res_type,
        const char *                    __dtype
        ) {
    // CQT is the special case of VQT with gamma=0
    return vqt(__y, __sr, __hop_length, __fmin, __n_bins, 0.f, __bins_per_octave, __tuning, __filter_scale, __norm, __sparsity, __window, __scale, __pad_mode, __res_type, __dtype);
}



} // namespace core
} // namespace rosacxx
