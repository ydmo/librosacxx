#ifndef ROSACXX_FEATURE_SPECTRAL_H
#define ROSACXX_FEATURE_SPECTRAL_H

#include <stdio.h>
#include <stdexcept>
#include <cfloat>
#include <cmath>

#include <3rd/numcxx/numcxx.h>
#include <rosacxx/core/constantq.h>
#include <rosacxx/filters.h>


namespace rosacxx {
namespace feature {

/// Function: chroma_cqt
/// ----------
/// Param Name              | Type          | Note
/// @param y                | NDArrayF32Ptr | Audio time series, LPCM data
/// @param sr               | float         | Sampling rate of ``y``
/// @param C                | NDArrayF32Ptr | Shape=(d, t), a pre-computed constant-Q spectrogram
/// @param hop_length       | int           | Number of samples between successive chroma frames
/// @param fmin             | float *       | minimum frequency to analyze in the CQT. Default: `C1 ~= 32.7 Hz`
/// @param norm             | int           | Column-wise normalization of the chromagram
/// @param threshold        | float         | Pre-normalization energy threshold.  Values below the threshold are discarded, resulting in a sparse chromagram
/// @param tuning           | float         | Deviation (in fractions of a CQT bin) from A440 tuning
/// @param n_chroma         | int           | Number of chroma bins to produce
/// @param n_octaves        | int           | Number of octaves to analyze above ``fmin``
/// @param window           | NDArrayF32Ptr | Optional window parameter to `filters.cq_to_chroma`
/// @param bins_per_octave  | int           | Number of bins per octave in the CQT. Must be an integer multiple of ``n_chroma``. Default: 36 (3 bins per semitone)
/// @param cqt_mode         | char *        | Constant-Q transform mode ['full', 'hybrid']
/// ----------
/// @result chromagram      | NDArrayF32Ptr | The output chromagram, shape=(n_chroma, t)
inline nc::NDArrayF32Ptr chroma_cqt(
        const nc::NDArrayF32Ptr& i_y,
        const float&                i_sr =                22050,
        const nc::NDArrayF32Ptr& i_C =                 nullptr,
        const int&                  i_hop_length =        512,
        const float&                i_fmin =              0,
        const int&                  i_norm =              0,
        const float&                i_threshold =         0,
        const float&                i_tuning =            0,
        const int&                  i_n_chroma =          12,
        const int&                  i_n_octaves =         7,
        const nc::NDArrayF32Ptr& i_window =            nullptr,
        const int&                  i_bins_per_octave =   36,
        const char *                i_cqt_mode =          "full"
        ) {

    if (i_y == nullptr) {
        throw std::invalid_argument("i_y is nullptr.");
        return nullptr;
    }

    int n_chroma = i_n_chroma;
    int bins_per_octave = i_bins_per_octave;
    if (bins_per_octave == 0) {
        bins_per_octave = n_chroma;
    }
    else {
        if (bins_per_octave % n_chroma) {
            throw std::invalid_argument("bins_per_octave must be an integer multiple of n_chroma.");
        }
    }

    // Build the CQT if we don't have one already
    nc::NDArrayF32Ptr C = nullptr;
    if (i_C == NULL) {
        if (strcmp("full", i_cqt_mode) == 0) {
            C = nullptr;
        }
        else if (strcmp("hybrid", i_cqt_mode) == 0) {
            C = nullptr;
        }
        else {
            throw std::invalid_argument("Invalid argument: i_cqt_mode");
        }
    }

    nc::NDArrayF32Ptr cq_to_chr = filters::cq_to_chroma<float>(
        C->shape()[0],
        i_bins_per_octave,
        i_n_chroma,
        i_fmin,
        i_window
        );

    nc::NDArrayF32Ptr chroma = cq_to_chr->dot(C);

    if (i_threshold != -FLT_MAX) {
        // chroma[chroma < threshold] = 0.0;
        float * ptr = chroma->data();
        for (auto i = 0; i < chroma->elemCount(); i++) {
            if (ptr[i] < i_threshold) ptr[i] = 0.0f;
        }
    }

    if (i_norm != INFINITY) {
        // chroma = util.normalize(chroma, norm=norm, axis=0)
    }

    return chroma;
}

} // namespace feature
} // namespace rosacxx

#endif // ROSACXX_FEATURE_SPECTRAL_H
