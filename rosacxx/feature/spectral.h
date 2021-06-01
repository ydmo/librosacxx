#ifndef ROSACXX_FEATURE_SPECTRAL_H
#define ROSACXX_FEATURE_SPECTRAL_H

#include <stdio.h>
#include <stdexcept>
#include <cfloat>
#include <cmath>

#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/core/constantq.h>
#include <rosacxx/filters.h>

#if ROSACXX_TEST
#   include <rosacxx/util/visualize.h>
#endif

namespace rosacxx {
namespace feature {

/// Function: chroma_cqt
/// ----------
/// Param Name              | Type          | Note
/// @param y                | NDArrayF32::Ptr | Audio time series, LPCM data
/// @param sr               | float         | Sampling rate of ``y``
/// @param C                | NDArrayF32::Ptr | Shape=(d, t), a pre-computed constant-Q spectrogram
/// @param hop_length       | int           | Number of samples between successive chroma frames
/// @param fmin             | float *       | minimum frequency to analyze in the CQT. Default: `C1 ~= 32.7 Hz`
/// @param norm             | float         | Column-wise normalization of the chromagram
/// @param threshold        | float         | Pre-normalization energy threshold.  Values below the threshold are discarded, resulting in a sparse chromagram
/// @param tuning           | float         | Deviation (in fractions of a CQT bin) from A440 tuning
/// @param n_chroma         | int           | Number of chroma bins to produce
/// @param n_octaves        | int           | Number of octaves to analyze above ``fmin``
/// @param window           | NDArrayF32::Ptr | Optional window parameter to `filters.cq_to_chroma`
/// @param bins_per_octave  | int           | Number of bins per octave in the CQT. Must be an integer multiple of ``n_chroma``. Default: 36 (3 bins per semitone)
/// @param cqt_mode         | char *        | Constant-Q transform mode ['full', 'hybrid']
/// ----------
/// @result chromagram      | NDArrayF32::Ptr | The output chromagram, shape=(n_chroma, t)
inline nc::NDArrayF32Ptr chroma_cqt(
        const nc::NDArrayF32Ptr&    __y,
        const float&                __sr =                22050,
        const nc::NDArrayF32Ptr&    __C =                 nullptr,
        const int&                  __hop_length =        512,
        const float&                __fmin =              INFINITY,
        const float&                __norm =              INFINITY,
        const float&                __threshold =         0,
        const float&                __tuning =            INFINITY,
        const int&                  __n_chroma =          12,
        const int&                  __n_octaves =         7,
        const nc::NDArrayF32Ptr&    __window =            nullptr,
        const int&                  __bins_per_octave =   36,
        const char *                __cqt_mode =          "full"
        ) {
    // --------

    nc::NDArrayF32Ptr y = __y;
    float sr = __sr;
    int hop_length = __hop_length;
    float fmin = __fmin;
    int n_octaves = __n_octaves;
    int n_chroma = __n_chroma;
    int bins_per_octave = __bins_per_octave;
    float tuning = __tuning;
    nc::NDArrayF32Ptr C = __C;

    // --------

    if (y == nullptr) {
        throw std::invalid_argument("i_y is nullptr.");
        return nullptr;
    }


    if (bins_per_octave <= 0) {
        bins_per_octave = n_chroma;
    }
    else if (bins_per_octave % n_chroma) {
        throw std::invalid_argument("bins_per_octave must be an integer multiple of n_chroma.");
    }

    // Build the CQT if we don't have one already
    if (!C) {
        if (strcmp("full", __cqt_mode) == 0) {
            C = nc::abs(core::cqt(
                            y,
                            sr,
                            hop_length,
                            fmin,
                            n_octaves * bins_per_octave,
                            bins_per_octave,
                            tuning
                            )
                        );
        }
        else if (strcmp("hybrid", __cqt_mode) == 0) {
            C = nullptr;
        }
        else {
            throw std::invalid_argument("Invalid argument: i_cqt_mode");
        }
    }

    nc::NDArrayF32Ptr cq_to_chr = filters::cq_to_chroma<float>(
        C.shape()[0],
        __bins_per_octave,
        __n_chroma,
        __fmin,
        __window
        );

    nc::NDArrayF32Ptr chroma = cq_to_chr.dot(C);

    if (__threshold != -FLT_MAX) {
        // chroma[chroma < threshold] = 0.0;
        float * ptr = chroma.data();
        for (auto i = 0; i < chroma.elemCount(); i++) {
            if (ptr[i] < __threshold) ptr[i] = 0.0f;
        }
    }

    util::ShowNDArray2DF32(chroma, "chroma");

    // chroma = util.normalize(chroma, norm=norm, axis=0)
    if (__norm == INFINITY) {
        double threshold = FLT_MIN; // 1.1754944e-38
        auto mag = nc::abs(chroma);
        // length = np.max(mag, axis=axis, keepdims=True)
        std::vector<float> maxs(mag.shape()[1], -FLT_MAX);
        auto ptr_mag = mag.data();
        for (auto r = 0; r < mag.shape()[0]; r++) {
            for (auto c = 0; c < mag.shape()[1]; c++) {
                maxs[c] = std::max(maxs[c], *ptr_mag++);
            }
        }
        for (auto c = 0; c < mag.shape()[1]; c++) {
            if (maxs[c] < threshold) maxs[c] = 1.0;
        }
        for (auto r = 0; r < mag.shape()[0]; r++) {
            auto ptr_Cr = chroma.at(r, 0);
            for (auto c = 0; c < mag.shape()[1]; c++) {
                ptr_Cr[c] /= maxs[c];
            }
        }
    }
    else {
        throw std::runtime_error("Not implemented.");

    }

    return chroma;
}

} // namespace feature
} // namespace rosacxx

#endif // ROSACXX_FEATURE_SPECTRAL_H
