#ifndef ROSACXX_FEATURE_SPECTRAL_H
#define ROSACXX_FEATURE_SPECTRAL_H

#include <stdio.h>

namespace rosacxx {
namespace feature {

/// Function: chroma_cqt
/// Param Name              | Type      | Note
/// @param y                | float *   | Audio time series, LPCM data
/// @param sr               | float     | Sampling rate of ``y``
/// @param C                | float *   | Shape=(d, t), a pre-computed constant-Q spectrogram
/// @param hop_length       | int       | Number of samples between successive chroma frames
/// @param fmin             | float *   | minimum frequency to analyze in the CQT. Default: `C1 ~= 32.7 Hz`
/// @param norm             | int       | Column-wise normalization of the chromagram
/// @param threshold        | float     | Pre-normalization energy threshold.  Values below the threshold are discarded, resulting in a sparse chromagram
/// @param tuning           | float     | Deviation (in fractions of a CQT bin) from A440 tuning
/// @param n_chroma         | int       | Number of chroma bins to produce
/// @param n_octaves        | int       | Number of octaves to analyze above ``fmin``
/// @param window           | float *   | Optional window parameter to `filters.cq_to_chroma`
/// @param bins_per_octave  | int       | Number of bins per octave in the CQT. Must be an integer multiple of ``n_chroma``. Default: 36 (3 bins per semitone)
/// @param cqt_mode         | char *    | Constant-Q transform mode ['full', 'hybrid']
/// @result chromagram      | float *   | The output chromagram, shape=(n_chroma, t)
void chroma_cqt(
        const float * y = NULL,
        const float sr = 22050,
        const float * C = NULL,
        const int hop_length = 512,
        const float * fmin = NULL,
        const int norm = 0,
        const float threshold = 0,
        const float tuning = 0,
        const int n_chroma = 12,
        const int n_octaves = 7,
        const float * window = NULL,
        const int bins_per_octave = 36,
        const char * cqt_mode = "full",
        float * chromagram = NULL
        );

} // namespace feature
} // namespace rosacxx

#endif // ROSACXX_FEATURE_SPECTRAL_H
