#include <feature/spectral.h>
#include <core/constantq.h>

#include <stdexcept>

namespace rosacxx {
namespace feature {

void chroma_cqt(
        const float *       i_y,
        const float         i_sr,
        const float *       i_C,
        const int           i_hop_length,
        const float *       i_fmin,
        const int           i_norm,
        const float         i_threshold,
        const float         i_tuning,
        const int           i_n_chroma,
        const int           i_n_octaves,
        const float *       i_window,
        const int           i_bins_per_octave,
        const char *        i_cqt_mode,
        float *             o_chromagram
        ) {
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
    int d = 0;
    int t = 0;
    float * C = (float *)malloc((d * t) * sizeof (float));
    if (i_C == NULL) {

    }
    return;
}

} // namespace feature
} // namespace rosacxx
