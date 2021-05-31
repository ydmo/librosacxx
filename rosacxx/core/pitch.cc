#include <rosacxx/core/pitch.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/spectrum.h>

#include <memory>
#include <cmath>
#include <map>

#if ROSACXX_TEST
#   include <rosacxx/util/visualize.h>
#endif

namespace rosacxx {
namespace core {

std::vector<nc::NDArrayF32Ptr> piptrack(
        const nc::NDArrayF32Ptr& __y,
        const float& __sr,
        nc::NDArrayF32Ptr& __S,
        const int& __n_fft,
        const int& __hop_length,
        const float& __fmin,
        const float& __fmax,
        const float& __threshold,
        const int& __win_length,
        const filters::STFTWindowType& __window,
        const bool& __center,
        const char * __pad_mode,
        float * __ref
        ) {
    float sr = __sr;
    int n_fft = __n_fft;
    nc::NDArrayF32Ptr S = __S;

    _spectrogram(__y, S, n_fft, __hop_length, 1, __win_length, __window, __center, __pad_mode);

    // pitch_mask
    float fmin = std::max(__fmin, 0.f);
    float fmax = std::min(__fmax, __sr / 2.f);

    auto fft_freqs = fft_frequencies(sr, n_fft);

    auto shapeS = S.shape();
    nc::NDArrayF32Ptr avg   = nc::NDArrayF32Ptr(new nc::NDArrayF32({shapeS[0]-2, shapeS[1]}));
    nc::NDArrayF32Ptr shift = nc::NDArrayF32Ptr(new nc::NDArrayF32({shapeS[0]-2, shapeS[1]}));
    float * ptr_avg = avg.data();
    float * ptr_shift = shift.data();
    for (auto i = 0; i < shapeS[0]-2; i++) {
        for (auto j = 0; j < shapeS[1]; j++) {
            *ptr_avg++ = 0.5 * (S.getitem(i+2, j) - S.getitem(i, j)); // *ptr_avg++ = (ptr_S[(i + 2) * shapeS[1] + j] - ptr_S[i * shapeS[1] + j]) * 0.5f; // avg = 0.5 * (S[2:] - S[:-2])
            *ptr_shift++ = 2.0 * S.getitem(i+1, j) - S.getitem(i+2, j) - S.getitem(i, j); // *ptr_shift++ = 2 * ptr_S[(i + 1) * shapeS[1] + j] - ptr_S[(i + 2) * shapeS[1] + j] - ptr_S[i * shapeS[1] + j];// shift = 2 * S[1:-1] - S[2:] - S[:-2]
        }
    }

    shift = avg / (shift + (nc::abs(shift) < 1.1754944e-38f)); // shift = avg / (shift + (np.abs(shift) < util.tiny(shift)))

    avg   = nc::pad(avg,   {{1, 1}, {0, 0}}); // avg   = np.pad(avg, ([1, 1], [0, 0]), mode="constant")
    shift = nc::pad(shift, {{1, 1}, {0, 0}}); // shift = np.pad(shift, ([1, 1], [0, 0]), mode="constant")

    auto dskew = .5f * avg * shift; // dskew = 0.5 * avg * shift

    auto freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape({-1, 1}); // freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))

    auto ref_value = __threshold * nc::max(S, 0);

    auto idx = nc::argwhere(nc::localmax(S * (S > ref_value), 0) & freq_mask); // idx = np.argwhere(freq_mask & util.localmax(S * (S > ref_value)))

    auto pitches = nc::zeros_like(S);
    auto mags = nc::zeros_like(S);

    auto p_pitches = pitches.data();
    auto p_mags = mags.data();
    auto p_dskew = dskew.data();
    auto p_S = S.data();
    auto p_shift = shift.data();
    auto strides_s = S.strides();
    int *ptr_coor = idx.data();
    for (int i = 0; i < idx.shape()[0]; i++) {
        int * coor = ptr_coor + (i << 1); // idx.at(i, 0);
        int offset = coor[0] * strides_s[0] + coor[1];
        p_pitches[offset] = (coor[0] + p_shift[offset]) * sr / n_fft;
        p_mags[offset] = p_S[offset] + p_dskew[offset];
    }

    return { pitches, mags };
}


float pitch_tuning(
        const nc::NDArrayF32Ptr& frequencies,
        const float& resolution,
        const int& bins_per_octave
        ) {

//    std::vector<int> indies_larger_than_zero(0);
//    float * ptr_freq = frequencies.data();
//    for (int i = 0; i < frequencies.elemCount(); i++) {
//        if (ptr_freq[i] > 0) {
//            indies_larger_than_zero.push_back(i);
//        }
//    }
//    auto freq = frequencies[frequencies > 0];

//    if (indies_larger_than_zero.size() == 0) {
//        printf("[Wraning] Trying to estimate tuning from empty frequency set.");
//        return 0.0f;
//        throw std::runtime_error("Trying to estimate tuning from empty frequency set.");
//    }

//    nc::NDArrayF32Ptr newFreq = nc::NDArrayF32Ptr(new nc::NDArrayF32({int(indies_larger_than_zero.size())}));
//    float * ptr_newfreq = newFreq.data();
//    for (auto i = 0; i < indies_larger_than_zero.size(); i++) {
//        ptr_newfreq[i] = ptr_freq[indies_larger_than_zero[i]];
//    }

    nc::NDArrayF32Ptr newFreq = frequencies[frequencies > 0];

    nc::NDArrayF32Ptr octs = hz_to_octs(newFreq);
    float * ptr_octs = octs.data();

    nc::NDArrayF32Ptr residual = nc::NDArrayF32Ptr(new nc::NDArrayF32(octs.shape()));
    float * ptr_residual = residual.data();
    for (auto i = 0; i < octs.elemCount(); i++) {
        ptr_residual[i] = bins_per_octave * ptr_octs[i];
        ptr_residual[i] = ptr_residual[i] - std::round(ptr_residual[i]);
        if (ptr_residual[i] >= .5f) {
            ptr_residual[i] -= 1.f;
        }
    }

    auto bins = nc::linspace<float>(-0.5f, 0.5f, int(std::round(1.0 / resolution)) + 1);
    auto counts = nc::histogram(residual, bins);

    auto vec_bins = bins.toStdVector1D();
    auto vec_counts = counts.toStdVector1D();

    return bins.getitem(counts.argmax());
}

float estimate_tuning(
        const nc::NDArrayF32Ptr& y, //  = nullptr,
        const float& sr, //             = 22050,
        nc::NDArrayF32Ptr& S, //  = nullptr,
        const int& n_fft, //            = 2048,
        const float& resolution, //     = 0.01f,
        const int& bins_per_octave, //  = 12,
        const int& hop_length, //       = -1,
        const float& fmin, //           = 150.0,
        const float& fmax, //           = 4000.0,
        const float& threshold, //      = 0.1,
        const int& win_length, //       = -1,
        const filters::STFTWindowType& window, //         = "hann",
        const bool& center, //          = true,
        const char * pad_mode, //       = "reflect",
        float * ref //                  = nullptr
        ) {
    auto pitch_mag = piptrack(y, sr, S, n_fft, hop_length, fmin, fmax, threshold, win_length, window, center, pad_mode, ref); // pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)
    nc::NDArrayF32Ptr pitch = pitch_mag[0];
    nc::NDArrayF32Ptr mag = pitch_mag[1];
    
    nc::NDArrayBoolPtr pitch_mask = pitch > 0;

    auto mag_tmp = mag[pitch_mask].toStdVector1D();

    float mag_threshold = 0.f;
    if (pitch_mask != nullptr) {
        mag_threshold = nc::median(mag[pitch_mask]);
    }

    nc::NDArrayF32Ptr frequencies = pitch[(mag >= mag_threshold) & pitch_mask];
    auto vec_freq = frequencies.toStdVector1D();

    return pitch_tuning(frequencies, resolution, bins_per_octave);
}



} // namespace core
} // namespace rosacxx
