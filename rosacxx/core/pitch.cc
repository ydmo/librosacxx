#include <rosacxx/core/pitch.h>
#include <rosacxx/core/convert.h>

#include <memory>
#include <cmath>
#include <map>

namespace rosacxx {
namespace core {

std::map<const char *, const nc::NDArrayF32Ptr> piptrack(
        const nc::NDArrayF32Ptr& y  = nullptr,
        const float& sr             = 22050,
        const nc::NDArrayF32Ptr& S  = nullptr,
        const int& n_fft            = 2048,
        const int& hop_length       = -1,
        const float& fmin           = 150.0,
        const float& fmax           = 4000.0,
        const float& threshold      = 0.1,
        const int& win_length       = -1,
        const char * window         = "hann",
        const bool& center          = true,
        const char * pad_mode       = "reflect",
        float * ref                 = nullptr
        ) {
    std::map<const char *, const nc::NDArrayF32Ptr> rets = { {"pitches", nullptr}, {"magnitudes", nullptr} };

    // pitch_mask

    return rets;
}

float estimate_tuning(
        const nc::NDArrayF32Ptr& y, //  = nullptr,
        const float& sr, //             = 22050,
        const nc::NDArrayF32Ptr& S, //  = nullptr,
        const int& n_fft, //            = 2048,
        const float& resolution, //     = 0.01f,
        const int& bins_per_octave, //  = 12,
        const int& hop_length, //       = -1,
        const float& fmin, //           = 150.0,
        const float& fmax, //           = 4000.0,
        const float& threshold, //      = 0.1,
        const int& win_length, //       = -1,
        const char * window, //         = "hann",
        const bool& center, //          = true,
        const char * pad_mode, //       = "reflect",
        float * ref //                  = nullptr
        ) {
    auto pitch_mag = piptrack(y, sr, S, n_fft, hop_length, fmin, fmax, threshold, win_length, window, center, pad_mode, ref); // pitch, mag = piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)
    nc::NDArrayF32Ptr pitch = pitch_mag["pitch"];
    nc::NDArrayF32Ptr mag = pitch_mag["magnitudes"];

    nc::NDArrayBoolPtr pitch_mask = pitch > 0;

    float mag_threshold = 0.f;
    if (pitch_mask != nullptr) {
        mag_threshold = nc::median(mag[pitch_mask]);
    }

    nc::NDArrayF32Ptr frequencies = pitch[(mag > mag_threshold) && pitch_mask];

    return pitch_tuning(frequencies, resolution, bins_per_octave);
}

float pitch_tuning(
        const nc::NDArrayF32Ptr& frequencies,
        const float& resolution,
        const int& bins_per_octave
        ) {

    std::vector<int> indies_larger_than_zero(0);
    float * ptr_freq = frequencies.data();
    for (int i = 0; i < frequencies.elemCount(); i++) {
        if (ptr_freq[i] > 0) {
            indies_larger_than_zero.push_back(i);
        }
    }

    if (indies_larger_than_zero.size() == 0) {
        printf("[Wraning] Trying to estimate tuning from empty frequency set.");
        return 0.0f;
        throw std::runtime_error("Trying to estimate tuning from empty frequency set.");
    }

    nc::NDArrayF32Ptr newFreq = nc::NDArrayF32Ptr(new nc::NDArrayF32({int(indies_larger_than_zero.size())}));
    float * ptr_newfreq = newFreq.data();
    for (auto i = 0; i < indies_larger_than_zero.size(); i++) {
        ptr_newfreq[i] = ptr_freq[indies_larger_than_zero[i]];
    }

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

    auto bins = nc::linspace<float>(-0.5f, 0.5f, int(std::ceil(1.0 / resolution)) + 1);
    auto counts = nc::histogram(residual, bins);
    return bins.getitem(counts.argmax());
}



} // namespace core
} // namespace rosacxx
