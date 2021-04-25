#ifndef ROSACXX_CORE_PITCH_H
#define ROSACXX_CORE_PITCH_H

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

float estimate_tuning(
        const nc::NDArrayF32Ptr& y  = nullptr,
        const float& sr             = 22050,
        const nc::NDArrayF32Ptr& S  = nullptr,
        const int& n_fft            = 2048,
        const float& resolution     = 0.01f,
        const int& bins_per_octave  = 12,
        const int& hop_length       = -1,
        const float& fmin           = 150.0,
        const float& fmax           = 4000.0,
        const float& threshold      = 0.1,
        const int& win_length       = -1,
        const char * window         = "hann",
        const bool& center          = true,
        const char * pad_mode       = "reflect",
        float * ref                 = nullptr
        );

float pitch_tuning(
        const nc::NDArrayF32Ptr& frequencies,
        const float& resolution=0.01f,
        const int& bins_per_octave=12
        );

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_PITCH_H
