#ifndef ROSACXX_CORE_PITCH_H
#define ROSACXX_CORE_PITCH_H

#include <rosacxx/numcxx/numcxx.h>
#include <map>
#include <rosacxx/filters.h>

namespace rosacxx {
namespace core {

float estimate_tuning(
        const nc::NDArrayF32Ptr& y,
        const float& sr,
        nc::NDArrayF32Ptr& S,
        const int& n_fft            = 2048,
        const float& resolution     = 0.01f,
        const int& bins_per_octave  = 12,
        const int& hop_length       = -1,
        const float& fmin           = 150.0,
        const float& fmax           = 4000.0,
        const float& threshold      = 0.1,
        const int& win_length       = -1,
        const filters::STFTWindowType& window = filters::STFTWindowType::Hanning,
        const bool& center          = true,
        const char * pad_mode       = "reflect",
        float * ref                  = nullptr
        );

float pitch_tuning(
        const nc::NDArrayF32Ptr& frequencies,
        const float& resolution=0.01f,
        const int& bins_per_octave=12
        );

std::vector<nc::NDArrayF32Ptr> piptrack(
        const nc::NDArrayF32Ptr& __y,
        const float& __sr,
        nc::NDArrayF32Ptr& __S,
        const int& __n_fft                      = 2048,
        const int& __hop_length                 = -1,
        const float& __fmin                     = 150.0,
        const float& __fmax                     = 4000.0,
        const float& __threshold                = 0.1,
        const int& __win_length                 = -1,
        const filters::STFTWindowType& __window = filters::STFTWindowType::Hanning,
        const bool& __center                    = true,
        const char * __pad_mode                 = "reflect",
        float * __ref                           = nullptr
        );

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_PITCH_H
