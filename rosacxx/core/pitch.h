#ifndef ROSACXX_CORE_PITCH_H
#define ROSACXX_CORE_PITCH_H

#include <3rd/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

float estimate_tuning(
        const nc::NDArrayF32Ptr& y = nullptr,
        const float& sr=22050,
        const nc::NDArrayF32Ptr& S= nullptr,
        const int& n_fft=2048,
        const float& resolution=0.01f,
        const int& bins_per_octave=12
        );

float pitch_tuning(
        const nc::NDArrayF32Ptr& frequencies,
        const float& resolution=0.01f,
        const int& bins_per_octave=12
        );

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_PITCH_H
