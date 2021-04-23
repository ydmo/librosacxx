#ifndef ROSACXX_CORE_SPECTRUM_H
#define ROSACXX_CORE_SPECTRUM_H

#include <3rd/numcxx/numcxx.h>
#include <vector>

namespace rosacxx {
namespace core {

nc::NDArrayF32Ptr stft(
    const nc::NDArrayF32Ptr& y,
    const int& n_fft = 2048,
    const int& hop_length = -1,
    const int& win_length = -1,
    const char * window="hann",
    const bool& center = true,
    const char * dtype = NULL,
    const char * pad_mode = "reflect"
    );

void _spectrogram(
        const nc::NDArrayF32Ptr& y,
        nc::NDArrayF32Ptr&       S,
        int&                     n_fft,
        const int&          hop_length  = 512,
        const float&        power       = 1,
        const int&          win_length  = -1,
        const char *        window      = "hann",
        const bool&         center      = true,
        const char *        pad_mode    = "reflect"
        );

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_SPECTRUM_H
