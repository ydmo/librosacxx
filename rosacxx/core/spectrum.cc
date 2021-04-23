#include <rosacxx/core/spectrum.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/filters.h>

#include <memory>
#include <cmath>
#include <map>

namespace rosacxx {
namespace core {

nc::NDArrayF32Ptr stft(
        const nc::NDArrayF32Ptr& __y,
        const int& __n_fft, // = 2048,
        const int& __hop_length, // = -1,
        const int& __win_length, // = -1,
        const filters::STFTWindowType& __window, //="hann",
        const bool& __center, // = true,
        const char * __dtype, // = NULL,
        const char * __pad_mode  // = "reflect"
        ) {

    int win_length = __win_length;
    if (win_length < 0) win_length = __n_fft;

    int hop_length = __hop_length;
    if (hop_length < 0) hop_length = win_length / 4;

    auto fft_window = filters::get_window(__window, win_length, true);

    return nullptr;
}

void _spectrogram(
        const nc::NDArrayF32Ptr& y,
        nc::NDArrayF32Ptr&       S,
        int&                     n_fft,
        const int&               hop_length, //  = 512,
        const float&             power, //       = 1,
        const int&               win_length, //  = -1,
        const char *             window, //      = "hann",
        const bool&              center, //      = true,
        const char *             pad_mode  //    = "reflect"
        ) {
    if (S != nullptr) {
        n_fft = 2 * (S.shape()[0] - 1);
    }
}

} // namespace rosacxx
} // namespace core
