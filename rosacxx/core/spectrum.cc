#include <rosacxx/util/utils.h>
#include <rosacxx/core/spectrum.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/fft.h>


#include <3rd/fft/kiss/kiss_fft.h>

#include <memory>
#include <cmath>
#include <map>
#include <iostream>

namespace rosacxx {
namespace core {

nc::NDArrayPtr<Complex<float>> stft(
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

    // # Pad the window out to n_fft size | fft_window = util.pad_center(fft_window, n_fft)
    if (__n_fft > win_length) {
        fft_window = utils::pad_center_1d(fft_window, __n_fft);
    }

    // # Pad the time series so that frames are centered
    nc::NDArrayF32Ptr y = __y;
    if (__center) {
        if (__n_fft > y.shape()[y.shape().size() - 1]) printf("n_fft=%d is too small for input signal of length=%d.\n", __n_fft, y.shape()[__y.shape().size() - 1]);
        if (strcmp(__pad_mode, "reflect") == 0) {
            y = nc::reflect_pad1d(y, {int(__n_fft / 2), int(__n_fft / 2)});
        }
    }
    else {
        if (__n_fft > y.shape()[y.shape().size() - 1]) throw std::runtime_error("n_fft is too small for input signal of length.");
    }

    int numFrames = 1 + (y.shape()[y.shape().size() - 1] - __n_fft) / hop_length;
    int numFFTOut = 1 + __n_fft / 2;

    nc::NDArrayPtr<Complex<float>> stft_mat = nc::NDArrayPtr<Complex<float>>(new nc::NDArray<Complex<float>>({numFrames, numFFTOut}));

    vkfft::FFTReal rfft(__n_fft);

    Complex<float> * ptr_stft_mat = stft_mat.data();
    float * ptr_frame = y.data();

    for (auto i = 0; i < numFrames; i++) {
        auto ri = nc::NDArrayF32Ptr(new nc::NDArrayF32({__n_fft}, ptr_frame)) * fft_window;
        rfft.forward(ri.data(), (float *)ptr_stft_mat);
        ptr_stft_mat += numFFTOut;
        ptr_frame += hop_length;
    }

    return stft_mat.T();
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
