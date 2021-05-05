#include <rosacxx/util/utils.h>
#include <rosacxx/core/spectrum.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/fft.h>
#include <rosacxx/fft/kiss/kiss_fft.h>

#include <memory>
#include <cmath>
#include <map>
#include <iostream>
#include <complex>

namespace rosacxx {
namespace core {

nc::NDArrayPtr<std::complex<float>> stft(
        const nc::NDArrayF32Ptr& __y,
        const int& __n_fft, // = 2048,
        const int& __hop_length, // = -1,
        const int& __win_length, // = -1,
        const filters::STFTWindowType& __window, //="hann",
        const bool& __center, // = true,
        const char * __pad_mode  // = "reflect"
        ) {

    int win_length = __win_length;
    if (win_length < 0) win_length = __n_fft;

    int hop_length = __hop_length;
    if (hop_length < 0) hop_length = win_length / 4;

    auto fft_window = filters::get_window<double>(__window, win_length, true);

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

    nc::NDArrayPtr<std::complex<float>> stft_mat = nc::NDArrayPtr<std::complex<float>>(new nc::NDArray<std::complex<float>>({numFrames, numFFTOut}));

    vkfft::FFTReal rfft(__n_fft);

    std::complex<float> * ptr_stft_mat = stft_mat.data();
    float * ptr_frame = y.data();

    kiss_fft_scalar_t * tmp_ri = (kiss_fft_scalar_t *)nc::alignedMalloc(32, __n_fft * sizeof(kiss_fft_scalar_t));
    kiss_fft_scalar_t * tmp_co = (kiss_fft_scalar_t *)nc::alignedMalloc(32, numFFTOut * sizeof(kiss_fft_scalar_t) * 2);

    for (auto i = 0; i < numFrames; i++) {

        for (auto j = 0; j < __n_fft; j++) {
            tmp_ri[j] = ptr_frame[j]  * fft_window.getitem(j);
        }

        rfft.forward(tmp_ri, tmp_co);

        for (auto j = 0; j < numFFTOut; j++) {
            ptr_stft_mat[j].real(tmp_co[j * 2 + 0]);
            ptr_stft_mat[j].imag(tmp_co[j * 2 + 1]);
        }

        ptr_stft_mat += numFFTOut;
        ptr_frame += hop_length;
    }

    nc::alignedFree(tmp_ri);
    nc::alignedFree(tmp_co);

    return stft_mat.T();
}

nc::NDArrayPtr<std::complex<double>> stft(
        const nc::NDArrayF64Ptr& __y,
        const int& __n_fft, // = 2048,
        const int& __hop_length, // = -1,
        const int& __win_length, // = -1,
        const filters::STFTWindowType& __window, //="hann",
        const bool& __center, // = true,
        const char * __pad_mode  // = "reflect"
        ) {

    int win_length = __win_length;
    if (win_length < 0) win_length = __n_fft;

    int hop_length = __hop_length;
    if (hop_length < 0) hop_length = win_length / 4;

    auto fft_window = filters::get_window<double>(__window, win_length, true);

    // # Pad the window out to n_fft size | fft_window = util.pad_center(fft_window, n_fft)
    if (__n_fft > win_length) {
        fft_window = utils::pad_center_1d(fft_window, __n_fft);
    }

    // # Pad the time series so that frames are centered
    nc::NDArrayF64Ptr y = __y;
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

    nc::NDArrayPtr<std::complex<double>> stft_mat = nc::NDArrayPtr<std::complex<double>>(new nc::NDArray<std::complex<double>>({numFrames, numFFTOut}));

    vkfft::FFTReal rfft(__n_fft);

    std::complex<double> * ptr_stft_mat = stft_mat.data();
    double * ptr_frame = y.data();

    kiss_fft_scalar_t * tmp_ri = (kiss_fft_scalar_t *)nc::alignedMalloc(32, __n_fft * sizeof(kiss_fft_scalar_t));
    kiss_fft_scalar_t * tmp_co = (kiss_fft_scalar_t *)nc::alignedMalloc(32, numFFTOut * sizeof(kiss_fft_scalar_t) * 2);

    for (auto i = 0; i < numFrames; i++) {

        for (auto j = 0; j < __n_fft; j++) {
            tmp_ri[j] = ptr_frame[j]  * fft_window.getitem(j);
        }

        rfft.forward(tmp_ri, tmp_co);

        for (auto j = 0; j < numFFTOut; j++) {
            ptr_stft_mat[j].real(tmp_co[j * 2 + 0]);
            ptr_stft_mat[j].imag(tmp_co[j * 2 + 1]);
        }

        ptr_stft_mat += numFFTOut;
        ptr_frame += hop_length;
    }

    nc::alignedFree(tmp_ri);
    nc::alignedFree(tmp_co);

    return stft_mat.T();
}

void _spectrogram(
        const nc::NDArrayF32Ptr&        __y,
        nc::NDArrayF32Ptr&              __S,
        int&                            __n_fft,
        const int&                      __hop_length,
        const float&                    __power,
        const int&                      __win_length,
        const filters::STFTWindowType&  __window,
        const bool&                     __center,
        const char *                    __pad_mode
        ) {
    if (__S != nullptr) {
        __n_fft = 2 * (__S.shape()[0] - 1);
    }
    else {
        auto tmp = stft(__y, __n_fft, __hop_length, __win_length, __window, __center, __pad_mode);
        auto S = nc::NDArrayF32Ptr(new nc::NDArrayF32(tmp.shape()));
        std::complex<float> * ptr_tmp = tmp.data();
        float * ptr_S = S.data();
        if (__power == 1) {
            for (auto i = 0; i < tmp.elemCount(); i++) {
                ptr_S[i] = std::sqrt( ptr_tmp[i].real() * ptr_tmp[i].real() + ptr_tmp[i].imag() * ptr_tmp[i].imag() );
            }
        }
        else if (__power == 2) {
            for (auto i = 0; i < tmp.elemCount(); i++) {
                // ptr_S[i] = ptr_tmp[i].r * ptr_tmp[i].r + ptr_tmp[i].i * ptr_tmp[i].i;
                ptr_S[i] = ptr_tmp[i].real() * ptr_tmp[i].real() + ptr_tmp[i].imag() * ptr_tmp[i].imag();
            }
        }
        else {
            for (auto i = 0; i < tmp.elemCount(); i++) {
                // ptr_S[i] = std::pow(std::sqrt( ptr_tmp[i].r * ptr_tmp[i].r + ptr_tmp[i].i * ptr_tmp[i].i ), __power);
                ptr_S[i] = std::sqrt( ptr_tmp[i].real() * ptr_tmp[i].real() + ptr_tmp[i].imag() * ptr_tmp[i].imag() );
                ptr_S[i] = std::pow(ptr_S[i], __power);
            }
        }
        __S = S;
    }
}

} // namespace core
} // namespace rosacxx
