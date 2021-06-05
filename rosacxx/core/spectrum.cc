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
#include <cfloat>

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
        fft_window = util::pad_center_1d(fft_window, __n_fft);
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
        fft_window = util::pad_center_1d(fft_window, __n_fft);
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

template<typename DType>
void __window_ss_fill(nc::NDArrayPtr<DType>& x, const nc::NDArrayPtr<DType>& win_sq, const int& n_frames, const int& hop_length) {
    int n = x.elemCount();
    int n_fft = win_sq.elemCount();
    DType * ptr_x = x.data();
    DType * ptr_w = win_sq.data();
    for (auto i = 0; i < n_frames; i++) {
        auto sample = i * hop_length;
        if (sample > n) {
            for (auto j = 0; j < n_fft; j++) {
                if (sample + j >= n) break;
                ptr_x[sample+j] += ptr_w[0];
            }
        }
        else {
            for (auto j = 0; j < n_fft; j++) {
                if (sample + j >= n) break;
                ptr_x[sample+j] += ptr_w[j];
            }
        }
    }
}

template<typename DType>
nc::NDArrayPtr<DType> __window_sumsquare(
        const filters::STFTWindowType& __window,
        const int& __n_frames,
        const int& __hop_length,
        const int& __win_length,
        const int& __n_fft,
        const char * __norm=NULL
        ) {
    int n_frames = __n_frames;
    int n_fft = __n_fft;
    int hop_length = __hop_length;
    int win_length = __win_length;
    if (win_length <= 0) {
        win_length = __n_fft;
    }
    int n = n_fft + hop_length * (n_frames - 1);
    nc::NDArrayPtr<DType> x = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({n}));
    auto win_sq = filters::get_window<DType>(__window, win_length);
    if (__norm == NULL) {
        win_sq = nc::pow(win_sq, DType(2));
    } else {
        throw std::runtime_error("Not implemented error.");
    }
    win_sq = util::pad_center_1d(win_sq, n_fft);
    __window_ss_fill(x, win_sq, n_frames, hop_length);
    return x;
}

nc::NDArrayPtr<float> istft(
    const nc::NDArrayPtr<std::complex<float>>& __stft_matrix,
    const int& __hop_length, // = -1,
    const int& __win_length, // = -1,
    const filters::STFTWindowType& window, // = filters::STFTWindowType::Hanning,
    const bool& center, // = true,
    const int& length // = 0
    ) {
    using DType = float;

    auto stft_matrix = __stft_matrix.T();

    int n_fft = 2 * (stft_matrix.shape()[1] - 1);
    int n_out = n_fft / 2 + 1;

    int win_length = __win_length;
    if ( win_length <= 0 ) {
        win_length = n_fft;
    }

    int hop_length = __hop_length;
    if ( hop_length <= 0 ) {
        hop_length = win_length / 4;
    }

    // ifft_window = get_window(window, win_length, fftbins=True)
    auto ifft_window = filters::get_window<DType>(window, win_length, true);

    ifft_window = util::pad_center_1d(ifft_window, n_fft);

    int n_frames = -1;
    if ( length > 0 ) {
        int padded_length = 0;
        if ( center ) {
            padded_length = length + n_fft;
        }
        else {
            padded_length = length;
        }
        n_frames = std::min(stft_matrix.shape()[0], int(std::ceil(float(padded_length) / hop_length)));
    }
    else {
        n_frames = stft_matrix.shape()[0];
    }

    int expected_signal_len = n_fft + hop_length * (n_frames - 1);

    nc::NDArrayPtr<DType> y = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({expected_signal_len}));

    vkfft::FFTReal rfft(n_fft);

    kiss_fft_scalar_t * tmp_ci = (kiss_fft_scalar_t *)nc::alignedMalloc(32, n_out * sizeof(kiss_fft_scalar_t) * 2);
    kiss_fft_scalar_t * tmp_ro = (kiss_fft_scalar_t *)nc::alignedMalloc(32, n_fft * sizeof(kiss_fft_scalar_t));

    DType * ptr_y = y.data();
    std::complex<DType> * ptr_mat = stft_matrix.data();
    DType * ptr_w = ifft_window.data();

    for (auto i = 0; i < n_frames; i++) {
        for (auto j = 0; j < n_out; j++) {
            tmp_ci[j * 2 + 0] = ptr_mat[j].real();
            tmp_ci[j * 2 + 1] = ptr_mat[j].imag();
        }
        rfft.inverse(tmp_ci, tmp_ro);
        auto offset = i * hop_length;
        for (auto j = 0; j < n_fft; j++) {
            // overlap add ...
            ptr_y[offset + j] += tmp_ro[j] * ptr_w[j];
        }
    }

    free(tmp_ci);
    free(tmp_ro);

    auto ifft_window_sum = __window_sumsquare<DType>(window, n_frames, hop_length, win_length, n_fft);
    DType * ptr_ifft_window_sum = ifft_window_sum.data();
    for (auto i = 0; i < y.elemCount(); i++) {
        if (typeid(DType) == typeid(float)) {
            if (ptr_ifft_window_sum[i] > FLT_MIN) {
                ptr_y[i] /= ptr_ifft_window_sum[i];
            }
        }
        else if (typeid(DType) == typeid(double)) {
            if (ptr_ifft_window_sum[i] > DBL_MIN) {
                ptr_y[i] /= ptr_ifft_window_sum[i];
            }
        }
    }

    if ( length <= 0 ) {
        if (center) {
            // crop: y = y[int(n_fft // 2) : -int(n_fft // 2)]
            nc::NDArrayPtr<DType> y1 = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({y.elemCount()-n_fft/2*2}));
            DType *ptr_y1 = y1.data();
            for (auto i = 0; i < y1.elemCount(); i++) {
                ptr_y1[i] = ptr_y[n_fft/2+i];
            }
            y = y1;
        }
    }
    else {
        int start = 0;
        if (center) {
            start = n_fft / 2;
        }
        // y = util.fix_length(y[start:], length)
        if (start) {
            nc::NDArrayPtr<DType> y1 = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({y.elemCount()-start}));
            DType *ptr_y1 = y1.data();
            for (auto i = 0; i < y1.elemCount(); i++) {
                ptr_y1[i] = ptr_y[start+i];
            }
            y = y1;
        }
        y = util::fix_length(y, length);
    }

    return y;
}

} // namespace core
} // namespace rosacxx

































