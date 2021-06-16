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

template<typename DType>
nc::NDArrayPtr<std::complex<DType>> stft(
        const nc::NDArrayPtr<DType>& __y,
        const int& __n_fft,
        const int& __hop_length,
        const int& __win_length,
        const filters::STFTWindowType& __window,
        const bool& __center,
        const char * __pad_mode,
        const bool& __seqFirst
        ) {

    int win_length = __win_length;
    if (win_length < 0) win_length = __n_fft;

    int hop_length = __hop_length;
    if (hop_length < 0) hop_length = win_length / 4;

    auto fft_window = filters::get_window<DType>(__window, win_length, true);
    auto vec_fftwin = fft_window.toStdVector1D();

    // # Pad the window out to n_fft size | fft_window = util.pad_center(fft_window, n_fft)
    if (__n_fft > win_length) {
        fft_window = util::pad_center_1d(fft_window, __n_fft);
    }

    // # Pad the time series so that frames are centered
    nc::NDArrayPtr<DType> y = __y;
    if (__center) {
        if (__n_fft > y.shape()[y.shape().size() - 1]) printf("n_fft=%d is too small for input signal of length=%d.\n", __n_fft, y.shape()[__y.shape().size() - 1]);
        if (strcmp(__pad_mode, "reflect") == 0) {
            y = nc::reflect_pad1d(y, {int(__n_fft / 2), int(__n_fft / 2)});
        }
    }
    else {
        if (__n_fft > y.shape()[y.shape().size() - 1])
            throw std::runtime_error("n_fft is too small for input signal of length.");
    }

    int numFrames = 1 + (y.shape()[y.shape().size() - 1] - __n_fft) / hop_length;
    int numFFTOut = 1 + __n_fft / 2;

    nc::NDArrayPtr<std::complex<DType>> stft_mat = nc::NDArrayPtr<std::complex<DType>>(new nc::NDArray<std::complex<DType>>({numFrames, numFFTOut}));

    vkfft::FFTReal rfft((unsigned int)__n_fft);

    // std::complex<DType> * ptr_stft_mat = stft_mat.data();
    // DType * ptr_frame = y.data();

    kiss_fft_scalar_t * tmp_ri = (kiss_fft_scalar_t *)nc::alignedMalloc(size_t(32), size_t(__n_fft      ) * sizeof(kiss_fft_scalar_t));
    kiss_fft_scalar_t * tmp_co = (kiss_fft_scalar_t *)nc::alignedMalloc(size_t(32), size_t(numFFTOut * 2) * sizeof(kiss_fft_scalar_t));

#   pragma omp parallel for
    for (auto i = 0; i < numFrames; i++) {

        std::complex<DType> * ptr_stft_mat = stft_mat.data() + i * numFFTOut;
        DType * ptr_frame = y.data() + i * hop_length;

        for (auto j = 0; j < __n_fft; j++) {
            tmp_ri[j] = ptr_frame[j]  * fft_window.getitem(j);
        }

        rfft.forward(tmp_ri, tmp_co);

        for (auto j = 0; j < numFFTOut; j++) {
            auto j2 = (j << 1);
            ptr_stft_mat[j].real(tmp_co[j2 + 0]);
            ptr_stft_mat[j].imag(tmp_co[j2 + 1]);
        }

        // ptr_stft_mat += numFFTOut;
        // ptr_frame += hop_length;
    }

    nc::alignedFree(tmp_ri);
    nc::alignedFree(tmp_co);

    if (__seqFirst) {
        return stft_mat;
    }

    return stft_mat.T();
}

template nc::NDArrayPtr<std::complex<float>> stft(
        const nc::NDArrayPtr<float>& __y,
        const int& __n_fft,
        const int& __hop_length,
        const int& __win_length,
        const filters::STFTWindowType& __window,
        const bool& __center,
        const char * __pad_mode,
        const bool& __seqFirst
        );

template<typename DType>
inline void __window_ss_fill(nc::NDArrayPtr<DType>& x, const nc::NDArrayPtr<DType>& win_sq, const int& n_frames, const int& hop_length) {
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
inline nc::NDArrayPtr<DType> __window_sumsquare(
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

template<typename DType>
nc::NDArrayPtr<DType> istft(
        const nc::NDArrayPtr<std::complex<DType>>& __stft_matrix,
        const int& __hop_length,
        const int& __win_length,
        const filters::STFTWindowType& window,
        const bool& center,
        const int& length,
        const bool& __seqFirst
        ) {
    auto stft_matrix = __stft_matrix;
    if (!__seqFirst)
        stft_matrix = stft_matrix.T();

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

#   if __DEBUG
    auto vec_ifft_window = ifft_window.toStdVector1D();
#   endif

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
        auto offset_mat_at_i = i * n_out;
        for (auto j = 0; j < n_out; j++) {
            tmp_ci[j * 2 + 0] = ptr_mat[offset_mat_at_i + j].real();
            tmp_ci[j * 2 + 1] = ptr_mat[offset_mat_at_i + j].imag();
        }
        rfft.inverse(tmp_ci, tmp_ro);
        auto offset_y_at_i = i * hop_length;
        for (auto j = 0; j < n_fft; j++) {
            // overlap add ...
            ptr_y[offset_y_at_i + j] += tmp_ro[j] * ptr_w[j];
        }
    }

    nc::alignedFree(tmp_ci);
    nc::alignedFree(tmp_ro);

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

template nc::NDArrayPtr<float> istft(
        const nc::NDArrayPtr<std::complex<float>>& __stft_matrix,
        const int& __hop_length,
        const int& __win_length,
        const filters::STFTWindowType& window,
        const bool& center,
        const int& length,
        const bool& __seqFirst
        );

} // namespace core
} // namespace rosacxx

































