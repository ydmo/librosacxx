#ifndef ROSACXX_CORE_SPECTRUM_H
#define ROSACXX_CORE_SPECTRUM_H

#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/fft/kiss/kiss_fft.h>

#include <rosacxx/filters.h>
#include <rosacxx/util/utils.h>
#include <rosacxx/core/fft.h>
#include <rosacxx/core/convert.h>

#include <memory>
#include <cmath>
#include <map>
#include <complex>
#include <vector>
#include <cfloat>

namespace rosacxx {
namespace core {

template<typename DType>
nc::NDArrayPtr<std::complex<DType>> stft(
        const nc::NDArrayPtr<DType>& __y,
        const int& __n_fft = 2048,
        const int& __hop_length = -1,
        const int& __win_length = -1,
        const filters::STFTWindowType& __window = filters::STFTWindowType::Hanning,
        const bool& __center = true,
        const char * __pad_mode = "reflect",
        const bool& __seqFirst = false
        );

template<typename DType>
nc::NDArrayPtr<DType> istft(
        const nc::NDArrayPtr<std::complex<DType>>& __stft_matrix,
        const int& __hop_length = -1,
        const int& __win_length = -1,
        const filters::STFTWindowType& window = filters::STFTWindowType::Hanning,
        const bool& center = true,
        const int& length = 0,
        const bool& __seqFirst = false
        );

template<typename DType>
struct SpectrogramRet {
    nc::NDArrayPtr<DType> S;
    int n_fft;
};

template<typename DType>
inline SpectrogramRet<DType> _spectrogram(
        const nc::NDArrayPtr<DType>&    __y,
        nc::NDArrayPtr<DType>&          __S,
        int&                            __n_fft,
        const int                       __hop_length,
        const double                    __power,
        const int                       __win_length,
        const filters::STFTWindowType&  __window,
        const bool                      __center,
        const char *                    __pad_mode
        ) {
    if (__S != nullptr) {
        __n_fft = 2 * (__S.shape()[0] - 1);
    }
    else {
        auto tmp = stft(__y, __n_fft, __hop_length, __win_length, __window, __center, __pad_mode);
        auto S = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(tmp.shape()));
        std::complex<DType> * ptr_tmp = tmp.data();
        float * ptr_S = S.data();
        if (int(std::round(__power)) == 1) {
            for (auto i = 0; i < tmp.elemCount(); i++) {
                ptr_S[i] = std::sqrt( ptr_tmp[i].real() * ptr_tmp[i].real() + ptr_tmp[i].imag() * ptr_tmp[i].imag() );
            }
        }
        else if (int(std::round(__power)) == 2) {
            for (auto i = 0; i < tmp.elemCount(); i++) {
                // ptr_S[i] = ptr_tmp[i].r * ptr_tmp[i].r + ptr_tmp[i].i * ptr_tmp[i].i;
                ptr_S[i] = ptr_tmp[i].real() * ptr_tmp[i].real() + ptr_tmp[i].imag() * ptr_tmp[i].imag();
            }
        }
        else {
            for (auto i = 0; i < tmp.elemCount(); i++) {
                // ptr_S[i] = std::pow(std::sqrt( ptr_tmp[i].r * ptr_tmp[i].r + ptr_tmp[i].i * ptr_tmp[i].i ), __power);
                ptr_S[i] = std::sqrt( ptr_tmp[i].real() * ptr_tmp[i].real() + ptr_tmp[i].imag() * ptr_tmp[i].imag() );
                ptr_S[i] = std::pow(ptr_S[i], DType(__power));
            }
        }
        __S = S;
    }
    SpectrogramRet<DType> ret;
    ret.S = __S;
    ret.n_fft = __n_fft;
    return ret;
}





template <typename DType>
inline nc::NDArrayPtr<DType> melspectrogram(
        const nc::NDArrayPtr<DType>&    __y,
        nc::NDArrayPtr<DType>&          __S,
        int&                            __n_fft,
        const double                    __sr = 22050,
        const int                       __hop_length = 512,
        const int                       __win_length = -1,
        const filters::STFTWindowType   __window = filters::STFTWindowType::Hanning,
        const bool                      __center = true,
        const char *                    __pad_mode = "reflect",
        const double                    __power = 2.0
        ) {
    int n_fft = __n_fft;
    nc::NDArrayF32Ptr S = __S;
    auto ret = _spectrogram(__y, S, n_fft, __hop_length, __power, __win_length, __window, __center, __pad_mode);


    return nullptr;
}

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_SPECTRUM_H
