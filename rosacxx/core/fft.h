#ifndef ROSACXX_CORE_FFT_H
#define ROSACXX_CORE_FFT_H

#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/fft/fft.h>

namespace rosacxx {
namespace core {

//nc::NDArrayPtr<std::complex<float >> rfft(const nc::NDArrayPtr<float >& __real_data, const int& __n_fft);

//nc::NDArrayPtr<std::complex<double>> rfft(const nc::NDArrayPtr<double>& __real_data, const int& __n_fft);

template<typename DType>
inline nc::NDArrayPtr<std::complex<DType>> rfft(const nc::NDArrayPtr<DType>& __real_data, const int& __n_fft) {
    nc::NDArrayPtr<std::complex<DType>> co = nc::NDArrayPtr<std::complex<DType>>(new nc::NDArray<std::complex<DType>>({__n_fft / 2 + 1}));
    DType * ptr_co = (DType *)co.data();
    DType * ptr_ri = __real_data.data();
    fft_scalar_t * ptr_co_f32 = (fft_scalar_t *)calloc((__n_fft / 2 + 1) * 2,   sizeof (fft_scalar_t));
    fft_scalar_t * ptr_ri_f32 = (fft_scalar_t *)calloc(__real_data.elemCount(), sizeof (fft_scalar_t));
    for (auto i = 0; i < __real_data.elemCount(); i++) {
        ptr_ri_f32[i] = ptr_ri[i];
    }
    RFFT_FORWARD(__n_fft, ptr_ri_f32, ptr_co_f32);
    for (auto i = 0; i < (__n_fft / 2 + 1) * 2; i++) {
        ptr_co[i] = ptr_co_f32[i];
    }
    free(ptr_co_f32);
    free(ptr_ri_f32);
    return co;
}

} // namespace core
} // namespace rosacxx

#endif
