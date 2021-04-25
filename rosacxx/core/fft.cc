#include "fft.h"
#include <3rd/fft/fft.h>

namespace rosacxx {
namespace core {

nc::NDArrayPtr<Complex<float>> rfft(const nc::NDArrayPtr<float>& __real_data, const int& __n_fft) {
    nc::NDArrayPtr<Complex<float>> co = nc::NDArrayPtr<Complex<float>>(new nc::NDArray<Complex<float>>({__n_fft / 2 + 1}));
    float * ptr_co = (float *)co.data();
    float * ptr_ri = __real_data.data();
#   if ROSACXX_FFT_PRECISION_F32
    RFFT_FORWARD(__n_fft, ptr_ri, ptr_co);
#   else
    double * ptr_co_f64 = (double *)calloc((__n_fft / 2 + 1) * 2,   sizeof (double));
    double * ptr_ri_f64 = (double *)calloc(__real_data.elemCount(), sizeof (double));
    for (auto i = 0; i < __real_data.elemCount(); i++) {
        ptr_ri_f64[i] = ptr_ri[i];
    }
    RFFT_FORWARD(__n_fft, ptr_ri_f64, ptr_co_f64);
    for (auto i = 0; i < (__n_fft / 2 + 1) * 2; i++) {
        ptr_co[i] = ptr_co_f64[i];
    }
    free(ptr_co_f64);
    free(ptr_ri_f64);
#   endif
    return co;
}

nc::NDArrayPtr<Complex<double>> rfft(const nc::NDArrayPtr<double>& __real_data, const int& __n_fft) {
    nc::NDArrayPtr<Complex<double>> co = nc::NDArrayPtr<Complex<double>>(new nc::NDArray<Complex<double>>({__n_fft / 2 + 1}));
    double * ptr_co = (double *)co.data();
    double * ptr_ri = __real_data.data();
#   if ROSACXX_FFT_PRECISION_F32
    float * ptr_co_f32 = (float *)calloc((__n_fft / 2 + 1) * 2,   sizeof (float));
    float * ptr_ri_f32 = (float *)calloc(__real_data.elemCount(), sizeof (float));
    for (auto i = 0; i < __real_data.elemCount(); i++) {
        ptr_ri_f32[i] = ptr_ri[i];
    }
    RFFT_FORWARD(__n_fft, ptr_ri_f32, ptr_co_f32);
    for (auto i = 0; i < (__n_fft / 2 + 1) * 2; i++) {
        ptr_co[i] = ptr_co_f32[i];
    }
    free(ptr_co_f32);
    free(ptr_ri_f32);
#   else
    RFFT_FORWARD(__n_fft, ptr_ri, ptr_co);
#   endif
    return co;
}

} // namespace core
} // namespace rosacxx
