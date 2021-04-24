#include "fft.h"
#include <3rd/fft/fft.h>

namespace rosacxx {
namespace core {

nc::NDArrayPtr<Complex<float>> rfft(const nc::NDArrayPtr<float>& __real_data, const int& __n_fft) {
    nc::NDArrayPtr<Complex<float>> co = nc::NDArrayPtr<Complex<float>>(new nc::NDArray<Complex<float>>({__n_fft / 2 + 1}));
    float * ptr_co = (float *)co.data();
    float * ptr_ri = __real_data.data();
    RFFT_FORWARD(__n_fft, ptr_ri, ptr_co);
    return co;
}

} // namespace core
} // namespace rosacxx
