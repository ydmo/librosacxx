#ifndef ROSACXX_CORE_FFT_H
#define ROSACXX_CORE_FFT_H

#include <rosacxx/numcxx/numcxx.h>
// #include <rosacxx/complex/complex.h>

namespace rosacxx {
namespace core {

nc::NDArrayPtr<std::complex<float >> rfft(const nc::NDArrayPtr<float >& __real_data, const int& __n_fft);

nc::NDArrayPtr<std::complex<double>> rfft(const nc::NDArrayPtr<double>& __real_data, const int& __n_fft);

} // namespace core
} // namespace rosacxx

#endif
