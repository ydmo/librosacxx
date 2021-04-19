#ifndef VAMP_KISS_FFT_SCALAR_H
#define VAMP_KISS_FFT_SCALAR_H

#include <cstdio>
#include <stdexcept>
#include <cmath>
#include <cstring>

#if ROSACXX_FFT_PRECISION_F32
#   define vamp_kiss_fft_scalar float
#else
#   define vamp_kiss_fft_scalar double
#endif

#endif // VAMP_KISS_FFT_SCALAR_H
