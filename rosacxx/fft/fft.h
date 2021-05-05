#ifndef ROSACXX_FFT_FFT_H
#define ROSACXX_FFT_FFT_H

#include <stdio.h>
#include <complex>

#if ROSACXX_FFT_PRECISION_F32
#   define fft_scalar_t float
#else // ROSACXX_FFT_PRECISION_F32
#   define fft_scalar_t double
#endif // ROSACXX_FFT_PRECISION_F32

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void FFT_FORWARD(unsigned int n, const fft_scalar_t *ri, const fft_scalar_t *ii, fft_scalar_t *ro, fft_scalar_t *io);
void FFT_INVERSE(unsigned int n, const fft_scalar_t *ri, const fft_scalar_t *ii, fft_scalar_t *ro, fft_scalar_t *io);

void RFFT_FORWARD(unsigned int n, const fft_scalar_t *ri, fft_scalar_t *co);
void RFFT_INVERSE(unsigned int n, const fft_scalar_t *ci, fft_scalar_t *ro);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // ROSACXX_FFT_FFT_H
