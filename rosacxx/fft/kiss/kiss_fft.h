#ifndef ROSACXX_FFT_KISS_FFT_H
#define ROSACXX_FFT_KISS_FFT_H

#include <stdio.h>
#include <complex>

#if ROSACXX_FFT_PRECISION_F32
#   define kiss_fft_scalar_t float
#else // ROSACXX_FFT_PRECISION_F32
#   define kiss_fft_scalar_t double
#endif // ROSACXX_FFT_PRECISION_F32

namespace vkfft {

void fft_forward(unsigned int n, const std::complex<kiss_fft_scalar_t> *ci, std::complex<kiss_fft_scalar_t> *co);
void fft_inverse(unsigned int n, const std::complex<kiss_fft_scalar_t> *ci, std::complex<kiss_fft_scalar_t> *co);

void fft_forward(unsigned int n, const kiss_fft_scalar_t *ri, const kiss_fft_scalar_t *ii, kiss_fft_scalar_t *ro, kiss_fft_scalar_t *io);
void fft_inverse(unsigned int n, const kiss_fft_scalar_t *ri, const kiss_fft_scalar_t *ii, kiss_fft_scalar_t *ro, kiss_fft_scalar_t *io);

void rfft_forward(unsigned int n, const kiss_fft_scalar_t *ri, kiss_fft_scalar_t *co);
void rfft_inverse(unsigned int n, const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *ro);

class FFTComplex {
public:
    FFTComplex(unsigned int n);
    ~FFTComplex();
    void forward(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *co);
    void inverse(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *co);
private:
    class D;
    D *m_d;
};

class FFTReal {
public:
    FFTReal(unsigned int n);
    ~FFTReal();
    void forward(const kiss_fft_scalar_t *ri, kiss_fft_scalar_t *co);
    void inverse(const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *ro);
private:
    class D;
    D *m_d;
};

} // namespace vkfft

#endif
