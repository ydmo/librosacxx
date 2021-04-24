#include "fft.h"

#ifdef USE_VAMP_KISS_FFT
#   include "kiss/kiss_fft.h"
#endif // USE_VAMP_KISS_FFT


void FFT_FORWARD(unsigned int n, const fft_scalar_t *ri, const fft_scalar_t *ii, fft_scalar_t *ro, fft_scalar_t *io) {
    vkfft::fft_forward(n, ri, ii, ro, io);
}

void FFT_INVERSE(unsigned int n, const fft_scalar_t *ri, const fft_scalar_t *ii, fft_scalar_t *ro, fft_scalar_t *io) {
    vkfft::fft_inverse(n, ri, ii, ro, io);
}

void RFFT_FORWARD(unsigned int n, const kiss_fft_scalar_t *ri, kiss_fft_scalar_t *co) {
    vkfft::rfft_forward(n, ri, co);
}

void RFFT_INVERSE(unsigned int n, const kiss_fft_scalar_t *ci, kiss_fft_scalar_t *ro) {
    vkfft::rfft_inverse(n, ci, ro);
}
