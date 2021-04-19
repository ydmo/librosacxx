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
