#ifndef ROSACXX_CORE_SPECTRUM_H
#define ROSACXX_CORE_SPECTRUM_H

#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/core/fft.h>
#include <rosacxx/filters.h>
#include <vector>

namespace rosacxx {
namespace core {

nc::NDArrayPtr<std::complex<float>> stft(
        const nc::NDArrayF32Ptr&        y,
        const int&                      n_fft = 2048,
        const int&                      hop_length = -1,
        const int&                      win_length = -1,
        const filters::STFTWindowType&  window = filters::STFTWindowType::Hanning,
        const bool&                     center = true,
        const char *                    pad_mode = "reflect"
    );

nc::NDArrayPtr<std::complex<double>> stft(
        const nc::NDArrayF64Ptr&        y,
        const int&                      n_fft = 2048,
        const int&                      hop_length = -1,
        const int&                      win_length = -1,
        const filters::STFTWindowType&  window = filters::STFTWindowType::Hanning,
        const bool&                     center = true,
        const char *                    pad_mode = "reflect"
    );

void _spectrogram(
        const nc::NDArrayF32Ptr&        y,
        nc::NDArrayF32Ptr&              S,
        int&                            n_fft,
        const int&                      hop_length  = 512,
        const float&                    power       = 1,
        const int&                      win_length  = -1,
        const filters::STFTWindowType&  window      = filters::STFTWindowType::Hanning,
        const bool&                     center      = true,
        const char *                    pad_mode    = "reflect"
        );

nc::NDArrayPtr<float> istft(
    const nc::NDArrayF32Ptr& stft_matrix,
    const int& hop_length = -1,
    const int& win_length = -1,
    const filters::STFTWindowType& window = filters::STFTWindowType::Hanning,
    const bool& center = true,
    const int& length = 0
    );

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_SPECTRUM_H
