#ifndef ROSACXX_FILTERS_H
#define ROSACXX_FILTERS_H

#include <stdio.h>

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace filters {

enum STFTWindowType {
    Rectangular, // Boxcar,
    Bartlett,
    Hamming,
    Hanning,
    Blackman,
    Nuttall,
    BlackmanHarris,
};

template<typename DType = float>
nc::NDArrayPtr<DType> get_window(const STFTWindowType& __windowType, const int& Nx, const bool& __fftbins=true) {
    if (__windowType == STFTWindowType::Rectangular) {
        return nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}, DType(1)));
    }
    else if (__windowType == STFTWindowType::Bartlett) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        float symNx = __fftbins? float(Nx) : float(Nx-1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 2 / symNx * ( symNx / 2 - std::abs(n - symNx / 2));
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Hamming) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        float scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.54 - 0.46 * std::cos( scale2 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Hanning) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        float scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.5 - 0.5 * std::cos( scale2 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Blackman) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        float scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        float scale4 = __fftbins? 4 * M_PI / Nx : 4 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.42 - 0.5 * std::cos( scale2 * n ) + 0.08 * std::cos( scale4 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Nuttall) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        float scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        float scale4 = __fftbins? 4 * M_PI / Nx : 4 * M_PI / (Nx - 1);
        float scale6 = __fftbins? 6 * M_PI / Nx : 6 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.3635819 - 0.4891775 * std::cos( scale2 * n ) + 0.1365995 * std::cos( scale4 * n ) - 0.0106411 * std::cos( scale6 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::BlackmanHarris) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        float scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        float scale4 = __fftbins? 4 * M_PI / Nx : 4 * M_PI / (Nx - 1);
        float scale6 = __fftbins? 6 * M_PI / Nx : 6 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.35875 - 0.48829 * std::cos( scale2 * n ) + 0.14128 * std::cos( scale4 * n ) - 0.01168 * std::cos( scale6 * n );
        }
        return w;
    }
    else {
        throw std::invalid_argument("Invaild window type.");
    }
    return nullptr;
}

/// Function: cq_to_chroma
/// ----------
/// Param Name              | Type          | Note
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// ----------
/// @result                 |               |
template<typename DType = float>
inline nc::NDArrayPtr<DType> cq_to_chroma(
        const int& n_input,
        const int& bins_per_octave =            12,
        const int& n_chroma =                   12,
        const float& fmin =                     0,
        const nc::NDArrayF32Ptr& window =    nullptr,
        const bool& base_c =                    true
        ) {
    throw std::runtime_error("Not implemented error");
    return nullptr;
}

} // namespace filters
} // namespace rosacxx

#endif // ROSACXX_FILTERS_H
