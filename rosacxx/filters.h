#ifndef ROSACXX_FILTERS_H
#define ROSACXX_FILTERS_H

#include <stdio.h>
#include <map>
#include <string>
#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace filters {

enum STFTWindowType : int {
    Rectangular =       0, // Boxcar,
    Bartlett =          1,
    Hamming =           2,
    Hanning =           3,
    Blackman =          4,
    BlackmanHarris =    5,
    Nuttall =           6,
    End =               7,
};

constexpr double WINDOW_BANDWIDTHS[7] = {
    /* STFTWindowType::Rectangular = */     1.0,
    /* STFTWindowType::Bartlett = */        1.3334961334912805,
    /* STFTWindowType::Hamming = */         1.3629455320350348,
    /* STFTWindowType::Hanning = */         1.50018310546875,
    /* STFTWindowType::Blackman = */        1.7269681554262326,
    /* STFTWindowType::blackmanharris = */  2.0045975283585014,
    /* STFTWindowType::Nuttall = */         1.9763500280946082,
};

//--- python-librosa -----------------------------------
//    WINDOW_BANDWIDTHS = {
//        "bart": 1.3334961334912805,
//        "barthann": 1.4560255965133932,
//        "bartlett": 1.3334961334912805,
//        "bkh": 2.0045975283585014,
//        "black": 1.7269681554262326,
//        "blackharr": 2.0045975283585014,
//        "blackman": 1.7269681554262326,
//        "blackmanharris": 2.0045975283585014,
//        "blk": 1.7269681554262326,
//        "bman": 1.7859588613860062,
//        "bmn": 1.7859588613860062,
//        "bohman": 1.7859588613860062,
//        "box": 1.0,
//        "boxcar": 1.0,
//        "brt": 1.3334961334912805,
//        "brthan": 1.4560255965133932,
//        "bth": 1.4560255965133932,
//        "cosine": 1.2337005350199792,
//        "flat": 2.7762255046484143,
//        "flattop": 2.7762255046484143,
//        "flt": 2.7762255046484143,
//        "halfcosine": 1.2337005350199792,
//        "ham": 1.3629455320350348,
//        "hamm": 1.3629455320350348,
//        "hamming": 1.3629455320350348,
//        "han": 1.50018310546875,
//        "hann": 1.50018310546875,
//        "hanning": 1.50018310546875,
//        "nut": 1.9763500280946082,
//        "nutl": 1.9763500280946082,
//        "nuttall": 1.9763500280946082,
//        "ones": 1.0,
//        "par": 1.9174603174603191,
//        "parz": 1.9174603174603191,
//        "parzen": 1.9174603174603191,
//        "rect": 1.0,
//        "rectangular": 1.0,
//        "tri": 1.3331706523555851,
//        "triang": 1.3331706523555851,
//        "triangle": 1.3331706523555851,
//    }
//--- python-librosa -----------------------------------

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

/// Function: window_bandwidth
/// Get the equivalent noise bandwidth of a window function.
/// ----------
/// Param Name              | Type           | Note
/// @param windowType       | STFTWindowType | Enum of a window function.
/// ----------
/// @return                 | float          | The equivalent noise bandwidth (in FFT bins) of the given window function.
inline float window_bandwidth(const STFTWindowType& __windowType, const int& __n = 1000) {
    if (__windowType < STFTWindowType::End) {
        return WINDOW_BANDWIDTHS[__windowType];
    }
    else {
        throw std::runtime_error("Not implemented of window_bandwidth.");
        return -1;
    }
}

} // namespace filters
} // namespace rosacxx

#endif // ROSACXX_FILTERS_H
