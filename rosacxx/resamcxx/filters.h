#ifndef RESAMCXX_FILTERS_H
#define RESAMCXX_FILTERS_H

#include <rosacxx/numcxx/ndarray.h>
#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/resamcxx/data.h>

#ifdef _WIN32
#   define _USE_MATH_DEFINES 1
#   include <math.h>
#   include <cmath>
#   ifndef M_PI
#       define M_PI 3.14159265358979323846
#   endif
#endif // _WIN32

namespace resam {

template<typename DType>
struct get_filter_ret {
    nc::NDArrayPtr<DType> half_window = nullptr;
    int precision = 0;
    float rolloff = 0;
};

typedef double * (*GetSTFTWindow)(const int&, const bool&);

double * GetSTFTWindow_BlackmanHarris(const int& Nx, const bool& __fftbins=true) {
    double *ptr_w = (double *)calloc(Nx, sizeof (double));
    float scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
    float scale4 = __fftbins? 4 * M_PI / Nx : 4 * M_PI / (Nx - 1);
    float scale6 = __fftbins? 6 * M_PI / Nx : 6 * M_PI / (Nx - 1);
    for (auto n = 0; n < Nx; n++) {
        ptr_w[n] = 0.35875 - 0.48829 * std::cos( scale2 * n ) + 0.14128 * std::cos( scale4 * n ) - 0.01168 * std::cos( scale6 * n );
    }
    return ptr_w;
}

template<typename DType>
inline DType sinc(const DType& x) {
    DType xpi = x * M_PI;
    return std::sin(xpi) / xpi;
}

template<typename DType>
get_filter_ret<DType> get_filter(
        const char * filter,
        const int& num_zeros=64,
        const int& precision=9,
        GetSTFTWindow window=GetSTFTWindow_BlackmanHarris,
        const float& rolloff=0.945
        ) {
    get_filter_ret<DType> ret;

    if (strcmp(filter, "sinc_window") == 0) {
        int num_bits = std::pow(2,precision);
        int n = num_bits * num_zeros;

        double * w = window(2 * n + 1, true);
        double * taper = w + n;

        nc::NDArrayPtr<DType> sinc_win = nc::linspace<DType>(0, num_zeros, n+1, true);
        DType * ptr_sinc_win = sinc_win.data();
        for (auto i = 0; i < n + 1; i++) {
            ptr_sinc_win[i] = rolloff * sinc(ptr_sinc_win[i] * rolloff) * taper[i];
        }

        free(w);


        ret.half_window = sinc_win;
        ret.precision = num_bits;
        ret.rolloff = rolloff;

    }
    else if (strcmp(filter, "kaiser_fast") == 0) {
        ret.half_window = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({kaiser_fast_half_window_len}));
        DType * ptr_ret = ret.half_window.data();
        for (auto i = 0; i < kaiser_fast_half_window_len; i++) {
            ptr_ret[i] = kaiser_fast_half_window_dat[i];
        }
        ret.precision = 512;
        ret.rolloff = 0.85;
    }
    else if (strcmp(filter, "kaiser_best") == 0) {
        ret.half_window = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({kaiser_best_half_window_len}));
        DType * ptr_ret = ret.half_window.data();
        for (auto i = 0; i < kaiser_best_half_window_len; i++) {
            ptr_ret[i] = kaiser_best_half_window_dat[i];
        }
        ret.precision = 512;
        ret.rolloff = 0.94759372;
    }
    else {
        throw std::runtime_error("Invaild filter.");
    }

    return ret;
}

} // namespace resam
#endif /* RESAMCXX_FILTERS_H */
