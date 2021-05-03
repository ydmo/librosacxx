#ifndef RESAMCXX_CORE_H
#define RESAMCXX_CORE_H

#include <rosacxx/numcxx/ndarray.h>
#include <rosacxx/resamcxx/filters.h>
#include <rosacxx/resamcxx/interpn.h>

namespace resam {

/// resample
/// Resample a signal x from sr_orig to sr_new along a given axis.
/// ----------
/// Parameters              | Type          | Note
/// ...
/// ----------
/// Return                  | Type          | Note
/// ...
template<typename DType>
inline nc::NDArrayPtr<DType> resample(
        const nc::NDArrayPtr<DType>& x,
        const float& sr_orig,
        const float& sr_new,
        const int& axis=-1,
        const char * filter="kaiser_best"
        ) {
    if (sr_orig <= 0) throw std::invalid_argument("Invaild sr_origin.");
    if (sr_new <= 0) throw std::invalid_argument("Invaild sr_new.");

    float sample_ratio = float(sr_new) / sr_orig;

    int dims = x.dims();
    std::vector<int> shape = x.shape();
    if (axis < 0) {
        shape[dims+axis] = int(shape[dims+axis] * sample_ratio);
        if (shape[dims+axis] < 1) throw std::runtime_error("'Input signal length is too small to resample.");
    } else {
        shape[axis] = int(shape[axis] * sample_ratio);
        if (shape[axis] < 1) throw std::runtime_error("'Input signal length is too small to resample.");
    }

    auto y = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(shape)); // nc::zeros(shape, dtype=x.dtype, order=order)

    auto ret = get_filter<DType>(filter);
    auto interp_win = ret.half_window;
    auto precision = ret.precision;

    if (sample_ratio < 1) {
        interp_win *= sample_ratio;
    }

    // interp_delta = np.zeros_like(interp_win)
    // interp_delta[:-1] = np.diff(interp_win)
    auto interp_delta = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(interp_win.shape()));
    auto ptr_interp_delta = interp_delta.data();
    auto ptr_interp_win = interp_win.data();
    for (auto i = 0; i < interp_delta.elemCount(); i++) {
        ptr_interp_delta[i] = ptr_interp_win[i+1] - ptr_interp_win[i];
    }

    if (x.dims() == 1) {
        auto x_2d = x.reshape(-1, 1);
        auto y_2d = y.reshape(-1, 1);
        resample_f(x_2d, y_2d, sample_ratio, interp_win, interp_delta, precision);
        return y_2d.reshape(-1);
    }
    else if (x.dims() == 2) {
        if (axis == 0) {
            auto x_2d = x.reshape(x.shape()[0], -1);
            auto y_2d = y.reshape(y.shape()[0], -1);
            resample_f(x_2d, y_2d, sample_ratio, interp_win, interp_delta, precision);
            return y_2d;
        }
        else {
            auto x_2d = x.T();
            auto y_2d = y.T();
            resample_f(x_2d, y_2d, sample_ratio, interp_win, interp_delta, precision);
            return y_2d.T();
        }
    }
    else {
        throw std::runtime_error("Not implemented.");
    }

    return nullptr;
}
} // namespace resam
#endif /* RESAMCXX_CORE_H */
