#ifndef ROSACXX_UTIL_UTILS_H
#define ROSACXX_UTIL_UTILS_H

#include <stdio.h>

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace utils {

template <typename DType>
nc::NDArrayPtr<DType> pad_center_1d(const nc::NDArrayPtr<DType>& __data, const int& __size) {
    if (__data.shape().size() != 1) throw std::invalid_argument("__data.shape().size() != 1");

    int n = __data.elemCount();
    int size = __size;
    int lpad = (size - n) / 2;
    int rpad = size - n - lpad;

    nc::NDArrayPtr<DType> padded_data = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({n+lpad+rpad}));

    auto ptr_window = __data.data();
    auto ptr_padded_window = padded_data.data();
    for (auto i = 0; i < n; i++) {
        ptr_padded_window[lpad+i] = ptr_window[i];
    }

    return padded_data;
}

template <typename DType>
nc::NDArrayBoolPtr localmax(const nc::NDArrayPtr<DType>& __data, const int& __axis=0) {
    auto shape = __data.shape();
    nc::NDArrayBoolPtr ret = nc::NDArrayBoolPtr(new nc::NDArrayBool(shape));
    if (shape.size() == 1) {
        // 1D

    }
    return ret;
}

} // namespace rosacxx
} // namespace utils

#endif // ROSACXX_UTIL_UTILS_H
