#ifndef ROSACXX_UTIL_UTILS_H
#define ROSACXX_UTIL_UTILS_H

#include <stdio.h>

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace utils {

template <typename DType = float>
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

template <typename DType = float>
nc::NDArrayBoolPtr localmax(const nc::NDArrayPtr<DType>& __data, const int& __axis=0) {
    return nc::localmax(__data, __axis);
}

template <typename DType = float>
nc::NDArrayPtr<DType> fix_length(const nc::NDArrayPtr<DType>& __data, const int& __size, const int& __axis=-1) {
    int axis = __axis;
    if (axis < 0) axis += __data.dims();
    int n = __data.shape()[axis];
    if (n > __size) {
        throw std::runtime_error("Not implemented.");
    }
    else if (n < __size) {
        std::vector<std::pair<int, int>> lengths(0);
        for (auto i = 0;i < __data.dims(); i++) {
            if (i == axis) {
                lengths.push_back(std::make_pair(0, __size-n));
                continue;
            }
            lengths.push_back(std::make_pair(0, 0));
        }
        return nc::pad(__data, lengths);
    }
    return nullptr;
}

} // namespace rosacxx
} // namespace utils

#endif // ROSACXX_UTIL_UTILS_H
