#ifndef ROSACXX_UTIL_UTILS_H
#define ROSACXX_UTIL_UTILS_H

#include <stdio.h>

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace utils {

template <typename DType = float>
inline nc::NDArrayPtr<DType> pad_center_1d(const nc::NDArrayPtr<DType>& __data, const int& __size) {
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
inline nc::NDArrayBoolPtr localmax(const nc::NDArrayPtr<DType>& __data, const int& __axis=0) {
    return nc::localmax(__data, __axis);
}

template <typename DType = float>
inline nc::NDArrayPtr<DType> fix_length(const nc::NDArrayPtr<DType>& __data, const int& __size, const int& __axis=-1) {
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

template <typename DType = float>
inline nc::NDArrayPtr<std::complex<DType>> sparsify_rows(
        const nc::NDArrayPtr<std::complex<DType>>& __x,
        const float& __quantile=1e-2
        ) {
    nc::NDArrayPtr<std::complex<DType>> x = __x;
    float quantile = __quantile;
    // --------
    if (x.dims() == 1) {
        x = x.reshape({1, -1});
    }
    else if (x.dims() > 2) {
        throw std::invalid_argument("Input must have 2 or fewer dimensions. ");
    }
    if (!(0 <= quantile && quantile <1)) {
        throw std::invalid_argument("Invalid quantile");
    }
    // --------
    auto mags = nc::abs(x);
    auto vec_mags = mags.toStdVector2D();
    auto vec_mags_sort = vec_mags;

    // norms = np.sum(mags, axis=1, keepdims=True)
    std::vector<DType> norms(vec_mags.size(), 0);
    for (auto r = 0; r < vec_mags.size(); r++) {
        auto vec_mag = vec_mags[r];
        for (auto c = 0; c < vec_mag.size(); c++) {
            norms[r] += vec_mag[c];
        }
        std::sort(vec_mags_sort[r].begin(), vec_mags_sort[r].end(), [](DType a, DType b) ->bool { return a < b; });
    }

    auto cumulative_mag = vec_mags_sort;
    std::vector<int> threshold_idx(cumulative_mag.size(), 0);
    for (auto r = 0; r < vec_mags_sort.size(); r++) {
        auto vec_mag_cums = cumulative_mag[r];
        auto vec_mag_sort = vec_mags_sort[r];
        auto norm = norms[r];
        vec_mag_cums[0] = vec_mag_sort[0] / norm;
        if (vec_mag_cums[0] < __quantile) threshold_idx[r] += 1;
        for (auto c = 1; c < vec_mag_cums.size(); c++) {
            vec_mag_cums[c] = vec_mag_sort[c] / norm + vec_mag_cums[c - 1];
            if (vec_mag_cums[c] < __quantile) threshold_idx[r] += 1;
        }
        cumulative_mag[r] = vec_mag_cums;
    }

    nc::NDArrayPtr<std::complex<DType>> x_sparse = nc::NDArrayPtr<std::complex<DType>>(new nc::NDArray<std::complex<DType>>(x.shape()));

    for (int i = 0; i < threshold_idx.size(); i++) {
        int j = threshold_idx[i];
        std::vector<int> idx(0);
        auto mags_i = vec_mags[i];
        auto mags_sort_i = vec_mags_sort[i];
        for (int c = 0; c < mags_i.size(); c++) {
            if (mags_i[c] >= mags_sort_i[j]) {
                idx.push_back(c);
                *x_sparse.at(i, c) = x.getitem(i, c);
            }
        }
    }

    auto vec_x_sparse = x_sparse.toStdVector2D();

    return x_sparse;
}

} // namespace rosacxx
} // namespace utils

#endif // ROSACXX_UTIL_UTILS_H
