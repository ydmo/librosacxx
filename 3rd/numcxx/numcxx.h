#ifndef NUMCXX_H
#define NUMCXX_H

#include "ndarray.h"

namespace nc {

template<typename DType>
inline std::ostream &operator << (std::ostream &__os, const std::shared_ptr<NDArray<DType>>& __arr) {
    const DType * ptr = __arr->data();
    const char * DTypeName = typeid(ptr[0]).name();
    __os.clear();
    __os << "nc::NDArray<";
    __os << DTypeName;
    __os << ">( ";
    if (__arr->shape().size() == 1) { // is 1D array.
        __os << "[ ";
        for (auto i = 0; i < __arr->elemCount(); i++) {
            __os << ptr[i] << ", ";
        }
        __os << "]";
    }
    else {
        throw std::runtime_error("Not Implemented.");
    }
    __os << " )" << std::endl;
    return __os;
}

template<typename DType>
inline std::shared_ptr<NDArray<DType>> scalar(const DType& val) {
    return std::shared_ptr<NDArray<DType>>(new NDArray<DType>({1}, val));
}

template<typename DType>
inline std::shared_ptr<NDArray<DType>> linspace(const DType& start, const DType& stop, const size_t& num, const bool& endpoint = true, DType * restep = NULL) {
    auto res = std::shared_ptr<NDArray<DType>>(new NDArray<DType>({int(num)}));
    auto ptr_res = res->data();
    DType step = 0;
    if (endpoint) {
        step = (stop - start) / (num - 1);
    }
    else {
        step = (stop - start) / num;
    }
    for (auto i = 0; i < num; i++) {
        ptr_res[i] = start + i * step;
    }
    if (restep) {
        *restep = step;
    }
    return res;
}

template<typename DType>
inline std::shared_ptr<NDArray<int>> histogram(const std::shared_ptr<NDArray<DType>>& a, const std::shared_ptr<NDArray<DType>>& bins) {
    auto ptr_a = a->data();
    auto ptr_bins = bins->data();
    std::shared_ptr<NDArray<int>> hist = std::shared_ptr<NDArray<int>>(new NDArray<int>({int(bins->elemCount()-1)}, 0));
    auto ptr_hist = hist->data();
    for (auto i = 0; i < a->elemCount(); i++) {
        auto va = ptr_a[i];
        for (auto h = 0; h < hist->elemCount(); h++) {
            if (ptr_bins[h] <= va && va < ptr_bins[h+1]) {
                ptr_hist[h] += 1;
            }
        }
    }
    return hist;
}

template<typename DType>
int argmax(const std::shared_ptr<NDArray<DType>>& a, int axis = -1) {
    if (axis >= 0) {
        throw std::runtime_error("Not implemented.");
    }
    else {
        auto ptr_a = a->data();
        int max_idx = 0;
        DType max_a = ptr_a[0];
        for (auto i = 0; i < a->elemCount(); i++) {
            if (ptr_a[i] > max_a) {
                max_a = ptr_a[i];
                max_idx = i;
            }
        }
        return max_idx;
    }
}

}

#endif // NUMCXX_H
