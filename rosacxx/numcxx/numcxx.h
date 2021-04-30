#ifndef NUMCXX_H
#define NUMCXX_H

#include "ndarray.h"
#include "pad.h"

#include <rosacxx/half/half.h>
#include <rosacxx/complex/complex.h>

namespace nc {

template<typename DType>
inline std::ostream &operator << (std::ostream &__os, const NDArrayPtr<DType>& __arr) {
    const DType * ptr = __arr.data();
    const char * DTypeName = typeid(ptr[0]).name();
    __os.clear();
    __os << "nc::NDArray<";
    __os << DTypeName;
    __os << ">( ";
    if (__arr.shape().size() == 1) { // is 1D array.
        __os << __arr.shape()[0] << " ) = ";
        __os << "[ ";
        for (auto i = 0; i < __arr.elemCount(); i++) {
            __os << ptr[i] << ", ";
        }
        __os << "]";
    }
    else if (__arr.shape().size() == 2) { // is 2D array.
        __os << __arr.shape()[0] << ", " << __arr.shape()[1] << " ) = ";
        __os << "[ " << std::endl;
        for (auto s0 = 0; s0 < __arr.shape()[0]; s0++) {
            __os << "\t[ ";
            for (auto s1 = 0; s1 < __arr.shape()[1]; s1++) {
                __os << ptr[s0 * __arr.shape()[1] + s1] << ", ";
            }
            __os << "], " << std::endl;
        }
        __os << "\t]";
    }
    else {
        throw std::runtime_error("Not Implemented.");
    }
    __os << std::endl;
    return __os;
}

template<typename DType>
inline NDArrayPtr<DType> linspace(const DType& start, const DType& stop, const size_t& num, const bool& endpoint = true, DType * restep = NULL) {
    auto res = NDArrayPtr<DType>(new NDArray<DType>({int(num)}));
    auto ptr_res = res.data();
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
inline NDArrayS32Ptr histogram(const NDArrayPtr<DType>& a, const NDArrayPtr<DType>& bins) {
    auto ptr_a = a.data();
    auto ptr_bins = bins.data();
    NDArrayS32Ptr hist = NDArrayS32Ptr(new NDArray<int>({int(bins.elemCount()-1)}, 0));
    auto ptr_hist = hist.data();
    for (auto i = 0; i < a.elemCount(); i++) {
        auto va = ptr_a[i];
        for (auto h = 0; h < hist.elemCount(); h++) {
            if (ptr_bins[h] <= va && va < ptr_bins[h+1]) {
                ptr_hist[h] += 1;
            }
        }
    }
    return hist;
}

template<typename DType>
inline NDArrayPtr<DType> arange(DType __x) {
    int cx = int(std::ceil(__x));
    std::vector<DType> vec;
    for (int i = 0; i < cx; i++) {
        vec.push_back(DType(i));
    }
    return NDArrayPtr<DType>::FromVec1D(vec);
}

template<typename DType>
inline NDArrayS32Ptr argmax(const NDArrayPtr<DType>& a, int axis = -1) {
    if (axis >= 0) {
        return a.argmax(axis);
    }
    else {
        return NDArrayS32Ptr::FromScalar(a.argmax());
    }
}

template<typename DType>
inline NDArrayPtr<DType> max(const NDArrayPtr<DType>& a, int axis = -1) {
    if (axis >= 0) {
        return a.max(axis);
    }
    else {
        return NDArrayPtr<DType>::FromScalar(a.max());
    }
}

template<typename DType>
inline NDArrayS32Ptr argmin(const NDArrayPtr<DType>& a, int axis = -1) {
    if (axis >= 0) {
        return a.argmin(axis);
    }
    else {
        return NDArrayS32Ptr::FromScalar(a.argmin());
    }
}

template<typename DType>
inline DType median(const NDArrayPtr<DType>& __arr) {
    DType sum = 0;
    for (int i = 0; i < __arr.elemCount(); i++) {
        sum += __arr.getitem(i);
    }
    return sum / __arr.elemCount();
}

template<typename DType>
inline NDArrayPtr<DType> abs(const NDArrayPtr<DType>& __arr) {
    NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>(__arr.shape()));
    DType * ptr_ret = ret.data();
    DType * ptr_src = __arr.data();
    for (int i = 0; i < __arr.elemCount(); i++) {
        ptr_ret[i] = std::abs(ptr_src[i]);
    }
    return ret;
}

template<typename DType>
inline NDArrayPtr<DType> abs(const NDArrayPtr<complex::Complex<DType>>& __arr) {
    NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>(__arr.shape()));
    DType * ptr_ret = ret.data();
    complex::Complex<DType> * ptr_src = __arr.data();
    for (int i = 0; i < __arr.elemCount(); i++) {
        ptr_ret[i] = std::sqrt(ptr_src[i].r * ptr_src[i].r + ptr_src[i].i * ptr_src[i].i);
    }
    return ret;
}

template<typename DType>
inline NDArrayPtr<DType> pow(const NDArrayPtr<DType>& __arr, DType __power) {
    NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>(__arr.shape()));
    DType * ptr_ret = ret.data();
    DType * ptr_src = __arr.data();
    for (int i = 0; i < __arr.elemCount(); i++) {
        ptr_ret = std::pow(ptr_src[i], __power);
    }
    return ret;
}

template<typename DType>
inline NDArrayPtr<DType> operator * (const DType& lhs, const NDArrayPtr<DType>& rhs) {
    auto r_data = rhs.data();
    auto r_shape = rhs.shape();
    auto ret = NDArrayPtr<DType>(new NDArray<DType>(r_shape));
    auto ptr_ret = ret.data();
    for (auto i = 0; i < rhs.elemCount(); i++) {
        ptr_ret[i] = r_data[i] * lhs;
    }
    return ret;
}

template<typename LType, typename RType>
NDArrayPtr<bool> operator >= (const LType& lhs, const NDArrayPtr<RType>& rhs) {
    auto r_data = rhs.data();
    auto r_shape = rhs.shape();
    NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArrayBool(r_shape));
    bool * ptr_ret = ret.data();
    for (auto i = 0; i < ret.elemCount(); i++) {
        if (lhs >= r_data[i]) ptr_ret[i] = true;
    }
    return ret;
}

template<typename LType, typename RType>
NDArrayPtr<bool> operator <= (const LType& lhs, const NDArrayPtr<RType>& rhs) {
    auto r_data = rhs.data();
    auto r_shape = rhs.shape();
    NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArrayBool(r_shape));
    bool * ptr_ret = ret.data();
    for (auto i = 0; i < ret.elemCount(); i++) {
        if (lhs <= r_data[i]) ptr_ret[i] = true;
    }
    return ret;
}

template<typename DType>
inline NDArrayPtr<DType> zeros_like(const NDArrayPtr<DType>& __arr) {
    return NDArrayPtr<DType>(new NDArray<DType>(__arr.shape()));
}

} // namepace nc

#endif // NUMCXX_H
