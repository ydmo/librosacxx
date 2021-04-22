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
        __os << __arr->shape()[0] << " ) = ";
        __os << "[ ";
        for (auto i = 0; i < __arr->elemCount(); i++) {
            __os << ptr[i] << ", ";
        }
        __os << "]";
    }
    else if (__arr->shape().size() == 2) { // is 2D array.
        __os << __arr->shape()[0] << ", " << __arr->shape()[1] << " ) = ";
        __os << "[ " << std::endl;
        for (auto s0 = 0; s0 < __arr->shape()[0]; s0++) {
            __os << "\t[ ";
            for (auto s1 = 0; s1 < __arr->shape()[1]; s1++) {
                __os << ptr[s0 * __arr->shape()[1] + s1] << ", ";
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
std::shared_ptr<NDArray<DType>> arange(DType __x) {
    int cx = int(std::ceil(__x));
    std::vector<DType> vec;
    for (int i = 0; i < cx; i++) {
        vec.push_back(DType(i));
    }
    return NDArray<DType>::FromVec1D(vec);
}

template<typename DType>
std::shared_ptr<NDArray<int>> argmax(const std::shared_ptr<NDArray<DType>>& a, int axis = -1) {
    if (axis >= 0) {
        return a->argmax(axis);
    }
    else {
        return NDArray<int>::FromScalar(a->argmax());
    }
}

template<typename DType>
std::shared_ptr<NDArray<int>> argmin(const std::shared_ptr<NDArray<DType>>& a, int axis = -1) {
    if (axis >= 0) {
        return a->argmin(axis);
    }
    else {
        return NDArray<int>::FromScalar(a->argmin());
    }
}

template<typename DType>
DType median(const std::shared_ptr<NDArray<DType>>& __arr) {
    DType sum = 0;
    for (int i = 0; i < __arr->elemCount(); i++) {
        sum += __arr->getitem(i);
    }
    return sum / __arr->elemCount();
}

template<typename DType>
std::shared_ptr<NDArray<DType>> operator + (const std::shared_ptr<NDArray<DType>>& lhs, const DType& rhs) {
    return lhs->add(rhs);
}

template<typename DType, typename RType>
NDArrayBool::Ptr operator < (const std::shared_ptr<NDArray<DType>>& lhs, const RType& rhs) {
    NDArrayBool::Ptr ret = NDArrayBool::Ptr(new NDArrayBool(lhs->shape()));
    bool * ptr_ret = ret->data();
    for (auto i = 0; i < ret->elemCount(); i++) {
        if (lhs->getitem(i) < rhs) ptr_ret[i] = true;
    }
    return ret;
}

template<typename DType, typename RType>
NDArrayBool::Ptr operator <= (const std::shared_ptr<NDArray<DType>>& lhs, const RType& rhs) {
    NDArrayBool::Ptr ret = NDArrayBool::Ptr(new NDArrayBool(lhs->shape()));
    bool * ptr_ret = ret->data();
    for (auto i = 0; i < ret->elemCount(); i++) {
        if (lhs->getitem(i) <= rhs) ptr_ret[i] = true;
    }
    return ret;
}

template<typename DType, typename RType>
NDArrayBool::Ptr operator > (const std::shared_ptr<NDArray<DType>>& lhs, const RType& rhs) {
    NDArrayBool::Ptr ret = NDArrayBool::Ptr(new NDArrayBool(lhs->shape()));
    bool * ptr_ret = ret->data();
    for (auto i = 0; i < ret->elemCount(); i++) {
        if (lhs->getitem(i) > rhs) ptr_ret[i] = true;
    }
    return ret;
}

template<typename DType, typename RType>
NDArrayBool::Ptr operator >= (const std::shared_ptr<NDArray<DType>>& lhs, const RType& rhs) {
    NDArrayBool::Ptr ret = NDArrayBool::Ptr(new NDArrayBool(lhs->shape()));
    bool * ptr_ret = ret->data();
    for (auto i = 0; i < ret->elemCount(); i++) {
        if (lhs->getitem(i) >= rhs) ptr_ret[i] = true;
    }
    return ret;
}

template<typename DType>
std::shared_ptr<NDArray<DType>> operator & (const std::shared_ptr<NDArray<DType>>& __lhs, const std::shared_ptr<NDArray<DType>>& __rhs) {
    if (__lhs->shape() != __rhs->shape()) throw std::runtime_error("Invaild input params");
    NDArrayBool::Ptr ret = NDArrayBool::Ptr(new NDArrayBool(__lhs->shape()));
    DType * ptr_ret = ret->data();
    DType * ptr_lhs = __lhs->data();
    DType * ptr_rhs = __rhs->data();
    for (auto i = 0; i < ret->elemCount(); i++) {
        ptr_ret[i] = (ptr_lhs[i] & ptr_rhs[i]);
    }
}

NDArrayBool::Ptr operator && (const NDArrayBool::Ptr& __lhs, const NDArrayBool::Ptr& __rhs) {
    if (__lhs->shape() != __rhs->shape()) throw std::runtime_error("Invaild input params");
    NDArrayBool::Ptr ret = NDArrayBool::Ptr(new NDArrayBool(__lhs->shape()));
    bool * ptr_ret = ret->data();
    bool * ptr_lhs = __lhs->data();
    bool * ptr_rhs = __rhs->data();
    for (auto i = 0; i < ret->elemCount(); i++) {
        ptr_ret[i] = (ptr_lhs[i] && ptr_rhs[i]);
    }
}

}

#endif // NUMCXX_H
