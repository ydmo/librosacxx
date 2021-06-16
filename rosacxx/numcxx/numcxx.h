#ifndef NUMCXX_H
#define NUMCXX_H

// #include <rosacxx/half/half.h>
// #include <rosacxx/complex/complex.h>

#include <rosacxx/numcxx/ndarray.h>
#include <rosacxx/numcxx/pad.h>

#include <iomanip>
#include <complex>

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
            __os << std::setprecision(8) << ptr[i] << ", ";
        }
        __os << "]";
    }
    else if (__arr.shape().size() == 2) { // is 2D array.
        __os << __arr.shape()[0] << ", " << __arr.shape()[1] << " ) = ";
        __os << "[ " << std::endl;
        for (auto s0 = 0; s0 < __arr.shape()[0]; s0++) {
            __os << "\t[ ";
            for (auto s1 = 0; s1 < __arr.shape()[1]; s1++) {
                __os << std::setprecision(8) << ptr[s0 * __arr.shape()[1] + s1] << ", ";
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
inline NDArrayPtr<DType> arange(const int& __from, const int& __to) {
    std::vector<DType> vec;
    for (int k = __from; k < __to; k++) {
        vec.push_back(DType(k));
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
    auto vec = __arr.toStdVector1D();
    std::sort(vec.begin(), vec.end());
    if (vec.size() % 2 == 0) {
        return (vec[vec.size()/2] + vec[vec.size()/2-1]) / 2;
    }
    return vec[vec.size()/2];
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

template<typename DType = float>
inline NDArrayPtr<DType> abs(const NDArrayPtr<std::complex<DType>>& __arr) {
    NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>(__arr.shape()));
    DType * ptr_ret = ret.data();
    std::complex<DType> * ptr_src = __arr.data();
    for (int i = 0; i < __arr.elemCount(); i++) {
        // ptr_ret[i] = std::sqrt(ptr_src[i].real * ptr_src[i].real + ptr_src[i].imag * ptr_src[i].imag);
        ptr_ret[i] = std::abs(ptr_src[i]);
    }
    return ret;
}

template<typename DType>
inline NDArrayPtr<DType> pow(const NDArrayPtr<DType>& __arr, DType __power) {
    NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>(__arr.shape()));
    DType * ptr_ret = ret.data();
    DType * ptr_src = __arr.data();
    for (int i = 0; i < __arr.elemCount(); i++) {
        ptr_ret[i] = std::pow(ptr_src[i], __power);
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

template<typename DType>
inline NDArrayPtr<DType> operator / (const DType& lhs, const NDArrayPtr<DType>& rhs) {
    auto r_data = rhs.data();
    auto r_shape = rhs.shape();
    auto ret = NDArrayPtr<DType>(new NDArray<DType>(r_shape));
    auto ptr_ret = ret.data();
    for (auto i = 0; i < rhs.elemCount(); i++) {
        ptr_ret[i] = lhs / r_data[i];
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

template<typename DType>
inline NDArrayBoolPtr localmax(const NDArrayPtr<DType>& __arr, const int& __axis) {
    auto iarr_shape = __arr.shape();
    auto iarr_strides = __arr.strides();

    NDArrayBoolPtr ret = NDArrayBoolPtr(new NDArrayBool(iarr_shape));
    bool * p_ret = ret.data();

    for (auto i = 0; i < ret.elemCount(); i++) {
        std::vector<int> coor_c = _get_coor_s32(i, iarr_strides);
        int s = coor_c[__axis];
        int l = std::max(s - 1, 0);
        int r = std::min(s + 1, iarr_shape[__axis] - 1);
        std::vector<int> coor_l = coor_c; coor_l[__axis] = l;
        std::vector<int> coor_r = coor_c; coor_r[__axis] = r;
        DType val_c = __arr.getitem(coor_c);
        DType val_l = __arr.getitem(coor_l);
        DType val_r = __arr.getitem(coor_r);
        p_ret[i] = (val_c > val_l) && (val_c >= val_r);
    }

    return ret;
}

template<typename DType>
inline NDArrayS32Ptr argwhere(const NDArrayPtr<DType>& __arr) {
    std::vector<std::vector<int>> coors(0);

    auto iarr_shape = __arr.shape();
    auto iarr_strides = __arr.strides();

    for (auto i = 0; i < __arr.elemCount(); i++) {
        std::vector<int> coor = _get_coor_s32(i, iarr_strides);
        if (__arr.getitem(coor)) {
            coors.emplace_back(coor);
        }
    }

    return NDArrayS32Ptr::FromVec2D(coors);
}

template<typename DType = float>
inline int len(const NDArrayPtr<DType>& __arr) {
    return __arr.shape()[0];
}

template<typename DType = float>
NDArrayPtr<DType> matmul(const NDArrayPtr<DType>& A, const NDArrayPtr<DType>& B) {
    assert(A.dims() == 2);
    assert(B.dims() == 2);
    assert(A.shape(1) == B.shape(0));
    int M = A.shape(0);
    int K = A.shape(1);
    int N = B.shape(1);
    NDArrayPtr<DType> C = NDArrayPtr<DType>(new NDArray<DType>({M, N}));

    #pragma omp parallel for
    for(int r = 0; r < M; ++r) {
        std::complex<float> * ptr_A = A.data() + r * K;
        int offsetCr = r * N;
        std::complex<float> * ptr_B = B.data();
        for(int k = 0; k < K; ++k){
            std::complex<float> * ptr_C = C.data() + offsetCr;
            for(int n = 0; n < N; ++n){
                *ptr_C++ += *ptr_A * *ptr_B++;
            }
            ptr_A += 1;
        }
    }

//    auto ptr_C = C.data();
//    auto ptr_A = A.data();
//    auto ptr_B = B.data();
//#   pragma omp parallel for
//    for(int i = 0; i < M; ++i) {
//        for(int k = 0; k < K; ++k){
//            DType A_PART = ptr_A[i*K+k];
//            for(int j = 0; j < N; ++j){
//                ptr_C[i*N+j] += A_PART * ptr_B[k*N+j];
//            }
//        }
//    }

    return C;
}

} // namepace nc

#endif // NUMCXX_H
