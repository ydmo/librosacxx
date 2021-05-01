#ifndef NUMCXX_PAD_H
#define NUMCXX_PAD_H

#include <rosacxx/numcxx/ndarray.h>
#include <vector>

namespace nc {

/// constant pad
/// @param __array ( NDArrayPtr<DType> ) : Input array
/// @param __pad_width (std::vector<std::pair<int, int>>) : Number of values padded to the edges of each axis. ((before_1, after_1), …
template <typename DType>
NDArrayPtr<DType> pad(const NDArrayPtr<DType>& __array, const std::vector<std::pair<int, int>>& __pad_width, const DType& __constan_val = DType(0)) {
    if (__array == nullptr) throw std::invalid_argument("Invaild input array.");
    if (__array.shape().size() != __pad_width.size()) throw std::invalid_argument("Invaild input pad width.");
    const std::vector<int> oldstrides = __array.strides();
    const std::vector<int> oldshape = __array.shape();
    std::vector<int> newshape = oldshape;
    const int dims = newshape.size();
    for (auto d = 0; d < dims; d++) {
        newshape[d] += (__pad_width[d].first + __pad_width[d].second);
    }
    NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>(newshape, __constan_val));
    const std::vector<int> newstrides = ret.strides();
    DType *ptr_ret = ret.data();
    DType *ptr_src = __array.data();
    for (auto i = 0; i < __array.elemCount(); i++) {
        std::vector<int> oldcoor(dims);
        int remainder = i;
        for(int d = 0; d < dims; d++) {
            oldcoor[d] = remainder / oldstrides[d];
            remainder -= (oldcoor[d] * oldstrides[d]);
        }
        int new_loc = 1;
        std::vector<int> newcoor = oldcoor;
        for (auto d = 0; d < dims; d++) {
            newcoor[d] += __pad_width[d].first;
            new_loc *= (newcoor[d] + newstrides[d]);
        }
        ptr_ret[new_loc] = ptr_src[i];
    }
    return ret;
}

enum ReflectPadType {
    Even,
    Odd
};

template <typename DType>
NDArrayPtr<DType> reflect_pad1d(const NDArrayPtr<DType>& __array, const std::pair<int, int>& __pad_width, const ReflectPadType& __reflect_type = ReflectPadType::Even) {
    int src_size = __array.elemCount();
    DType * ptr_src = __array.data();

    NDArrayPtr<DType> reflect = NDArrayPtr<DType>(new NDArray<DType>({ int(src_size * 2 - 2) }));
    DType * ptr_reflect = reflect.data();
    memcpy(reflect.data(), ptr_src, src_size * sizeof(DType));
    for (auto i = src_size; i < src_size * 2 - 2; i++) {
        ptr_reflect[i] = ptr_src[src_size - 2 - (i - src_size)];
    }
    int reflect_size = reflect.elemCount();


    NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>({ src_size + __pad_width.first + __pad_width.second }));
    DType * ptr_ret = ret.data();
    memcpy(ptr_ret + __pad_width.first, ptr_src, src_size * sizeof(DType));

    // write pad left ...
    for (int i = 0; i < __pad_width.first; i++) {
        ptr_ret[__pad_width.first - 1 - i] = ptr_reflect[(i + 1) % reflect_size];
    }

    // write pad right ...
    for (int i = 0; i < __pad_width.second; i++) {
        ptr_ret[__pad_width.first + src_size + i] = ptr_reflect[(src_size + i) % reflect_size];
    }

    return ret;
}

} // namespace nc

#endif // NC_PAD_H
