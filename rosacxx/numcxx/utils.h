#ifndef NUMCXX_UTILS_H
#define NUMCXX_UTILS_H

#include <vector>
#include <algorithm>

namespace nc {

inline std::vector<int> _get_coor_s32(const int& __flattenedIdx, const std::vector<int>& __strides) {
    int dims = int(__strides.size());
    int remainder = __flattenedIdx;
    std::vector<int> coor(dims);
    for(int d = 0; d < dims; d++) {
        coor[d] = remainder / __strides[d];
        remainder -= (coor[d] * __strides[d]);
    }
    return coor;
}

inline std::vector<int> _get_broadcast_op_shape(const std::vector<int>& __lshape, const std::vector<int>& __rshape) {
    auto dims0 = __lshape;
    auto dims1 = __rshape;
    auto msize = std::max(__lshape.size(), __rshape.size());
    std::vector<int> odims(msize);
    if (dims0.size() == dims1.size()) {
        // get output shape ...
        for (auto i = 0; i < msize; i++) {
            odims[i] = std::max(dims0[i], dims1[i]);
        }
    }
    else {
        // unsqueeze dims0 ...
        std::vector<int> newDims0(0);
        for (auto i = 0; i < msize - dims0.size(); i++) {
            newDims0.push_back(int(1));
        }
        for (auto i = 0; i < dims0.size(); i++) {
            newDims0.push_back(dims0[i]);
        }
        // unsqueeze dims1 ...
        std::vector<int> newDims1(0);
        for (auto i = 0; i < msize - dims1.size(); i++) {
            newDims1.push_back(int(1));
        }
        for (auto i = 0; i < dims1.size(); i++) {
            newDims1.push_back(dims1[i]);
        }
        // get output shape ...
        for (auto i = 0; i < msize; i++) {
            odims[i] = std::max(newDims0[i], newDims1[i]);
        }
    }
    return odims;
}

} // namespace nc

#endif // NUMCXX_UTILS_H
