#ifndef ROSACXX_UTIL_ARRAY_H
#define ROSACXX_UTIL_ARRAY_H

#include <stdio.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>

namespace rosacxx {

inline size_t alignUp(const size_t& __size, const size_t& __align) {
    const size_t alignment_mask = __align - 1;
    return ((__size + alignment_mask) & (~alignment_mask));
}

inline void * alignedMalloc(size_t __alignment, size_t __size) {
#   ifdef _WIN32
    return _alignedMalloc(__alignment, alignUp(__size, __alignment));
#   else
    void *p = NULL;
    if (__alignment) {
        int ret = posix_memalign(&p, __alignment, alignUp(__size, __alignment));
        if(ret) { //  failed.
            throw std::runtime_error("posix_memalign failed.");
            return NULL;
        }
    } else {
        p = malloc(__size);
    }
    return p;
#   endif
}

inline void * alignedCalloc(size_t __alignment, size_t __size) {
#   ifdef _WIN32
    size_t alignedSize = alignUp(__size, __alignment);
    void * ptr = _alignedMalloc(__alignment, alignedSize);
    memset(ptr, 0x00, alignedSize);
    return ptr;
#   else
    void *p = NULL;
    if (__alignment) {
        size_t alignedSize = alignUp(__size, __alignment);
        int ret = posix_memalign(&p, __alignment, alignedSize);
        memset(p, 0x00, alignedSize);
        if(ret) { //  failed.
            throw std::runtime_error("posix_memalign failed.");
            return NULL;
        }
    } else {
        p = calloc(__size, 1);
    }
    return p;
#   endif
}

inline void alignedFree(void * __p) {
#   ifdef _WIN32
    _aligned_free(__p);
#   else
    free(__p);
#   endif
}

template<typename DType>
class NDArray {
public:
    ~NDArray() {
        _shape.clear();
        if (_data) {
            alignedFree(_data); _data = NULL;
        }
    }

    NDArray() {
        _shape.clear();
        _data = NULL;
    }

    NDArray(const std::vector<int>& i_shape, const DType& val = 0) : _shape(i_shape) {
        int elemCnt = 1;
        for (const int& s : i_shape) {
           elemCnt *= s;
        }
        if (val == 0) {
            _data = (DType *)alignedCalloc(32, elemCnt * sizeof (DType));
        }
        else {
            _data = (DType *)alignedMalloc(32, elemCnt * sizeof (DType));
            for (int n = 0; n < elemCnt; n++) {
                _data[n] = val;
            }
        }
    }

    inline size_t elemCount() const {
        size_t elemCnt = 1;
        for (const int& s : _shape) {
           elemCnt *= s;
        }
        return elemCnt;
    }

    inline size_t bytesCount() const {
        return this->elemCount() * sizeof (DType);
    }

    inline std::shared_ptr<NDArray> clone() const {
        auto sptr_clone = std::make_shared<NDArray>(this->_shape);
        memcpy(sptr_clone->_data, this->_data, this->bytesCount());
        return sptr_clone;
    }

    inline std::vector<int> shape() const {
        return _shape;
    }

    inline DType * data() const {
        return  _data;
    }

    inline std::shared_ptr<NDArray> dot(const std::shared_ptr<NDArray>& other) {
        if (this->_shape.size() == 1 && other->_shape.size() == 1) {
            if (this->_shape[0] != this->_shape[0]) {
                throw std::runtime_error("invalid shape");
            }
            DType sum = 0;
            for (auto i = 0; i < this->_shape[0]; i++) {
                sum += (this->_data[i] * other->_data[i]);
            }
            auto ptr = std::shared_ptr<NDArray>(new NDArray({1}, sum));
            return ptr;
        }
        else if (this->_shape.size() == 2 && other->_shape.size() == 2) {

        }
        else {
            throw std::runtime_error("Not implemented error");
        }
    }

private:
    std::vector<int> _shape;
    DType *          _data;
};

typedef NDArray<float> NDArrayF32;
typedef NDArray<int>   NDArrayS32;

typedef std::shared_ptr<NDArrayF32> NDArrayF32Ptr;
typedef std::shared_ptr<NDArrayS32> NDArrayS32Ptr;

} // namespace rosacxx

#endif
