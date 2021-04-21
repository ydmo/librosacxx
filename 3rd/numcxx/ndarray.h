#ifndef ROSACXX_UTIL_ARRAY_H
#define ROSACXX_UTIL_ARRAY_H

#include "alignmalloc.h"

namespace nc {

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

    size_t elemCount() const {
        size_t elemCnt = 1;
        for (const int& s : _shape) {
           elemCnt *= s;
        }
        return elemCnt;
    }

    size_t bytesCount() const {
        return this->elemCount() * sizeof (DType);
    }

    inline std::shared_ptr<NDArray> clone() const {
        auto sptr_clone = std::make_shared<NDArray>(this->_shape);
        memcpy(sptr_clone->_data, this->_data, this->bytesCount());
        return sptr_clone;
    }

    std::vector<int> shape() const {
        return _shape;
    }

    DType * data() const {
        return  _data;
    }

    DType loc(const std::vector<int>& location) const {
        std::vector<size_t> stride(_shape.size());
        size_t s = 1;
        for (int k = int(stride.size() - 1); k >= 0; k--) {
            stride[k] = s;
            s *= _shape[k];
        }
        size_t flattenIndex = 0;
        for (auto k = 0; k < _shape.size(); k++) {
            flattenIndex += location[k] * stride[k];
        }
        return _data[flattenIndex];
    }

    std::shared_ptr<NDArray> dot(const std::shared_ptr<NDArray>& other) const {
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

public: // static methods ...

    static std::shared_ptr<NDArray> FromVec1D(const std::vector<DType>& __vec) {
        auto p = std::shared_ptr<NDArray<DType>>(new NDArray<DType>({int(__vec.size())}));
        memcpy(p->_data, __vec.data(), __vec.size() * sizeof (DType));
        return p;
    }

private:
    std::vector<int> _shape;
    DType *          _data;
};

typedef NDArray<float> NDArrayF32;
typedef NDArray<int>   NDArrayS32;

typedef std::shared_ptr<NDArrayF32> NDArrayF32Ptr;
typedef std::shared_ptr<NDArrayS32> NDArrayS32Ptr;

} // namespace nc

#endif
