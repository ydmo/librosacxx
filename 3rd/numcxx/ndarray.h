#ifndef ROSACXX_UTIL_ARRAY_H
#define ROSACXX_UTIL_ARRAY_H

#include "alignmalloc.h"
#include <half.h>

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

    NDArray(const std::vector<int>& __shape, const DType& __val = 0) : _shape(__shape) {
        int elemCnt = 1;
        for (const int& s : __shape) {
           elemCnt *= s;
        }
        if (__val == 0) {
            _data = (DType *)alignedCalloc(32, elemCnt * sizeof (DType));
        }
        else {
            _data = (DType *)alignedMalloc(32, elemCnt * sizeof (DType));
            for (int n = 0; n < elemCnt; n++) {
                _data[n] = __val;
            }
        }
        _strides.resize(__shape.size());
        int tmp = 1;
        for (int i = _strides.size() - 1; i >= 0; i--) {
            _strides[i] = tmp;
            tmp *= __shape[i];
        }
    }

    int elemCount() const {
        int elemCnt = 1;
        for (const int& s : _shape) {
           elemCnt *= s;
        }
        return elemCnt;
    }

    int bytesCount() const {
        return this->elemCount() * sizeof (DType);
    }

    inline std::shared_ptr<NDArray> clone() const {
        auto sptr_clone = std::make_shared<NDArray>(this->_shape);
        memcpy(sptr_clone->_data, this->_data, this->bytesCount());
        return sptr_clone;
    }

    int dims() const {
        return int(_shape.size());
    }

    std::vector<int> shape() const {
        return _shape;
    }

    std::vector<int> strides() const {
        return _strides;
    }

    DType * data() const {
        return _data;
    }

    DType scalar() const {
        return *_data;
    }

    DType * at(const std::vector<int>& __coor) const {
        if (_shape.size() >= __coor.size()) throw std::runtime_error("Location error: _shape.size() != s.size()");
        size_t location = 0;
        for (auto i = 0; i < __coor.size(); i++) {
            location += __coor[i] * _strides[i];
        }
        return _data + location;
    }

    DType * at(const int& s) const {
        return _data + (s);
    }

    DType * at(const int& s0, const int& s1) const {
        if (_shape.size() !=  2) throw std::runtime_error("Location error: _shape.size() != 2");
        return _data + (s0 * _strides[0] + s1);
    }

    DType * at(const int& s0, const int& s1, const int& s2) const {
        if (_shape.size() !=  3) throw std::runtime_error("Location error: _shape.size() != 3");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3) const {
        if (_shape.size() !=  4) throw std::runtime_error("Location error: _shape.size() != 4");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4) const {
        if (_shape.size() !=  5) throw std::runtime_error("Location error: _shape.size() != 5");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5) const {
        if (_shape.size() !=  6) throw std::runtime_error("Location error: _shape.size() != 6");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6) const {
        if (_shape.size() !=  7) throw std::runtime_error("Location error: _shape.size() != 7");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7) const {
        if (_shape.size() !=  8) throw std::runtime_error("Location error: _shape.size() != 8");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8) const {
        if (_shape.size() !=  9) throw std::runtime_error("Location error: _shape.size() != 9");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8, const int& s9) const {
        if (_shape.size() != 10) throw std::runtime_error("Location error: _shape.size() != 10");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8 * _strides[8] + s9);
    }

    DType& getitem(const std::vector<int>& __loc) const {
        if (_shape.size() != __loc.size()) throw std::runtime_error("Location error: _shape.size() != s.size()");
        size_t location = 0;
        for (auto i = 0; i < __loc.size(); i++) {
            location += __loc[i] * _strides[i];
        }
        return _data[location];
    }

    DType& getitem(const int& s0) const {
        return _data[s0];
    }

    DType& getitem(const int& s0, const int& s1) const {
        if (_shape.size() !=  2) throw std::runtime_error("Location error: _shape.size() != 2");
        return _data[s0 * _strides[0] + s1];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2) const {
        if (_shape.size() !=  3) throw std::runtime_error("Location error: _shape.size() != 3");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2, const int& s3) const {
        if (_shape.size() !=  4) throw std::runtime_error("Location error: _shape.size() != 4");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4) const {
        if (_shape.size() !=  5) throw std::runtime_error("Location error: _shape.size() != 5");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5) const {
        if (_shape.size() !=  6) throw std::runtime_error("Location error: _shape.size() != 6");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6) const {
        if (_shape.size() !=  7) throw std::runtime_error("Location error: _shape.size() != 7");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7) const {
        if (_shape.size() !=  8) throw std::runtime_error("Location error: _shape.size() != 8");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8) const {
        if (_shape.size() !=  9) throw std::runtime_error("Location error: _shape.size() != 9");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8];
    }

    DType& getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8, const int& s9) const {
        if (_shape.size() != 10) throw std::runtime_error("Location error: _shape.size() != 10");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8 * _strides[8] + s9];
    }

    std::shared_ptr<NDArray> getitems(const std::shared_ptr<NDArray<bool>>& __mask) const {
        if (__mask->shape().size() > _shape.size()) throw std::runtime_error("Invaild input mask");
        std::vector<DType> vec_res(0);
        if (__mask->shape().size() == _shape.size()) {
            for (auto i = 0; i < __mask->elemCount(); i++) {
                if (__mask->at(i)) vec_res.push_back(_data[i]);
            }
            if (vec_res.size() == 0) return nullptr;
            auto p = std::shared_ptr<NDArray<DType>>(new NDArray<DType>({int(vec_res.size())}));
            memcpy(p->_data, vec_res.data(), vec_res.size() * sizeof (DType));
            return p;
        }
        else {
            throw std::runtime_error("Not implemented.");
        }
        return nullptr;
    }

    std::shared_ptr<NDArray> dot(const std::shared_ptr<NDArray>& __other) const {
        if (this->_shape.size() == 1 && __other->_shape.size() == 1) {
            if (this->_shape[0] != this->_shape[0]) {
                throw std::runtime_error("invalid shape");
            }
            DType sum = 0;
            for (auto i = 0; i < this->_shape[0]; i++) {
                sum += (this->_data[i] * __other->_data[i]);
            }
            auto ptr = std::shared_ptr<NDArray>(new NDArray({1}, sum));
            return ptr;
        }
        else if (this->_shape.size() == 2 && __other->_shape.size() == 2) {

        }
        else {
            throw std::runtime_error("Not implemented error");
        }
    }

    DType min() const {
        DType min_val = _data[0];
        for (int i = 0; i < elemCount(); i++) {
            if (min_val > _data[i]) min_val = _data[i];
        }
        return min_val;
    }

    int argmin() const {
        int   min_idx = 0;
        DType min_val = _data[0];
        for (int i = 0; i < elemCount(); i++) {
            if (min_val > _data[i]) {
                min_val = _data[i];
                min_idx = i;
            }
        }
        return min_idx;
    }

    DType max() const {
        DType max_val = _data[0];
        for (int i = 0; i < elemCount(); i++) {
            if (max_val < _data[i]) max_val = _data[i];
        }
        return max_val;
    }

    int argmax() const {
        int   max_idx = 0;
        DType max_val = _data[0];
        for (int i = 0; i < elemCount(); i++) {
            if (max_val < _data[i]) {
                max_val = _data[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    std::shared_ptr<NDArray<int>> argmax(int __axis) const {
        std::vector<int> shape_am;
        for (int i = 0; i < _shape.size(); i++) {
            if (i == __axis) continue;
            shape_am.push_back(i);
        }

        std::shared_ptr<NDArray<DType>> arr_max_val = std::shared_ptr<NDArray<DType>>(new NDArray<DType>(shape_am, min()));
        std::shared_ptr<NDArray<int  >> arr_max_idx = std::shared_ptr<NDArray<int  >>(new NDArray<int  >(shape_am));

        for (auto i = 0; i < elemCount(); i++) {

            std::vector<int> coor(_shape.size());
            int remainder = i;
            for(int d = 0; d < _shape.size(); d++) {
                coor[d] = remainder / _strides[d];
                remainder -= (coor[d] * _strides[d]);
                if (d == __axis) {
                    continue;
                }
            }

            DType curr_val = _data[i];
            int   curr_idx = coor[__axis];

            std::vector<int> am_coor = coor; am_coor.erase(am_coor.begin()+__axis);

            if (arr_max_val->getitem(am_coor) < curr_val) {
                arr_max_val->getitem(am_coor) = curr_val;
                arr_max_idx->getitem(am_coor) = curr_idx;
            }
        }

        return arr_max_idx;
    }

    std::shared_ptr<NDArray<int>> argmin(int __axis) const {
        std::vector<int> shape_am;
        for (int i = 0; i < _shape.size(); i++) {
            if (i == __axis) continue;
            shape_am.push_back(i);
        }

        std::shared_ptr<NDArray<DType>> arr_min_val = std::shared_ptr<NDArray<DType>>(new NDArray<DType>(shape_am, max()));
        std::shared_ptr<NDArray<int  >> arr_min_idx = std::shared_ptr<NDArray<int  >>(new NDArray<int  >(shape_am));

        for (auto i = 0; i < elemCount(); i++) {

            std::vector<int> coor(_shape.size());
            int remainder = i;
            for(int d = 0; d < _shape.size(); d++) {
                coor[d] = remainder / _strides[d];
                remainder -= (coor[d] * _strides[d]);
                if (d == __axis) {
                    continue;
                }
            }

            DType curr_val = _data[i];
            int   curr_idx = coor[__axis];

            std::vector<int> am_coor = coor; am_coor.erase(am_coor.begin()+__axis);

            if (arr_min_val->getitem(am_coor) > curr_val) {
                arr_min_val->getitem(am_coor) = curr_val;
                arr_min_idx->getitem(am_coor) = curr_idx;
            }
        }

        return arr_min_idx;
    }

    void iadd(const DType& __other) const {
        for (auto i = 0; i < elemCount(); i++) {
            _data[i] += __other;
        }
    }

    std::shared_ptr<NDArray<DType>> add(const DType& __other) const {
        std::shared_ptr<NDArray<DType>> res = std::shared_ptr<NDArray<DType>>(new NDArray<DType>(_shape));
        DType * ptr_res = res->data();
        for (auto i = 0; i < elemCount(); i++) {
            ptr_res[i] = _data[i] + __other;
        }
        return res;
    }

    std::shared_ptr<NDArray<DType>> add(const std::shared_ptr<NDArray<DType>>& __other) const {
        if (__other->shape() == _shape) {
            std::shared_ptr<NDArray<DType>> res = std::shared_ptr<NDArray<DType>>(new NDArray<DType>(_shape));
            DType * ptr_res   = res->data();
            DType * ptr_other = __other->data();
            for (auto i = 0; i < elemCount(); i++) {
                ptr_res[i] = _data[i] + ptr_other[i];
            }
            return res;
        }
        throw std::runtime_error("Broadcast operator was not implemented.");
        return nullptr;
    }

public: // static methods ...

    static std::shared_ptr<NDArray> FromScalar(const DType& __scalar) {
        return std::shared_ptr<NDArray>(new NDArray({1}, __scalar));
    }

    static std::shared_ptr<NDArray> FromVec1D(const std::vector<DType>& __vec) {
        if (__vec.size() == 0) {
            throw std::runtime_error("Invaild shape at dimension 0");
        }
        auto p = std::shared_ptr<NDArray<DType>>(new NDArray<DType>({int(__vec.size())}));
        memcpy(p->_data, __vec.data(), __vec.size() * sizeof (DType));
        return p;
    }

    static std::shared_ptr<NDArray> FromVec2D(const std::vector<std::vector<DType>>& __vec2d) {
        if (__vec2d.size() == 0) {
            throw std::runtime_error("Invaild shape at dimension 0");
        }
        if (__vec2d[0].size() == 0) {
            throw std::runtime_error("Invaild shape at dimension 1");
        }
        for (auto i = 1; i < __vec2d.size(); i++) {
            if (__vec2d[0].size() != __vec2d[i].size()) {
                throw std::runtime_error("Invaild input shape.");
            }
        }
        auto p = std::shared_ptr<NDArray<DType>>(new NDArray<DType>({int(__vec2d.size()), int(__vec2d[0].size())}));
        auto ptr_p = p->_data;
        for (auto i = 0; i < p->_shape[0]; i++) {
            for (auto j = 0; j < p->_shape[1]; j++) {
                ptr_p[i * p->_shape[1] + j] = __vec2d[i][j];
            }
        }
        return p;
    }

    static std::shared_ptr<NDArray> FromVec3D(const std::vector<std::vector<std::vector<DType>>>& __vec3d) {
        auto p = std::shared_ptr<NDArray<DType>>(new NDArray<DType>({int(__vec3d.size()), int(__vec3d[0].size()), int(__vec3d[0][0].size()), }));
        auto ptr_p = p->_data;
        for (auto i = 0; i < p->_shape[0]; i++) {
            for (auto j = 0; j < p->_shape[1]; j++) {
                for (auto k = 0; k < p->_shape[2]; k++) {
                    ptr_p[i * p->_strides[0] + j * p->_strides[1] + k] = __vec3d[i][j][k];
                }
            }
        }
        return p;
    }

private:
    std::vector<int> _strides;
    std::vector<int> _shape;
    DType *          _data;

public: // typedef
    typedef std::shared_ptr<NDArray> Ptr;
    typedef std::vector<Ptr>         PtrVec;
};

#ifdef HALF_HALF_HPP
typedef NDArray<half_float::half>   NDArrayF16;
#endif // HALF_HALF_HPP

typedef NDArray<float>              NDArrayF32;
typedef NDArray<double>             NDArrayF64;
typedef NDArray<char>               NDArrayS8;
typedef NDArray<short>              NDArrayS16;
typedef NDArray<int>                NDArrayS32;
typedef NDArray<unsigned char>      NDArrayU8;
typedef NDArray<unsigned short>     NDArrayU16;
typedef NDArray<unsigned int>       NDArrayU32;
typedef NDArray<bool>               NDArrayBool;

} // namespace nc

#endif
