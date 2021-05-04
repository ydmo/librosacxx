#ifndef NUMCXX_NDARRAY_H
#define NUMCXX_NDARRAY_H

#include <rosacxx/numcxx/alignmalloc.h>
#include <rosacxx/numcxx/utils.h>
#include <rosacxx/half/half.h>

namespace nc {

template<typename DType>
class NDArrayPtr;

template<typename DType>
class NDArray {
    friend NDArrayPtr<DType>;

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

    NDArray(const std::vector<int>& __shape) : _shape(__shape) {
        int elemCnt = 1;
        for (const int& s : __shape) {
           elemCnt *= s;
        }
        _data = (DType *)alignedCalloc(32, elemCnt * sizeof (DType));
        _strides.resize(__shape.size());
        int tmp = 1;
        for (int i = _strides.size() - 1; i >= 0; i--) {
            _strides[i] = tmp;
            tmp *= __shape[i];
        }
    }


    NDArray(const std::vector<int>& __shape, const DType& __val) : _shape(__shape) {
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

    NDArray(const std::vector<int>& __shape, const DType * __dat) : _shape(__shape) {
        int elemCnt = 1;
        for (const int& s : __shape) {
           elemCnt *= s;
        }
        if (__dat == NULL) {
            _data = (DType *)alignedCalloc(32, elemCnt * sizeof (DType));
        }
        else {
            _data = (DType *)alignedMalloc(32, elemCnt * sizeof (DType));
            for (int n = 0; n < elemCnt; n++) {
                _data[n] = __dat[n];
            }
        }
        _strides.resize(__shape.size());
        int tmp = 1;
        for (int i = _strides.size() - 1; i >= 0; i--) {
            _strides[i] = tmp;
            tmp *= __shape[i];
        }
    }

private:
    std::vector<int> _strides;
    std::vector<int> _shape;
    DType *          _data;
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

template<typename DType>
class NDArrayPtr : public std::shared_ptr<NDArray<DType>> {
public:
    using elem_type = NDArray<DType>;
    using std::shared_ptr<elem_type>::shared_ptr;
    using std::shared_ptr<elem_type>::operator*;
    using std::shared_ptr<elem_type>::get;

public: // static methods ......

    static NDArrayPtr FromScalar(const DType& __scalar) {
        return NDArrayPtr(new NDArray<DType>({1}, __scalar));
    }

    static NDArrayPtr FromVec1D(const std::vector<DType>& __vec) {
        if (__vec.size() == 0) {
            throw std::runtime_error("Invaild shape at dimension 0");
        }
        auto p = NDArrayPtr(new NDArray<DType>({int(__vec.size())}));
        memcpy(p->_data, __vec.data(), __vec.size() * sizeof (DType));
        return p;
    }

    static NDArrayPtr FromVec2D(const std::vector<std::vector<DType>>& __vec2d) {
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
        auto p = NDArrayPtr(new NDArray<DType>({int(__vec2d.size()), int(__vec2d[0].size())}));
        auto ptr_p = p->_data;
        for (auto i = 0; i < p->_shape[0]; i++) {
            for (auto j = 0; j < p->_shape[1]; j++) {
                ptr_p[i * p->_shape[1] + j] = __vec2d[i][j];
            }
        }
        return p;
    }

    static NDArrayPtr FromVec3D(const std::vector<std::vector<std::vector<DType>>>& __vec3d) {
        auto p = NDArrayPtr(new NDArray<DType>({int(__vec3d.size()), int(__vec3d[0].size()), int(__vec3d[0][0].size()), }));
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

public: // dynamic methods .....

    inline NDArrayPtr clone() const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto sptr_clone = NDArrayPtr(new NDArray<DType>(_shape));
        memcpy(sptr_clone->_data, _data, bytesCount());
        return sptr_clone;
    }

    int elemCount() const {
        int elemCnt = 1;
        for (const int& s : get()->_shape) {
           elemCnt *= s;
        }
        return elemCnt;
    }

    int bytesCount() const {
        return elemCount() * sizeof (DType);
    }

    int dims() const {
        return int(get()->_shape.size());
    }

    std::vector<int> shape() const {
        return get()->_shape;
    }

    std::vector<int> strides() const {
        return get()->_strides;
    }

    DType * data() const {
        return get()->_data;
    }

    DType scalar() const {
        return *(get()->_data);
    }

    DType * at(const std::vector<int>& __coor) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() != __coor.size())
            throw std::runtime_error("Location error: _shape.size() != __coor.size()");
        size_t location = 0;
        for (auto i = 0; i < __coor.size(); i++) {
            location += __coor[i] * _strides[i];
        }
        return _data + location;
    }

    DType * at(const int& s) const {
        auto _data = get()->_data;
        return _data + (s);
    }

    DType * at(const int& s0, const int& s1) const {
        DType * data = get()->_data;
        if (get()->_shape.size() !=  2) throw std::runtime_error("Location error: _shape.size() != 2");
        return data + (s0 * get()->_strides[0] + s1);
    }

    DType * at(const int& s0, const int& s1, const int& s2) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  3) throw std::runtime_error("Location error: _shape.size() != 3");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  4) throw std::runtime_error("Location error: _shape.size() != 4");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  5) throw std::runtime_error("Location error: _shape.size() != 5");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  6) throw std::runtime_error("Location error: _shape.size() != 6");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  7) throw std::runtime_error("Location error: _shape.size() != 7");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  8) throw std::runtime_error("Location error: _shape.size() != 8");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  9) throw std::runtime_error("Location error: _shape.size() != 9");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8);
    }

    DType * at(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8, const int& s9) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() != 10) throw std::runtime_error("Location error: _shape.size() != 10");
        return _data + (s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8 * _strides[8] + s9);
    }

    DType getitem(const std::vector<int>& __loc) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() != __loc.size()) throw std::runtime_error("Location error: _shape.size() != s.size()");
        size_t location = 0;
        for (auto i = 0; i < __loc.size(); i++) {
            location += __loc[i] * _strides[i];
        }
        return _data[location];
    }

    DType getitem(const int& s0) const {
        return get()->_data[s0];
    }

    DType getitem(const int& s0, const int& s1) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  2) throw std::runtime_error("Location error: _shape.size() != 2");
        return _data[s0 * _strides[0] + s1];
    }

    DType getitem(const int& s0, const int& s1, const int& s2) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  3) throw std::runtime_error("Location error: _shape.size() != 3");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2];
    }

    DType getitem(const int& s0, const int& s1, const int& s2, const int& s3) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  4) throw std::runtime_error("Location error: _shape.size() != 4");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3];
    }

    DType getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  5) throw std::runtime_error("Location error: _shape.size() != 5");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4];
    }

    DType getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  6) throw std::runtime_error("Location error: _shape.size() != 6");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5];
    }

    DType getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  7) throw std::runtime_error("Location error: _shape.size() != 7");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6];
    }

    DType getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  8) throw std::runtime_error("Location error: _shape.size() != 8");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7];
    }

    DType getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() !=  9) throw std::runtime_error("Location error: _shape.size() != 9");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8];
    }

    DType getitem(const int& s0, const int& s1, const int& s2, const int& s3, const int& s4, const int& s5, const int& s6, const int& s7, const int& s8, const int& s9) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape.size() != 10) throw std::runtime_error("Location error: _shape.size() != 10");
        return _data[s0 * _strides[0] + s1 * _strides[1] + s2 * _strides[2] + s3 * _strides[3] + s4 * _strides[4] + s5 * _strides[5] + s6 * _strides[6] + s7 * _strides[7] + s8 * _strides[8] + s9];
    }

    NDArrayPtr getitems(const NDArrayPtr<bool>& __mask) const {
        if (__mask.shape().size() > get()->_shape.size()) throw std::runtime_error("Invaild input mask");
        std::vector<DType> vec_res(0);
        if (__mask.shape().size() == get()->_shape.size()) {
            for (auto i = 0; i < __mask.elemCount(); i++) {
                if (__mask.at(i)) vec_res.push_back(get()->_data[i]);
            }
            if (vec_res.size() == 0) return nullptr;
            auto p = NDArrayPtr(new NDArray<DType>({int(vec_res.size())}));
            memcpy(p->_data, vec_res.data(), vec_res.size() * sizeof (DType));
            return p;
        }
        else {
            throw std::runtime_error("Not implemented.");
        }
        return nullptr;
    }

    template<typename RType>
    void operator *= (const RType& rhs) {
        auto _data = get()->_data;
        for (auto i = 0; i < elemCount(); i++) {
            _data[i] *= rhs;
        }
    }

    template<typename RType>
    void operator /= (const RType& rhs) {
        auto _data = get()->_data;
        for (auto i = 0; i < elemCount(); i++) {
            _data[i] /= rhs;
        }
    }

    template<typename RType>
    NDArrayPtr<DType> operator + (const RType& rhs) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        auto ret = NDArrayPtr<DType>(new NDArray<DType>(_shape));
        auto ptr_ret = ret.data();
        for (auto i = 0; i < elemCount(); i++) {
            ptr_ret[i] = _data[i] + rhs;
        }
        return ret;
    }

    template<typename RType>
    NDArrayPtr<DType> operator * (const RType& rhs) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        auto ret = NDArrayPtr<DType>(new NDArray<DType>(_shape));
        auto ptr_ret = ret.data();
        for (auto i = 0; i < elemCount(); i++) {
            ptr_ret[i] = _data[i] * rhs;
        }
        return ret;
    }

    NDArrayPtr<DType> operator * (const NDArrayPtr<DType>& rhs) const {
        DType * _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape == rhs.shape()) { // is elementwise operation
            auto ret = NDArrayPtr<DType>(new NDArray<DType>(_shape));
            DType * ptr_ret = ret.data();
            DType * ptr_rhs = rhs.data();
            for (auto i = 0; i < elemCount(); i++) {
                ptr_ret[i] = _data[i] * ptr_rhs[i];
            }
            return ret;
        }
        else {
            throw std::runtime_error("Not implemented.");
            return nullptr;
        }
    }

    template<typename RType>
    NDArrayPtr<DType> operator * (const NDArrayPtr<RType>& rhs) const {
        DType * _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        // -------
        if (_shape == rhs.shape()) { // is elementwise operation
            auto ret = NDArrayPtr<DType>(new NDArray<DType>(_shape));
            DType * ptr_ret = ret.data();
            RType * ptr_rhs = rhs.data();
            for (auto i = 0; i < elemCount(); i++) {
                ptr_ret[i] = _data[i] * ptr_rhs[i];
            }
            return ret;
        }
        else {
            throw std::runtime_error("Not implemented.");

            std::vector<int> dims0 = _shape;
            std::vector<int> dims1 = rhs.shape();
            std::vector<int> dims2 = _get_broadcast_op_shape(_shape, dims1);

            NDArrayPtr<DType> ret = NDArrayPtr<DType>(new NDArray<DType>(dims2));

            int msize = dims2.size();

            // unsqueeze dims0 ...
            std::vector<int> newDims0(0);
            for (auto i = 0; i < msize - dims0.size(); i++) {
                newDims0.push_back(int(1));
            }
            for (auto i = 0; i < dims0.size(); i++) {
                newDims0.push_back(dims0[i]);
            }
            dims0 = newDims0;

            // unsqueeze dims1 ...
            std::vector<int> newDims1(0);
            for (auto i = 0; i < msize - dims1.size(); i++) {
                newDims1.push_back(int(1));
            }
            for (auto i = 0; i < dims1.size(); i++) {
                newDims1.push_back(dims1[i]);
            }
            dims1 = newDims1;

            std::vector<int> strides0(msize);
            std::vector<int> strides1(msize);
            std::vector<int> strides2(msize);

            for (int i = 0; i < msize; i++) {
                strides0[i] = 1;
                strides1[i] = 1;
                strides2[i] = 1;
                for (int j = i+1; j < msize; j++) {
                    strides0[i] *= dims0[j];
                    strides1[i] *= dims1[j];
                    strides2[i] *= dims2[j];
                }
            }

            DType * in0_ptr = _data;
            RType * in1_ptr = rhs.data();
            DType * out_ptr = ret.data();

            std::vector<int> coor0(msize);
            std::vector<int> coor1(msize);
            std::vector<int> coor2(msize);
            int remainder = 0, loc0 = 0, loc1 = 0;
            for (int n = 0; n < ret.elemCount(); ++n) {
                // cal coor2 --> cal coor0 and coor1 --> cal the loc0 and loc1
                remainder = n; loc0 = 0; loc1 = 0;
                for(auto d = 0; d < msize; d++) {
                    coor2[d] = remainder / strides2[d];
                    remainder -= (coor2[d] * strides2[d]);
                    coor0[d] = dims2[d] == dims0[d]? coor2[d] : 0;
                    coor1[d] = dims2[d] == dims1[d]? coor2[d] : 0;
                    loc0 += coor0[d] * strides0[d];
                    loc1 += coor1[d] * strides1[d];
                }
                // operator ...
                out_ptr[n] = in0_ptr[loc0] * in1_ptr[loc1];
            }

            return ret;
        }
    }

    template<typename RType>
    NDArrayPtr<bool> operator > (const NDArrayPtr<RType>& rhs) const {
        DType * _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        // -------
        if (_shape == rhs.shape()) { // is elementwise operation
            auto ret = NDArrayPtr<bool>(new NDArray<bool>(_shape));
            bool * ptr_ret = ret.data();
            DType * ptr_rhs = rhs.data();
            for (auto i = 0; i < elemCount(); i++) {
                ptr_ret[i] = _data[i] > ptr_rhs[i];
            }
            return ret;
        }
        else {
            std::vector<int> dims0 = _shape;
            std::vector<int> dims1 = rhs.shape();
            std::vector<int> dims2 = _get_broadcast_op_shape(_shape, dims1);

            NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArray<bool>(dims2));

            int msize = dims2.size();

            // unsqueeze dims0 ...
            std::vector<int> newDims0(0);
            for (auto i = 0; i < msize - dims0.size(); i++) {
                newDims0.push_back(int(1));
            }
            for (auto i = 0; i < dims0.size(); i++) {
                newDims0.push_back(dims0[i]);
            }
            dims0 = newDims0;

            // unsqueeze dims1 ...
            std::vector<int> newDims1(0);
            for (auto i = 0; i < msize - dims1.size(); i++) {
                newDims1.push_back(int(1));
            }
            for (auto i = 0; i < dims1.size(); i++) {
                newDims1.push_back(dims1[i]);
            }
            dims1 = newDims1;

            std::vector<int> strides0(msize);
            std::vector<int> strides1(msize);
            std::vector<int> strides2(msize);

            for (int i = 0; i < msize; i++) {
                strides0[i] = 1;
                strides1[i] = 1;
                strides2[i] = 1;
                for (int j = i+1; j < msize; j++) {
                    strides0[i] *= dims0[j];
                    strides1[i] *= dims1[j];
                    strides2[i] *= dims2[j];
                }
            }

            DType * in0_ptr = _data;
            RType * in1_ptr = rhs.data();
            bool * out_ptr = ret.data();

            std::vector<int> coor0(msize);
            std::vector<int> coor1(msize);
            std::vector<int> coor2(msize);
            int remainder = 0, loc0 = 0, loc1 = 0;
            for (int n = 0; n < ret.elemCount(); ++n) {
                // cal coor2 --> cal coor0 and coor1 --> cal the loc0 and loc1
                remainder = n; loc0 = 0; loc1 = 0;
                for(auto d = 0; d < msize; d++) {
                    coor2[d] = remainder / strides2[d];
                    remainder -= (coor2[d] * strides2[d]);
                    coor0[d] = dims2[d] == dims0[d]? coor2[d] : 0;
                    coor1[d] = dims2[d] == dims1[d]? coor2[d] : 0;
                    loc0 += coor0[d] * strides0[d];
                    loc1 += coor1[d] * strides1[d];
                }
                // operator ...
                out_ptr[n] = in0_ptr[loc0] > in1_ptr[loc1];
            }

            return ret;
        }
    }

    template<typename RType>
    NDArrayPtr<bool> operator < (const NDArrayPtr<RType>& rhs) const {
        DType * _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        // -------
        if (_shape == rhs.shape()) { // is elementwise operation
            auto ret = NDArrayPtr<bool>(new NDArray<bool>(_shape));
            bool * ptr_ret = ret.data();
            DType * ptr_rhs = rhs.data();
            for (auto i = 0; i < elemCount(); i++) {
                ptr_ret[i] = _data[i] < ptr_rhs[i];
            }
            return ret;
        }
        else {
            std::vector<int> dims0 = _shape;
            std::vector<int> dims1 = rhs.shape();
            std::vector<int> dims2 = _get_broadcast_op_shape(_shape, dims1);

            NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArray<bool>(dims2));

            int msize = dims2.size();

            // unsqueeze dims0 ...
            std::vector<int> newDims0(0);
            for (auto i = 0; i < msize - dims0.size(); i++) {
                newDims0.push_back(int(1));
            }
            for (auto i = 0; i < dims0.size(); i++) {
                newDims0.push_back(dims0[i]);
            }
            dims0 = newDims0;

            // unsqueeze dims1 ...
            std::vector<int> newDims1(0);
            for (auto i = 0; i < msize - dims1.size(); i++) {
                newDims1.push_back(int(1));
            }
            for (auto i = 0; i < dims1.size(); i++) {
                newDims1.push_back(dims1[i]);
            }
            dims1 = newDims1;

            std::vector<int> strides0(msize);
            std::vector<int> strides1(msize);
            std::vector<int> strides2(msize);

            for (int i = 0; i < msize; i++) {
                strides0[i] = 1;
                strides1[i] = 1;
                strides2[i] = 1;
                for (int j = i+1; j < msize; j++) {
                    strides0[i] *= dims0[j];
                    strides1[i] *= dims1[j];
                    strides2[i] *= dims2[j];
                }
            }

            DType * in0_ptr = _data;
            RType * in1_ptr = rhs.data();
            bool * out_ptr = ret.data();

            std::vector<int> coor0(msize);
            std::vector<int> coor1(msize);
            std::vector<int> coor2(msize);
            int remainder = 0, loc0 = 0, loc1 = 0;
            for (int n = 0; n < ret.elemCount(); ++n) {
                // cal coor2 --> cal coor0 and coor1 --> cal the loc0 and loc1
                remainder = n; loc0 = 0; loc1 = 0;
                for(auto d = 0; d < msize; d++) {
                    coor2[d] = remainder / strides2[d];
                    remainder -= (coor2[d] * strides2[d]);
                    coor0[d] = dims2[d] == dims0[d]? coor2[d] : 0;
                    coor1[d] = dims2[d] == dims1[d]? coor2[d] : 0;
                    loc0 += coor0[d] * strides0[d];
                    loc1 += coor1[d] * strides1[d];
                }
                // operator ...
                out_ptr[n] = in0_ptr[loc0] < in1_ptr[loc1];
            }

            return ret;
        }
    }

    NDArrayPtr<DType> operator / (const NDArrayPtr<DType>& rhs) const {
        DType * _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape == rhs.shape()) { // is elementwise operation
            auto ret = NDArrayPtr<DType>(new NDArray<DType>(_shape));
            DType * ptr_ret = ret.data();
            DType * ptr_rhs = rhs.data();
            for (auto i = 0; i < elemCount(); i++) {
                ptr_ret[i] = _data[i] / ptr_rhs[i];
            }
            return ret;
        }
        else {
            throw std::runtime_error("Not implemented.");
            return nullptr;
        }
    }

    NDArrayPtr<DType> operator + (const NDArrayPtr<bool>& rhs) const {
        DType * _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape == rhs.shape()) { // is elementwise operation
            auto ret = NDArrayPtr<DType>(new NDArray<DType>(_shape));
            DType * ptr_ret = ret.data();
            bool * ptr_rhs = rhs.data();
            for (auto i = 0; i < elemCount(); i++) {
                ptr_ret[i] = _data[i] + (ptr_rhs[i] ? DType(1) : DType(0));
            }
            return ret;
        }
        else {
            throw std::runtime_error("Not implemented.");
            return nullptr;
        }
    }

    template<typename RType>
    NDArrayPtr<bool> operator < (const RType& rhs) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto ret = NDArrayPtr<bool>(new NDArrayBool(_shape));
        bool * ptr_ret = ret.data();
        for (auto i = 0; i < ret.elemCount(); i++) {
            if (_data[i] < rhs) ptr_ret[i] = true;
        }
        return ret;
    }

    template<typename RType>
    NDArrayPtr<bool> operator <= (const RType& rhs) {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArrayBool(_shape));
        bool * ptr_ret = ret.data();
        for (auto i = 0; i < ret.elemCount(); i++) {
            if (_data[i] <= rhs) ptr_ret[i] = true;
        }
        return ret;
    }

    template<typename RType>
    NDArrayPtr<bool> operator > (const RType& rhs) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto ret = NDArrayPtr<bool>(new NDArrayBool(_shape));
        bool * ptr_ret = ret.data();
        for (auto i = 0; i < ret.elemCount(); i++) {
            if (_data[i] > rhs) ptr_ret[i] = true;
        }
        return ret;
    }

    template<typename RType>
    NDArrayPtr<bool> operator >= (const RType& rhs) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArrayBool(_shape));
        bool * ptr_ret = ret.data();
        for (auto i = 0; i < ret.elemCount(); i++) {
            if (_data[i] >= rhs) ptr_ret[i] = true;
        }
        return ret;
    }

    NDArrayPtr operator & (const NDArrayPtr& __rhs) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape != __rhs.shape()) throw std::runtime_error("Invaild input params");
        NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArrayBool(_shape));
        DType * ptr_ret = ret.data();
        DType * ptr_lhs = _data;
        DType * ptr_rhs = __rhs.data();
        for (auto i = 0; i < ret.elemCount(); i++) {
            ptr_ret[i] = (ptr_lhs[i] & ptr_rhs[i]);
        }
        return ret;
    }

    NDArrayPtr<bool> operator && (const NDArrayPtr<bool>& __rhs) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        if (_shape != __rhs.shape()) throw std::runtime_error("Invaild input params");
        NDArrayPtr<bool> ret = NDArrayPtr<bool>(new NDArrayBool(_shape));
        bool * ptr_ret = ret.data();
        bool * ptr_lhs = _data;
        bool * ptr_rhs = __rhs.data();
        for (auto i = 0; i < ret.elemCount(); i++) {
            ptr_ret[i] = (ptr_lhs[i] && ptr_rhs[i]);
        }
        return ret;
    }

    NDArrayPtr operator [] (const NDArrayPtr<bool>& __mask) const {
        return getitems(__mask);
    }

    DType min() const {
        auto _data = get()->_data;
        DType min_val = _data[0];
        for (int i = 0; i < elemCount(); i++) {
            if (min_val > _data[i]) min_val = _data[i];
        }
        return min_val;
    }

    int argmin() const {
        auto _data = get()->_data;
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
        auto _data = get()->_data;
        DType max_val = _data[0];
        for (int i = 0; i < elemCount(); i++) {
            if (max_val < _data[i]) max_val = _data[i];
        }
        return max_val;
    }

    int argmax() const {
        auto _data = get()->_data;
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

    NDArrayPtr dot(const NDArrayPtr& __other) const {
        if (get()->_shape.size() == 1 && __other.shape().size() == 1) {
            if (get()->_shape[0] != get()->_shape[0]) {
                throw std::runtime_error("invalid shape");
            }
            DType sum = 0;
            for (auto i = 0; i < get()->_shape[0]; i++) {
                sum += (get()->_data[i] * __other->_data[i]);
            }
            auto ptr = NDArrayPtr(new NDArray<DType>({1}, sum));
            return ptr;
        }
        else if (get()->_shape.size() == 2 && __other.shape().size() == 2) {

        }
        else {
            throw std::runtime_error("Not implemented error");
        }
    }

    NDArrayPtr<int> argmax(int __axis) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;

        std::vector<int> shape_am;
        for (int i = 0; i < _shape.size(); i++) {
            if (i == __axis) continue;
            shape_am.push_back(_shape[i]);
        }

        auto arr_max_val = NDArrayPtr<DType>(new NDArray<DType>(shape_am, min()));
        auto arr_max_idx = NDArrayPtr<int  >(new NDArray<int  >(shape_am));

        for (auto i = 0; i < elemCount(); i++) {

//            std::vector<int> coor(_shape.size());
//            int remainder = i;
//            for(int d = 0; d < _shape.size(); d++) {
//                coor[d] = remainder / _strides[d];
//                remainder -= (coor[d] * _strides[d]);
//                if (d == __axis) {
//                    continue;
//                }
//            }
            std::vector<int> coor = _get_coor_s32(i, _strides);

            DType curr_val = _data[i];
            int   curr_idx = coor[__axis];

            std::vector<int> am_coor = coor; am_coor.erase(am_coor.begin()+__axis);

            if (arr_max_val.getitem(am_coor) < curr_val) {
                arr_max_val.at(am_coor)[0] = curr_val;
                arr_max_idx.at(am_coor)[0] = curr_idx;
            }
        }

        return arr_max_idx;
    }

    NDArrayPtr max(int __axis) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;

        std::vector<int> shape_am;
        for (int i = 0; i < _shape.size(); i++) {
            if (i == __axis) continue;
            shape_am.push_back(_shape[i]);
        }

        auto arr_max_val = NDArrayPtr<DType>(new NDArray<DType>(shape_am, min()));
        auto arr_max_idx = NDArrayPtr<int  >(new NDArray<int  >(shape_am));

        for (auto i = 0; i < elemCount(); i++) {

//            std::vector<int> coor(_shape.size());
//            int remainder = i;
//            for(int d = 0; d < _shape.size(); d++) {
//                coor[d] = remainder / _strides[d];
//                remainder -= (coor[d] * _strides[d]);
//            }
            std::vector<int> coor = _get_coor_s32(i, _strides);

            DType curr_val = _data[i];
            int   curr_idx = coor[__axis];

            std::vector<int> am_coor = coor; am_coor.erase(am_coor.begin()+__axis);

            if (arr_max_val.getitem(am_coor) < curr_val) {
                arr_max_val.at(am_coor)[0] = curr_val;
                arr_max_idx.at(am_coor)[0] = curr_idx;
            }
        }

        return arr_max_val;
    }

    NDArrayPtr<int> argmin(int __axis) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;

        std::vector<int> shape_am;
        for (int i = 0; i < _shape.size(); i++) {
            if (i == __axis) continue;
            shape_am.push_back(_shape[i]);
        }

        auto arr_min_val = NDArrayPtr<DType>(new NDArray<DType>(shape_am, max()));
        auto arr_min_idx = NDArrayPtr<int  >(new NDArray<int  >(shape_am));

        for (auto i = 0; i < elemCount(); i++) {

//            std::vector<int> coor(_shape.size());
//            int remainder = i;
//            for(int d = 0; d < _shape.size(); d++) {
//                coor[d] = remainder / _strides[d];
//                remainder -= (coor[d] * _strides[d]);
//            }
            std::vector<int> coor = _get_coor_s32(i, _strides);

            DType curr_val = _data[i];
            int   curr_idx = coor[__axis];

            std::vector<int> am_coor = coor; am_coor.erase(am_coor.begin()+__axis);

            if (arr_min_val.getitem(am_coor) > curr_val) {
                arr_min_val.at(am_coor)[0] = curr_val;
                arr_min_idx.at(am_coor)[0] = curr_idx;
            }
        }

        return arr_min_idx;
    }

    NDArrayPtr min(int __axis) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;

        std::vector<int> shape_am;
        for (int i = 0; i < _shape.size(); i++) {
            if (i == __axis) continue;
            shape_am.push_back(_shape[i]);
        }

        auto arr_min_val = NDArrayPtr<DType>(new NDArray<DType>(shape_am, max()));
        auto arr_min_idx = NDArrayPtr<int  >(new NDArray<int  >(shape_am));

        for (auto i = 0; i < elemCount(); i++) {

//            std::vector<int> coor(_shape.size());
//            int remainder = i;
//            for(int d = 0; d < _shape.size(); d++) {
//                coor[d] = remainder / _strides[d];
//                remainder -= (coor[d] * _strides[d]);
//                if (d == __axis) {
//                    continue;
//                }
//            }
            std::vector<int> coor = _get_coor_s32(i, _strides);

            DType curr_val = _data[i];
            int   curr_idx = coor[__axis];

            std::vector<int> am_coor = coor; am_coor.erase(am_coor.begin()+__axis);

            if (arr_min_val.getitem(am_coor) > curr_val) {
                arr_min_val.at(am_coor)[0] = curr_val;
                arr_min_idx.at(am_coor)[0] = curr_idx;
            }
        }

        return arr_min_val;
    }

    NDArrayPtr T() const {
        auto _shape = get()->_shape;
        if (_shape.size() != 2) throw std::runtime_error("Only 2D array (Matrix2D) can use T() method.");
        auto _data = get()->_data;
        auto _strides = get()->_strides;
        auto ret = NDArrayPtr<DType>(new NDArray<DType>({_shape[1], _shape[0]}));
        auto ptr_ret = ret.data();
        for (auto i = 0; i < _shape[0]; i++) {
            for (auto j = 0; j < _shape[1]; j++) {
                ptr_ret[j * _shape[0] + i] = _data[i * _shape[1] + j];
            }
        }
        return ret;
    }

    NDArrayPtr reshape(const std::vector<int>& __newshape) const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        // ------------------------------------------------------------------
        std::vector<int> new_shape(__newshape.size());
        int k = 1;
        int neg_dim = 0;
        int neg_cnt = 0;
        for (auto i = 0; i < __newshape.size(); i++) {
            if (__newshape[i] > 0) {
                k *= __newshape[i];
                new_shape[i] = __newshape[i];
            } else {
                neg_dim = i;
                neg_cnt += 1;
            }
        }
        const int elemCnt = elemCount();
        if (elemCnt < k || (elemCnt % k) != 0 || neg_cnt > 1) {
            throw std::invalid_argument("Invalid input new shape.");
        }
        if(neg_cnt > 0) {
            new_shape[neg_dim] = elemCnt / k;
        }
        // ------------------------------------------------------------------
        return NDArrayPtr<DType>(new NDArray<DType>(new_shape, _data));;
    }

    template<typename RType>
    NDArrayPtr<RType> astype() const {
        auto _data = get()->_data;
        auto _shape = get()->_shape;
        auto _strides = get()->_strides;
        auto ret = NDArrayPtr<RType>(new NDArray<RType>(_shape));
        auto ptr_ret = ret.data();
        for (auto i = 0; i < elemCount(); i++) {
            ptr_ret[i] = _data[i];
        }
        return ret;
    }
};

#ifdef HALF_HALF_HPP
typedef NDArrayPtr<half_float::half>   NDArrayF16Ptr;
#endif // HALF_HALF_HPP
typedef NDArrayPtr<float>              NDArrayF32Ptr;
typedef NDArrayPtr<double>             NDArrayF64Ptr;
typedef NDArrayPtr<char>               NDArrayS8Ptr;
typedef NDArrayPtr<short>              NDArrayS16Ptr;
typedef NDArrayPtr<int>                NDArrayS32Ptr;
typedef NDArrayPtr<unsigned char>      NDArrayU8Ptr;
typedef NDArrayPtr<unsigned short>     NDArrayU16Ptr;
typedef NDArrayPtr<unsigned int>       NDArrayU32Ptr;
typedef NDArrayPtr<bool>               NDArrayBoolPtr;

} // namespace nc

#endif
