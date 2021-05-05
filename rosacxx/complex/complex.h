#ifndef ROSACXX_COMPLEX_H
#define ROSACXX_COMPLEX_H

#include <iostream>
#include <complex>

namespace cpx {

template <typename DType>
struct complex {

    complex(const DType& __ri) : _r(__ri), _i(__ri) { }
    complex(const DType& __r, const DType& __i) : _r(__r), _i(__i) { }
    complex(const complex& __other) : complex(__other._r, __other._i) { }

    DType imag() const { return _i; }
    DType real() const { return _r; }

    void imag(const DType& __i) {
        _i = __i;
    }

    void real(const DType& __r) {
        _r = __r;
    }

    bool operator == (const DType& __other) const {
        return (this->_r == __other) && (this->_i == __other);
    }

    bool operator == (const complex<DType>& __other) const {
        return (this->_r == __other._r) && (this->_i == __other._i);
    }

    DType _r;
    DType _i;
};

template<typename DType>
inline std::ostream &operator << (std::ostream &__os, const complex<DType>& __cpx) {
    __os << int((__cpx._r) * 1e6) * 1e-6 << " + " << int((__cpx._i) * 1e6) * 1e-6 << "j";
    return __os;
}

} // namespace cpx

#endif // ROSACXX_COMPLEX_H
