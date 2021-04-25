#ifndef ROSACXX_COMPLEX_H
#define ROSACXX_COMPLEX_H

#include <iostream>

namespace complex {

template <typename DType>
struct Complex {

    Complex(const DType& __ri) : r(__ri), i(__ri) { }
    Complex(const DType& __r, const DType& __i) : r(__r), i(__i) { }
    Complex(const Complex& __other) : Complex(__other.r, __other.i) { }

    bool operator == (const DType& __other) const {
        return (this->r == __other) && (this->i == __other);
    }

    bool operator == (const Complex<DType>& __other) const {
        return (this->r == __other.r) && (this->i == __other.i);
    }

    DType r;
    DType i;
};

template<typename DType>
inline std::ostream &operator << (std::ostream &__os, const Complex<DType>& __cpx) {
    __os << int((__cpx.r) * 1e6) * 1e-6 << " + " << int((__cpx.i) * 1e6) * 1e-6 << "j";
    return __os;
}

} // namespace complex

#endif // ROSACXX_COMPLEX_H
