#ifndef ROSACXX_CORE_FFT_H
#define ROSACXX_CORE_FFT_H

#include <3rd/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

template <typename DType>
struct Complex {
    DType r;
    DType i;

    Complex(const DType& __ri) : r(__ri), i(__ri) { }

    Complex(const DType& __r, const DType& __i) : r(__r), i(__i) { }

    Complex(const Complex& __other) : Complex(__other.r, __other.i) { }

    bool operator == (const DType& __other) const {
        return (this->r == __other) && (this->i == __other);
    }

    bool operator == (const Complex<DType>& __other) const {
        return (this->r == __other.r) && (this->i == __other.i);
    }

//    Complex<DType> operator = (const Complex<DType>& __val) {
//        return Complex<DType>(__val.r, __val.i);
//    }

//    Complex<DType> operator = (const DType& __val) {
//        return Complex<DType>(__val);
//    }
};

template<typename DType>
inline std::ostream &operator << (std::ostream &__os, const Complex<DType>& __cpx) {
    __os << int((__cpx.r) * 1e6) * 1e-6 << "+" << int((__cpx.i) * 1e6) * 1e-6 << "j";
    return __os;
}

nc::NDArrayPtr<Complex<float>> rfft(const nc::NDArrayPtr<float>& __real_data, const int& __n_fft);

} // namespace core
} // namespace rosacxx

#endif
