#ifndef ROSACXX_FILTERS_H
#define ROSACXX_FILTERS_H

#include <stdio.h>

#include <util/ndarray.h>

namespace rosacxx {
namespace filters {

/// Function: cq_to_chroma
/// ----------
/// Param Name              | Type          | Note
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// @param                  |               |
/// ----------
/// @result                 |               |
template<typename DType = float>
inline std::shared_ptr<NDArray<DType>> cq_to_chroma(
        const int& n_input,
        const int& bins_per_octave =    12,
        const int& n_chroma =           12,
        const float& fmin =             0,
        const NDArrayF32Ptr& window =   nullptr,
        const bool& base_c =            true
        ) {
    throw std::runtime_error("Not implemented error");
    return nullptr;
}

} // namespace filters
} // namespace rosacxx

#endif // ROSACXX_FILTERS_H
