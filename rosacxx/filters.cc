#include <filters.h>

namespace rosacxx {
namespace filters {

template<typename DType>
std::shared_ptr<NDArray<DType>> cq_to_chroma(
        const int& n_input,
        const int& bins_per_octave,
        const int& n_chroma,
        const float& fmin,
        const NDArrayF32::Ptr& window,
        const bool& base_c
        ) {

}

} // namespace core
} // namespace rosacxx
