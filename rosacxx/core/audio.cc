#include <rosacxx/core/audio.h>
#include <rosacxx/resamcxx/core.h>

namespace rosacxx {
namespace core {

nc::NDArrayF32Ptr tone(const float& __freq, const float& __sr, const int * __length, const float * __duration, const float * __phi) {
    if (__length == NULL && __duration == NULL) throw std::invalid_argument("__length == NULL && __duration == NULL");
    int length = __length == NULL? int(__sr * __duration[0]) : *__length;
    float phi = __phi == NULL? -M_PI * .5f : *__phi;
    nc::NDArrayF32Ptr ret = nc::NDArrayF32Ptr(new nc::NDArrayF32({length}));
    float *p_ret = ret.data();
    for (int i = 0; i < length; i++) {
        p_ret[i] = std::cos(2 * M_PI * __freq * i / __sr + phi);
    }
    return ret;
}

nc::NDArrayF32Ptr resample(const nc::NDArrayF32Ptr& y, const float& origin_sr, const float& target_sr, const char * res_type, const bool& scale) {
    return nullptr;
}

} // namespace core
} // namespace rosacxx
