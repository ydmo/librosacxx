#ifndef ROSACXX_CORE_AUDIO_H
#define ROSACXX_CORE_AUDIO_H

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

constexpr double BW_BEST = 0.94759372;
constexpr double BW_FASTEST = 0.85;

nc::NDArrayF32Ptr tone(const float& __freq, const float& __sr, const int * __length = NULL, const float * __duration = NULL, const float * __phi = NULL);

nc::NDArrayF32Ptr resample(const nc::NDArrayF32Ptr& y, const float& origin_sr, const float& target_sr, const char * res_type, const bool& scale = false);

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_AUDIO_H
