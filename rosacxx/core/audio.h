#ifndef ROSACXX_CORE_AUDIO_H
#define ROSACXX_CORE_AUDIO_H

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

nc::NDArrayF32Ptr tone(const float& __freq, const float& __sr, const int * __length = NULL, const float * __duration = NULL, const float * __phi = NULL);

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_AUDIO_H
