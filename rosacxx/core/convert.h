#ifndef ROSACXX_CORE_CONVERT_H
#define ROSACXX_CORE_CONVERT_H

#include <3rd/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

int note_to_midi(const char * note);

float midi_to_hz(const int& midi);

float note_to_hz(const char * note);

float hz_to_octs(const float& freq, const float& tuning=0.0, const int& bins_per_octave=12);

nc::NDArrayF32Ptr hz_to_octs(const nc::NDArrayF32Ptr& freq, const float& tuning=0.0, const int& bins_per_octave=12);

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_CONVERT_H
