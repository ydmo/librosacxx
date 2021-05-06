#ifndef ROSACXX_CORE_CONVERT_H
#define ROSACXX_CORE_CONVERT_H

#include <rosacxx/numcxx/numcxx.h>

namespace rosacxx {
namespace core {

int note_to_midi(const char * note);

float midi_to_hz(const float& midi);

float note_to_hz(const char * note);

float hz_to_octs(const float& freq, const float& tuning=0.0, const int& bins_per_octave=12);

float hz_to_midi(const float& freq);

nc::NDArrayF32Ptr hz_to_octs(const nc::NDArrayF32Ptr& freq, const float& tuning=0.0, const int& bins_per_octave=12);

nc::NDArrayF32Ptr midi_to_hz(const nc::NDArrayF32Ptr& midi);

nc::NDArrayF32Ptr midi_to_hz(const nc::NDArrayS32Ptr& midi);

nc::NDArrayF32Ptr hz_to_midi(const nc::NDArrayF32Ptr& freq);

nc::NDArrayF32Ptr fft_frequencies(const float& sr=22050, const int& n_fft=2048);

nc::NDArrayF32Ptr cqt_frequencies(const int& __n_bins, const float& __fmin, const int& __bins_per_octave=12, const float& __tuning=0.f);

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_CONVERT_H
