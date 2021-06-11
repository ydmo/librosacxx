#ifndef ROSACXX_CORE_CONVERT_H
#define ROSACXX_CORE_CONVERT_H

#include <rosacxx/numcxx/numcxx.h>

#include <unordered_map>
#include <cmath>

namespace rosacxx {
namespace core {

// int note_to_midi(const char * note);
inline int note_to_midi(const char * note) {
    const std::unordered_map<std::string, int> map_note_2_midi = { {"A0", 21}, {"A#0/Bb0", 22}, {"B0", 23}, {"C1", 24}, {"C#1/Db1", 25}, {"D1", 26}, {"D#1/Eb1", 27}, {"E1", 28}, {"F1", 29}, {"F#1/Gb1", 30}, {"G1", 31}, {"G#1/Ab1", 32}, {"A1", 33}, {"A#1/Bb1", 34}, {"B1", 35}, {"C2", 36}, {"C#2/Db2", 37}, {"D2", 38}, {"D#2/Eb2", 39}, {"E2", 40}, {"F2", 41}, {"F#2/Gb2", 42}, {"G2", 43}, {"G#2/Ab2", 44}, {"A2", 45}, {"A#2/Bb2", 46}, {"B2", 47}, {"C3", 48}, {"C#3/Db3", 49}, {"D3", 50}, {"D#3/Eb3", 51}, {"E3", 52}, {"F3", 53}, {"F#3/Gb3", 54}, {"G3", 55}, {"G#3/Ab3", 56}, {"A3", 57}, {"A#3/Bb3", 58}, {"B3", 59}, {"C4 (middle C)", 60}, {"C#4/Db4", 61}, {"D4", 62}, {"D#4/Eb4", 63}, {"E4", 64}, {"F4", 65}, {"F#4/Gb4", 66}, {"G4", 67}, {"G#4/Ab4", 68}, {"A4 concert pitch", 69}, {"A#4/Bb4", 70}, {"B4", 71}, {"C5", 72}, {"C#5/Db5", 73}, {"D5", 74}, {"D#5/Eb5", 75}, {"E5", 76}, {"F5", 77}, {"F#5/Gb5", 78}, {"G5", 79}, {"G#5/Ab5", 80}, {"A5", 81}, {"A#5/Bb5", 82}, {"B5", 83}, {"C6", 84}, {"C#6/Db6", 85}, {"D6", 86}, {"D#6/Eb6", 87}, {"E6", 88}, {"F6", 89}, {"F#6/Gb6", 90}, {"G6", 91}, {"G#6/Ab6", 92}, {"A6", 93}, {"A#6/Bb6", 94}, {"B6", 95}, {"C7", 96}, {"C#7/Db7", 97}, {"D7", 98}, {"D#7/Eb7", 99}, {"E7", 100}, {"F7", 101}, {"F#7/Gb7", 102}, {"G7", 103}, {"G#7/Ab7", 104}, {"A7", 105}, {"A#7/Bb7", 106}, {"B7", 107}, {"C8", 108}, {"C#8/Db8", 109}, {"D8", 110}, {"D#8/Eb8", 111}, {"E8", 112}, {"F8", 113}, {"F#8/Gb8", 114}, {"G8", 115}, {"G#8/Ab8", 116}, {"A8", 117}, {"A#8/Bb8", 118}, {"B8", 119}, {"C9", 120}, {"C#9/Db9", 121}, {"D9", 122}, {"D#9/Eb9", 123}, {"E9", 124}, {"F9", 125}, {"F#9/Gb9", 126}, {"G9", 127}, {"G#9/Ab9", 128} };
    const std::unordered_map<int, std::string> map_midi_2_note = { {21, "A0"}, {22, "A#0/Bb0"}, {23, "B0"}, {24, "C1"}, {25, "C#1/Db1"}, {26, "D1"}, {27, "D#1/Eb1"}, {28, "E1"}, {29, "F1"}, {30, "F#1/Gb1"}, {31, "G1"}, {32, "G#1/Ab1"}, {33, "A1"}, {34, "A#1/Bb1"}, {35, "B1"}, {36, "C2"}, {37, "C#2/Db2"}, {38, "D2"}, {39, "D#2/Eb2"}, {40, "E2"}, {41, "F2"}, {42, "F#2/Gb2"}, {43, "G2"}, {44, "G#2/Ab2"}, {45, "A2"}, {46, "A#2/Bb2"}, {47, "B2"}, {48, "C3"}, {49, "C#3/Db3"}, {50, "D3"}, {51, "D#3/Eb3"}, {52, "E3"}, {53, "F3"}, {54, "F#3/Gb3"}, {55, "G3"}, {56, "G#3/Ab3"}, {57, "A3"}, {58, "A#3/Bb3"}, {59, "B3"}, {60, "C4 (middle C)"}, {61, "C#4/Db4"}, {62, "D4"}, {63, "D#4/Eb4"}, {64, "E4"}, {65, "F4"}, {66, "F#4/Gb4"}, {67, "G4"}, {68, "G#4/Ab4"}, {69, "A4 concert pitch"}, {70, "A#4/Bb4"}, {71, "B4"}, {72, "C5"}, {73, "C#5/Db5"}, {74, "D5"}, {75, "D#5/Eb5"}, {76, "E5"}, {77, "F5"}, {78, "F#5/Gb5"}, {79, "G5"}, {80, "G#5/Ab5"}, {81, "A5"}, {82, "A#5/Bb5"}, {83, "B5"}, {84, "C6"}, {85, "C#6/Db6"}, {86, "D6"}, {87, "D#6/Eb6"}, {88, "E6"}, {89, "F6"}, {90, "F#6/Gb6"}, {91, "G6"}, {92, "G#6/Ab6"}, {93, "A6"}, {94, "A#6/Bb6"}, {95, "B6"}, {96, "C7"}, {97, "C#7/Db7"}, {98, "D7"}, {99, "D#7/Eb7"}, {100, "E7"}, {101, "F7"}, {102, "F#7/Gb7"}, {103, "G7"}, {104, "G#7/Ab7"}, {105, "A7"}, {106, "A#7/Bb7"}, {107, "B7"}, {108, "C8"}, {109, "C#8/Db8"}, {110, "D8"}, {111, "D#8/Eb8"}, {112, "E8"}, {113, "F8"}, {114, "F#8/Gb8"}, {115, "G8"}, {116, "G#8/Ab8"}, {117, "A8"}, {118, "A#8/Bb8"}, {119, "B8"}, {120, "C9"}, {121, "C#9/Db9"}, {122, "D9"}, {123, "D#9/Eb9"}, {124, "E9"}, {125, "F9"}, {126, "F#9/Gb9"}, {127, "G9"}, {128, "G#9/Ab9"} };
    auto iter = map_note_2_midi.find(note);
    return iter->second;
}

// float midi_to_hz(const float& midi);
template <typename DType, typename MType>
inline DType midi_to_hz(const MType& midi) {
    return DType(440.0 * std::pow(2.0, ((double(midi) - 69.0) / 12.0)));
}

// float note_to_hz(const char * note);
template <typename DType>
inline DType note_to_hz(const char * note) {
    return midi_to_hz<DType>(note_to_midi(note));
};

// float hz_to_octs(const float& freq, const float& tuning=0.0, const int& bins_per_octave=12);
template <typename DType>
inline DType hz_to_octs(const DType freq, const DType tuning, const int bins_per_octave=12) {
    auto A440 = 440.0 * std::pow(2.0, (tuning / bins_per_octave));
    return std::log2(freq / (A440 / 16));
}

// float hz_to_midi(const float& freq);
template <typename DType>
inline int hz_to_midi(const DType& frequencies) {
    return int(12 * (std::log2(frequencies) - std::log2(440.0)) + 69);
}

// nc::NDArrayF32Ptr hz_to_octs(const nc::NDArrayF32Ptr& freq, const float& tuning=0.0, const int& bins_per_octave=12);
template <typename DType>
inline nc::NDArrayPtr<DType> hz_to_octs(const nc::NDArrayPtr<DType>& freq, const DType& tuning=0, const int& bins_per_octave=12) {
    nc::NDArrayPtr<DType> octs = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(freq.shape()));
    DType *ptr_freq = freq.data();
    DType *ptr_octs = octs.data();
    for (auto i = 0; i < freq.elemCount(); i++) {
        ptr_octs[i] = hz_to_octs(ptr_freq[i], tuning, bins_per_octave);
    }
    return octs;
}

// nc::NDArrayF32Ptr midi_to_hz(const nc::NDArrayF32Ptr& midi);
template <typename DType>
nc::NDArrayPtr<DType> midi_to_hz(const nc::NDArrayPtr<DType>& midi) {
    DType *ptr_midi = midi.data();
    nc::NDArrayPtr<DType> hz = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(midi.shape()));
    DType *ptr_hz = hz.data();
    for (auto i = 0; i < hz.elemCount(); i++) {
        ptr_hz[i] = midi_to_hz<DType>(ptr_midi[i]);
    }
    return hz;
}

nc::NDArrayF32Ptr midi_to_hz(const nc::NDArrayS32Ptr& midi);
template <typename DType>
nc::NDArrayPtr<DType> midi_to_hz(const nc::NDArrayS32Ptr& midi) {
    int *ptr_midi = midi.data();
    nc::NDArrayPtr<DType> hz = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(midi.shape()));
    DType *ptr_hz = hz.data();
    for (auto i = 0; i < hz.elemCount(); i++) {
        ptr_hz[i] = midi_to_hz<DType>(ptr_midi[i]);
    }
    return hz;
}

nc::NDArrayF32Ptr hz_to_midi(const nc::NDArrayF32Ptr& freq);

// nc::NDArrayF32Ptr fft_frequencies(const float& sr=22050, const int& n_fft=2048);

template <typename DType>
inline nc::NDArrayPtr<DType> fft_frequencies(const DType& __sr, const int& __n_fft) {
    return nc::linspace(DType(0.0), __sr * DType(0.5), (__n_fft >> 1) + 1, true);
}

// nc::NDArrayF32Ptr cqt_frequencies(const int& __n_bins, const float& __fmin, const int& __bins_per_octave=12, const float& __tuning=0.f);

template <typename DType>
inline nc::NDArrayPtr<DType> cqt_frequencies(const int& __n_bins, const DType& __fmin, const int& __bins_per_octave=12, const DType& __tuning=0) {
    double tuning = double(__tuning);
    double correction = std::pow(2.0, (tuning / __bins_per_octave));
    nc::NDArrayPtr<DType> frequencies = nc::NDArrayF32Ptr(new nc::NDArrayF32({__n_bins}));
    DType *ptr_frequencies = frequencies.data();
    for (auto i = 0; i < __n_bins; i++) {
        ptr_frequencies[i] = DType(double(correction) * double(__fmin) * std::pow(2.0, double(i) / __bins_per_octave));
    }
    return frequencies;
}

template <typename DType>
inline DType hz_to_mel(const DType& __frequencies, const bool& htk=false) {
    double frequencies = double(__frequencies);
    if (htk) {
        return DType( 2595.0 * std::log10(1.0 + frequencies / 700.0) );
    }
    constexpr double f_min = 0.0;
    constexpr double f_sp = 200.0 / 3;
    double mels = (frequencies - f_min) / f_sp;
    double min_log_hz = 1000.0;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    double logstep = std::log(6.4) / 27.0;
    if (frequencies >= min_log_hz) {
        mels = min_log_mel + std::log(frequencies / min_log_hz) / logstep;
    }
    return DType(mels);
}

template <typename DType>
inline nc::NDArrayPtr<DType> hz_to_mel(const nc::NDArrayPtr<DType>& __frequencies, const bool& htk=false) {
    nc::NDArrayPtr<DType> melsArr = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(__frequencies.shape()));
    DType *ptr_mels = melsArr.data();
    DType *ptr_freq = __frequencies.data();
    if (htk) {
        for (auto i = 0; i < melsArr.elemCount(); i++) {
            ptr_mels[i] = DType( 2595.0 * std::log10(1.0 + double(ptr_freq[i]) / 700.0) );
        }
    }
    else {
        constexpr double f_min = 0.0;
        constexpr double f_sp = 200.0 / 3;
        constexpr double min_log_hz = 1000.0;
        constexpr double logstep = 0.06875177742094912; // std::log(6.4) / 27.0;
        for (auto i = 0; i < melsArr.elemCount(); i++) {
            double mels = (double(ptr_freq[i]) - f_min) / f_sp;
            double min_log_mel = (min_log_hz - f_min) / f_sp;
            if (double(ptr_freq[i]) >= min_log_hz) {
                mels = min_log_mel + std::log(double(ptr_freq[i]) / min_log_hz) / logstep;
            }
            ptr_mels[i] = DType(mels);
        }
    }
    return melsArr;
}

template <typename DType>
inline DType mel_to_hz(const DType& __mel, const bool& htk=false) {
    constexpr double f_min = 0.0;
    constexpr double f_sp = 200.0 / 3;
    constexpr double min_log_hz = 1000.0;
    constexpr double logstep = 0.06875177742094912; // np.log(6.4) / 27.0;
    // start ..
    double mels = double(__mel);
    if ( htk ) {
        return 700.0 * (std::pow(10.0, (mels / 2595.0)) - 1.0);
    }
    double freqs = f_min + f_sp * mels;
    double min_log_mel = (min_log_hz - f_min) / f_sp;
    if (mels >= min_log_mel) {
        freqs = min_log_hz * std::exp(logstep * (mels - min_log_mel));
    }
    return freqs;
}

template <typename DType>
inline nc::NDArrayPtr<DType> mel_to_hz(const nc::NDArrayPtr<DType>& __mels, const bool& htk=false) {
    nc::NDArrayPtr<DType> hzs = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(__mels.shape()));
    DType *ptr_hz = hzs.data();
    DType *ptr_mel = __mels.data();
    for (auto i = 0; i < hzs.elemCount(); i++) {
        ptr_hz[i] = mel_to_hz(ptr_mel[i], htk);
    }
    return hzs;
}

template <typename DType>
nc::NDArrayPtr<DType> mel_frequencies(const int& n_mels=128, const DType& fmin=0, const DType& fmax=11025, const bool& htk=false) {
    DType min_mel = hz_to_mel(fmin, htk);
    DType max_mel = hz_to_mel(fmax, htk);
    nc::NDArrayPtr<DType> mels = nc::linspace(min_mel, max_mel, n_mels);
    return mel_to_hz(mels, htk);
}

template <typename DType>
inline nc::NDArrayPtr<DType> frames_to_samples(const nc::NDArrayPtr<DType>& frames, const int& hop_length=512, const int& n_fft=-1) {
    int offset = 0;
    if (n_fft > 0) {
        offset = int(n_fft / 2);
    }
    return frames * hop_length + offset;
}

} // namespace core
} // namespace rosacxx

#endif // ROSACXX_CORE_CONVERT_H
