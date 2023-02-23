#ifndef ROSACXX_FILTERS_H
#define ROSACXX_FILTERS_H

#include <stdio.h>
#include <map>
#include <string>
#include <cfloat>

#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/util/utils.h>

#ifdef _WIN32
#   define _USE_MATH_DEFINES 1
#   include <math.h>
#   ifndef M_PI
#       define M_PI 3.141592653589793
#   endif
#endif // _WIN32

namespace rosacxx {
namespace filters {

enum STFTWindowType : int {
    Rectangular =       0, // Boxcar,
    Bartlett =          1,
    Hamming =           2,
    Hanning =           3,
    Blackman =          4,
    BlackmanHarris =    5,
    Nuttall =           6,
    Ones =              7,
    End =               8,
};

constexpr double WINDOW_BANDWIDTHS[STFTWindowType::End] = {
    /* STFTWindowType::Rectangular = */     1.0,
    /* STFTWindowType::Bartlett = */        1.3334961334912805,
    /* STFTWindowType::Hamming = */         1.3629455320350348,
    /* STFTWindowType::Hanning = */         1.50018310546875,
    /* STFTWindowType::Blackman = */        1.7269681554262326,
    /* STFTWindowType::blackmanharris = */  2.0045975283585014,
    /* STFTWindowType::Nuttall = */         1.9763500280946082,
    /* STFTWindowType::Ones = */            1.0,
};

//--- python-librosa -----------------------------------
//    WINDOW_BANDWIDTHS = {
//        "bart": 1.3334961334912805,
//        "barthann": 1.4560255965133932,
//        "bartlett": 1.3334961334912805,
//        "bkh": 2.0045975283585014,
//        "black": 1.7269681554262326,
//        "blackharr": 2.0045975283585014,
//        "blackman": 1.7269681554262326,
//        "blackmanharris": 2.0045975283585014,
//        "blk": 1.7269681554262326,
//        "bman": 1.7859588613860062,
//        "bmn": 1.7859588613860062,
//        "bohman": 1.7859588613860062,
//        "box": 1.0,
//        "boxcar": 1.0,
//        "brt": 1.3334961334912805,
//        "brthan": 1.4560255965133932,
//        "bth": 1.4560255965133932,
//        "cosine": 1.2337005350199792,
//        "flat": 2.7762255046484143,
//        "flattop": 2.7762255046484143,
//        "flt": 2.7762255046484143,
//        "halfcosine": 1.2337005350199792,
//        "ham": 1.3629455320350348,
//        "hamm": 1.3629455320350348,
//        "hamming": 1.3629455320350348,
//        "han": 1.50018310546875,
//        "hann": 1.50018310546875,
//        "hanning": 1.50018310546875,
//        "nut": 1.9763500280946082,
//        "nutl": 1.9763500280946082,
//        "nuttall": 1.9763500280946082,
//        "ones": 1.0,
//        "par": 1.9174603174603191,
//        "parz": 1.9174603174603191,
//        "parzen": 1.9174603174603191,
//        "rect": 1.0,
//        "rectangular": 1.0,
//        "tri": 1.3331706523555851,
//        "triang": 1.3331706523555851,
//        "triangle": 1.3331706523555851,
//    }
//--- python-librosa -----------------------------------

template<typename DType = float>
nc::NDArrayPtr<DType> get_window(const STFTWindowType& __windowType, const int& Nx, const bool& __fftbins=true) {
    if (__windowType == STFTWindowType::Rectangular || __windowType == STFTWindowType::Ones) {
        return nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}, DType(1)));
    }
    else if (__windowType == STFTWindowType::Bartlett) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        double symNx = __fftbins? double(Nx) : double(Nx-1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 2 / symNx * ( symNx / 2 - std::abs(n - symNx / 2));
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Hamming) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        double scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.54 - 0.46 * std::cos( scale2 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Hanning) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        double scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.5 - 0.5 * std::cos( scale2 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Blackman) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        double scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        double scale4 = __fftbins? 4 * M_PI / Nx : 4 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.42 - 0.5 * std::cos( scale2 * n ) + 0.08 * std::cos( scale4 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::Nuttall) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        double scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        double scale4 = __fftbins? 4 * M_PI / Nx : 4 * M_PI / (Nx - 1);
        double scale6 = __fftbins? 6 * M_PI / Nx : 6 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.3635819 - 0.4891775 * std::cos( scale2 * n ) + 0.1365995 * std::cos( scale4 * n ) - 0.0106411 * std::cos( scale6 * n );
        }
        return w;
    }
    else if (__windowType == STFTWindowType::BlackmanHarris) {
        auto w = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({Nx}));
        auto ptr_w = w.data();
        double scale2 = __fftbins? 2 * M_PI / Nx : 2 * M_PI / (Nx - 1);
        double scale4 = __fftbins? 4 * M_PI / Nx : 4 * M_PI / (Nx - 1);
        double scale6 = __fftbins? 6 * M_PI / Nx : 6 * M_PI / (Nx - 1);
        for (auto n = 0; n < Nx; n++) {
            ptr_w[n] = 0.35875 - 0.48829 * std::cos( scale2 * n ) + 0.14128 * std::cos( scale4 * n ) - 0.01168 * std::cos( scale6 * n );
        }
        return w;
    }
    else {
        throw std::invalid_argument("Invaild window type.");
    }
    return nullptr;
}

template<typename DType = float>
nc::NDArrayPtr<DType> __float_window(const STFTWindowType& __windowType, const float& n) {
    int n_min = int(std::floor(n));
    int n_max = int(std::ceil(n));
    auto window = get_window(__windowType, n_min);
    if (n_min < n_max) {
        window = nc::pad(window, {std::make_pair(0, n_max - n_min)});
    }
    return window;
}

/// Function: window_bandwidth
/// Get the equivalent noise bandwidth of a window function.
/// ----------
/// Param Name              | Type           | Note
/// @param windowType       | STFTWindowType | Enum of a window function.
/// ----------
/// @return                 | float          | The equivalent noise bandwidth (in FFT bins) of the given window function.
inline float window_bandwidth(const STFTWindowType& __windowType, const int& __n = 1000) {
    if (__windowType < STFTWindowType::End) {
        return WINDOW_BANDWIDTHS[__windowType];
    }
    else {
        throw std::runtime_error("Not implemented of window_bandwidth.");
        return -1;
    }
}

inline nc::NDArrayF32Ptr __compute_frequencies(const float& fmin, const int& n_bins, const int& bins_per_octave) {
    std::vector<float> vec_freq(n_bins);
    for (auto i = 0; i < n_bins; i++) {
        vec_freq[i] = float(fmin * std::pow(2.0, double(i) / bins_per_octave));
    }
    return nc::NDArrayF32Ptr::FromVec1D(vec_freq);
}

inline nc::NDArrayF32Ptr constant_q_lengths(
        const float& sr,
        const float& fmin,
        const int& n_bins=84,
        const int& bins_per_octave=12,
        const STFTWindowType& window=STFTWindowType::Hanning,
        const float& filter_scale=1,
        const float& gamma=0
        ) {

    if (fmin <= 0) throw std::invalid_argument("fmin must be positive");
    if (bins_per_octave <= 0) throw std::invalid_argument("bins_per_octave must be positive");
    if (filter_scale <= 0) throw std::invalid_argument("filter_scale must be positive");
    if (n_bins <= 0) throw std::invalid_argument("n_bins must be a positive integer");

    // Q should be capitalized here, so we suppress the name warning
    // pylint: disable=invalid-name
    float alpha = std::pow(2.0, (1.0 / bins_per_octave)) - 1.0;
    float Q = float(filter_scale) / alpha;

    // Compute the frequencies
    nc::NDArrayF32Ptr freq = __compute_frequencies(fmin, n_bins, bins_per_octave);

    if (freq.getitem(freq.elemCount()-1) * (1 + 0.5 * window_bandwidth(window) / Q) > sr / 2.0) throw std::invalid_argument("Filter pass-band lies beyond Nyquist");

    // Convert frequencies to filter lengths
    auto lengths = (Q * sr) / (freq + (gamma / alpha));

    return lengths;
}

template<typename DType = float>
struct RetConstantQ {
    std::vector<nc::NDArrayPtr<std::complex<float>>> filters;
    nc::NDArrayPtr<DType> lengths;
};

template<typename DType = float>
inline RetConstantQ<DType> constant_q(
        const float& sr,
        const float& __fmin = INFINITY,
        const int& n_bins = 84,
        const int& bins_per_octave = 12,
        const STFTWindowType& window=STFTWindowType::Hanning,
        const float& filter_scale = 1,
        const bool& pad_fft = true,
        const float& norm = 1,
        const float& gamma = 0
        ) {
    float fmin = __fmin;
    if (fmin == INFINITY) {
        fmin = core::note_to_hz<float>("C1");
    }
    nc::NDArrayF32Ptr lengths = constant_q_lengths(sr, fmin, n_bins, bins_per_octave, window, filter_scale, gamma);

    nc::NDArrayF32Ptr freqs = __compute_frequencies(fmin, n_bins, bins_per_octave);

    // Build the filters
    std::vector<nc::NDArrayPtr<std::complex<float>>> filters(0);
    for (auto i = 0; i < freqs.elemCount(); i++) {
        auto ilen = lengths.getitem(i);
        auto freq = freqs.getitem(i);

        std::vector<std::complex<float>> vec_sig(0);
        for (int k = int(std::floor(-ilen / 2)); k < int(std::floor(ilen/2)); k++) {
            std::complex<float> x;
            x.real(0);
            x.imag(k * 2 * M_PI * freq / sr);
            x = std::exp(x);
            vec_sig.push_back(x);
        }

        nc::NDArrayPtr<std::complex<float>> sig = nc::NDArrayPtr<std::complex<float>>::FromVec1D(vec_sig);

        sig = sig * __float_window(window, len(sig));

        // Normalize ---------------------
        auto mag = nc::abs(sig);
        float sum = nc::pow(mag, norm).sum();
        float length = std::pow(sum, (1.0 / norm));
        sig /= length;
        // Normalize ---------------------

        filters.push_back(sig);
    }

    auto max_len_f = lengths.max();
    int max_len = int(std::pow(2.0, (std::ceil(std::log2(max_len_f)))));

    for (auto i = 0; i < filters.size(); i++) {
        filters[i] = util::pad_center_1d(filters[i], max_len);
    }

    RetConstantQ<DType> ret;
    ret.filters = filters;
    ret.lengths = lengths;

    return ret;
}

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
inline nc::NDArrayPtr<DType> cq_to_chroma(
        const int& __n_input,
        const int& __bins_per_octave =      12,
        const int& __n_chroma =             12,
        const float& __fmin =               INFINITY,
        const nc::NDArrayF32Ptr& __window = nullptr,
        const bool& __base_c =              true
        ) {
    // --------
    int n_input = __n_input;
    int bins_per_octave = __bins_per_octave;
    int n_chroma = __n_chroma;
    float fmin = __fmin;
    bool base_c = __base_c;
    nc::NDArrayF32Ptr window = __window;
    // --------

    double n_merge = double(bins_per_octave) / n_chroma;

    if (fmin == INFINITY) {
        fmin = core::note_to_hz<float>("C1");
    }

    if (n_merge - int(n_merge / 1) * 1 != 0) {
        throw std::runtime_error(
            "Incompatible CQ merge: "
            "input bins must be an "
            "integer multiple of output bins."
            );
    }

    // # Tile the identity to merge fractional bins
    // cq_to_ch = np.repeat(np.eye(n_chroma), n_merge, axis=1)
    nc::NDArrayPtr<DType> cq_to_ch = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({n_chroma, n_chroma * int(n_merge)}));
    auto ptr_cq_to_ch = cq_to_ch.data();
    for (auto r = 0; r < n_chroma; r++) {
        for (auto c = 0; c < n_chroma; c++) {
            DType val = r == c? DType(1) : 0;
            for (auto m = 0; m < int(n_merge); m++) {
                *ptr_cq_to_ch++ = val;
            }
        }
    }

    // # Roll it left to center on the target bin
    // cq_to_ch = np.roll(cq_to_ch, -int(n_merge // 2), axis=1)
    nc::NDArrayPtr<DType> cq_to_ch_roll = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(cq_to_ch.shape()));
    for (int r = 0; r < cq_to_ch.shape()[0]; r++) {
        auto ptr_cq_to_ch_r = cq_to_ch.at(r, 0);
        auto ptr_cq_to_ch_roll_r = cq_to_ch_roll.at(r, 0);
        for (int c = 0; c < cq_to_ch.shape()[1]; c++) {
            int c_from = c;
            int c_to = c - int(n_merge) / 2;
            if (c_to < 0) c_to += cq_to_ch.shape()[1];
            ptr_cq_to_ch_roll_r[c_to] = ptr_cq_to_ch_r[c_from];
        }
    }
    cq_to_ch = cq_to_ch_roll;

    //  n_octaves = np.ceil(np.float(n_input) / bins_per_octave)
    int  n_octaves = std::ceil(float(n_input) / bins_per_octave);

    // # Repeat and trim
    // cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]
    nc::NDArrayPtr<DType> cq_to_ch_tiled = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({ cq_to_ch.shape()[0], n_input }));
    for (int r = 0; r < cq_to_ch_tiled.shape()[0]; r++) {
        auto ptr_cq_to_ch_r       = cq_to_ch.at(r, 0);
        auto ptr_cq_to_ch_tiled_r = cq_to_ch_tiled.at(r, 0);
        for (auto c = 0; c < n_input; c++) {
            int c_to = c;
            int c_from = c % cq_to_ch.shape()[1];
            ptr_cq_to_ch_tiled_r[c_to] = ptr_cq_to_ch_r[c_from];
        }
    }
    cq_to_ch = cq_to_ch_tiled;

    // midi_0 = np.mod(hz_to_midi(fmin), 12)
    float midi_0_tmp = core::hz_to_midi(fmin);
    float midi_0 = midi_0_tmp - int(midi_0_tmp) / 12;

    float rollf = midi_0 - 9;
    if (base_c) {
        // # rotate to C
        rollf = midi_0;
    }
    int roll = int(std::round(rollf * (n_chroma / 12.0)));

    // # Apply the roll
    // cq_to_ch = np.roll(cq_to_ch, roll, axis=0).astype(dtype)
    if (roll) {
        nc::NDArrayPtr<DType> cq_to_ch_roll_new = nc::NDArrayPtr<DType>(new nc::NDArray<DType>(cq_to_ch.shape()));
        for (int r = 0; r < cq_to_ch.shape()[0]; r++) {
            int r_from = r;
            int r_to = (r + roll) % cq_to_ch.shape()[0];
            memcpy(cq_to_ch_roll_new.at(r_to, 0), cq_to_ch.at(r_from), cq_to_ch.shape()[1] * sizeof (DType));
        }
    }

    if (window) {
        throw std::runtime_error("Not implemented when window is not null.");
    }

    return cq_to_ch;
}


template <typename DType>
nc::NDArrayPtr<DType> mel(
        const DType& __sr,
        const int& __n_fft,
        const int& __n_mels=128,
        const DType& __fmin=0.0,
        const DType& __fmax=-DBL_MAX,
        const bool& __htk=false,
        const char * __norm="slaney"
        ) {
    DType sr = __sr;
    DType fmax = __fmax;
    if (fmax < 0) {
        fmax = sr / 2;
    }
    int n_mels = __n_mels;
    int n_fft = __n_fft;
    nc::NDArrayPtr<DType> weights = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({n_mels, int(1 + n_fft / 2)}));
    nc::NDArrayPtr<DType> fftfreqs = core::fft_frequencies(sr, n_fft);
    nc::NDArrayPtr<DType> mel_f = core::mel_frequencies(n_mels + 2, __fmin, fmax, __htk);

    // fdiff = np.diff(mel_f)
    nc::NDArrayPtr<DType> fdiff = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({mel_f.elemCount()-1}));
    DType * ptr_fdiff = fdiff.data();
    for (auto i = 0; i < fdiff.elemCount(); i++) {
        ptr_fdiff[i] = mel_f.getitem(i+1) - mel_f.getitem(i);
    }

    // ramps = np.subtract.outer(mel_f, fftfreqs)
    nc::NDArrayPtr<DType> ramps = nc::NDArrayPtr<DType>(new nc::NDArray<DType>({mel_f.elemCount(), fftfreqs.elemCount()}));
    for (auto r = 0; r < mel_f.elemCount(); r++) {
        for (auto c = 0; c < fftfreqs.elemCount(); c++) {
            ramps.at(r, c)[0] = mel_f.getitem(r) - fftfreqs.getitem(c);
        }
    }

    DType * ptr_w = weights.data();
    for (auto i = 0; i < n_mels; i++) {
        for (auto j = 0; j < fftfreqs.elemCount(); j++) {
            DType lower = -ramps.getitem(i, j) / fdiff.getitem(i);
            DType upper = ramps.getitem(i+2, j) / fdiff.getitem(i+1);
            *ptr_w++ = std::max(0, std::min(lower, upper));
        }
    }

    if (strcmp(__norm, "slaney") == 0) {
        std::vector<DType> enorm(n_mels);
        for (auto i = 0; i < n_mels; i++) {
            enorm[i] = 2.0 / (mel_f.getitem(i+2)-mel_f.getitem(i));
        }
        DType * ptr_w0 = weights.data();
        for (auto i = 0; i < weights.shape(0); i++) {
            for (auto j = 0; j < weights.shape(1); j++) {
                *ptr_w0++ *= enorm[i];
            }
        }
    }
    else {
        throw std::runtime_error("Not implemented.");
    }

    return weights;
}


} // namespace filters
} // namespace rosacxx

#endif // ROSACXX_FILTERS_H
