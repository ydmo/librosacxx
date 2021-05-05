#include <rosacxx/core/constantq.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/audio.h>
#include <rosacxx/core/spectrum.h>
#include <rosacxx/filters.h>
#include <rosacxx/fft/kiss/kiss_fft.h>

#include <cmath>

namespace rosacxx {
namespace core {

/// __num_two_factors
/// Return how many times integer x can be evenly divided by 2.
/// Returns 0 for non-positive integers.
inline int __num_two_factors(const int& __x) {
    if (__x <= 0) return 0;
    int num_twos = 0;
    int x = __x;
    while (x % 2 == 0) {
        num_twos += 1;
        x = x / 2;
    }
    return num_twos;
}

/// __early_downsample_count
/// Compute the number of early downsampling operations
inline int __early_downsample_count(const float& nyquist, const float& filter_cutoff, const int& hop_length, const int& n_octaves) {
    int downsample_count1 = std::max(0, int(std::ceil(std::log2(BW_FASTEST * nyquist / filter_cutoff)) - 1) - 1);
    int num_twos = __num_two_factors(hop_length);
    int downsample_count2 = std::max(0, num_twos - n_octaves + 1);
    return std::min(downsample_count1, downsample_count2);
}

/// __early_downsample
/// Perform early downsampling on an audio signal, if it applies.
inline void __early_downsample(
        nc::NDArrayF32Ptr& y,
        float& sr,
        int& hop_length,
        const char * res_type,
        const int& n_octaves,
        const float& nyquist,
        const float& filter_cutoff,
        const bool& scale
        ) {
    int downsample_count = __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves);
    if (downsample_count > 0 && strcmp(res_type, "kaiser_fast") == 0) {
        auto downsample_factor = std::pow(2, downsample_count);
        auto new_hop_length = hop_length / downsample_factor;
        if (y.elemCount() < downsample_factor) throw std::invalid_argument("Input signal is too short to for CQT.");
        float new_sr = sr / float(downsample_factor);
        nc::NDArrayF32Ptr new_y = resample(y, sr, new_sr, res_type, true, true);
        if (!scale) {
            new_y *= std::sqrt(downsample_factor);
        }
        // re-write input arguments ...
        y = new_y;
        sr = new_sr;
        hop_length = new_hop_length;
    }
}

struct RetCQTFilterFFT {
    nc::NDArrayCpxF32Ptr fft_basis;
    int n_fft;
    nc::NDArrayPtr<float> lengths;
};

RetCQTFilterFFT __cqt_filter_fft(
        const float& sr,
        const float& fmin,
        const int& n_bins,
        const int& bins_per_octave,
        const float& filter_scale,
        const float& norm,
        const float& sparsity,
        const int& hop_length=-1,
        const filters::STFTWindowType& window = filters::STFTWindowType::Hanning,
        const float& gamma=0.0
        ) {
    auto ret_cq = filters::constant_q(sr, fmin, n_bins, bins_per_octave, window, filter_scale, true, norm, gamma);
    auto basis = ret_cq.filters;
    auto lengths = ret_cq.lengths;

    int n_fft = len(basis[0]);

    if (hop_length > 0 && n_fft < std::pow(2.0, (1 + std::ceil(std::log2(hop_length))))) {
        n_fft = std::pow(2.0, (1 + std::ceil(std::log2(hop_length))));
    }

    // basis *= lengths[:, np.newaxis] / float(n_fft)
    // fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

    auto fft_basis_full = nc::NDArrayPtr<std::complex<float>>(new nc::NDArray<std::complex<float>>({int(basis.size()), n_fft}));

    for (auto i = 0; i < basis.size(); i++) {
        basis[i] *= (lengths.getitem(i) / n_fft);
        vkfft::fft_forward(n_fft, basis[i].data(), fft_basis_full.at(i, 0));
    }

    auto fft_basis = nc::NDArrayPtr<std::complex<float>>(new nc::NDArray<std::complex<float>>({ int(basis.size()), n_fft/2+1 }));
    for (auto r = 0; r < fft_basis.shape()[0]; r++) {
        memcpy(fft_basis.at(r, 0), fft_basis_full.at(r, 0), (n_fft/2+1) * sizeof(std::complex<float>));
    }

    fft_basis = utils::sparsify_rows(fft_basis, sparsity);

    RetCQTFilterFFT ret;
    ret.fft_basis = fft_basis;
    ret.n_fft = n_fft;
    ret.lengths = lengths;

    return ret;
}

nc::NDArrayCpxF32Ptr __cqt_response(
        const nc::NDArrayF32Ptr& y,
        const int& n_fft,
        const int& hop_length,
        const nc::NDArrayCpxF32Ptr& fft_basis,
        const filters::STFTWindowType&  window = filters::STFTWindowType::Ones,
        const char* pad_mode = "reflect"
        ) {
    auto D = stft(y, n_fft, hop_length, -1, window, true, pad_mode);
    return matmul(fft_basis, D);
}

/// Helper function to trim and stack a collection of CQT responses
nc::NDArrayCpxF32Ptr __trim_stack(
        const std::vector<nc::NDArrayCpxF32Ptr>& cqt_resp,
        const int& n_bins
        ) {

    // max_col = min(c_i.shape[-1] for c_i in cqt_resp)
    int max_col = 0;
    for (nc::NDArrayCpxF32Ptr c_i : cqt_resp) {
        max_col = std::max(c_i.shape().back(), max_col);
    }

    // cqt_out = np.empty((n_bins, max_col), dtype=dtype, order="F")
    nc::NDArrayCpxF32Ptr cqt_out = nc::NDArrayCpxF32Ptr(new nc::NDArray<std::complex<float>>({n_bins, max_col}));

    int end = n_bins;
    for (nc::NDArrayCpxF32Ptr c_i : cqt_resp) {
        // # By default, take the whole octave
        int n_oct = c_i.shape()[0];

        // # If the whole octave is more than we can fit,
        // # take the highest bins from c_i
        if (end < n_oct) {
            for (auto r = 0; r < end; r++) {
                auto i_to = r;
                auto i_from = c_i.shape()[0] - end + r;
                auto ptr_cqt_out_i = cqt_out.at(i_to, 0);
                auto ptr_c_i = c_i.at(i_to, 0);
                for (auto c = 0; c < max_col; c++) {
                    ptr_cqt_out_i[c] = ptr_c_i[c];
                }
            }
        }
        else {
            for (auto r = 0; r < end; r++) {
                auto i_to = r + end - n_oct;
                auto i_from = r;
                auto ptr_cqt_out_i = cqt_out.at(i_to, 0);
                auto ptr_c_i = c_i.at(i_to, 0);
                for (auto c = 0; c < max_col; c++) {
                    ptr_cqt_out_i[c] = ptr_c_i[c];
                }
            }
        }
        end -= n_oct;
    }

    return cqt_out;
}

nc::NDArrayCpxF32Ptr vqt(
        const nc::NDArrayF32Ptr&        __y,
        const float                     __sr,
        const int                       __hop_length,
        const float                     __fmin,
        const int                       __n_bins,
        const float                     __gamma,
        const int                       __bins_per_octave,
        const float                     __tuning,
        const float                     __filter_scale,
        const float                     __norm,
        const float                     __sparsity,
        const filters::STFTWindowType&  __window,
        const bool                      __scale,
        const char *                    __pad_mode,
        const char *                    __res_type
        ) {
    nc::NDArrayF32Ptr y = __y;
    int hop_length = __hop_length;
    float sr = __sr;
    int bins_per_octave = __bins_per_octave;
    int n_bins = __n_bins;
    float filter_scale = __filter_scale;
    filters::STFTWindowType window = __window;
    std::string res_type = __res_type ? __res_type : "";
    bool scale = __scale;
    float norm = __norm;
    float sparsity = __sparsity;

    int n_octaves = int(std::ceil(float(n_bins) / bins_per_octave));
    int n_filters = std::min(bins_per_octave, n_bins);

    int len_orig = __y.shape()[0];
    float alpha = std::pow(2.0, (1.0 / bins_per_octave)) - 1;

    float fmin = __fmin;
    if (fmin == INFINITY) {
        fmin = note_to_hz("C1");
    }

    float tuning = __tuning;
    if (tuning == INFINITY) {
        nc::NDArrayF32Ptr S = nullptr;
        tuning = estimate_tuning(__y, __sr, S, 2048, 0.01f, __bins_per_octave);
    }

    float gamma = __gamma;
    if (gamma == INFINITY) {
        gamma = 24.7 * alpha / 0.108;
    }

    fmin = fmin * std::pow(2.0, (tuning / bins_per_octave));

    auto freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);

    float fmin_t = freqs.min();
    float fmax_t = freqs.max();

    float Q = filter_scale / alpha;
    float filter_cutoff = fmax_t * (1 + 0.5 * filters::window_bandwidth(window) / Q) + 0.5 * gamma;
    float nyquist = sr / 2.0;

    bool auto_resample = false;
    if (!__res_type) {
        auto_resample = true;
        if (filter_cutoff < 0.85 * nyquist) {
            res_type = "kaiser_fast";
        }
        else {
            res_type = "kaiser_best";
        }
    }

    __early_downsample(y, sr, hop_length, res_type.c_str(), n_octaves, nyquist, filter_cutoff, scale);

    // Skip this block for now
    if (auto_resample && strcmp(res_type.c_str(), "kaiser_fast") != 0) {
        // Do the top octave before resampling to allow for fast resampling

    }

    int num_twos = __num_two_factors(hop_length);
    if (num_twos < n_octaves - 1) throw std::runtime_error("hop_length must be a positive integer ");

//    my_y, my_sr, my_hop = y, sr, hop_length
    auto my_y = y.clone();
    auto my_sr = sr;
    auto my_hop = hop_length;

    std::vector<nc::NDArrayCpxF32Ptr> vqt_resp(0);
    for (auto i = 0; i < n_octaves; i++) {
        if (i > 0) {
            if (len(my_y) < 2)  throw std::runtime_error("Input signal length is too short for CQT/VQT.");
            my_y = resample(my_y, 2, 1, res_type.c_str(), scale=true);
            my_sr = my_sr / 2;
            my_hop = my_hop / 2;
        }
        auto ret = __cqt_filter_fft(my_sr, fmin * std::pow(2.0, double(-i)), n_filters, bins_per_octave, filter_scale, norm, sparsity, -1, window, gamma);
        nc::NDArrayCpxF32Ptr fft_basis = ret.fft_basis;
        int n_fft = ret.n_fft;
        fft_basis *= std::sqrt(std::pow(2, i));
        vqt_resp.push_back(__cqt_response(my_y, n_fft, my_hop, fft_basis, filters::STFTWindowType::Ones, __pad_mode));
    }

    auto V = __trim_stack(vqt_resp, n_bins);

    if (scale) {
        auto lengths = filters::constant_q_lengths(sr, fmin, n_bins, bins_per_octave, window, filter_scale, gamma);
        for (auto i = 0; i < lengths.elemCount(); i++) {
            auto ptr_V_i = V.at(i, 0);
            float scale = 1. / std::sqrt(lengths.getitem(i));
            for (auto j = 0; j < V.shape()[1]; j++) {
                ptr_V_i[j] *= scale;
            }
        }
    }

    return V;
}

nc::NDArrayCpxF32Ptr cqt(
        const nc::NDArrayF32Ptr&        __y,
        const float                     __sr,
        const int                       __hop_length,
        const float                     __fmin,
        const int                       __n_bins,
        const int                       __bins_per_octave,
        const float                     __tuning,
        const float                     __filter_scale,
        const float                     __norm,
        const float                     __sparsity,
        const filters::STFTWindowType&  __window,
        const bool                      __scale,
        const char *                    __pad_mode,
        const char *                    __res_type
        ) {
    // CQT is the special case of VQT with gamma=0
    return vqt(__y,
               __sr,
               __hop_length,
               __fmin,
               __n_bins,
               0.0f,
               __bins_per_octave,
               __tuning,
               __filter_scale,
               __norm,
               __sparsity,
               __window,
               __scale,
               __pad_mode,
               __res_type);
}



} // namespace core
} // namespace rosacxx
