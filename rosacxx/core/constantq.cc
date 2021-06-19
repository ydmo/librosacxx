#include <rosacxx/rosacxx.h>
#include <rosacxx/core/constantq.h>
#include <rosacxx/core/convert.h>
#include <rosacxx/core/audio.h>
#include <rosacxx/core/spectrum.h>
#include <rosacxx/filters.h>
#include <rosacxx/fft/kiss/kiss_fft.h>

#include <cmath>

#if ROSACXX_TEST
#   include <rosacxx/util/visualize.h>
#endif

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

    fft_basis = util::sparsify_rows(fft_basis, sparsity);

    RetCQTFilterFFT ret;
    ret.fft_basis = fft_basis;
    ret.n_fft = n_fft;
    ret.lengths = lengths;

    return ret;
}

#if __SSE__
nc::NDArrayCpxF32Ptr matmul_cpxf32_sse(const nc::NDArrayCpxF32Ptr& A, const nc::NDArrayCpxF32Ptr& B) {

    // Complex A = a + bj
    // Complex B = c + dj
    // A * B = (a*c) + (a*d)j + (b*c)j - (b*d) = (a*c-b*d) + (a*d+b*c)j

    //    float arr[4] = {0.f, 1.f, 2.f, 3.f};
    //    __m128 row0_ = _mm_loadu_ps(arr);

    //    __m128 row0 = _mm_setr_ps( 0.f,  1.f,  2.f,  3.f);
    //    __m128 row1 = _mm_setr_ps( 4.f,  5.f,  6.f,  7.f);
    //    __m128 row2 = _mm_setr_ps( 8.f,  9.f, 10.f, 11.f);
    //    __m128 row3 = _mm_setr_ps(12.f, 13.f, 14.f, 15.f);

    //    __m128 tmp3, tmp2, tmp1, tmp0;

    //    tmp0 = _mm_unpacklo_ps(row0, row1);
    //    tmp2 = _mm_unpacklo_ps(row2, row3);
    //    tmp1 = _mm_unpackhi_ps(row0, row1);
    //    tmp3 = _mm_unpackhi_ps(row2, row3);
    //    row0 = _mm_movelh_ps(tmp0, tmp2);
    //    row1 = _mm_movehl_ps(tmp2, tmp0);
    //    row2 = _mm_movelh_ps(tmp1, tmp3);
    //    row3 = _mm_movehl_ps(tmp3, tmp1);

    //    for(int r = 0; r < M; ++r) {
    //        std::complex<float> * ptr_A = A.data() + r * K;
    //        int offsetCr = r * N;
    //        std::complex<float> * ptr_B = B.data();
    //        for(int k = 0; k < K; ++k){
    //            std::complex<float> * ptr_C = C.data() + offsetCr;
    //            for(int n = 0; n < N; ++n){
    //                *ptr_C++ += *ptr_A * *ptr_B++;
    //            }
    //            ptr_A += 1;
    //        }
    //    }

    assert(A.shape(1) == B.shape(0));

    const int M = A.shape(0);
    const int K = A.shape(1);
    const int N = B.shape(1);

    nc::NDArrayCpxF32Ptr C = nc::NDArrayCpxF32Ptr(new nc::NDArray<std::complex<float>>({M, N}));

    int Kdiv4 = 0; // K / 4;
    int Kmod4 = K; // K % 4;

    int Ndiv4 = N / 4;
    int Nmod4 = N % 4;

#   pragma omp parallel for
    for (int r = 0; r < M; r++) {

        std::complex<float> * ptr_A = A.data() + r * K;
        int offsetCr = r * N;
        std::complex<float> * ptr_B = B.data();

        __m128 v4f_res = _mm_set1_ps(0.f);

        for (int k = 0; k < Kdiv4; k++) {

            for (int n = 0; n < Ndiv4; n++) {

            }

            for (int n = 0; n < Nmod4; n++) {
                // Complex A = a + bj
                // Complex B = c + dj
                // A * B = (a*c) + (a*d)j + (b*c)j - (b*d) = (a*c-b*d) + (a*d+b*c)j
            }

        }

        for (int k = 0; k < Kmod4; k++) {

            std::complex<float> * ptr_C = C.data() + offsetCr;

            float realA = ptr_A->real();
            float imagA = ptr_A->imag();

            const __m128 v4f_realA = _mm_set1_ps(realA);
            const __m128 v4f_imagA = _mm_set1_ps(imagA);

            for (int n = 0; n < Ndiv4; n++) {
                __m128 v4f_Br0i0r1i1 = _mm_loadu_ps( (float *)ptr_B     );
                __m128 v4f_Br2i2r3i3 = _mm_loadu_ps(((float *)ptr_B) + 4);

                __m128 v4f_Cr0i0r1i1 = _mm_loadu_ps( (float *)ptr_C     );
                __m128 v4f_Cr2i2r3i3 = _mm_loadu_ps(((float *)ptr_C) + 4);

                __m128 v4f_realB = _mm_shuffle_ps(v4f_Br0i0r1i1, v4f_Br2i2r3i3, 0x88); // 0b-10-00-10-00 // 2-0-2-0
                __m128 v4f_imagB = _mm_shuffle_ps(v4f_Br0i0r1i1, v4f_Br2i2r3i3, 0xdd); // 0b-11-01-11-01 // 3-1-3-1

                // __m128 tmp00 = _mm_mul_ps(v4f_realA, v4f_realB);
                // __m128 tmp01 = _mm_mul_ps(v4f_imagA, v4f_imagB);
                // __m128 tmp10 = _mm_mul_ps(v4f_realA, v4f_imagB);
                // __m128 tmp11 = _mm_mul_ps(v4f_imagA, v4f_realB);
                // __m128 v4f_realC = _mm_sub_ps(tmp00, tmp01);
                // __m128 v4f_imagC = _mm_add_ps(tmp10, tmp11);

                __m128 v4f_realC = _mm_sub_ps(_mm_mul_ps(v4f_realA, v4f_realB), _mm_mul_ps(v4f_imagA, v4f_imagB));
                __m128 v4f_imagC = _mm_add_ps(_mm_mul_ps(v4f_realA, v4f_imagB), _mm_mul_ps(v4f_imagA, v4f_realB));

                __m128 v4f_Cr0i0r1i1_ = _mm_unpacklo_ps(v4f_realC, v4f_imagC);
                __m128 v4f_Cr2i2r3i3_ = _mm_unpackhi_ps(v4f_realC, v4f_imagC);

                v4f_Cr0i0r1i1 = _mm_add_ps(v4f_Cr0i0r1i1, v4f_Cr0i0r1i1_);
                v4f_Cr2i2r3i3 = _mm_add_ps(v4f_Cr2i2r3i3, v4f_Cr2i2r3i3_);

                _mm_storeu_ps( (float *)ptr_C     , v4f_Cr0i0r1i1);
                _mm_storeu_ps(((float *)ptr_C) + 4, v4f_Cr2i2r3i3);

                ptr_C += 4;
                ptr_B += 4;
            }

            for (int n = 0; n < Nmod4; n++) {
                *ptr_C++ += *ptr_A * *ptr_B++;
            }

            ptr_A += 1;
        }

    }

    return C;
}
#endif // __SSE__

nc::NDArrayCpxF32Ptr __cqt_response(
        const nc::NDArrayF32Ptr& y,
        const int& n_fft,
        const int& hop_length,
        const nc::NDArrayCpxF32Ptr& fft_basis,
        const filters::STFTWindowType&  window = filters::STFTWindowType::Ones,
        const char* pad_mode = "reflect"
        ) {
    auto D = stft(y, n_fft, hop_length, -1, window, true, pad_mode);
#   if __SSE__
    auto res = matmul_cpxf32_sse(fft_basis, D);
    return res;
#   endif
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
                auto ptr_c_i = c_i.at(i_from, 0);
                for (auto c = 0; c < max_col; c++) {
                    ptr_cqt_out_i[c] = ptr_c_i[c];
                }
            }
        }
        else {
            for (auto r = 0; r < n_oct; r++) {
                auto i_to = r + end - n_oct;
                auto i_from = r;
                auto ptr_cqt_out_i = cqt_out.at(i_to, 0);
                auto ptr_c_i = c_i.at(i_from, 0);
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
    LOGTIC(vqt_prepare);
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
        fmin = note_to_hz<float>("C1");
    }
    LOGTOC(vqt_prepare);
    LOGTIC(vqt_estimate_tuning);
    float tuning = __tuning;
    if (tuning == INFINITY) {
        nc::NDArrayF32Ptr S = nullptr;
        tuning = estimate_tuning(__y, __sr, S, 2048, 0.01f, __bins_per_octave);// cost 65% time !!
        // printf("tuning = %f\n", double(tuning));
    }
    LOGTOC(vqt_estimate_tuning);

    LOGTIC(vqt_others_243to281);

    float gamma = __gamma;
    if (gamma == INFINITY) {
        gamma = 24.7 * alpha / 0.108;
    }

    fmin = fmin * std::pow(2.0, (tuning / bins_per_octave));

    auto freqs = cqt_frequencies(n_bins, fmin, bins_per_octave);
    auto vec_freqs = freqs.toStdVector1D();
    vec_freqs = std::vector<float>(vec_freqs.end()-bins_per_octave, vec_freqs.end());
    std::sort(vec_freqs.end()-bins_per_octave, vec_freqs.end());
    float fmin_t = vec_freqs[0];
    float fmax_t = vec_freqs[vec_freqs.size()-1];

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
        throw std::runtime_error("Not implemented.");
    }

    int num_twos = __num_two_factors(hop_length);
    if (num_twos < n_octaves - 1) throw std::runtime_error("hop_length must be a positive integer ");

//    my_y, my_sr, my_hop = y, sr, hop_length
    auto my_y = y.clone();
    auto my_sr = sr;
    auto my_hop = hop_length;

    LOGTOC(vqt_others_243to281);

    std::vector<nc::NDArrayCpxF32Ptr> vqt_resp(0);
    std::vector<std::string> names = {
        "vqt_resp[0]", "vqt_resp[1]", "vqt_resp[2]", "vqt_resp[3]", "vqt_resp[4]", "vqt_resp[5]", "vqt_resp[6]",
    };

    TimeMetrics tm0, tm1, tm2;

    for (auto i = 0; i < n_octaves; i++) {
        tm0.tic();
        if (i > 0) {
            if (len(my_y) < 2)  throw std::runtime_error("Input signal length is too short for CQT/VQT.");
            my_y = resample(my_y, 2, 1, res_type.c_str(), true, scale); // hot-spot 462
            my_sr = my_sr / 2;
            my_hop = my_hop / 2;
        }
        tm0.toc();

        tm1.tic();
        auto ret = __cqt_filter_fft(my_sr, fmin_t * std::pow(2.0, double(-i)), n_filters, bins_per_octave, filter_scale, norm, sparsity, -1, window, gamma); // hop-spot 102
        tm1.toc();

        tm2.tic();
        nc::NDArrayCpxF32Ptr fft_basis = ret.fft_basis;
        int n_fft = ret.n_fft;
        fft_basis *= std::sqrt(std::pow(2, i));
        vqt_resp.push_back(__cqt_response(my_y, n_fft, my_hop, fft_basis, filters::STFTWindowType::Ones, __pad_mode)); // hop-spot 4537
        tm2.toc();
    }

//    printf("[ROSACXX] Event[%s] cost %f ms\n", "vqt_mainblock_tm0", tm0.sum() * 1e3);
//    printf("[ROSACXX] Event[%s] cost %f ms\n", "vqt_mainblock_tm1", tm1.sum() * 1e3);
//    printf("[ROSACXX] Event[%s] cost %f ms\n", "vqt_mainblock_tm2", tm2.sum() * 1e3);

    LOGTIC(vqt___trim_stack);

    auto V = __trim_stack(vqt_resp, n_bins);

    LOGTOC(vqt___trim_stack);

    LOGTIC(vqt_scale);

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

    LOGTOC(vqt_scale);

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


void test_matmul_cpxf32_sse() {
    const int M = 9;
    const int K = 13;
    const int N = 18;
    nc::NDArrayCpxF32Ptr A = nc::NDArrayCpxF32Ptr(new nc::NDArray<std::complex<float>>({M, K}));
    nc::NDArrayCpxF32Ptr B = nc::NDArrayCpxF32Ptr(new nc::NDArray<std::complex<float>>({K, N}));
    for (auto k = 0; k < 1000; k++) {
        for (auto i = 0; i < A.elemCount(); i++) {
            *A.at(i) = std::complex<float>((k+1)*i*2, (k+1)*i*2+1);
        }
        for (auto i = 0; i < B.elemCount(); i++) {
            *B.at(i) = std::complex<float>((k+1)*i*2, (k+1)*i*2+1);
        }
        nc::NDArrayCpxF32Ptr C_pred = rosacxx::core::matmul_cpxf32_sse(A, B);
        nc::NDArrayCpxF32Ptr C_gt = matmul(A, B);
        assert(C_pred.elemCount() == C_gt.elemCount());
        for (auto i = 0; i < C_gt.elemCount(); i++) {
            assert(1e-9 > double(std::abs(C_pred.getitem(i)-C_gt.getitem(i))));
        }
    }
    printf("[TEST] test_matmul_cpxf32_sse ok.\n");
}
