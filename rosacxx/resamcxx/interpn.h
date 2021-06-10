#ifndef RESAMCXX_INTERPN_H
#define RESAMCXX_INTERPN_H

#include <rosacxx/numcxx/ndarray.h>
#include <rosacxx/numcxx/numcxx.h>

namespace resam {

template <typename DType>
inline void resample_f(
        const nc::NDArrayPtr<DType>& x,
        nc::NDArrayPtr<DType>& y,
        const float& sample_ratio,
        const nc::NDArrayPtr<DType>& interp_win,
        const nc::NDArrayPtr<DType>& interp_delta,
        const int& num_table
        ) {
    float scale = std::min(1.f, sample_ratio);
    float time_increment = 1./sample_ratio;
    int index_step = int(scale * num_table);
    float time_register = 0.0;

    int n = 0;
    float frac = 0.0;
    float index_frac = 0.0;
    int offset = 0;
    float eta = 0.0;
    float weight = 0.0;

    int nwin = interp_win.shape()[0];
    int n_orig = x.shape()[0];
    int n_out = y.shape()[0];
    int n_channels = y.shape()[1];

    if (n_channels == 1) {
        DType * ptr_y = y.data();
        DType * ptr_x = x.data();
        DType * ptr_interp_win = interp_win.data();
        DType * ptr_interp_delta = interp_delta.data();

        #pragma omp parallel for
        for (int t = 0; t < n_out; t++) {
            time_register = t * time_increment;
            n = int(time_register);
            frac = scale * (time_register - n);
            index_frac = frac * num_table;
            offset = int(index_frac);
            eta = index_frac - offset;
            int i_max = std::min(n + 1, (nwin - offset) / index_step);
            for (int i = 0; i < i_max; i++) {
                int idx = offset + i * index_step;
                // float weight = interp_win.getitem(idx) + eta * interp_delta.getitem(idx);
                float weight = ptr_interp_win[idx] + eta * ptr_interp_delta[idx];
                ptr_y[t] += weight * ptr_x[n - i];
            }
            frac = scale - frac;
            index_frac = frac * num_table;
            offset = int(index_frac);
            eta = index_frac - offset;
            int k_max = std::min(n_orig - n - 1, (nwin - offset)/index_step);
            for (int k = 0; k < k_max; k++) {
                int idx = offset + k * index_step;
                // float weight = interp_win.getitem(idx) + eta * interp_delta.getitem(idx);
                float weight = ptr_interp_win[idx] + eta * ptr_interp_delta[idx];
                ptr_y[t] += weight * ptr_x[n + k + 1];
            }
        }
    }
    else {
        for (int t = 0; t < n_out; t++) {
            time_register = t * time_increment;
            n = int(time_register);
            frac = scale * (time_register - n);
            index_frac = frac * num_table;
            offset = int(index_frac);
            eta = index_frac - offset;
            int i_max = std::min(n + 1, (nwin - offset) / index_step);
            for (int i = 0; i < i_max; i++) {
                int idx = offset + i * index_step;
                float weight = interp_win.getitem(idx) + eta * interp_delta.getitem(idx);
                for (int j = 0; j < n_channels; j++) {
                    *y.at(t, j) += weight * x.getitem(n - i, j);
                }
            }
            frac = scale - frac;
            index_frac = frac * num_table;
            offset = int(index_frac);
            eta = index_frac - offset;
            int k_max = std::min(n_orig - n - 1, (nwin - offset)/index_step);
            for (int k = 0; k < k_max; k++) {
                int idx = offset + k * index_step;
                float weight = interp_win.getitem(idx) + eta * interp_delta.getitem(idx);
                for (int j = 0; j < n_channels; j++) {
                    *y.at(t, j) += weight * x.getitem(n + k + 1, j);
                }
            }
        }
    }

}

} // namespace resam
#endif /* RESAMCXX_INTERPN_H */
