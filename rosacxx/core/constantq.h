#ifndef ROSACXX_CORE_CONSTANTQ_H
#define ROSACXX_CORE_CONSTANTQ_H

#include <stdio.h>

#include <util/ndarray.h>

namespace rosacxx {
namespace core {

/// Function: cqt
/// ----------
/// Parameters              | Type      | Note
/// @param y                | float *   | audio time series, shape = (n)
/// @param sr               | float     | sampling rate of ``y``
/// @param hop_length       | int       | number of samples between successive CQT columns.
/// @param fmin             | float     | Minimum frequency. Defaults to `C1 ~= 32.70 Hz`
/// @param n_bins           | int       | Number of frequency bins, starting at ``fmin``
/// @param bins_per_octave  | int       | Number of bins per octave
/// @param tuning           | float     | Tuning offset in fractions of a bin. If ``0.0``, tuning will be automatically estimated from the signal. The minimum frequency of the resulting CQT will be modified to ``fmin * 2**(tuning / bins_per_octave)``
/// @param filter_scale     | float     | Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
/// @param norm             | float     | Type of norm to use for basis function normalization.  See `rosacxx::util::normalize`.
/// @param sparsity         | flaot     | Sparsify the CQT basis by discarding up to ``sparsity`` fraction of the energy in each basis. Set ``sparsity=0`` to disable sparsification.
/// @param window           | char *    | Window specification for the basis filters.  See `rosacxx::filters::get_window` for details.
/// @param scale            | bool      | If ``True``, scale the CQT response by square-root the length of each channel's filter.  This is analogous to ``norm='ortho'`` in FFT. If ``False``, do not scale the CQT. This is analogous to ``norm=None`` in FFT.
/// @param pad_mode         | char *    | Padding mode for centered frame analysis.
/// @param res_type         | char *    | By default, `cqt` will adaptively select a resampling mode which trades off accuracy at high frequencies for efficiency at low frequencies.
/// @param dtype            | char *    | The (complex) data type of the output array.
/// ----------
/// Returns                 | Type      | Note
/// @result CQT             | float *   | Constant-Q value each frequency at each time.
void cqt(
        const float * y = NULL,
        const float sr=22050,
        const int hop_length=512,
        const float fmin=0,
        const int n_bins=84,
        const int bins_per_octave=12,
        const float tuning=0.0,
        const float filter_scale=1,
        const float norm=1,
        const float sparsity=0.01,
        const char * window="hann",
        const bool scale=true,
        const char * pad_mode="reflect",
        const char * res_type="",
        const char * dtype="",
        float * CQT = NULL
        );

/// Function: hybrid_cqt
/// ----------
/// Parameters              | Type      | Note
/// @param y                | float *   | audio time series, shape = (n)
/// @param sr               | float     | sampling rate of ``y``
/// @param hop_length       | int       | number of samples between successive CQT columns.
/// @param fmin             | float     | Minimum frequency. Defaults to `C1 ~= 32.70 Hz`
/// @param n_bins           | int       | Number of frequency bins, starting at ``fmin``
/// @param bins_per_octave  | int       | Number of bins per octave
/// @param tuning           | float     | Tuning offset in fractions of a bin. If ``0.0``, tuning will be automatically estimated from the signal. The minimum frequency of the resulting CQT will be modified to ``fmin * 2**(tuning / bins_per_octave)``
/// @param filter_scale     | float     | Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
/// @param norm             | float     | Type of norm to use for basis function normalization.  See `rosacxx::util::normalize`.
/// @param sparsity         | flaot     | Sparsify the CQT basis by discarding up to ``sparsity`` fraction of the energy in each basis. Set ``sparsity=0`` to disable sparsification.
/// @param window           | char *    | Window specification for the basis filters.  See `rosacxx::filters::get_window` for details.
/// @param scale            | bool      | If ``True``, scale the CQT response by square-root the length of each channel's filter.  This is analogous to ``norm='ortho'`` in FFT. If ``False``, do not scale the CQT. This is analogous to ``norm=None`` in FFT.
/// @param pad_mode         | char *    | Padding mode for centered frame analysis.
/// @param res_type         | char *    | By default, `cqt` will adaptively select a resampling mode which trades off accuracy at high frequencies for efficiency at low frequencies.
/// @param dtype            | char *    | The (complex) data type of the output array.
/// ----------
/// Returns                 | Type      | Note
/// @result CQT             | float *   | Constant-Q value each frequency at each time.
void hybrid_cqt(
        const float * y = NULL,
        const float sr=22050,
        const int hop_length=512,
        const float fmin=0,
        const int n_bins=84,
        const int bins_per_octave=12,
        const float tuning=0.0,
        const float filter_scale=1,
        const float norm=1,
        const float sparsity=0.01,
        const char * window="hann",
        const bool scale=true,
        const char * pad_mode="reflect",
        const char * res_type="",
        const char * dtype="",
        float * CQT = NULL
        );

} // namespace core
} // namespace rosacxx

#endif
