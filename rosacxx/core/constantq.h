#ifndef ROSACXX_CORE_CONSTANTQ_H
#define ROSACXX_CORE_CONSTANTQ_H

#include <stdio.h>

#include <rosacxx/numcxx/numcxx.h>
#include <rosacxx/core/pitch.h>

namespace rosacxx {
namespace core {

/// Function: cqt
/// ----------
/// Parameters              | Type          | Note
/// @param y                | NDArrayF32Ptr | audio time series, shape = (n)
/// @param sr               | float         | sampling rate of ``y``
/// @param hop_length       | int           | number of samples between successive CQT columns.
/// @param fmin             | float         | Minimum frequency. Defaults to `C1 ~= 32.70 Hz`
/// @param n_bins           | int           | Number of frequency bins, starting at ``fmin``
/// @param bins_per_octave  | int           | Number of bins per octave
/// @param tuning           | float         | Tuning offset in fractions of a bin. If ``0.0``, tuning will be automatically estimated from the signal. The minimum frequency of the resulting CQT will be modified to ``fmin * 2**(tuning / bins_per_octave)``
/// @param filter_scale     | float         | Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
/// @param norm             | float         | Type of norm to use for basis function normalization.  See `rosacxx::util::normalize`.
/// @param sparsity         | flaot         | Sparsify the CQT basis by discarding up to ``sparsity`` fraction of the energy in each basis. Set ``sparsity=0`` to disable sparsification.
/// @param window           | char *        | Window specification for the basis filters.  See `rosacxx::filters::get_window` for details.
/// @param scale            | bool          | If ``True``, scale the CQT response by square-root the length of each channel's filter.  This is analogous to ``norm='ortho'`` in FFT. If ``False``, do not scale the CQT. This is analogous to ``norm=None`` in FFT.
/// @param pad_mode         | char *        | Padding mode for centered frame analysis.
/// @param res_type         | char *        | By default, `cqt` will adaptively select a resampling mode which trades off accuracy at high frequencies for efficiency at low frequencies.
/// ----------
/// Returns                 | Type          | Note
/// @result CQT             | NDArrayF32::Ptr | Constant-Q value each frequency at each time.
nc::NDArrayCpxF32Ptr cqt(
        const nc::NDArrayF32Ptr&        __y,
        const float                     __sr = 22050,
        const int                       __hop_length = 512,
        const float                     __fmin = INFINITY,
        const int                       __n_bins = 84,
        const int                       __bins_per_octave = 12,
        const float                     __tuning = 0,
        const float                     __filter_scale = 1,
        const float                     __norm =1 ,
        const float                     __sparsity = 1e-2,
        const filters::STFTWindowType&  __window  = filters::STFTWindowType::Hanning,
        const bool                      __scale = true,
        const char *                    __pad_mode = "reflect",
        const char *                    __res_type = NULL
        );

/// Function: hybrid_cqt
/// ----------
/// Parameters              | Type          | Note
/// @param y                | NDArrayF32Ptr | audio time series, shape = (n)
/// @param sr               | float         | sampling rate of ``y``
/// @param hop_length       | int           | number of samples between successive CQT columns.
/// @param fmin             | float         | Minimum frequency. Defaults to `C1 ~= 32.70 Hz`
/// @param n_bins           | int           | Number of frequency bins, starting at ``fmin``
/// @param bins_per_octave  | int           | Number of bins per octave
/// @param tuning           | float         | Tuning offset in fractions of a bin. If ``0.0``, tuning will be automatically estimated from the signal. The minimum frequency of the resulting CQT will be modified to ``fmin * 2**(tuning / bins_per_octave)``
/// @param filter_scale     | float         | Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
/// @param norm             | float         | Type of norm to use for basis function normalization.  See `rosacxx::util::normalize`.
/// @param sparsity         | flaot         | Sparsify the CQT basis by discarding up to ``sparsity`` fraction of the energy in each basis. Set ``sparsity=0`` to disable sparsification.
/// @param window           | char *        | Window specification for the basis filters.  See `rosacxx::filters::get_window` for details.
/// @param scale            | bool          | If ``True``, scale the CQT response by square-root the length of each channel's filter.  This is analogous to ``norm='ortho'`` in FFT. If ``False``, do not scale the CQT. This is analogous to ``norm=None`` in FFT.
/// @param pad_mode         | char *        | Padding mode for centered frame analysis.
/// @param res_type         | char *        | By default, `cqt` will adaptively select a resampling mode which trades off accuracy at high frequencies for efficiency at low frequencies.
/// ----------
/// Returns                 | Type          | Note
/// @result CQT             | NDArrayF32::Ptr | Constant-Q value each frequency at each time.
nc::NDArrayCpxF32Ptr hybrid_cqt(
        const nc::NDArrayF32Ptr&        __y,
        const float                     __sr = 22050,
        const int                       __hop_length = 512,
        const float                     __fmin = INFINITY,
        const int                       __n_bins = 84,
        const int                       __bins_per_octave = 12,
        const float                     __tuning = 0,
        const float                     __filter_scale = 1,
        const float                     __norm =1 ,
        const float                     __sparsity = 1e-2,
        const filters::STFTWindowType&  __window  = filters::STFTWindowType::Hanning,
        const bool                      __scale = true,
        const char *                    __pad_mode = "reflect",
        const char *                    __res_type = NULL
        );


/// Function: vqt
/// ----------
/// Parameters              | Type          | Note
/// @param y                | NDArrayF32Ptr | audio time series, shape = (n)
/// @param sr               | float         | sampling rate of ``y``
/// @param hop_length       | int           | number of samples between successive CQT columns.
/// @param fmin             | float         | Minimum frequency. Defaults to `C1 ~= 32.70 Hz`
/// @param n_bins           | int           | Number of frequency bins, starting at ``fmin``
/// @param gamma            | bool          | Bandwidth offset for determining filter lengths.
/// @param bins_per_octave  | int           | Number of bins per octave
/// @param tuning           | float         | Tuning offset in fractions of a bin. If ``0.0``, tuning will be automatically estimated from the signal. The minimum frequency of the resulting CQT will be modified to ``fmin * 2**(tuning / bins_per_octave)``
/// @param filter_scale     | float         | Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
/// @param norm             | float         | Type of norm to use for basis function normalization.  See `rosacxx::util::normalize`.
/// @param sparsity         | flaot         | Sparsify the CQT basis by discarding up to ``sparsity`` fraction of the energy in each basis. Set ``sparsity=0`` to disable sparsification.
/// @param window           | char *        | Window specification for the basis filters.  See `rosacxx::filters::get_window` for details.
/// @param scale            | bool          | If ``True``, scale the CQT response by square-root the length of each channel's filter.  This is analogous to ``norm='ortho'`` in FFT. If ``False``, do not scale the CQT. This is analogous to ``norm=None`` in FFT.
/// @param pad_mode         | char *        | Padding mode for centered frame analysis.
/// @param res_type         | char *        | By default, `cqt` will adaptively select a resampling mode which trades off accuracy at high frequencies for efficiency at low frequencies.
/// ----------
/// Returns                 | Type          | Note
/// @result CQT             | NDArrayF32::Ptr | Constant-Q value each frequency at each time.
nc::NDArrayCpxF32Ptr vqt(
        const nc::NDArrayF32Ptr&        __y,
        const float                     __sr = 22050,
        const int                       __hop_length = 512,
        const float                     __fmin = INFINITY,
        const int                       __n_bins = 84,
        const float                     __gamma = INFINITY,
        const int                       __bins_per_octave = 12,
        const float                     __tuning = 0,
        const float                     __filter_scale = 1,
        const float                     __norm =1 ,
        const float                     __sparsity = 1e-2,
        const filters::STFTWindowType&  __window  = filters::STFTWindowType::Hanning,
        const bool                      __scale = true,
        const char *                    __pad_mode = "reflect",
        const char *                    __res_type = NULL
        );

} // namespace core
} // namespace rosacxx

#endif
