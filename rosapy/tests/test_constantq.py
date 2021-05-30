#!/usr/bin/env python
"""
CREATED:2015-03-01 by Eric Battenberg <ebattenberg@gmail.com>
unit tests for librosa core.constantq
"""
from __future__ import division

import warnings

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

import librosa
from librosa import audio
from librosa.core.fft import get_fftlib
from librosa.core.convert import cqt_frequencies, note_to_hz, hz_to_midi, hz_to_octs, fft_frequencies
from librosa.core.spectrum import stft, istft, _spectrogram
from librosa.core.pitch import estimate_tuning
from librosa import filters
from librosa import util

import numpy as np

import pytest

from .test_core import srand

import resampy
import scipy

def _pitch_tuning(frequencies, resolution=0.01, bins_per_octave=12):
    
    frequencies = np.atleast_1d(frequencies)

    # Trim out any DC components
    frequencies = frequencies[frequencies > 0]

    if not np.any(frequencies):
        warnings.warn("Trying to estimate tuning from empty frequency set.")
        return 0.0

    # Compute the residual relative to the number of bins
    residual = np.mod(bins_per_octave * hz_to_octs(frequencies), 1.0)

    # Are we on the wrong side of the semitone?
    # A residual of 0.95 is more likely to be a deviation of -0.05
    # from the next tone up.
    residual[residual >= 0.5] -= 1.0

    bins = np.linspace(-0.5, 0.5, int(np.ceil(1.0 / resolution)) + 1)

    counts, tuning = np.histogram(residual, bins)

    # return the histogram peak
    return tuning[np.argmax(counts)]

def _piptrack(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=None,
    fmin=150.0,
    fmax=4000.0,
    threshold=0.1,
    win_length=None,
    window="hann",
    center=True,
    pad_mode="reflect",
    ref=None,
    ):

    # Check that we received an audio time series or STFT
    S, n_fft = _spectrogram(
        y=y,
        S=S,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )

    # Make sure we're dealing with magnitudes
    S = np.abs(S)

    # Truncate to feasible region
    fmin = np.maximum(fmin, 0)
    fmax = np.minimum(fmax, float(sr) / 2)

    fft_freqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])

    shift = 2 * S[1:-1] - S[2:] - S[:-2]

    import pdb; pdb.set_trace()

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (np.abs(shift) < util.tiny(shift)))

    # Pad back up to the same shape as S
    avg = np.pad(avg, ([1, 1], [0, 0]), mode="constant")
    shift = np.pad(shift, ([1, 1], [0, 0]), mode="constant")

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = ((fmin <= fft_freqs) & (fft_freqs < fmax)).reshape((-1, 1))

    # Compute the column-wise local max of S after thresholding
    # Find the argmax coordinates
    if ref is None:
        ref = np.max

    if callable(ref):
        ref_value = threshold * ref(S, axis=0)
    else:
        ref_value = np.abs(ref)

    idx = np.argwhere(freq_mask & util.localmax(S * (S > ref_value)))

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = (
        (idx[:, 0] + shift[idx[:, 0], idx[:, 1]]) * float(sr) / n_fft
    )

    mags[idx[:, 0], idx[:, 1]] = S[idx[:, 0], idx[:, 1]] + dskew[idx[:, 0], idx[:, 1]]

    return pitches, mags

def _estimate_tuning(
    y=None, sr=22050, S=None, n_fft=2048, resolution=0.01, bins_per_octave=12, **kwargs
    ):
    pitch, mag = _piptrack(y=y, sr=sr, S=S, n_fft=n_fft, **kwargs)
    import pdb; pdb.set_trace()

    # Only count magnitude where frequency is > 0
    pitch_mask = pitch > 0

    if pitch_mask.any():
        threshold = np.median(mag[pitch_mask])
    else:
        threshold = 0.0

    return _pitch_tuning(
        pitch[(mag >= threshold) & pitch_mask],
        resolution=resolution,
        bins_per_octave=bins_per_octave,
    )

def __make_signal(sr, duration, fmin="C1", fmax="C8"):
    """ Generates a linear sine sweep """

    if fmin is None:
        fmin = 0.01
    else:
        fmin = librosa.note_to_hz(fmin) / sr

    if fmax is None:
        fmax = 0.5
    else:
        fmax = librosa.note_to_hz(fmax) / sr

    return np.sin(
        np.cumsum(
            2
            * np.pi
            * np.logspace(np.log10(fmin), np.log10(fmax), num=int(duration * sr))
        )
    )

def __sparsify_rows(x, quantile=0.01, dtype=None):
    if x.ndim == 1:
        x = x.reshape((1, -1))

    elif x.ndim > 2:
        raise ParameterError(
            "Input must have 2 or fewer dimensions. "
            "Provided x.shape={}.".format(x.shape)
        )

    if not 0.0 <= quantile < 1:
        raise ParameterError("Invalid quantile {:.2f}".format(quantile))

    if dtype is None:
        dtype = x.dtype

    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=dtype)
    x_sparse_np = np.zeros(x.shape, dtype=dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]
        x_sparse_np[i, idx] = x[i, idx]

    return x_sparse.tocsr()

def __normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    
    # Avoid div-by-zero
    if threshold is None:
        threshold = util.utils.tiny(S)

    elif threshold <= 0:
        raise ParameterError(
            "threshold={} must be strictly " "positive".format(threshold)
        )

    if fill not in [None, False, True]:
        raise ParameterError("fill={} must be None or boolean".format(fill))

    if not np.all(np.isfinite(S)):
        raise ParameterError("Input must be finite")

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            raise ParameterError("Cannot normalize with norm=0 and fill=True")

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag ** norm, axis=axis, keepdims=True) ** (1.0 / norm)

        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    elif norm is None:
        return S

    else:
        raise ParameterError("Unsupported norm: {}".format(repr(norm)))

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm

def __constant_q(
    sr,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    window="hann",
    filter_scale=1,
    pad_fft=True,
    norm=1,
    dtype=np.complex64,
    gamma=0,
    **kwargs,
    ):

    if fmin is None:
        fmin = note_to_hz("C1")

    # Pass-through parameters to get the filter lengths
    lengths = librosa.filters.constant_q_lengths(
        sr,
        fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        window=window,
        filter_scale=filter_scale,
        gamma=gamma,
    )

    freqs = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.arange(-ilen // 2, ilen // 2, dtype=float) * 1j * 2 * np.pi * freq / sr;
        sig = np.exp(sig)

        # Apply the windowing function
        sig = sig * librosa.filters.__float_window(window)(len(sig))

        # Normalize
        sig = __normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0 ** (np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray(
        [util.pad_center(filt, max_len, **kwargs) for filt in filters], dtype=dtype
    )

    return filters, np.asarray(lengths)

def __resample(y, orig_sr, target_sr, res_type="kaiser_best", fix=True, scale=False, **kwargs):
    # First, validate the audio buffer
    util.valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    elif res_type == "polyphase":
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise ParameterError(
                "polyphase resampling is only supported for integer-valued sampling rates."
            )
        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    elif res_type in (
        "linear",
        "zero_order_hold",
        "sinc_best",
        "sinc_fastest",
        "sinc_medium",
        ):
        import samplerate
        # We have to transpose here to match libsamplerate
        y_hat = samplerate.resample(y.T, ratio, converter_type=res_type).T
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)

def __trim_stack(cqt_resp, n_bins, dtype):
    """Helper function to trim and stack a collection of CQT responses"""

    max_col = min(c_i.shape[-1] for c_i in cqt_resp)
    cqt_out = np.empty((n_bins, max_col), dtype=dtype, order="F")

    # Copy per-octave data into output array
    end = n_bins
    for c_i in cqt_resp:
        # By default, take the whole octave
        n_oct = c_i.shape[0]
        # If the whole octave is more than we can fit,
        # take the highest bins from c_i
        if end < n_oct:
            cqt_out[:end] = c_i[-end:, :max_col]
        else:
            cqt_out[end - n_oct : end] = c_i[:, :max_col]

        end -= n_oct

    return cqt_out

def __num_two_factors(x):
    """Return how many times integer x can be evenly divided by 2.

    Returns 0 for non-positive integers.
    """
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos

def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    """Compute the number of early downsampling operations"""

    downsample_count1 = max(
        0, int(np.ceil(np.log2(audio.BW_FASTEST * nyquist / filter_cutoff)) - 1) - 1
    )

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)

def __early_downsample(
    y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale
    ):
    """Perform early downsampling on an audio signal, if it applies."""

    downsample_count = __early_downsample_count(
        nyquist, filter_cutoff, hop_length, n_octaves
    )

    if downsample_count > 0 and res_type == "kaiser_fast":
        downsample_factor = 2 ** (downsample_count)

        hop_length //= downsample_factor

        if len(y) < downsample_factor:
            raise RuntimeError(
                "Input signal length={:d} is too short for "
                "{:d}-octave CQT".format(len(y), n_octaves)
            )

        new_sr = sr / float(downsample_factor)
        # y_hat_tmp1 = resampy.resample(y, sr, new_sr, filter=res_type, axis=-1)
        # y_hat_tmp2 = __resample(y, sr, new_sr, res_type=res_type, scale=True)
        y_hat = audio.resample(y, sr, new_sr, res_type=res_type, scale=True)

        # If we're not going to length-scale after CQT, we
        # need to compensate for the downsampling factor here
        y = y_hat
        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

    return y, sr, hop_length

def __cqt_filter_fft(
    sr,
    fmin,
    n_bins,
    bins_per_octave,
    filter_scale,
    norm,
    sparsity,
    hop_length=None,
    window="hann",
    gamma=0.0,
    dtype=np.complex,
    ):
    """Generate the frequency domain constant-Q filter basis."""

    basis, lengths = __constant_q(
        sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        filter_scale=filter_scale,
        norm=norm,
        pad_fft=True,
        window=window,
        gamma=gamma,
    )

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if hop_length is not None and n_fft < 2.0 ** (1 + np.ceil(np.log2(hop_length))):

        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft = get_fftlib()
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, : (n_fft // 2) + 1]

    # sparsify the basis
    fft_basis = __sparsify_rows(fft_basis, quantile=sparsity, dtype=dtype)

    return fft_basis, n_fft, lengths

def __cqt_response(y, n_fft, hop_length, fft_basis, mode, dtype=None):
    """Compute the filter response with a target STFT hop."""

    # Compute the STFT matrix
    D = stft(
        y, n_fft=n_fft, hop_length=hop_length, window="ones", pad_mode=mode, dtype=dtype
    )

    # And filter response energy
    return fft_basis.dot(D)

def __vqt(
    y,
    sr=22050,
    hop_length=512,
    fmin=None,
    n_bins=84,
    gamma=None,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="reflect",
    res_type=None,
    dtype=None,
    ):

    # How many octaves are we dealing with?
    n_octaves = int(np.ceil(float(n_bins) / bins_per_octave))
    n_filters = min(bins_per_octave, n_bins)

    len_orig = len(y)

    # Relative difference in frequency between any two consecutive bands
    alpha = 2.0 ** (1.0 / bins_per_octave) - 1

    if fmin is None:
        # C1 by default
        fmin = note_to_hz("C1")

    if tuning is None:
        tuning = _estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

    if gamma is None:
        gamma = 24.7 * alpha / 0.108

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Apply tuning correction
    fmin = fmin * 2.0 ** (tuning / bins_per_octave)

    # First thing, get the freqs of the top octave
    freqs = cqt_frequencies(n_bins, fmin, bins_per_octave=bins_per_octave)[
        -bins_per_octave:
    ]

    fmin_t = np.min(freqs)
    fmax_t = np.max(freqs)

    # Determine required resampling quality
    Q = float(filter_scale) / alpha
    filter_cutoff = (
        fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q) + 0.5 * gamma
    )
    nyquist = sr / 2.0

    auto_resample = False
    if not res_type:
        auto_resample = True
        if filter_cutoff < audio.BW_FASTEST * nyquist:
            res_type = "kaiser_fast"
        else:
            res_type = "kaiser_best"

    y, sr, hop_length = __early_downsample(
        y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale
    )

    vqt_resp = []

    # Skip this block for now
    if auto_resample and res_type != "kaiser_fast":

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = __cqt_filter_fft(
            sr,
            fmin_t,
            n_filters,
            bins_per_octave,
            filter_scale,
            norm,
            sparsity,
            window=window,
            gamma=gamma,
            dtype=dtype,
        )

        # Compute the VQT filter response and append it to the stack
        vqt_resp.append(
            __cqt_response(y, n_fft, hop_length, fft_basis, pad_mode, dtype=dtype)
        )

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * filters.window_bandwidth(window) / Q)

        res_type = "kaiser_fast"

    # Make sure our hop is long enough to support the bottom octave
    num_twos = __num_two_factors(hop_length)
    if num_twos < n_octaves - 1:
        raise RuntimeError(
            "hop_length must be a positive integer "
            "multiple of 2^{0:d} for {1:d}-octave CQT/VQT".format(
                n_octaves - 1, n_octaves
            )
        )

    # Now do the recursive bit
    my_y, my_sr, my_hop = y, sr, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):
        # Resample (except first time)
        if i > 0:
            if len(my_y) < 2:
                raise RuntimeError(
                    "Input signal length={} is too short for "
                    "{:d}-octave CQT/VQT".format(len_orig, n_octaves)
                )

            my_y = audio.resample(my_y, 2, 1, res_type=res_type, scale=True)

            my_sr /= 2.0
            my_hop //= 2

        fft_basis, n_fft, _ = __cqt_filter_fft(
            my_sr,
            fmin_t * 2.0 ** -i,
            n_filters,
            bins_per_octave,
            filter_scale,
            norm,
            sparsity,
            window=window,
            gamma=gamma,
            dtype=dtype,
        )
        # Re-scale the filters to compensate for downsampling
        fft_basis[:] *= np.sqrt(2 ** i)

        # Compute the vqt filter response and append to the stack
        vqt_resp.append(
            __cqt_response(my_y, n_fft, my_hop, fft_basis, pad_mode, dtype=dtype)
        )

    V = __trim_stack(vqt_resp, n_bins, dtype)

    if scale:
        lengths = filters.constant_q_lengths(
            sr,
            fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            window=window,
            filter_scale=filter_scale,
            gamma=gamma,
        )
        V /= np.sqrt(lengths[:, np.newaxis])

    return V

def __cqt(
    y,
    sr=22050,
    hop_length=512,
    fmin=None,
    n_bins=84,
    bins_per_octave=12,
    tuning=0.0,
    filter_scale=1,
    norm=1,
    sparsity=0.01,
    window="hann",
    scale=True,
    pad_mode="reflect",
    res_type=None,
    dtype=None,
    ):

    # CQT is the special case of VQT with gamma=0
    return __vqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        gamma=0,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        window=window,
        scale=scale,
        pad_mode=pad_mode,
        res_type=res_type,
        dtype=dtype,
    )

def __cq_to_chroma(
    n_input,
    bins_per_octave=12,
    n_chroma=12,
    fmin=None,
    window=None,
    base_c=True,
    dtype=np.float32,
    ):

    # How many fractional bins are we merging?
    n_merge = float(bins_per_octave) / n_chroma

    if fmin is None:
        fmin = note_to_hz("C1")

    if np.mod(n_merge, 1) != 0:
        raise ParameterError(
            "Incompatible CQ merge: "
            "input bins must be an "
            "integer multiple of output bins."
        )

    # Tile the identity to merge fractional bins
    cq_to_ch = np.repeat(np.eye(n_chroma), n_merge, axis=1)

    # Roll it left to center on the target bin
    cq_to_ch = np.roll(cq_to_ch, -int(n_merge // 2), axis=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(np.float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]

    # What's the note number of the first bin in the CQT?
    # midi uses 12 bins per octave here
    midi_0 = np.mod(hz_to_midi(fmin), 12)

    if base_c:
        # rotate to C
        roll = midi_0
    else:
        # rotate to A
        roll = midi_0 - 9

    # Adjust the roll in terms of how many chroma we want out
    # We need to be careful with rounding here
    roll = int(np.round(roll * (n_chroma / 12.0)))

    # Apply the roll
    cq_to_ch = np.roll(cq_to_ch, roll, axis=0).astype(dtype)

    if window is not None:
        cq_to_ch = scipy.signal.convolve(cq_to_ch, np.atleast_2d(window), mode="same")

    return cq_to_ch

def __chroma_cqt(
    y=None,
    sr=22050,
    C=None,
    hop_length=512,
    fmin=None,
    norm=np.inf,
    threshold=0.0,
    tuning=None,
    n_chroma=12,
    n_octaves=7,
    window=None,
    bins_per_octave=36,
    cqt_mode="full",
    ):

    cqt_func = {"full": __cqt, "hybrid": __cqt}

    if bins_per_octave is None:
        bins_per_octave = n_chroma
    elif np.remainder(bins_per_octave, n_chroma) != 0:
        raise ParameterError(
            "bins_per_octave={} must be an integer "
            "multiple of n_chroma={}".format(bins_per_octave, n_chroma)
        )

    # Build the CQT if we don't have one already
    if C is None:
        C = np.abs(
            cqt_func[cqt_mode](
                y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_octaves * bins_per_octave,
                bins_per_octave=bins_per_octave,
                tuning=tuning,
            )
        )

    # Map to chroma
    cq_to_chr = __cq_to_chroma(
        C.shape[0],
        bins_per_octave=bins_per_octave,
        n_chroma=n_chroma,
        fmin=fmin,
        window=window,
    )
    chroma = cq_to_chr.dot(C)

    if threshold is not None:
        chroma[chroma < threshold] = 0.0

    # Normalize
    if norm is not None:
        chroma = util.normalize(chroma, norm=norm, axis=0)

    return chroma

# <gen_test_vqt_data/>
def gen_test_vqt_data():

    sr = 11025
    fmin = None
    n_bins = 12
    gamma = None
    bins_per_octave = 12
    tuning = 0
    filter_scale = 1
    norm = 1
    res_type = None
    sparsity = 0.01
    hop_length = 512

    y = __make_signal(sr, 2.0)

    C = __vqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        gamma=gamma,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        res_type=res_type,
        )

    contents = """
    #ifndef tests_data_vqt
    #define tests_data_vqt

    constexpr float  ROSACXXTest_vqt_sr = {0};
    constexpr int    ROSACXXTest_vqt_y_len = {1};
    constexpr double ROSACXXTest_vqt_y_dat[ROSACXXTest_vqt_y_len] = {{ {2} }};
    
    constexpr int    ROSACXXTest_vqt_C_dims = {3};
    constexpr int    ROSACXXTest_vqt_C_shape[ROSACXXTest_vqt_C_dims] = {{ {4} }};
    constexpr double ROSACXXTest_vqt_C_real_dat[{5}] = {{ {6} }};
    constexpr double ROSACXXTest_vqt_C_imag_dat[{5}] = {{ {7} }};

    #endif // tests_data_vqt
    """.format(
        sr, # 0
        y.size, # 1
        str(y.tolist()).replace('[', '').replace(']', ''), # 2
        len(C.shape), # 3
        str(C.shape).replace('(', '').replace(')', '').replace('[', '').replace(']', ''), # 4
        C.size, # 5
        str(C.real.reshape(-1).tolist()).replace('[', '').replace(']', ''), # 6
        str(C.imag.reshape(-1).tolist()).replace('[', '').replace(']', ''), # 7
        )
    with open("/home/yuda/Documents/Projects/raspai/song_scoring/heart/scorpio/3rd/librosacxx/tests/tests_data_vqt.h", "w+") as fp:
        fp.write(contents)
    # end-with

    return True
# </gen_test_vqt_data>

# <gen_test_cqt_data/>
def gen_test_cqt_data():

    sr = 11025
    fmin = None
    n_bins = 12
    bins_per_octave = 12
    tuning = 0
    filter_scale = 1
    norm = 1
    res_type = None
    sparsity = 0.01
    hop_length = 512

    y = __make_signal(sr, 60.0)

    V = __vqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        gamma=0.0,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        res_type=res_type,
        )
    
    # C = librosa.cqt(
    #     y=y,
    #     sr=sr,
    #     hop_length=hop_length,
    #     fmin=fmin,
    #     n_bins=n_bins,
    #     bins_per_octave=bins_per_octave,
    #     tuning=tuning,
    #     filter_scale=filter_scale,
    #     norm=norm,
    #     sparsity=sparsity,
    #     res_type=res_type,
    #     )
    # print(np.abs(C-V).max())
    C = V

    contents = """
    #ifndef tests_data_cqt
    #define tests_data_cqt

    constexpr float  ROSACXXTest_cqt_sr = {0};
    constexpr int    ROSACXXTest_cqt_y_len = {1};
    constexpr double ROSACXXTest_cqt_y_dat[ROSACXXTest_cqt_y_len] = {{ {2} }};
    
    constexpr int    ROSACXXTest_cqt_C_dims = {3};
    constexpr int    ROSACXXTest_cqt_C_shape[ROSACXXTest_cqt_C_dims] = {{ {4} }};
    constexpr double ROSACXXTest_cqt_C_real_dat[{5}] = {{ {6} }};
    constexpr double ROSACXXTest_cqt_C_imag_dat[{5}] = {{ {7} }};

    #endif // tests_data_cqt
    """.format(
        sr, # 0
        y.size, # 1
        str(y.tolist()).replace('[', '').replace(']', ''), # 2
        len(C.shape), # 3
        str(C.shape).replace('(', '').replace(')', '').replace('[', '').replace(']', ''), # 4
        C.size, # 5
        str(C.real.reshape(-1).tolist()).replace('[', '').replace(']', ''), # 6
        str(C.imag.reshape(-1).tolist()).replace('[', '').replace(']', ''), # 7
        )
    with open("/home/yuda/Documents/Projects/raspai/song_scoring/heart/scorpio/3rd/librosacxx/tests/tests_data_cqt.h", "w+") as fp:
        fp.write(contents)
    # end-with

    return True
# </gen_test_cqt_data>

# <gen_test_chroma_cqt_data/>
def gen_test_chroma_cqt_data(
    song_path = "/home/yuda/Documents/Datasets/heshan/set02_20210331/professional/spleeter_outputs/小白船/vocals.wav", 
    # song_path = "D:\DataSets\set02_20210331\professional\spleeter_outputs\小白船\\vocals.wav",
    sr = 22050, 
    hop_length = 1024
    ):
    
    y, sr = librosa.load(
        song_path,
        sr = sr,
        mono = True
        )
    y = y[int(sr*20):int(sr*30)]
    print("y.shape = ", y.shape)
    print("sr = ", sr)
    
    chroma = __chroma_cqt(
        y=y, sr=sr, hop_length=hop_length)
    print("chroma.shape = ", chroma.shape)

    contents = """
    #ifndef tests_data_chroma_cqt
    #define tests_data_chroma_cqt
    namespace tests {{
    namespace ROSACXXTest {{
    namespace chroma_cqt {{
    // -------- //
    constexpr float y_sr = {0};
    constexpr int   y_len = {1};
    constexpr float y_dat[y_len] = {2};
    constexpr int   hop_lenght = {3};
    constexpr int   chroma_dims = {4};
    constexpr int   chroma_shape[chroma_dims] = {5};
    constexpr float chroma_dat[{6}] = {7};
    // -------- //
    }}
    }}  
    }}
    #endif // tests_data_chroma_cqt
    """.format(
        sr, # 0
        y.size, # 1
        str(y.tolist()).replace('[', '{').replace(']', '}'), # 2
        hop_length, # 3
        len(chroma.shape), # 4
        str(list(chroma.shape)).replace('[', '{').replace(']', '}'), # 5
        chroma.size, # 6
        str(chroma.reshape(-1).tolist()).replace('[', '{').replace(']', '}'), # 7
        )
    with open(
        # "/home/yuda/Documents/Projects/raspai/song_scoring/heart/scorpio/3rd/librosacxx/tests/tests_data_chroma_cqt.h", 
        "./tests/tests_data_chroma_cqt.h", 
        "w+") as fp:
        fp.write(contents)
    # end-with
    
# </gen_test_chroma_cqt_data>

if __name__ == '__main__':
    # python -m rosapy.tests.test_constantq
    gen_test_chroma_cqt_data()
    

    
    