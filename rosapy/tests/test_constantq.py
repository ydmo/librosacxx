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
from librosa.core.convert import cqt_frequencies, note_to_hz
from librosa.core.spectrum import stft, istft
from librosa.core.pitch import estimate_tuning
from librosa import filters
from librosa import util

import numpy as np

import pytest

from test_core import srand

import resampy
import scipy

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

def vqt(
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
        tuning = estimate_tuning(y=y, sr=sr, bins_per_octave=bins_per_octave)

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

def __test_cqt_size(
    y,
    sr,
    hop_length,
    fmin,
    n_bins,
    bins_per_octave,
    tuning,
    filter_scale,
    norm,
    sparsity,
    res_type,
):

    cqt_output = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
            filter_scale=filter_scale,
            norm=norm,
            sparsity=sparsity,
            res_type=res_type,
        )
    )

    assert cqt_output.shape[0] == n_bins

    return cqt_output


def make_signal(sr, duration, fmin="C1", fmax="C8"):
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


@pytest.fixture(scope="module")
def sr_cqt():
    return 11025


@pytest.fixture(scope="module")
def y_cqt(sr_cqt):
    return make_signal(sr_cqt, 2.0)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("hop_length", [-1, 0, 16, 63, 65])
@pytest.mark.parametrize("bpo", [12, 24])
def test_cqt_bad_hop(y_cqt, sr_cqt, hop_length, bpo):
    # incorrect hop lengths for a 6-octave analysis
    # num_octaves = 6, 2**(6-1) = 32 > 15
    librosa.cqt(
        y=y_cqt, sr=sr_cqt, hop_length=hop_length, n_bins=bpo * 6, bins_per_octave=bpo
    )


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("bpo", [12, 24])
def test_cqt_exceed_passband(y_cqt, sr_cqt, bpo):
    # Filters going beyond nyquist: 500 Hz -> 4 octaves = 8000 > 11025/2
    librosa.cqt(y=y_cqt, sr=sr_cqt, fmin=500, n_bins=4 * bpo, bins_per_octave=bpo)


@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24, 76])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [None, 0, 0.25])
@pytest.mark.parametrize("filter_scale", [1, 2])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("res_type", [None, "polyphase"])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("hop_length", [256, 512])
def test_cqt(
    y_cqt,
    sr_cqt,
    hop_length,
    fmin,
    n_bins,
    bins_per_octave,
    tuning,
    filter_scale,
    norm,
    res_type,
    sparsity,
):

    C = librosa.cqt(
        y=y_cqt,
        sr=sr_cqt,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=filter_scale,
        norm=norm,
        sparsity=sparsity,
        res_type=res_type,
    )

    # type is complex
    assert np.iscomplexobj(C)

    # number of bins is correct
    assert C.shape[0] == n_bins


@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [12, 24])
@pytest.mark.parametrize("gamma", [None, 0, 2.5])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [0])
@pytest.mark.parametrize("filter_scale", [1])
@pytest.mark.parametrize("norm", [1])
@pytest.mark.parametrize("res_type", ["polyphase"])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("hop_length", [512])
def test_vqt(
    y_cqt,
    sr_cqt,
    hop_length,
    fmin,
    n_bins,
    gamma,
    bins_per_octave,
    tuning,
    filter_scale,
    norm,
    res_type,
    sparsity,
):

    C = librosa.vqt(
        y=y_cqt,
        sr=sr_cqt,
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

    # type is complex
    assert np.iscomplexobj(C)

    # number of bins is correct
    assert C.shape[0] == n_bins


@pytest.fixture(scope="module")
def y_hybrid():
    return make_signal(11025, 5.0, None)


@pytest.mark.parametrize("sr", [11025])
@pytest.mark.parametrize("hop_length", [512])
@pytest.mark.parametrize("sparsity", [0.01])
@pytest.mark.parametrize("fmin", [None, librosa.note_to_hz("C2")])
@pytest.mark.parametrize("n_bins", [1, 12, 24, 48, 72, 74, 76])
@pytest.mark.parametrize("bins_per_octave", [12, 24])
@pytest.mark.parametrize("tuning", [None, 0, 0.25])
@pytest.mark.parametrize("resolution", [1, 2])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("res_type", [None, "polyphase"])
def test_hybrid_cqt(
    y_hybrid,
    sr,
    hop_length,
    fmin,
    n_bins,
    bins_per_octave,
    tuning,
    resolution,
    norm,
    sparsity,
    res_type,
):
    # This test verifies that hybrid and full cqt agree down to 1e-4
    # on 99% of bins which are nonzero (> 1e-8) in either representation.

    C2 = librosa.hybrid_cqt(
        y_hybrid,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=tuning,
        filter_scale=resolution,
        norm=norm,
        sparsity=sparsity,
        res_type=res_type,
    )

    C1 = np.abs(
        librosa.cqt(
            y_hybrid,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            tuning=tuning,
            filter_scale=resolution,
            norm=norm,
            sparsity=sparsity,
            res_type=res_type,
        )
    )

    assert C1.shape == C2.shape

    # Check for numerical comparability
    idx1 = C1 > 1e-4 * C1.max()
    idx2 = C2 > 1e-4 * C2.max()

    perc = 0.99

    thresh = 1e-3

    idx = idx1 | idx2

    assert np.percentile(np.abs(C1[idx] - C2[idx]), perc) < thresh * max(
        C1.max(), C2.max()
    )


@pytest.mark.parametrize("note_min", [12, 18, 24, 30, 36])
@pytest.mark.parametrize("sr", [22050])
@pytest.mark.parametrize(
    "y", [np.sin(2 * np.pi * librosa.midi_to_hz(60) * np.arange(2 * 22050) / 22050.0)]
)
def test_cqt_position(y, sr, note_min):

    C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.midi_to_hz(note_min))) ** 2

    # Average over time
    Cbar = np.median(C, axis=1)

    # Find the peak
    idx = np.argmax(Cbar)

    assert idx == 60 - note_min

    # Make sure that the max outside the peak is sufficiently small
    Cscale = Cbar / Cbar[idx]
    Cscale[idx] = np.nan
    assert np.nanmax(Cscale) < 6e-1, Cscale

    Cscale[idx - 1 : idx + 2] = np.nan
    assert np.nanmax(Cscale) < 5e-2, Cscale


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_early():

    # sampling rate is sufficiently above the top octave to trigger early downsampling
    y = np.zeros(16)
    librosa.cqt(y, sr=44100, n_bins=36)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_cqt_fail_short_late():

    y = np.zeros(16)
    librosa.cqt(y, sr=22050)


@pytest.fixture(scope="module", params=[11025, 16384, 22050, 32000, 44100])
def sr_impulse(request):
    return request.param


@pytest.fixture(scope="module", params=range(1, 9))
def hop_impulse(request):
    return 64 * request.param


@pytest.fixture(scope="module")
def y_impulse(sr_impulse, hop_impulse):
    x = np.zeros(sr_impulse)
    center = int((len(x) / (2.0 * float(hop_impulse))) * hop_impulse)
    x[center] = 1
    return x


def test_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    # Test to resolve issue #348
    # Updated in #417 to use integrated energy, rather than frame-wise max

    C = np.abs(librosa.cqt(y=y_impulse, sr=sr_impulse, hop_length=hop_impulse))

    response = np.mean(C ** 2, axis=1)

    continuity = np.abs(np.diff(response))

    # Test that integrated energy is approximately constant
    assert np.max(continuity) < 5e-4, continuity


def test_hybrid_cqt_impulse(y_impulse, sr_impulse, hop_impulse):
    # Test to resolve issue #341
    # Updated in #417 to use integrated energy instead of pointwise max

    hcqt = librosa.hybrid_cqt(
        y=y_impulse, sr=sr_impulse, hop_length=hop_impulse, tuning=0
    )

    response = np.mean(np.abs(hcqt) ** 2, axis=1)

    continuity = np.abs(np.diff(response))

    assert np.max(continuity) < 5e-4, continuity


@pytest.fixture(scope="module")
def sr_white():
    return 22050


@pytest.fixture(scope="module")
def y_white(sr_white):
    srand()
    return np.random.randn(10 * sr_white)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fmin", list(librosa.note_to_hz(["C1", "C2"])))
@pytest.mark.parametrize("n_bins", [24, 36])
def test_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):

    C = np.abs(
        librosa.cqt(y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale)
    )

    if not scale:
        lengths = librosa.filters.constant_q_lengths(sr_white, fmin, n_bins=n_bins)
        C /= np.sqrt(lengths[:, np.newaxis])

    # Only compare statistics across the time dimension
    # we want ~ constant mean and variance across frequencies
    assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)


@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fmin", list(librosa.note_to_hz(["C1", "C2"])))
@pytest.mark.parametrize("n_bins", [72, 84])
def test_hybrid_cqt_white_noise(y_white, sr_white, fmin, n_bins, scale):
    C = librosa.hybrid_cqt(
        y=y_white, sr=sr_white, fmin=fmin, n_bins=n_bins, scale=scale
    )

    if not scale:
        lengths = librosa.filters.constant_q_lengths(sr_white, fmin, n_bins=n_bins)
        C /= np.sqrt(lengths[:, np.newaxis])

    assert np.allclose(np.mean(C, axis=1), 1.0, atol=2.5e-1), np.mean(C, axis=1)
    assert np.allclose(np.std(C, axis=1), 0.5, atol=5e-1), np.std(C, axis=1)


@pytest.fixture(scope="module", params=[22050, 44100])
def sr_icqt(request):
    return request.param


@pytest.fixture(scope="module")
def y_icqt(sr_icqt):
    return make_signal(sr_icqt, 1.5, fmin="C2", fmax="C4")


@pytest.mark.parametrize("over_sample", [1, 3])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("hop_length", [384, 512])
@pytest.mark.parametrize("length", [None, True])
@pytest.mark.parametrize("res_type", ["scipy", "kaiser_fast", "polyphase"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_icqt(y_icqt, sr_icqt, scale, hop_length, over_sample, length, res_type, dtype):

    bins_per_octave = over_sample * 12
    n_bins = 7 * bins_per_octave

    C = librosa.cqt(
        y_icqt,
        sr=sr_icqt,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        scale=scale,
        hop_length=hop_length,
    )

    if length:
        _len = len(y_icqt)
    else:
        _len = None
    yinv = librosa.icqt(
        C,
        sr=sr_icqt,
        scale=scale,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        length=_len,
        res_type=res_type,
        dtype=dtype,
    )

    assert yinv.dtype == dtype

    # Only test on the middle section
    if length:
        assert len(y_icqt) == len(yinv)
    else:
        yinv = librosa.util.fix_length(yinv, len(y_icqt))

    y_icqt = y_icqt[sr_icqt // 2 : -sr_icqt // 2]
    yinv = yinv[sr_icqt // 2 : -sr_icqt // 2]

    residual = np.abs(y_icqt - yinv)
    # We'll tolerate 10% RMSE
    # error is lower on more recent numpy/scipy builds

    resnorm = np.sqrt(np.mean(residual ** 2))
    assert resnorm <= 0.1, resnorm


@pytest.fixture
def y_chirp():
    sr = 22050
    y = librosa.chirp(55, 55 * 2 ** 3, length=sr // 8, sr=sr)
    return y


@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.parametrize("window", ["hann", "hamming"])
@pytest.mark.parametrize("use_length", [False, True])
@pytest.mark.parametrize("over_sample", [1, 3])
@pytest.mark.parametrize("res_type", ["polyphase"])
@pytest.mark.parametrize("pad_mode", ["reflect"])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("momentum", [0, 0.99])
@pytest.mark.parametrize("random_state", [None, 0, np.random.RandomState()])
@pytest.mark.parametrize("fmin", [40.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("init", [None, "random"])
def test_griffinlim_cqt(
    y_chirp,
    hop_length,
    window,
    use_length,
    over_sample,
    fmin,
    res_type,
    pad_mode,
    scale,
    momentum,
    init,
    random_state,
    dtype,
):

    if use_length:
        length = len(y_chirp)
    else:
        length = None

    sr = 22050
    bins_per_octave = 12 * over_sample
    n_bins = 6 * bins_per_octave
    C = librosa.cqt(
        y_chirp,
        sr=sr,
        hop_length=hop_length,
        window=window,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        n_bins=n_bins,
        scale=scale,
        pad_mode=pad_mode,
        res_type=res_type,
    )

    Cmag = np.abs(C)

    y_rec = librosa.griffinlim_cqt(
        Cmag,
        hop_length=hop_length,
        window=window,
        sr=sr,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
        scale=scale,
        pad_mode=pad_mode,
        n_iter=3,
        momentum=momentum,
        random_state=random_state,
        length=length,
        res_type=res_type,
        init=init,
        dtype=dtype,
    )

    y_inv = librosa.icqt(
        Cmag,
        sr=sr,
        fmin=fmin,
        hop_length=hop_length,
        window=window,
        bins_per_octave=bins_per_octave,
        scale=scale,
        length=length,
        res_type=res_type,
    )

    # First check for length
    if use_length:
        assert len(y_rec) == length

    assert y_rec.dtype == dtype

    # Check that the data is okay
    assert np.all(np.isfinite(y_rec))


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_badinit():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, init="garbage")


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_griffinlim_cqt_momentum():
    x = np.zeros((33, 3))
    librosa.griffinlim_cqt(x, momentum=-1)


def test_griffinlim_cqt_momentum_warn():
    x = np.zeros((33, 3))
    with pytest.warns(UserWarning):
        librosa.griffinlim_cqt(x, momentum=2)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cqt_precision(y_cqt, sr_cqt, dtype):
    C = librosa.cqt(y=y_cqt, sr=sr_cqt, dtype=dtype)
    assert np.dtype(C.dtype) == np.dtype(dtype)

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

    y = make_signal(sr, 2.0)

    C = vqt(
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

    y = make_signal(sr, 2.0)

    V = vqt(
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

# </gen_test_cqt_data>

if __name__ == '__main__':
    gen_test_cqt_data()
    

    
    