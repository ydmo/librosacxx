
import numpy as np
import librosa
from librosa.core.spectrum import _spectrogram
from librosa.core import convert
from librosa import util
import base64 as b64

def piptrack(
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

    fft_freqs = convert.fft_frequencies(sr=sr, n_fft=n_fft)

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    avg = 0.5 * (S[2:] - S[:-2])

    shift = 2 * S[1:-1] - S[2:] - S[:-2]

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by localmax anyway
    shift = avg / (shift + (np.abs(shift) < util.tiny(shift)))

    # Pad back up to the same shape as S
    avg = np.pad(avg, ([1, 1], [0, 0]), mode="constant")
    shift = np.pad(shift, ([1, 1], [0, 0]), mode="constant")

    dskew = 0.5 * avg * shift

    # Pre-allocate output
    pitches = np.zeros_like(S).astype(np.float32)
    mags = np.zeros_like(S).astype(np.float32)

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
    pitches[idx[:, 0], ididx[:, 1]] = (
        (idx[:, 0] + shift[idx[:, 0], x[:, 1]]) * float(sr) / n_fft
    )

    mags[idx[:, 0], idx[:, 1]] = S[idx[:, 0], idx[:, 1]] + dskew[idx[:, 0], idx[:, 1]]

    return pitches, mags


def test_piptrack(freq = 440, n_fft = 1024):
    
    # import base64 as b64
    # b64.b64encode(inp.tobytes()).decode('utf-8')

    y = librosa.tone(freq, sr=22050, duration=0.2).astype(np.float32)
    S = np.abs(librosa.stft(y, n_fft=n_fft, center=False))

    pitches, mags = piptrack(S=S, fmin=100)
    print(b64.b64encode(pitches.astype(np.float32).tobytes()).decode('utf-8'))
    print(b64.b64encode(mags.astype(np.float32).tobytes()).decode('utf-8'))

    idx = mags > 0

    assert len(idx) > 0

    recovered_pitches = pitches[idx]

    # We should be within one cent of the target
    assert np.all(np.abs(np.log2(recovered_pitches) - np.log2(freq)) <= 1e-2)


test_piptrack()