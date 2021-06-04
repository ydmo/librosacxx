import numpy as np
import librosa
from librosa.filters import get_window
from librosa import util
from numpy import fft
from numpy.lib.type_check import real


def stft(
    y,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="reflect",
    ):

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    import pdb; pdb.set_trace()

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    import pdb; pdb.set_trace()

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            print(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, y.shape[-1]
                )
            )

        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise ParameterError(
            "n_fft={} is too small for input signal of length={}".format(
                n_fft, y.shape[-1]
            )
        )

    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)

    if dtype is None:
        dtype = util.dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.rfft(
            fft_window * y_frames[:, bl_s:bl_t], axis=0
        )
    return stft_matrix

def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Helper function for window sum-square calculation."""
    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]


def window_sumsquare(
    window,
    n_frames,
    hop_length=512,
    win_length=None,
    n_fft=2048,
    dtype=np.float32,
    norm=None,
    ):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = util.normalize(win_sq, norm=norm) ** 2
    win_sq = util.pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x

def istft(
    stft_matrix,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    length=None,
    ):

    def __overlap_add(y, ytmp, hop_length):
        # numba-accelerated overlap add for inverse stft
        # y is the pre-allocated output buffer
        # ytmp is the windowed inverse-stft frames
        # hop_length is the hop-length of the STFT analysis
        n_fft = ytmp.shape[0]
        for frame in range(ytmp.shape[1]):
            sample = frame * hop_length
            y[sample : (sample + n_fft)] += ytmp[:, frame]

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft, and add a broadcasting axis
    ifft_window = util.pad_center(ifft_window, n_fft)[:, np.newaxis]

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[1]

    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    if dtype is None:
        dtype = util.dtype_c2r(stft_matrix.dtype)

    y = np.zeros(expected_signal_len, dtype=dtype)

    n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

        # Overlap-add the istft block starting at the i'th frame
        __overlap_add(y[frame * hop_length :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window,
        n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2) : -int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = util.fix_length(y[start:], length)

    return y

# python -m rosapy.tests.test_stft
if __name__ == '__main__':
    outSc_real = np.fromfile('./rosapy/tests/data/outSc.real.bin', dtype=np.float32).reshape(2049, 5027)
    outSc_imag = np.fromfile('./rosapy/tests/data/outSc.imag.bin', dtype=np.float32).reshape(2049, 5027)
    outSc = outSc_real + outSc_imag * np.array(0+1j)
    Yc_pred = istft(outSc, hop_length=1024, center=False)
    Yc = np.fromfile('./rosapy/tests/data/Yc.bin', dtype=np.float32)
    import pdb; pdb.set_trace()
    print("end.")



    

