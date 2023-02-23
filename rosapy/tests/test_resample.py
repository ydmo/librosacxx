import numpy as np
import librosa
from librosa.core.spectrum import _spectrogram
from librosa.core import convert
from librosa import util
import base64 as b64
import resampy
from resampy.interpn import resample_f
from resampy.filters import get_filter

def resampy_resample(x, sr_orig, sr_new, axis=-1, filter='kaiser_best', **kwargs):

    if sr_orig <= 0:
        raise ValueError('Invalid sample rate: sr_orig={}'.format(sr_orig))

    if sr_new <= 0:
        raise ValueError('Invalid sample rate: sr_new={}'.format(sr_new))

    sample_ratio = float(sr_new) / sr_orig

    # Set up the output shape
    shape = list(x.shape)
    shape[axis] = int(shape[axis] * sample_ratio)

    if shape[axis] < 1:
        raise ValueError('Input signal length={} is too small to '
                         'resample from {}->{}'.format(x.shape[axis], sr_orig, sr_new))

    # Preserve contiguity of input (if it exists)
    # If not, revert to C-contiguity by default
    if x.flags['F_CONTIGUOUS']:
        order = 'F'
    else:
        order = 'C'

    y = np.zeros(shape, dtype=x.dtype, order=order)

    interp_win, precision, _ = get_filter(filter, **kwargs)

    if sample_ratio < 1:
        interp_win *= sample_ratio

    interp_delta = np.zeros_like(interp_win)
    interp_delta[:-1] = np.diff(interp_win)

    # Construct 2d views of the data with the resampling axis on the first dimension
    x_2d = x.swapaxes(0, axis).reshape((x.shape[axis], -1))
    y_2d = y.swapaxes(0, axis).reshape((y.shape[axis], -1))
    resample_f(x_2d, y_2d, sample_ratio, interp_win, interp_delta, precision)

    return y

def resample(
    y, orig_sr, target_sr, res_type="kaiser_best", fix=True, scale=False, **kwargs
    ):
    
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
        y_hat = resampy_resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)

# half_window, precision, rolloff = resampy.filters.get_filter("kaiser_best")
# contents_kaiser_best = """
# #ifndef data_kaiser_best_h
# #define data_kaiser_best_h
# namespace resam {{
# constexpr float kaiser_best_precision = {0};
# constexpr float kaiser_best_rolloff = {1};
# constexpr int kaiser_best_half_window_len = {2};
# constexpr double kaiser_best_half_window_dat[kaiser_best_half_window_len] = {{
#     {3}
# }};
# }} // namespace resam
# #endif
# """.format(
#     precision,
#     rolloff,
#     half_window.size,
#     str(half_window.tolist()).replace('[', '').replace(']', '')
#     )
# with open("/home/yuda/Documents/Projects/raspai/song_scoring/heart/scorpio/3rd/librosacxx/rosacxx/resamcxx/data_kaiser_best.h", "w+") as fp:
#     fp.write(contents_kaiser_best)
# # end-with

# half_window, precision, rolloff = resampy.filters.get_filter("kaiser_fast")
# contents_kaiser_fast = """
# #ifndef data_kaiser_fast_h
# #define data_kaiser_fast_h
# namespace resam {{
# constexpr float kaiser_fast_precision = {0};
# constexpr float kaiser_fast_rolloff = {1};
# constexpr int kaiser_fast_half_window_len = {2};
# constexpr double kaiser_fast_half_window_dat[kaiser_fast_half_window_len] = {{
#     {3}
# }};
# }} // namespace resam
# #endif
# """.format(
#     precision,
#     rolloff,
#     half_window.size,
#     str(half_window.tolist()).replace('[', '').replace(']', '')
#     )
# with open("/home/yuda/Documents/Projects/raspai/song_scoring/heart/scorpio/3rd/librosacxx/rosacxx/resamcxx/data_kaiser_fast.h", "w+") as fp:
#     fp.write(contents_kaiser_fast)
# # end-with

sr_dst = 11025
res_type = "kaiser_fast"
fn = librosa.ex("trumpet")
src, sr_src = librosa.load(fn, sr=None, res_type=res_type)
dst = resample(src, sr_src, sr_dst, res_type=res_type, scale=True)
dst_gt = librosa.resample(src, sr_src, sr_dst, res_type=res_type, scale=True)
print(dst_gt - dst)

contents_kaiser_fast = """
#ifndef tests_data_resample_kaiser_fast_h
#define tests_data_resample_kaiser_fast_h

constexpr float  ROSACXXTest_resample_kaiser_fast_src_sr = {0};
constexpr int    ROSACXXTest_resample_kaiser_fast_src_len = {1};
constexpr double ROSACXXTest_resample_kaiser_fast_src_dat[ROSACXXTest_resample_kaiser_fast_src_len] = {{ {2} }};

constexpr float  ROSACXXTest_resample_kaiser_fast_dst_sr = {3};
constexpr int    ROSACXXTest_resample_kaiser_fast_dst_len = {4};
constexpr double ROSACXXTest_resample_kaiser_fast_dst_dat[ROSACXXTest_resample_kaiser_fast_dst_len] = {{ {5} }};

#endif // tests_data_resample_kaiser_fast_h
""".format(
    sr_src,
    src.size,
    str(src.tolist()).replace('[', '').replace(']', ''),
    sr_dst,
    dst.size,
    str(dst.tolist()).replace('[', '').replace(']', ''),
    )
with open("/home/yuda/Documents/Projects/raspai/song_scoring/heart/scorpio/3rd/librosacxx/tests/tests_data_resample_kaiser_fast.h", "w+") as fp:
    fp.write(contents_kaiser_fast)
# end-with