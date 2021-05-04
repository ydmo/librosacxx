import numpy as np
import librosa
from librosa.core.spectrum import _spectrogram
from librosa.core import convert
from librosa import util
import base64 as b64
import resampy

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
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = util.fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.asfortranarray(y_hat, dtype=y.dtype)

sr_out = [11025, 22050, 44100]
res_type = "kaiser_best"
fn = librosa.ex("trumpet")
y1, sr = librosa.load(fn, sr=None, res_type=res_type)

with open("/home/myd/文档/librosacxx/tests/tests_data_resample_src.h", "w+") as fp:
    contents = """
    #include <string>
    constexpr float  ROSACXXTest_resample_src_sr = {};
    constexpr float  ROSACXXTest_resample_src_length = {};
    constexpr char * ROSACXXTest_resample_src_f32b64 = \"{}\";
    """.format(
        sr,
        y1.reshape(-1).shape[0],
        b64.b64encode(y1.astype(np.float32).tobytes()).decode('utf-8'),
        )
    fp.write(contents)
# end-with

y2 = librosa.resample(y1, sr, sr_out[0], res_type=res_type, scale=True)
y3 = resample(y1, sr, sr_out[0], res_type=res_type, scale=True)

print(y3-y2)