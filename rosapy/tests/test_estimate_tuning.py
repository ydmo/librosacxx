import numpy as np
import librosa
from librosa.core.spectrum import _spectrogram
from librosa.core import convert
from librosa import util
import base64 as b64

def test_estimate_tuning(sr=11025, center_note=69, tuning=-0.5, bins_per_octave=12, resolution=1e-2):

    target_hz = librosa.midi_to_hz(center_note + tuning)

    y = librosa.tone(target_hz, duration=0.5, sr=sr)

    tuning_est = librosa.estimate_tuning(
        resolution=resolution,
        bins_per_octave=bins_per_octave,
        y=y,
        sr=sr,
        n_fft=2048,
        fmin=librosa.note_to_hz("C4"),
        fmax=librosa.note_to_hz("G#9"),
    )

    # Round to the proper number of decimals
    deviation = np.around(tuning - tuning_est, int(-np.log10(resolution)))

    # Take the minimum floating point for positive and negative deviations
    max_dev = np.min([np.mod(deviation, 1.0), np.mod(-deviation, 1.0)])

    # We'll accept an answer within three bins of the resolution
    assert max_dev <= 3 * resolution

test_estimate_tuning()