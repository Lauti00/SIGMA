import io

import numpy as np
from scipy.io import wavfile


def read_wav_file(uploaded_file):
    """Lee un WAV y devuelve frecuencia, audio mono normalizado y canales originales."""
    sample_rate, raw_audio = wavfile.read(io.BytesIO(uploaded_file.getvalue()))

    if raw_audio.size == 0:
        raise ValueError("El archivo WAV no contiene muestras de audio.")

    if raw_audio.ndim == 1:
        original_channels = 1
    elif raw_audio.ndim == 2:
        original_channels = raw_audio.shape[1]
    else:
        raise ValueError("El WAV debe tener uno o dos canales.")

    if original_channels > 2:
        raise ValueError("Esta versión admite archivos mono o estéreo.")

    if np.issubdtype(raw_audio.dtype, np.unsignedinteger):
        # PCM WAV de 8 bits es unsigned y su silencio está centrado en 128.
        info = np.iinfo(raw_audio.dtype)
        midpoint = (info.max + 1) / 2
        audio = (raw_audio.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(raw_audio.dtype, np.signedinteger):
        info = np.iinfo(raw_audio.dtype)
        scale = max(abs(info.min), abs(info.max))
        audio = raw_audio.astype(np.float64) / scale
    else:
        audio = raw_audio.astype(np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if not np.all(np.isfinite(audio)):
        raise ValueError("El audio contiene valores no válidos.")

    return int(sample_rate), audio, original_channels
