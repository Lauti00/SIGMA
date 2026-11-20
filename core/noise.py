import numpy as np


NOISE_TYPES = ("Ruido blanco", "Tono sinusoidal")


def add_noise_at_snr(clean_audio, sample_rate, snr_db, noise_type="Ruido blanco", tone_frequency=50.0, seed=0):
    """Mezcla ruido conocido con una señal limpia a una SNR objetivo.

    Devuelve (señal limpia escalada, mezcla contaminada, ruido escalado).
    El limitador aplica la misma ganancia a las tres señales para evitar
    clipping sin cambiar la relación señal-ruido.
    """
    clean_audio = np.asarray(clean_audio, dtype=np.float64)
    if clean_audio.ndim != 1 or clean_audio.size == 0:
        raise ValueError("El audio limpio debe ser un vector mono no vacío.")
    if not np.all(np.isfinite(clean_audio)):
        raise ValueError("El audio limpio contiene valores no finitos.")
    if sample_rate <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")
    if noise_type not in NOISE_TYPES:
        raise ValueError("Tipo de ruido no reconocido.")
    if not np.isfinite(snr_db):
        raise ValueError("La SNR debe ser un número finito.")

    signal_rms = float(np.sqrt(np.mean(clean_audio**2)))
    if signal_rms == 0:
        raise ValueError("No se puede contaminar una señal silenciosa con SNR definida.")

    if noise_type == "Ruido blanco":
        noise = np.random.default_rng(seed).standard_normal(clean_audio.size)
    else:
        if not 0 < tone_frequency < sample_rate / 2:
            raise ValueError("La frecuencia del tono debe estar entre 0 y Nyquist.")
        times = np.arange(clean_audio.size, dtype=np.float64) / sample_rate
        noise = np.sin(2 * np.pi * tone_frequency * times)
        noise -= np.mean(noise)

    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if noise_rms == 0:
        raise ValueError("No se pudo generar ruido con potencia distinta de cero para este audio.")

    target_noise_rms = signal_rms / (10 ** (float(snr_db) / 20))
    noise = noise * (target_noise_rms / noise_rms)
    mixture = clean_audio + noise
    peak = float(np.max(np.abs(mixture)))
    gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0

    return clean_audio * gain, mixture * gain, noise * gain
