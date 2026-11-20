import numpy as np
from scipy.signal import find_peaks


WINDOW_TYPES = ("Rectangular", "Hann", "Hamming")


def compute_spectrum(audio, sample_rate, window="Rectangular"):
    """Calcula el espectro de amplitud de un solo lado usando rFFT.

    Normalizamos por la suma de la ventana para conservar la amplitud de los
    tonos. Hann y Hamming reducen leakage espectral, a cambio de ensanchar los
    picos. Las componentes interiores se duplican por usar solo frecuencias
    no negativas.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("La señal debe ser un vector mono no vacío.")
    if sample_rate <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")

    sample_count = audio.size
    if window == "Rectangular":
        window_values = np.ones(sample_count)
    elif window == "Hann":
        indices = np.arange(sample_count)
        window_values = 0.5 - 0.5 * np.cos(2 * np.pi * indices / sample_count)
    elif window == "Hamming":
        indices = np.arange(sample_count)
        window_values = 0.54 - 0.46 * np.cos(2 * np.pi * indices / sample_count)
    else:
        raise ValueError(f"Ventana no reconocida: {window}.")

    # Para una señal de una muestra, Hann da peso cero. No hay leakage que
    # reducir en ese caso, así que usamos una ventana rectangular.
    if np.sum(window_values) <= np.finfo(float).eps:
        window_values = np.ones(sample_count)

    spectrum = np.fft.rfft(audio * window_values)
    amplitude = np.abs(spectrum) / np.sum(window_values)

    if sample_count > 1:
        if sample_count % 2 == 0:
            amplitude[1:-1] *= 2
        else:
            amplitude[1:] *= 2

    frequencies = np.fft.rfftfreq(sample_count, d=1 / sample_rate)
    return frequencies, amplitude


def find_dominant_frequencies(frequencies, amplitude, count=5):
    """Devuelve hasta `count` picos locales más altos, excluyendo DC (0 Hz)."""
    frequencies = np.asarray(frequencies)
    amplitude = np.asarray(amplitude)
    if frequencies.size != amplitude.size:
        raise ValueError("Frecuencias y amplitudes deben tener el mismo tamaño.")
    if count <= 0 or frequencies.size < 3:
        return []

    peak_indices, _ = find_peaks(amplitude)
    peak_indices = peak_indices[frequencies[peak_indices] > 0]
    peak_indices = sorted(peak_indices, key=lambda index: amplitude[index], reverse=True)

    return [
        {"frequency": float(frequencies[index]), "amplitude": float(amplitude[index])}
        for index in peak_indices[:count]
    ]
