import numpy as np
from scipy.signal import fftconvolve, firwin


FILTER_TYPES = {
    "Pasabajo": "lowpass",
    "Pasaalto": "highpass",
    "Pasabanda": "bandpass",
    "Rechaza banda": "bandstop",
}
FIR_DIRECT_MAX_SAMPLES = 200_000


def fir_implementation_name(sample_count):
    if sample_count <= FIR_DIRECT_MAX_SAMPLES:
        return "Convolución discreta NumPy (`np.convolve`)."
    return "Convolución FFT optimizada de SciPy (`fftconvolve`) para archivos largos."


def design_fir(filter_type, sample_rate, order, cutoff_hz, kaiser_beta=8.6):
    """Diseña coeficientes FIR Kaiser con firwin de SciPy.

    `order` es el orden del filtro; la cantidad de coeficientes (taps) es
    order + 1. Se usa orden par para obtener un número impar de taps.
    """
    kind = FILTER_TYPES.get(filter_type)
    if kind is None:
        raise ValueError("Tipo de filtro FIR no reconocido.")
    if sample_rate <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")

    order = int(order)
    if order < 2:
        raise ValueError("El orden FIR debe ser al menos 2.")
    if order % 2:
        order += 1

    nyquist = sample_rate / 2
    cutoffs = np.atleast_1d(cutoff_hz).astype(float)
    if kind in ("lowpass", "highpass") and cutoffs.size != 1:
        raise ValueError("Este filtro requiere una frecuencia de corte.")
    if kind in ("bandpass", "bandstop") and cutoffs.size != 2:
        raise ValueError("Este filtro requiere dos frecuencias de corte.")
    if np.any(cutoffs <= 0) or np.any(cutoffs >= nyquist):
        raise ValueError(f"Las frecuencias deben estar entre 0 y Nyquist ({nyquist:g} Hz).")
    if cutoffs.size == 2 and cutoffs[0] >= cutoffs[1]:
        raise ValueError("La frecuencia inferior debe ser menor que la superior.")
    if not 0 <= kaiser_beta <= 20:
        raise ValueError("Beta Kaiser debe estar entre 0 y 20.")

    pass_zero = kind in ("lowpass", "bandstop")
    cutoff_argument = float(cutoffs[0]) if cutoffs.size == 1 else cutoffs
    coefficients = firwin(
        numtaps=order + 1,
        cutoff=cutoff_argument,
        pass_zero=pass_zero,
        window=("kaiser", float(kaiser_beta)),
        fs=sample_rate,
    )
    return coefficients


def apply_fir(audio, coefficients):
    """Aplica FIR, con convolución directa o FFT según el tamaño de entrada."""
    audio = np.asarray(audio, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("La señal debe ser un vector mono no vacío.")
    if audio.size <= FIR_DIRECT_MAX_SAMPLES:
        full = np.convolve(audio, coefficients, mode="full")
    else:
        full = fftconvolve(audio, coefficients, mode="full")
    start = (coefficients.size - 1) // 2
    return full[start : start + audio.size]
