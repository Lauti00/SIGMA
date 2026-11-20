import numpy as np


def _validate_signals(reference, estimate):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 1 or estimate.ndim != 1:
        raise ValueError("Las señales de referencia y estimada deben ser vectores mono.")
    if reference.size == 0 or reference.shape != estimate.shape:
        raise ValueError("Las señales deben tener la misma cantidad de muestras y no estar vacías.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(estimate)):
        raise ValueError("Las señales contienen valores no finitos.")
    return reference, estimate


def mean_squared_error(reference, estimate):
    """Calcula MSE entre dos señales mono alineadas muestra a muestra."""
    reference, estimate = _validate_signals(reference, estimate)
    error = reference - estimate
    return float(np.mean(error**2))


def signal_to_noise_ratio_db(reference, estimate):
    """Calcula SNR en dB tratando `reference` como señal limpia conocida."""
    reference, estimate = _validate_signals(reference, estimate)
    error = reference - estimate
    signal_power = float(np.mean(reference**2))
    noise_power = float(np.mean(error**2))
    if signal_power == 0:
        raise ValueError("La señal de referencia no puede tener potencia cero.")
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))
