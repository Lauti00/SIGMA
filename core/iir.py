import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from core.fir import FILTER_TYPES

IIR_MANUAL_MAX_SAMPLES = 200_000


def iir_implementation_name(sample_count):
    if sample_count <= IIR_MANUAL_MAX_SAMPLES:
        return "Ecuaciones en diferencias implementadas muestra a muestra (forma directa II transpuesta)."
    return "Ecuaciones en diferencias optimizadas por SciPy (`sosfilt`) para archivos largos."


def design_iir(filter_type, sample_rate, order, cutoff_hz):
    """Diseña un Butterworth en secciones de segundo orden (SOS) con SciPy."""
    kind = FILTER_TYPES.get(filter_type)
    if kind is None:
        raise ValueError("Tipo de filtro IIR no reconocido.")
    if sample_rate <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")
    order = int(order)
    if order < 1 or order > 12:
        raise ValueError("El orden IIR debe estar entre 1 y 12.")

    nyquist = sample_rate / 2
    cutoffs = np.atleast_1d(cutoff_hz).astype(float)
    expected = 1 if kind in ("lowpass", "highpass") else 2
    if cutoffs.size != expected:
        raise ValueError("Cantidad incorrecta de frecuencias de corte.")
    if np.any(cutoffs <= 0) or np.any(cutoffs >= nyquist):
        raise ValueError(f"Las frecuencias deben estar entre 0 y Nyquist ({nyquist:g} Hz).")
    if cutoffs.size == 2 and cutoffs[0] >= cutoffs[1]:
        raise ValueError("La frecuencia inferior debe ser menor que la superior.")

    cutoff_argument = float(cutoffs[0]) if cutoffs.size == 1 else cutoffs
    return butter(order, cutoff_argument, btype=kind, fs=sample_rate, output="sos")


def apply_iir(audio, sos):
    """Aplica IIR causal mediante ecuaciones en diferencias por secciones.

    SciPy diseña el filtro y lo entrega como secciones de segundo orden
    (SOS). El cálculo muestra a muestra se implementa aquí en forma directa
    II transpuesta. Los estados iniciales representan la respuesta
    estacionaria a una entrada constante igual a la primera muestra.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("La señal debe ser un vector mono no vacío.")

    sections = np.asarray(sos, dtype=np.float64)
    if sections.ndim != 2 or sections.shape[1] != 6 or sections.shape[0] == 0:
        raise ValueError("Los coeficientes IIR deben estar en formato SOS (secciones x 6).")

    if audio.size > IIR_MANUAL_MAX_SAMPLES:
        initial_state = sosfilt_zi(sections) * audio[0]
        return sosfilt(sections, audio, zi=initial_state)[0]

    normalized = np.empty_like(sections)
    states = np.zeros((sections.shape[0], 2), dtype=np.float64)
    steady_input = float(audio[0])

    for section_index, section in enumerate(sections):
        b0, b1, b2, a0, a1, a2 = section
        if a0 == 0:
            raise ValueError("El coeficiente a0 de una sección no puede ser cero.")
        b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
        a1, a2 = a1 / a0, a2 / a0
        normalized[section_index] = (b0, b1, b2, 1.0, a1, a2)

        dc_denominator = 1.0 + a1 + a2
        steady_output = (
            0.0
            if abs(dc_denominator) < np.finfo(float).eps
            else steady_input * (b0 + b1 + b2) / dc_denominator
        )
        state_2 = b2 * steady_input - a2 * steady_output
        state_1 = b1 * steady_input - a1 * steady_output + state_2
        states[section_index] = (state_1, state_2)
        steady_input = steady_output

    output = np.empty_like(audio)
    for sample_index, sample in enumerate(audio):
        section_input = sample
        for section_index, (b0, b1, b2, _, a1, a2) in enumerate(normalized):
            state_1, state_2 = states[section_index]
            section_output = b0 * section_input + state_1
            states[section_index, 0] = b1 * section_input - a1 * section_output + state_2
            states[section_index, 1] = b2 * section_input - a2 * section_output
            section_input = section_output
        output[sample_index] = section_input

    return output
