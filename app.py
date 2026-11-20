import hashlib
import io

import numpy as np
import streamlit as st
from scipy.io import wavfile

from core.audio import read_wav_file
from core.filter_response import fir_response, iir_response
from core.fir import FILTER_TYPES, apply_fir, design_fir, fir_implementation_name
from core.fourier import WINDOW_TYPES, compute_spectrum, find_dominant_frequencies
from core.iir import apply_iir, design_iir, iir_implementation_name
from core.metrics import mean_squared_error, signal_to_noise_ratio_db
from visualization.interactive_plots import (
    filter_response_figure,
    spectrum_figure,
    time_comparison_figure,
)


st.set_page_config(
    page_title="SIGMA | Audio Lab",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def to_wav_bytes(audio, sample_rate):
    """Serializa el audio mono procesado en formato WAV de 32 bits."""
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, np.asarray(audio, dtype=np.float32))
    return buffer.getvalue()


def show_interactive_chart(figure, key):
    st.plotly_chart(
        figure,
        theme="streamlit",
        width="stretch",
        key=key,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )


def cached_spectrum(cache_key, signal, sample_rate, window):
    cache = st.session_state.setdefault("sigma_spectrum_cache", {})
    if cache_key not in cache:
        cache[cache_key] = compute_spectrum(signal, sample_rate, window)
        # Keep only a few entries so changing audio does not retain large arrays.
        while len(cache) > 6:
            cache.pop(next(iter(cache)))
    return cache[cache_key]


with st.sidebar:
    st.markdown("## SIGMA")
    st.caption("Laboratorio de análisis y procesamiento digital de audio")
    uploaded_file = st.file_uploader("Archivo de audio", type=["wav"], help="WAV mono o estéreo. El audio estéreo se convierte a mono para el procesamiento.")

st.title("SIGMA Audio Lab")
st.caption("Explorá la señal, ajustá un filtro y compará el resultado con controles interactivos.")

if uploaded_file is None:
    st.info("Cargá un archivo WAV desde el panel lateral para empezar el análisis.")
    st.markdown("### Flujo de trabajo")
    st.markdown("1. Cargá el audio.  2. Explorá el espectro y elegí un filtro.  3. Compará, escuchá y descargá el resultado.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    upload_id = hashlib.sha256(file_bytes).hexdigest()
    sample_rate, audio, channels = read_wav_file(uploaded_file)
except Exception as error:
    st.error(f"No se pudo abrir el WAV: {error}")
    st.stop()

if sample_rate < 4:
    st.error("La frecuencia de muestreo es demasiado baja para diseñar estos filtros.")
    st.stop()

nyquist = sample_rate / 2
duration = len(audio) / sample_rate
current_result = st.session_state.get("sigma_result")
if current_result is not None and current_result.get("upload_id") != upload_id:
    current_result = None

# Compact metadata cards keep the main content focused on the signal.
metric_cols = st.columns(4)
metric_cols[0].metric("Duración", f"{duration:.2f} s")
metric_cols[1].metric("Muestreo", f"{sample_rate:,} Hz")
metric_cols[2].metric("Muestras", f"{len(audio):,}")
metric_cols[3].metric("Canales", f"{channels} → mono" if channels == 2 else "Mono")

if channels == 2:
    st.caption("La señal estéreo se promedió a mono para el análisis y el procesamiento.")

# Analysis controls update the FFT and its plots as soon as they change.
with st.sidebar:
    st.divider()
    st.subheader("Análisis")
    max_fft_seconds = min(duration, 30.0, 2_000_000 / sample_rate)
    fft_seconds = st.slider(
        "Tramo para FFT",
        min_value=0.1,
        max_value=max(0.1, float(max_fft_seconds)),
        value=max(0.1, min(10.0, float(max_fft_seconds))),
        step=0.1,
        help="La FFT usa el inicio del audio. El filtrado sigue usando la señal completa.",
    )
    fft_window = st.selectbox("Ventana", WINDOW_TYPES, help="Hann y Hamming reducen leakage a cambio de ensanchar los picos.")

    st.divider()
    st.subheader("Filtro digital")
    method = st.selectbox("Familia", ["FIR · Kaiser", "IIR · Butterworth"])
    filter_label = st.selectbox("Respuesta", list(FILTER_TYPES))
    filter_kind = FILTER_TYPES[filter_label]
    if filter_kind in ("lowpass", "highpass"):
        cutoff = [
            st.number_input(
                "Frecuencia de corte (Hz)",
                min_value=1.0,
                max_value=float(nyquist - 1),
                value=float(min(1000, nyquist * 0.4)),
                step=100.0,
            )
        ]
    else:
        lower_default = max(1.0, min(500.0, nyquist * 0.2, nyquist - 2))
        upper_default = max(lower_default + 1, min(2000.0, nyquist * 0.7, nyquist - 1))
        lower_cut = st.number_input("Corte inferior (Hz)", min_value=1.0, max_value=float(nyquist - 2), value=lower_default, step=100.0)
        upper_cut = st.number_input("Corte superior (Hz)", min_value=2.0, max_value=float(nyquist - 1), value=upper_default, step=100.0)
        cutoff = [lower_cut, upper_cut]

    is_fir = method.startswith("FIR")
    order = st.slider("Orden", min_value=2 if is_fir else 1, max_value=200 if is_fir else 12, value=64 if is_fir else 4)
    beta = st.slider("Beta de Kaiser", min_value=0.0, max_value=14.0, value=8.6, step=0.1) if is_fir else None
    auto_update = st.toggle(
        "Actualizar al cambiar parámetros",
        value=len(audio) <= 200_000,
        help="En archivos largos se recomienda aplicar el filtro con el botón para evitar procesamientos repetidos.",
    )
    apply_button = st.button("Aplicar / actualizar filtro", type="primary", width="stretch")

    with st.expander("Métricas con referencia limpia"):
        st.caption("La referencia debe estar limpia, alineada y tener la misma frecuencia de muestreo y duración.")
        reference_file = st.file_uploader("WAV de referencia", type=["wav"], key="clean_reference_upload")

clean_reference = None
reference_error = None
if reference_file is not None:
    try:
        reference_rate, clean_reference, _ = read_wav_file(reference_file)
        if reference_rate != sample_rate or len(clean_reference) != len(audio):
            clean_reference = None
            reference_error = "La referencia debe tener la misma frecuencia de muestreo y cantidad de muestras."
    except Exception as error:
        clean_reference = None
        reference_error = f"No se pudo leer la referencia: {error}"

analysis_count = min(len(audio), max(1, int(fft_seconds * sample_rate)), 2_000_000)
analysis_audio = audio[:analysis_count]
fft_key = (upload_id, analysis_count, fft_window)
frequencies, amplitude = cached_spectrum(fft_key, analysis_audio, sample_rate, fft_window)
peaks = find_dominant_frequencies(frequencies, amplitude, count=5)
processed_spectrum = None
if current_result:
    processed_key = (upload_id, current_result["filter_key"], analysis_count, fft_window)
    processed_f, processed_a = cached_spectrum(
        processed_key,
        current_result["processed"][:analysis_count],
        sample_rate,
        fft_window,
    )
    processed_spectrum = (processed_f, processed_a)

filter_key = (upload_id, method, filter_label, tuple(cutoff), int(order), beta)
previous_key = current_result.get("filter_key") if current_result else None
should_apply = apply_button or (auto_update and previous_key != filter_key)

if should_apply:
    try:
        if is_fir:
            coefficients = design_fir(filter_label, sample_rate, order, cutoff, beta)
            processed = apply_fir(audio, coefficients)
            response_f, response_db = fir_response(coefficients, sample_rate)
            implementation = fir_implementation_name(len(audio)) + " Diseño Kaiser con SciPy."
            effective_order = len(coefficients) - 1
        else:
            coefficients = design_iir(filter_label, sample_rate, order, cutoff)
            processed = apply_iir(audio, coefficients)
            response_f, response_db = iir_response(coefficients, sample_rate)
            implementation = iir_implementation_name(len(audio)) + " Diseño Butterworth con SciPy."
            effective_order = order

        current_result = {
            "upload_id": upload_id,
            "filter_key": filter_key,
            "processed": processed,
            "response_f": response_f,
            "response_db": response_db,
            "title": f"Respuesta · {method} · {filter_label}",
            "implementation": implementation,
            "order": effective_order,
        }
        st.session_state["sigma_result"] = current_result
    except Exception as error:
        st.error(f"No se pudo aplicar el filtro: {error}")

if analysis_count < len(audio):
    st.caption(f"Análisis FFT sobre los primeros {analysis_count / sample_rate:.1f} s. Los filtros se aplican al archivo completo.")

analysis_tab, compare_tab, export_tab = st.tabs(["Análisis", "Filtro y comparación", "Audio y métricas"])

with analysis_tab:
    st.subheader("Forma de onda")
    show_interactive_chart(time_comparison_figure(audio, sample_rate), key="sigma_time_chart")

    st.subheader("Espectro de frecuencias")
    show_interactive_chart(
        spectrum_figure(frequencies, amplitude, peaks=peaks),
        key="sigma_frequency_chart",
    )

    if peaks:
        st.subheader("Picos predominantes")
        st.dataframe(
            [{"Frecuencia (Hz)": round(peak["frequency"], 1), "Amplitud": round(peak["amplitude"], 4)} for peak in peaks],
            hide_index=True,
            width="stretch",
        )

with compare_tab:
    if current_result is None:
        st.info("Elegí los parámetros del filtro en el panel lateral y presioná «Aplicar / actualizar filtro».")
    else:
        st.subheader("Respuesta en frecuencia")
        st.caption(current_result["implementation"] + f" Orden efectivo: {current_result['order']}.")
        show_interactive_chart(
            filter_response_figure(current_result["response_f"], current_result["response_db"], current_result["title"]),
            key="sigma_filter_response_chart",
        )
        st.subheader("Comparación directa")
        st.caption("En las gráficas podés acercar, desplazar, ocultar curvas desde la leyenda y recuperar la vista original.")
        compare_left, compare_right = st.columns(2)
        with compare_left:
            show_interactive_chart(time_comparison_figure(audio, sample_rate, current_result["processed"]), key="sigma_compare_time_chart")
        with compare_right:
            show_interactive_chart(
                spectrum_figure(frequencies, amplitude, processed=processed_spectrum),
                key="sigma_compare_frequency_chart",
            )

with export_tab:
    if current_result is None:
        st.info("Aplicá un filtro para habilitar la escucha y descarga del resultado.")
    else:
        original_col, processed_col = st.columns(2)
        with original_col:
            st.markdown("**Audio original**")
            st.audio(to_wav_bytes(audio, sample_rate), format="audio/wav")
        with processed_col:
            st.markdown("**Audio procesado**")
            st.audio(to_wav_bytes(current_result["processed"], sample_rate), format="audio/wav")
            st.download_button(
                "Descargar WAV procesado",
                data=to_wav_bytes(current_result["processed"], sample_rate),
                file_name="sigma_procesado.wav",
                mime="audio/wav",
                width="stretch",
            )

        if reference_error:
            st.error(reference_error)
        elif clean_reference is not None:
            mse = mean_squared_error(clean_reference, current_result["processed"])
            snr = signal_to_noise_ratio_db(clean_reference, current_result["processed"])
            metric_left, metric_right = st.columns(2)
            metric_left.metric("MSE vs. referencia limpia", f"{mse:.6g}")
            metric_right.metric("SNR vs. referencia limpia", "∞ dB" if np.isinf(snr) else f"{snr:.2f} dB")
        else:
            st.info("MSE y SNR aparecen al cargar una referencia limpia, alineada y con igual Fs y duración.")
