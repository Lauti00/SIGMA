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
from visualization.frequency_plot import plot_spectrum
from visualization.response_plot import plot_filter_response, plot_signal_comparison
from visualization.time_plot import plot_waveform


def to_wav_bytes(audio, sample_rate):
    """Serializa audio mono float como WAV de 32 bits."""
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, np.asarray(audio, dtype=np.float32))
    return buffer.getvalue()


st.set_page_config(page_title="SIGMA | Audio", page_icon="🎵", layout="wide")
st.title("SIGMA — Análisis y Procesamiento Digital de Audio")
st.write("Cargá un WAV para analizarlo y experimentar con filtros digitales.")

uploaded_file = st.file_uploader("Seleccioná un archivo WAV", type=["wav"])

if uploaded_file is not None:
    try:
        upload_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        sample_rate, audio, channels = read_wav_file(uploaded_file)
        nyquist = sample_rate / 2
        duration = len(audio) / sample_rate

        st.subheader("Información del audio")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frecuencia de muestreo", f"{sample_rate:,} Hz")
        col2.metric("Muestras", f"{len(audio):,}")
        col3.metric("Duración", f"{duration:.2f} s")
        col4.metric("Canales originales", str(channels))
        if channels == 2:
            st.caption("Los canales estéreo se promediaron para obtener una señal mono.")
        st.audio(to_wav_bytes(audio, sample_rate), format="audio/wav")

        clean_reference = None
        with st.expander("Métricas opcionales con referencia limpia", expanded=False):
            st.caption("La referencia debe estar limpia, alineada con el audio cargado y tener la misma frecuencia de muestreo y duración.")
            reference_file = st.file_uploader("WAV de referencia limpia", type=["wav"], key="clean_reference_upload")
            if reference_file is not None:
                try:
                    reference_rate, clean_reference, _ = read_wav_file(reference_file)
                    if reference_rate != sample_rate or len(clean_reference) != len(audio):
                        clean_reference = None
                        st.error("La referencia debe tener la misma frecuencia de muestreo y cantidad de muestras que el audio cargado.")
                except Exception as error:
                    clean_reference = None
                    st.error(f"No se pudo leer la referencia limpia: {error}")

        st.subheader("Análisis temporal")
        st.pyplot(plot_waveform(audio, sample_rate), clear_figure=True)
        st.subheader("Análisis frecuencial")
        analysis_max_seconds = min(duration, 30.0, 2_000_000 / sample_rate)
        analysis_seconds = st.slider(
            "Duración del tramo para FFT (desde el inicio)",
            min_value=0.1,
            max_value=max(0.1, float(analysis_max_seconds)),
            value=max(0.1, min(10.0, float(analysis_max_seconds))),
            step=0.1,
            help="Para archivos largos analizamos solo este tramo y reservamos la señal completa para aplicar filtros.",
        )
        analysis_count = min(len(audio), max(1, int(analysis_seconds * sample_rate)))
        analysis_audio = audio[:analysis_count]
        if analysis_count < len(audio):
            st.caption(f"FFT y frecuencias predominantes calculadas sobre los primeros {analysis_count / sample_rate:.1f} s; los filtros se aplican al audio completo.")
        fft_window = st.selectbox(
            "Ventana para FFT",
            WINDOW_TYPES,
            help="Hann y Hamming reducen leakage espectral. Rectangular conserva la señal sin ponderar.",
        )
        frequencies, amplitude = compute_spectrum(analysis_audio, sample_rate, fft_window)
        st.pyplot(plot_spectrum(frequencies, amplitude, title=f"Espectro de amplitud — ventana {fft_window}"), clear_figure=True)
        peaks = find_dominant_frequencies(frequencies, amplitude)
        if peaks:
            st.markdown("**Frecuencias predominantes (picos locales):**")
            st.dataframe(
                [{"Frecuencia (Hz)": round(p["frequency"], 1), "Amplitud": round(p["amplitude"], 4)} for p in peaks],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No se encontraron picos frecuenciales claros en esta señal.")

        st.subheader("Diseño y aplicación de filtros")
        with st.form("filter_form"):
            method = st.selectbox("Familia del filtro", ["FIR (ventana Kaiser)", "IIR (Butterworth)"])
            filter_label = st.selectbox("Tipo de filtro", list(FILTER_TYPES))
            if FILTER_TYPES[filter_label] in ("lowpass", "highpass"):
                cutoff_1 = st.number_input("Frecuencia de corte (Hz)", min_value=1.0, max_value=float(nyquist - 1), value=float(min(1000, nyquist * 0.4)), step=100.0)
                cutoff = [cutoff_1]
            else:
                lower = float(min(500, nyquist * 0.2))
                upper = float(min(2000, nyquist * 0.7))
                lower = max(1.0, min(lower, nyquist - 2))
                upper = max(lower + 1, min(upper, nyquist - 1))
                cutoff_low = st.number_input("Corte inferior (Hz)", min_value=1.0, max_value=float(nyquist - 2), value=lower, step=100.0)
                cutoff_high = st.number_input("Corte superior (Hz)", min_value=2.0, max_value=float(nyquist - 1), value=upper, step=100.0)
                cutoff = [cutoff_low, cutoff_high]
            order_min, order_max = (2, 200) if method.startswith("FIR") else (1, 12)
            default_order = 64 if method.startswith("FIR") else 4
            order = st.slider("Orden", min_value=order_min, max_value=order_max, value=default_order, step=1)
            if method.startswith("FIR"):
                beta = st.slider("Beta de Kaiser", min_value=0.0, max_value=14.0, value=8.6, step=0.1, help="Beta mayor suele reducir lóbulos laterales, a costa de una transición más ancha.")
            apply_button = st.form_submit_button("Aplicar filtro")

        if apply_button:
            try:
                if method.startswith("FIR"):
                    coefficients = design_fir(filter_label, sample_rate, order, cutoff, beta)
                    processed = apply_fir(audio, coefficients)
                    response_f, response_db = fir_response(coefficients, sample_rate)
                    implementation = fir_implementation_name(len(audio)) + " Coeficientes diseñados con `scipy.signal.firwin`."
                    effective_order = len(coefficients) - 1
                else:
                    coefficients = design_iir(filter_label, sample_rate, order, cutoff)
                    processed = apply_iir(audio, coefficients)
                    response_f, response_db = iir_response(coefficients, sample_rate)
                    implementation = iir_implementation_name(len(audio)) + " Diseño Butterworth con `scipy.signal.butter`."
                    effective_order = order

                st.session_state["sigma_result"] = {
                    "upload_id": upload_id,
                    "processed": processed,
                    "sample_rate": sample_rate,
                    "response_f": response_f,
                    "response_db": response_db,
                    "title": f"Respuesta en frecuencia — {method}, {filter_label}",
                    "implementation": implementation,
                    "order": effective_order,
                }
            except Exception as error:
                st.error(f"No se pudo aplicar el filtro: {error}")

        result = st.session_state.get("sigma_result")
        if result is not None and result["upload_id"] == upload_id:
            processed = result["processed"]
            st.subheader("Resultados")
            st.caption(result["implementation"] + f" Orden efectivo: {result['order']}.")
            st.pyplot(plot_filter_response(result["response_f"], result["response_db"], result["title"]), clear_figure=True)
            st.pyplot(plot_signal_comparison(audio, processed, sample_rate), clear_figure=True)
            original_f, original_a = compute_spectrum(analysis_audio, sample_rate, fft_window)
            processed_f, processed_a = compute_spectrum(processed[:analysis_count], sample_rate, fft_window)
            st.markdown("**Espectros: original y procesado**")
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(10, 3.5))
            axis.plot(original_f, original_a, label="Original", linewidth=0.8)
            axis.plot(processed_f, processed_a, label="Procesado", linewidth=0.8)
            axis.set_xlabel("Frecuencia (Hz)")
            axis.set_ylabel("Amplitud")
            axis.set_xlim(0, nyquist)
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            st.pyplot(figure, clear_figure=True)
            st.markdown("**Escucha y descarga**")
            st.audio(to_wav_bytes(processed, sample_rate), format="audio/wav")
            st.download_button("Descargar audio procesado (WAV)", data=to_wav_bytes(processed, sample_rate), file_name="sigma_procesado.wav", mime="audio/wav")
            if clean_reference is not None:
                mse = mean_squared_error(clean_reference, processed)
                snr = signal_to_noise_ratio_db(clean_reference, processed)
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("MSE respecto de la referencia limpia", f"{mse:.6g}")
                metric_col2.metric("SNR respecto de la referencia limpia", "∞ dB" if np.isinf(snr) else f"{snr:.2f} dB")
            else:
                st.caption("SNR y MSE requieren una referencia limpia, alineada y de igual duración; por eso no se calculan a partir del audio original filtrado.")
    except Exception as error:
        st.error(f"No se pudo procesar el archivo. Verificá que sea un WAV válido. Detalle: {error}")


