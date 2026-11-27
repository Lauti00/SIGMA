
import hashlib
import io

import numpy as np
import streamlit as st
from scipy.io import wavfile

from core.audio import read_wav_file
from core.filter_response import fir_response, iir_response
from core.fir import (
    FILTER_TYPES,
    apply_fir,
    design_fir,
    fir_implementation_name,
)
from core.fourier import (
    WINDOW_TYPES,
    compute_spectrum,
    find_dominant_frequencies,
)
from core.iir import (
    IIR_FAMILIES,
    apply_iir,
    design_iir,
    iir_implementation_name,
)
from core.metrics import mean_squared_error, signal_to_noise_ratio_db
from core.noise import NOISE_TYPES, add_noise_at_snr

from ui.styles import (
    apply_styles,
    brand,
    chips,
    footer,
    hero,
    section,
    side_label,
)
from ui.welcome import render_welcome, select_active_file

from visualization.interactive_plots import (
    filter_response_figure,
    spectrum_figure,
    time_comparison_figure,
)


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="SIGMA | Audio Lab",
    page_icon="🎚️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_styles()


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def to_wav_bytes(audio, sample_rate):
    """Serializa el audio mono procesado en WAV de 32 bits."""
    buffer = io.BytesIO()

    wavfile.write(
        buffer,
        sample_rate,
        np.asarray(audio, dtype=np.float32),
    )

    return buffer.getvalue()


def show_interactive_chart(figure, key):
    """Muestra un gráfico Plotly interactivo."""
    st.plotly_chart(
        figure,
        theme="streamlit",
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "responsive": True,
        },
    )


def cached_spectrum(cache_key, signal, sample_rate, window):
    """Almacena temporalmente los espectros calculados."""
    cache = st.session_state.setdefault(
        "sigma_spectrum_cache",
        {},
    )

    if cache_key not in cache:
        cache[cache_key] = compute_spectrum(
            signal,
            sample_rate,
            window,
        )

        while len(cache) > 6:
            cache.pop(next(iter(cache)))

    return cache[cache_key]


def format_snr(value):
    """Formatea la relación señal-ruido."""
    if np.isinf(value):
        return "∞ dB"

    if np.isnan(value):
        return "No disponible"

    return f"{value:.2f} dB"


def render_audio_card(title, description, audio_bytes):
    """Presenta un reproductor de audio dentro de una tarjeta."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(description)
        st.audio(audio_bytes, format="audio/wav")


def dashboard_item(label, value, detail=""):
    """Genera un indicador visual del Dashboard."""
    detail_html = f"<small>{detail}</small>" if detail else ""

    return (
        '<div class="sigma-dashboard-item">'
        f"<span>{label}</span>"
        f"<strong>{value}</strong>"
        f"{detail_html}"
        "</div>"
    )


def render_dashboard(
    sample_rate,
    duration,
    samples,
    channels,
    method,
    filter_label,
    order,
    cutoff,
    beta,
    nyquist,
    display_result,
    current_result,
    clean_reference,
    reference_error,
):
    """Dashboard visual con el estado y resumen del procesamiento."""

    if display_result is not None:
        status_class = "sigma-status-ready"
        status_text = "Procesamiento completado"
        status_message = (
            "El resultado corresponde a la configuración actual "
            "del filtro."
        )

    elif current_result is not None:
        status_class = "sigma-status-pending"
        status_text = "Configuración pendiente"
        status_message = (
            "Cambiaste los parámetros. Aplicá el filtro "
            "para actualizar el resultado."
        )

    else:
        status_class = "sigma-status-idle"
        status_text = "Análisis sin procesar"
        status_message = (
            "La señal está cargada y lista para configurar "
            "el procesamiento."
        )

    st.markdown(
        '<div class="sigma-dashboard">'
        '<div class="sigma-dashboard-heading">'
        '<div>'
        '<h3>Resumen de la sesión</h3>'
        '<p>Estado general del análisis de audio</p>'
        '</div>'
        f'<div class="sigma-status {status_class}">'
        '<span class="sigma-status-dot"></span>'
        f'{status_text}'
        '</div>'
        '</div>'
        '<div class="sigma-dashboard-grid">'
        + dashboard_item(
            "Frecuencia de muestreo",
            f"{sample_rate:,} Hz",
            "Sampling rate",
        )
        + dashboard_item(
            "Duración",
            f"{duration:.2f} s",
            "Tiempo total de señal",
        )
        + dashboard_item(
            "Muestras",
            f"{samples:,}",
            "Cantidad de muestras",
        )
        + dashboard_item(
            "Canales",
            f"{channels} → mono" if channels == 2 else "Mono",
            "Formato de procesamiento",
        )
        + '</div>'
        '<div class="sigma-dashboard-note">'
        f'{status_message}'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    section(
        "Configuración del procesamiento",
        "Parámetros del filtro seleccionado y estado de ejecución.",
    )

    result_order = (
        str(display_result["order"])
        if display_result is not None
        else "Pendiente"
    )

    cutoff_description = " – ".join(
        f"{frequency:g} Hz"
        for frequency in cutoff
    )

    config_html = (
        '<div class="sigma-dashboard">'
        '<div class="sigma-dashboard-grid">'
        + dashboard_item(
            "Familia",
            method,
            "Arquitectura del filtro",
        )
        + dashboard_item(
            "Respuesta",
            filter_label,
            "Tipo de filtro",
        )
        + dashboard_item(
            "Orden efectivo",
            result_order,
            f"Orden solicitado: {order}",
        )
        + dashboard_item(
            "Frecuencia de corte",
            cutoff_description,
            "Frecuencia en Hz",
        )
        + '</div>'
        '</div>'
    )

    st.markdown(config_html, unsafe_allow_html=True)

    if method.startswith("FIR"):
        st.caption(
            f"Beta de Kaiser: {beta:.1f} · "
            f"Frecuencia de Nyquist: {nyquist:.0f} Hz"
        )
    else:
        st.caption(
            f"{method} · Frecuencia de Nyquist: {nyquist:.0f} Hz"
        )

    section(
        "Evaluación de calidad",
        "Comparación cuantitativa contra una referencia limpia.",
    )

    if reference_error:
        st.warning(
            "No se pueden calcular las métricas: "
            + reference_error
        )

    elif clean_reference is not None and display_result is not None:
        mse = mean_squared_error(
            clean_reference,
            display_result["processed"],
        )

        snr = signal_to_noise_ratio_db(
            clean_reference,
            display_result["processed"],
        )

        metric_cols = st.columns(2)

        metric_cols[0].metric(
            "Error cuadrático medio (MSE)",
            f"{mse:.6g}",
            help=(
                "Mide el error cuadrático medio entre "
                "la referencia limpia y la señal procesada."
            ),
        )

        metric_cols[1].metric(
            "Relación señal-ruido (SNR)",
            format_snr(snr),
            help=(
                "Compara la potencia de la referencia "
                "con el error de la señal procesada."
            ),
        )

        st.caption(
            "Las métricas se calculan utilizando la referencia "
            "limpia y la señal procesada."
        )

    elif display_result is not None:
        st.info(
            "Para calcular MSE y SNR necesitás una referencia "
            "limpia alineada con el audio procesado."
        )

    else:
        st.caption(
            "Las métricas estarán disponibles después de aplicar "
            "el filtro y contar con una referencia limpia válida."
        )


# =========================================================
# PANEL LATERAL: CARGA DE AUDIO
# =========================================================

with st.sidebar:
    brand()

    st.markdown("### Fuente de audio")

    uploaded_file = st.file_uploader(
        "Seleccioná un archivo WAV",
        type=["wav"],
        help=(
            "WAV mono o estéreo. "
            "El audio estéreo se convierte a mono "
            "para el procesamiento."
        ),
    )

    active_file = select_active_file(uploaded_file)


# =========================================================
# ENCABEZADO PRINCIPAL
# =========================================================

hero(
    "SIGMA Audio Lab",
    (
        "Analizá, filtrá y compará señales de audio. "
        "Explorá qué sucede antes y después del procesamiento "
        "en el dominio temporal y frecuencial."
    ),
    compact=active_file is not None,
)


# =========================================================
# PANTALLA DE BIENVENIDA
# =========================================================

if active_file is None:
    st.session_state.pop("sigma_result", None)
    st.session_state.pop("sigma_spectrum_cache", None)
    st.session_state.pop("sigma_upload_id", None)

    render_welcome()
    st.stop()


# =========================================================
# LECTURA DEL ARCHIVO WAV
# =========================================================

try:
    file_bytes = active_file.getvalue()

    upload_id = hashlib.sha256(
        file_bytes
    ).hexdigest()

    sample_rate, source_audio, channels = read_wav_file(
        active_file
    )

except Exception as error:
    st.error(f"No se pudo abrir el WAV: {error}")
    st.stop()


previous_upload_id = st.session_state.get(
    "sigma_upload_id"
)

if previous_upload_id != upload_id:
    st.session_state.pop("sigma_result", None)
    st.session_state.pop("sigma_spectrum_cache", None)
    st.session_state["sigma_upload_id"] = upload_id


if sample_rate < 4:
    st.error(
        "La frecuencia de muestreo es demasiado baja "
        "para diseñar estos filtros."
    )
    st.stop()


# =========================================================
# INFORMACIÓN GENERAL DEL AUDIO
# =========================================================

nyquist = sample_rate / 2
duration = len(source_audio) / sample_rate

section(
    "Señal de entrada",
    "Información técnica del archivo cargado.",
)

metric_cols = st.columns(4)

metric_cols[0].metric(
    "Duración",
    f"{duration:.2f} s",
)

metric_cols[1].metric(
    "Frecuencia de muestreo",
    f"{sample_rate:,} Hz",
)

metric_cols[2].metric(
    "Cantidad de muestras",
    f"{len(source_audio):,}",
)

metric_cols[3].metric(
    "Canales",
    f"{channels} → mono" if channels == 2 else "Mono",
)

if channels == 2:
    st.caption(
        "La señal estéreo se promedió a mono "
        "para el análisis y el procesamiento."
    )


# =========================================================
# PANEL LATERAL: EXPERIMENTOS Y FFT
# =========================================================

with st.sidebar:
    st.divider()
    side_label("Experimentos")

    with st.expander(
        "Agregar ruido sintético",
        expanded=False,
    ):
        add_synthetic_noise = st.toggle(
            "Generar contaminación sintética",
            value=False,
            key=f"sigma_add_noise_{upload_id}",
            help=(
                "Usa el WAV cargado como referencia limpia "
                "y genera una mezcla controlada."
            ),
        )

        noise_type = st.selectbox(
            "Tipo de ruido",
            NOISE_TYPES,
            key=f"sigma_noise_type_{upload_id}",
        )

        target_snr_db = st.slider(
            "SNR objetivo (dB)",
            min_value=-5,
            max_value=40,
            value=10,
            step=1,
            key=f"sigma_noise_snr_{upload_id}",
        )

        tone_frequency = 50.0

        if noise_type == "Tono sinusoidal":
            min_tone_frequency = max(
                0.001,
                nyquist * 1e-6,
            )

            max_tone_frequency = (
                nyquist - min_tone_frequency
            )

            tone_key = f"sigma_noise_frequency_{upload_id}"

            tone_default = float(
                np.clip(
                    50.0,
                    min_tone_frequency,
                    max_tone_frequency,
                )
            )

            tone_input = {
                "min_value": float(min_tone_frequency),
                "max_value": float(max_tone_frequency),
                "step": float(
                    min(
                        10.0,
                        max(0.01, nyquist / 100),
                    )
                ),
                "key": tone_key,
            }

            if tone_key in st.session_state:
                st.session_state[tone_key] = float(
                    np.clip(
                        st.session_state[tone_key],
                        min_tone_frequency,
                        max_tone_frequency,
                    )
                )
            else:
                tone_input["value"] = tone_default

            tone_frequency = st.number_input(
                "Frecuencia del tono (Hz)",
                **tone_input,
            )

    st.divider()
    side_label("Análisis espectral")

    max_fft_seconds = min(
        duration,
        30.0,
        2_000_000 / sample_rate,
    )

    fft_seconds = st.slider(
        "Tramo para FFT",
        min_value=0.1,
        max_value=max(0.1, float(max_fft_seconds)),
        value=max(
            0.1,
            min(10.0, float(max_fft_seconds)),
        ),
        step=0.1,
        key=f"sigma_fft_seconds_{upload_id}",
        help=(
            "La FFT usa el inicio del audio. "
            "El filtrado sigue usando la señal completa."
        ),
    )

    fft_window = st.selectbox(
        "Ventana",
        WINDOW_TYPES,
        key=f"sigma_fft_window_{upload_id}",
    )


# =========================================================
# GENERACIÓN DE LA SEÑAL DE ANÁLISIS
# =========================================================

audio = source_audio

clean_reference = None
generated_noise = None

if add_synthetic_noise:
    try:
        clean_reference, audio, generated_noise = add_noise_at_snr(
            source_audio,
            sample_rate,
            target_snr_db,
            noise_type=noise_type,
            tone_frequency=tone_frequency,
            seed=int(upload_id[:8], 16),
        )

    except ValueError as error:
        st.error(
            f"No se pudo generar el ruido de prueba: {error}"
        )
        st.stop()

    noise_rms = float(
        np.sqrt(np.mean(generated_noise**2))
    )

    st.info(
        f"Se agregó {noise_type.lower()} "
        f"a {target_snr_db} dB de SNR objetivo. "
        f"RMS del ruido: {noise_rms:.5f}."
    )


signal_config = (
    upload_id,
    add_synthetic_noise,
    noise_type if add_synthetic_noise else None,
    target_snr_db if add_synthetic_noise else None,
    tone_frequency if add_synthetic_noise else None,
)

analysis_id = hashlib.sha256(
    repr(signal_config).encode("utf-8")
).hexdigest()

previous_signal_id = st.session_state.get(
    "sigma_signal_id"
)

if previous_signal_id != analysis_id:
    st.session_state.pop("sigma_result", None)
    st.session_state.pop("sigma_spectrum_cache", None)
    st.session_state["sigma_signal_id"] = analysis_id


# =========================================================
# LABORATORIO DE FILTROS
# =========================================================

section(
    "Laboratorio de filtros",
    "Configurá los parámetros del procesamiento digital.",
)

with st.container(border=True):
    st.markdown("#### 1. Configuración del filtro")

    col_method, col_response, col_order = st.columns(
        [1.2, 1.2, 1]
    )

    with col_method:
        method = st.selectbox(
            "Familia del filtro",
            ["FIR · Kaiser"]
            + [f"IIR · {family}" for family in IIR_FAMILIES],
            key="sigma_filter_method",
            help=(
                "FIR utiliza un diseño por ventana Kaiser. "
                "IIR ofrece diseños Butterworth, Chebyshev I, "
                "Chebyshev II y elíptico."
            ),
        )

    with col_response:
        filter_label = st.selectbox(
            "Tipo de respuesta",
            list(FILTER_TYPES),
            key="sigma_filter_response",
        )

    filter_kind = FILTER_TYPES[filter_label]
    is_fir = method.startswith("FIR")
    iir_family = method.split(" · ", 1)[1] if not is_fir else None

    with col_order:
        order = st.slider(
            "Orden del filtro",
            min_value=2 if is_fir else 1,
            max_value=200 if is_fir else 12,
            value=64 if is_fir else 4,
            key="sigma_filter_order",
        )

    passband_ripple = 1.0
    stopband_attenuation = 40.0

    st.divider()
    st.markdown("#### 2. Parámetros de frecuencia")

    minimum_cutoff = max(
        nyquist * 0.01,
        np.finfo(float).eps,
    )

    input_step = min(
        100.0,
        max(minimum_cutoff, nyquist / 20),
    )

    if filter_kind in ("lowpass", "highpass"):
        cutoff_col, beta_col = st.columns(2)

        max_single_cutoff = (
            nyquist - minimum_cutoff
        )

        single_default = float(
            np.clip(
                min(1000.0, nyquist * 0.4),
                minimum_cutoff,
                max_single_cutoff,
            )
        )

        with cutoff_col:
            cutoff = [
                st.number_input(
                    "Frecuencia de corte (Hz)",
                    min_value=float(minimum_cutoff),
                    max_value=float(max_single_cutoff),
                    value=single_default,
                    step=float(input_step),
                    key=f"sigma_single_cutoff_{upload_id}",
                )
            ]

        with beta_col:
            if is_fir:
                beta = st.slider(
                    "Beta de Kaiser",
                    min_value=0.0,
                    max_value=14.0,
                    value=8.6,
                    step=0.1,
                    key="sigma_filter_beta",
                )
            else:
                beta = None
                st.info(
                    "Los filtros IIR no utilizan el parámetro Beta."
                )

    else:
        cutoff_col, upper_col = st.columns(2)

        minimum_gap = minimum_cutoff

        lower_max = (
            nyquist - minimum_gap - minimum_cutoff
        )

        lower_default = float(
            np.clip(
                min(500.0, nyquist * 0.2),
                minimum_cutoff,
                lower_max,
            )
        )

        with cutoff_col:
            lower_cut = st.number_input(
                "Frecuencia de corte inferior (Hz)",
                min_value=float(minimum_cutoff),
                max_value=float(lower_max),
                value=lower_default,
                step=float(input_step),
                key=f"sigma_lower_cutoff_{upload_id}",
            )

        upper_min = lower_cut + minimum_gap
        upper_max = nyquist - minimum_cutoff

        upper_default = float(
            np.clip(
                min(2000.0, nyquist * 0.7),
                upper_min,
                upper_max,
            )
        )

        upper_key = f"sigma_upper_cutoff_{upload_id}"

        upper_input = {
            "min_value": float(upper_min),
            "max_value": float(upper_max),
            "step": float(input_step),
            "key": upper_key,
        }

        if upper_key in st.session_state:
            st.session_state[upper_key] = float(
                np.clip(
                    st.session_state[upper_key],
                    upper_min,
                    upper_max,
                )
            )
        else:
            upper_input["value"] = upper_default

        with upper_col:
            upper_cut = st.number_input(
                "Frecuencia de corte superior (Hz)",
                **upper_input,
            )

        cutoff = [
            lower_cut,
            upper_cut,
        ]

        if is_fir:
            beta = st.slider(
                "Beta de Kaiser",
                min_value=0.0,
                max_value=14.0,
                value=8.6,
                step=0.1,
                key="sigma_filter_beta",
            )
        else:
            beta = None
            st.info(
                "Los filtros IIR no utilizan el parámetro Beta."
            )

    if not is_fir:
        needs_ripple = iir_family in ("Chebyshev I", "Elíptico")
        needs_attenuation = iir_family in ("Chebyshev II", "Elíptico")

        if needs_ripple and needs_attenuation:
            ripple_col, attenuation_col = st.columns(2)
            with ripple_col:
                passband_ripple = st.number_input(
                    "Ondulación de banda pasante (dB)",
                    min_value=0.01,
                    max_value=20.0,
                    value=1.0,
                    step=0.1,
                    key="sigma_iir_passband_ripple",
                    help="Parámetro rp de Chebyshev I y elíptico.",
                )
            with attenuation_col:
                stopband_attenuation = st.number_input(
                    "Atenuación de banda eliminada (dB)",
                    min_value=0.01,
                    max_value=120.0,
                    value=40.0,
                    step=1.0,
                    key="sigma_iir_stopband_attenuation",
                    help="Parámetro rs de Chebyshev II y elíptico.",
                )
        elif needs_ripple:
            passband_ripple = st.number_input(
                "Ondulación de banda pasante (dB)",
                min_value=0.01,
                max_value=20.0,
                value=1.0,
                step=0.1,
                key="sigma_iir_passband_ripple",
                help="Parámetro rp de Chebyshev I.",
            )
        elif needs_attenuation:
            stopband_attenuation = st.number_input(
                "Atenuación de banda eliminada (dB)",
                min_value=0.01,
                max_value=120.0,
                value=40.0,
                step=1.0,
                key="sigma_iir_stopband_attenuation",
                help="Parámetro rs de Chebyshev II.",
            )

    st.divider()
    st.markdown("#### 3. Resumen de configuración")

    config_cols = st.columns(4)

    config_cols[0].metric(
        "Familia",
        method,
    )

    config_cols[1].metric(
        "Respuesta",
        filter_label,
    )

    config_cols[2].metric(
        "Orden solicitado",
        str(order),
    )

    config_cols[3].metric(
        "Frecuencia de Nyquist",
        f"{nyquist:.0f} Hz",
    )

    st.divider()

    update_col, button_col = st.columns([1.4, 1])

    with update_col:
        auto_update = st.toggle(
            "Actualizar automáticamente al cambiar parámetros",
            value=len(audio) <= 200_000,
            key="sigma_filter_auto_update",
            help=(
                "En archivos largos conviene aplicar "
                "el filtro manualmente."
            ),
        )

    with button_col:
        apply_button = st.button(
            "Aplicar filtro",
            type="primary",
            width="stretch",
            key="sigma_apply_filter",
        )

    st.caption(
        "El filtro se aplica sobre el audio completo. "
        "La respuesta en frecuencia permite analizar "
        "su comportamiento."
    )

    with st.expander(
        "Configuración de referencia limpia",
        expanded=False,
    ):
        reference_file = None

        if add_synthetic_noise:
            st.caption(
                "Se utilizará automáticamente el WAV original, "
                "antes de agregar ruido, como referencia limpia."
            )
        else:
            st.caption(
                "Subí una señal limpia alineada con el audio "
                "analizado, con igual frecuencia de muestreo "
                "y duración."
            )

            reference_file = st.file_uploader(
                "Archivo WAV de referencia",
                type=["wav"],
                key="clean_reference_upload",
            )


# =========================================================
# LECTURA DE REFERENCIA LIMPIA
# =========================================================

current_result = st.session_state.get(
    "sigma_result"
)

reference_error = None

if reference_file is not None:
    try:
        reference_rate, clean_reference, _ = read_wav_file(
            reference_file
        )

        if (
            reference_rate != sample_rate
            or len(clean_reference) != len(audio)
        ):
            clean_reference = None

            reference_error = (
                "La referencia debe tener la misma frecuencia "
                "de muestreo y cantidad de muestras."
            )

    except Exception as error:
        clean_reference = None

        reference_error = (
            f"No se pudo leer la referencia: {error}"
        )


# =========================================================
# ANÁLISIS FFT
# =========================================================

analysis_count = min(
    len(audio),
    max(1, int(fft_seconds * sample_rate)),
    2_000_000,
)

analysis_audio = audio[:analysis_count]

fft_key = (
    analysis_id,
    analysis_count,
    fft_window,
)

frequencies, amplitude = cached_spectrum(
    fft_key,
    analysis_audio,
    sample_rate,
    fft_window,
)

peaks = find_dominant_frequencies(
    frequencies,
    amplitude,
    count=5,
)


# =========================================================
# APLICACIÓN DEL FILTRO
# =========================================================

filter_key = (
    analysis_id,
    method,
    filter_label,
    tuple(cutoff),
    int(order),
    beta,
    iir_family,
    passband_ripple,
    stopband_attenuation,
)

previous_key = (
    current_result.get("filter_key")
    if current_result
    else None
)

should_apply = (
    apply_button
    or (
        auto_update
        and previous_key != filter_key
    )
)

if should_apply:
    try:
        if is_fir:
            coefficients = design_fir(
                filter_label,
                sample_rate,
                order,
                cutoff,
                beta,
            )

            processed = apply_fir(
                audio,
                coefficients,
            )

            response_f, response_db = fir_response(
                coefficients,
                sample_rate,
            )

            implementation = (
                fir_implementation_name(len(audio))
                + " Diseño Kaiser con SciPy."
            )

            effective_order = len(coefficients) - 1

        else:
            coefficients = design_iir(
                filter_label,
                sample_rate,
                order,
                cutoff,
                family=iir_family,
                passband_ripple_db=passband_ripple,
                stopband_attenuation_db=stopband_attenuation,
            )

            processed = apply_iir(
                audio,
                coefficients,
            )

            response_f, response_db = iir_response(
                coefficients,
                sample_rate,
            )

            implementation = (
                iir_implementation_name(len(audio))
                + f" Diseño {iir_family} con SciPy."
            )

            effective_order = order

        current_result = {
            "upload_id": analysis_id,
            "filter_key": filter_key,
            "processed": processed,
            "response_f": response_f,
            "response_db": response_db,
            "title": (
                f"Respuesta · {method} · {filter_label}"
            ),
            "implementation": implementation,
            "order": effective_order,
        }

        st.session_state["sigma_result"] = current_result

    except Exception as error:
        st.error(
            f"No se pudo aplicar el filtro: {error}"
        )


# =========================================================
# PREPARACIÓN DE RESULTADOS
# =========================================================

result_is_current = (
    current_result is not None
    and current_result.get("filter_key") == filter_key
)

display_result = (
    current_result
    if result_is_current
    else None
)

processed_spectrum = None

if display_result is not None:
    processed_key = (
        analysis_id,
        display_result["filter_key"],
        analysis_count,
        fft_window,
    )

    processed_f, processed_a = cached_spectrum(
        processed_key,
        display_result["processed"][:analysis_count],
        sample_rate,
        fft_window,
    )

    processed_spectrum = (
        processed_f,
        processed_a,
    )

if analysis_count < len(audio):
    st.caption(
        f"Análisis FFT sobre los primeros "
        f"{analysis_count / sample_rate:.1f} s. "
        "Los filtros se aplican al archivo completo."
    )


# =========================================================
# ESPECTRO DE REFERENCIA LIMPIA
# =========================================================

reference_spectrum = None

if clean_reference is not None:
    reference_id = hashlib.sha256(
        clean_reference.tobytes()
    ).hexdigest()

    reference_spectrum = cached_spectrum(
        (
            analysis_id,
            "reference",
            reference_id,
            analysis_count,
            fft_window,
        ),
        clean_reference[:analysis_count],
        sample_rate,
        fft_window,
    )


# =========================================================
# DASHBOARD
# =========================================================

render_dashboard(
    sample_rate=sample_rate,
    duration=duration,
    samples=len(audio),
    channels=channels,
    method=method,
    filter_label=filter_label,
    order=order,
    cutoff=cutoff,
    beta=beta,
    nyquist=nyquist,
    display_result=display_result,
    current_result=current_result,
    clean_reference=clean_reference,
    reference_error=reference_error,
)


# =========================================================
# PESTAÑAS PRINCIPALES
# =========================================================

analysis_tab, compare_tab, export_tab = st.tabs(
    [
        "📊 Análisis",
        "🎛️ Filtro y comparación",
        "🔊 Audio y métricas",
    ]
)


# =========================================================
# PESTAÑA 1: ANÁLISIS
# =========================================================

with analysis_tab:
    section(
        "Forma de onda",
        "Amplitud normalizada en función del tiempo.",
    )

    st.caption(
        "Observá la evolución de la señal a lo largo del tiempo. "
        "Podés acercar, desplazar y explorar los valores del gráfico."
    )

    show_interactive_chart(
        time_comparison_figure(
            audio,
            sample_rate,
        ),
        key="sigma_time_chart",
    )

    st.divider()

    section(
        "Espectro de frecuencias",
        "FFT de un lado: amplitud por bin de frecuencia.",
    )

    st.caption(
        "El espectro permite identificar qué frecuencias "
        "componen la señal y dónde se concentra su energía."
    )

    show_interactive_chart(
        spectrum_figure(
            frequencies,
            amplitude,
            peaks=peaks,
        ),
        key="sigma_frequency_chart",
    )

    if peaks:
        st.divider()

        section(
            "Picos predominantes",
            "Los cinco picos locales más altos del espectro.",
        )

        chips(
            [
                (
                    f"{peak['frequency']:.1f} Hz "
                    f"· amp {peak['amplitude']:.3f}"
                )
                for peak in peaks
            ]
        )

        with st.expander("Consultar tabla de frecuencias"):
            st.dataframe(
                [
                    {
                        "Frecuencia (Hz)": round(
                            peak["frequency"],
                            1,
                        ),
                        "Amplitud": round(
                            peak["amplitude"],
                            4,
                        ),
                    }
                    for peak in peaks
                ],
                hide_index=True,
                width="stretch",
            )


# =========================================================
# PESTAÑA 2: FILTRO Y COMPARACIÓN
# =========================================================

with compare_tab:
    if display_result is None:
        st.info(
            "Todavía no hay un resultado actualizado. "
            "Configurá los parámetros del filtro en el laboratorio "
            "y presioná «Aplicar filtro»."
        )

    else:
        section(
            "Resultado del procesamiento",
            "Resumen del filtro aplicado sobre la señal.",
        )

        st.success(
            "El filtro se aplicó correctamente. "
            "Ya podés analizar y escuchar el resultado."
        )

        result_cols = st.columns(4)

        result_cols[0].metric(
            "Familia",
            method,
        )

        result_cols[1].metric(
            "Tipo de filtro",
            filter_label,
        )

        result_cols[2].metric(
            "Orden efectivo",
            str(display_result["order"]),
        )

        result_cols[3].metric(
            "Muestras procesadas",
            f"{len(display_result['processed']):,}",
        )

        st.caption(
            display_result["implementation"]
        )

        st.divider()

        section(
            "Respuesta en frecuencia",
            "Magnitud del filtro en dB; 0 dB es ganancia unitaria.",
        )

        st.caption(
            "Esta curva muestra cómo responde el filtro "
            "ante las distintas frecuencias de entrada."
        )

        show_interactive_chart(
            filter_response_figure(
                display_result["response_f"],
                display_result["response_db"],
                display_result["title"],
            ),
            key="sigma_filter_response_chart",
        )

        st.divider()

        section(
            "Antes y después del filtrado",
            "Comparación de la señal original y la señal procesada.",
        )

        comparison_help = (
            "Cada señal aparece en su propio panel alineado y con "
            "la misma escala, para que las curvas no se tapen. "
            "Podés acercar y desplazar las gráficas."
        )

        if clean_reference is not None:
            comparison_help += (
                " La referencia limpia se muestra en un tercer panel."
            )

        st.caption(comparison_help)

        st.markdown("**Dominio temporal**")

        show_interactive_chart(
            time_comparison_figure(
                audio,
                sample_rate,
                display_result["processed"],
                reference=clean_reference,
                separate=True,
            ),
            key="sigma_compare_time_chart",
        )

        st.markdown("**Dominio frecuencial**")

        show_interactive_chart(
            spectrum_figure(
                frequencies,
                amplitude,
                processed=processed_spectrum,
                reference=reference_spectrum,
                separate=True,
            ),
            key="sigma_compare_frequency_chart",
        )


# =========================================================
# PESTAÑA 3: AUDIO Y MÉTRICAS
# =========================================================

with export_tab:
    if display_result is None:
        st.info(
            "Aplicá un filtro para habilitar "
            "la escucha y descarga del resultado."
        )

    else:
        section(
            "Escucha de señales",
            "Compará auditivamente el audio de entrada y el resultado.",
        )

        processed_bytes = to_wav_bytes(
            display_result["processed"],
            sample_rate,
        )

        if clean_reference is not None:
            clean_col, noisy_col, processed_col = st.columns(3)

            with clean_col:
                render_audio_card(
                    "Referencia limpia",
                    "Señal original antes de agregar contaminación.",
                    to_wav_bytes(
                        clean_reference,
                        sample_rate,
                    ),
                )

            with noisy_col:
                render_audio_card(
                    "Señal contaminada",
                    "Audio de entrada utilizado para el filtrado.",
                    to_wav_bytes(
                        audio,
                        sample_rate,
                    ),
                )

            with processed_col:
                render_audio_card(
                    "Audio procesado",
                    "Resultado obtenido después de aplicar el filtro.",
                    processed_bytes,
                )

        else:
            original_col, processed_col = st.columns(2)

            with original_col:
                render_audio_card(
                    "Audio original",
                    "Señal de entrada antes del procesamiento.",
                    to_wav_bytes(
                        audio,
                        sample_rate,
                    ),
                )

            with processed_col:
                render_audio_card(
                    "Audio procesado",
                    "Resultado obtenido después de aplicar el filtro.",
                    processed_bytes,
                )

        st.divider()

        section(
            "Exportar resultado",
            "Descargá la señal procesada en formato WAV.",
        )

        download_col, info_col = st.columns([1, 2])

        with download_col:
            st.download_button(
                "⬇️ Descargar WAV procesado",
                data=processed_bytes,
                file_name="sigma_procesado.wav",
                mime="audio/wav",
                width="stretch",
            )

        with info_col:
            st.caption(
                "El archivo exportado contiene el audio procesado "
                "en formato WAV de 32 bits flotantes, "
                f"con una frecuencia de muestreo de {sample_rate:,} Hz."
            )

        st.divider()

        section(
            "Evaluación de calidad",
            "Métricas de comparación entre la referencia limpia "
            "y la señal procesada.",
        )

        if reference_error:
            st.error(reference_error)

        elif clean_reference is not None:
            mse = mean_squared_error(
                clean_reference,
                display_result["processed"],
            )

            snr = signal_to_noise_ratio_db(
                clean_reference,
                display_result["processed"],
            )

            metric_left, metric_right = st.columns(2)

            metric_left.metric(
                "Error cuadrático medio (MSE)",
                f"{mse:.6g}",
                help=(
                    "Cuantifica el error cuadrático medio "
                    "entre la referencia limpia y el audio procesado. "
                    "Un valor menor indica menor diferencia."
                ),
            )

            metric_right.metric(
                "Relación señal-ruido (SNR)",
                format_snr(snr),
                help=(
                    "Compara la potencia de la referencia "
                    "con el error de la señal procesada. "
                    "Un valor mayor suele indicar mejor similitud."
                ),
            )

            st.caption(
                "Las métricas se calculan utilizando la referencia "
                "limpia y el resultado del filtrado. "
                "No representan una evaluación subjetiva "
                "de la calidad auditiva."
            )

        else:
            st.info(
                "Para calcular MSE y SNR necesitás una referencia "
                "limpia alineada con el audio procesado, "
                "con igual frecuencia de muestreo y duración."
            )

            st.caption(
                "Podés cargarla desde «Configuración de referencia "
                "limpia» en el laboratorio o activar el experimento "
                "de ruido sintético para utilizar automáticamente "
                "el WAV original como referencia."
            )


# =========================================================
# PIE DE PÁGINA
# =========================================================

footer()
