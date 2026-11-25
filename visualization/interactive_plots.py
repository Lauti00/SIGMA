
import math

import numpy as np
import plotly.graph_objects as go


COLORS = {
    "original": "#6685FF",
    "processed": "#FF9B45",
    "response": "#20C9A6",
    "peaks": "#D879FF",
    "grid": "rgba(160,175,210,0.12)",
    "text": "#DCE3F5",
    "muted": "#8995AD",
}


def _layout(figure, title, x_title, y_title, height=410):
    """Estilo visual unificado para los gráficos de SIGMA."""

    figure.update_layout(
        template="plotly_dark",
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.97,
            "font": {
                "size": 18,
                "color": COLORS["text"],
                "family": "Inter, Segoe UI, sans-serif",
            },
        },
        height=height,
        margin={"l": 65, "r": 30, "t": 78, "b": 65},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "Inter, Segoe UI, sans-serif",
            "size": 12,
            "color": COLORS["text"],
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0.01,
            "font": {"size": 11, "color": COLORS["text"]},
            "bgcolor": "rgba(18,24,38,0.65)",
            "bordercolor": "rgba(160,175,210,0.12)",
            "borderwidth": 1,
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#171F30",
            "bordercolor": "#34415C",
            "font": {
                "family": "Inter, sans-serif",
                "color": "#FFFFFF",
                "size": 12,
            },
        },
        uirevision="sigma-chart",
        dragmode="zoom",
    )

    figure.update_xaxes(
        title_text=x_title,
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        zeroline=False,
        showline=True,
        linecolor="rgba(160,175,210,0.2)",
        ticks="outside",
        ticklen=5,
        tickcolor="rgba(160,175,210,0.3)",
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"], "size": 12},
        automargin=True,
    )

    figure.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=COLORS["grid"],
        gridwidth=1,
        zeroline=False,
        showline=False,
        tickfont={"color": COLORS["muted"]},
        title_font={"color": COLORS["muted"], "size": 12},
        automargin=True,
    )

    return figure


def _minmax_decimate(signal, sample_rate, max_points=50_000):
    """Reduce puntos conservando máximos y mínimos de cada intervalo."""

    signal = np.asarray(signal, dtype=np.float64)

    if signal.size <= max_points:
        times = np.arange(signal.size) / sample_rate
        return times, signal, signal, False

    bucket_size = math.ceil(signal.size / max_points)
    usable = signal.size // bucket_size * bucket_size

    blocks = signal[:usable].reshape(-1, bucket_size)

    times = (
        np.arange(len(blocks)) * bucket_size + bucket_size / 2
    ) / sample_rate

    return times, blocks.min(axis=1), blocks.max(axis=1), True


def _add_time_signal(
    figure,
    signal,
    sample_rate,
    label,
    color,
    show_legend=True,
):
    """Agrega una señal temporal al gráfico."""

    times, lows, highs, reduced = _minmax_decimate(
        signal, sample_rate
    )

    group = label.lower()

    if not reduced:
        figure.add_trace(
            go.Scattergl(
                x=times,
                y=highs,
                mode="lines",
                name=label,
                showlegend=show_legend,
                line={
                    "color": color,
                    "width": 1.4,
                },
                hovertemplate=(
                    "%{x:.4f} s<br>"
                    "Amplitud: %{y:.5f}"
                    "<extra>" + label + "</extra>"
                ),
            )
        )
        return

    figure.add_trace(
        go.Scattergl(
            x=times,
            y=highs,
            mode="lines",
            name=label,
            legendgroup=group,
            showlegend=show_legend,
            line={
                "color": color,
                "width": 1.0,
            },
            hovertemplate=(
                "%{x:.4f} s<br>"
                "Máximo: %{y:.5f}"
                "<extra>" + label + "</extra>"
            ),
        )
    )

    fill_colors = {
        "Original": "rgba(102,133,255,0.13)",
        "Procesada": "rgba(255,155,69,0.13)",
        "Referencia limpia": "rgba(32,201,166,0.13)",
    }

    figure.add_trace(
        go.Scattergl(
            x=times,
            y=lows,
            mode="lines",
            name=label + " mín.",
            legendgroup=group,
            showlegend=False,
            line={
                "color": color,
                "width": 0.5,
            },
            fill="tonexty",
            fillcolor=fill_colors.get(
                label, "rgba(128,128,128,0.12)"
            ),
            hovertemplate=(
                "%{x:.4f} s<br>"
                "Mínimo: %{y:.5f}"
                "<extra>" + label + "</extra>"
            ),
        )
    )


def time_comparison_figure(
    original,
    sample_rate,
    processed=None,
    reference=None,
):
    """Gráfico temporal de la señal original y sus comparaciones."""

    figure = go.Figure()

    _add_time_signal(
        figure,
        original,
        sample_rate,
        "Original",
        COLORS["original"],
    )

    if processed is not None:
        _add_time_signal(
            figure,
            processed,
            sample_rate,
            "Procesada",
            COLORS["processed"],
        )

    if reference is not None:
        _add_time_signal(
            figure,
            reference,
            sample_rate,
            "Referencia limpia",
            COLORS["response"],
        )

    title = (
        "Señal en el tiempo"
        if processed is None
        else "Comparación temporal · original y procesada"
    )

    _layout(
        figure,
        title,
        "Tiempo (s)",
        "Amplitud normalizada",
        height=440,
    )

    figure.update_xaxes(
        rangeslider={
            "visible": True,
            "thickness": 0.065,
            "bgcolor": "rgba(120,140,180,0.06)",
            "bordercolor": "rgba(160,175,210,0.15)",
            "borderwidth": 1,
        }
    )

    return figure


def _peak_preserving_decimate(
    frequencies,
    amplitude,
    max_points=30_000,
):
    """Reduce el espectro conservando el máximo de cada intervalo."""

    frequencies = np.asarray(frequencies)
    amplitude = np.asarray(amplitude)

    if len(frequencies) <= max_points:
        return frequencies, amplitude

    bucket_size = math.ceil(len(frequencies) / max_points)

    output_frequencies = []
    output_amplitude = []

    for start in range(0, len(frequencies), bucket_size):
        end = min(start + bucket_size, len(frequencies))

        index = start + int(
            np.argmax(amplitude[start:end])
        )

        output_frequencies.append(frequencies[index])
        output_amplitude.append(amplitude[index])

    return (
        np.asarray(output_frequencies),
        np.asarray(output_amplitude),
    )


def spectrum_figure(
    frequencies,
    amplitude,
    peaks=None,
    processed=None,
    reference=None,
):
    """Espectro FFT con comparación y marcadores de frecuencias dominantes."""

    frequencies, amplitude = _peak_preserving_decimate(
        frequencies,
        amplitude,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scattergl(
            x=frequencies,
            y=amplitude,
            mode="lines",
            name="Original",
            line={
                "color": COLORS["original"],
                "width": 1.7,
            },
            hovertemplate=(
                "Frecuencia: %{x:.2f} Hz<br>"
                "Amplitud: %{y:.6f}"
                "<extra>Original</extra>"
            ),
        )
    )

    if processed is not None:
        processed_f, processed_a = _peak_preserving_decimate(
            *processed
        )

        figure.add_trace(
            go.Scattergl(
                x=processed_f,
                y=processed_a,
                mode="lines",
                name="Procesada",
                line={
                    "color": COLORS["processed"],
                    "width": 1.7,
                },
                hovertemplate=(
                    "Frecuencia: %{x:.2f} Hz<br>"
                    "Amplitud: %{y:.6f}"
                    "<extra>Procesada</extra>"
                ),
            )
        )

    if reference is not None:
        reference_f, reference_a = _peak_preserving_decimate(
            *reference
        )

        figure.add_trace(
            go.Scattergl(
                x=reference_f,
                y=reference_a,
                mode="lines",
                name="Referencia limpia",
                line={
                    "color": COLORS["response"],
                    "width": 1.5,
                    "dash": "dash",
                },
                hovertemplate=(
                    "Frecuencia: %{x:.2f} Hz<br>"
                    "Amplitud: %{y:.6f}"
                    "<extra>Referencia limpia</extra>"
                ),
            )
        )

    if peaks:
        figure.add_trace(
            go.Scattergl(
                x=[p["frequency"] for p in peaks],
                y=[p["amplitude"] for p in peaks],
                mode="markers+text",
                name="Picos principales",
                text=[
                    f'{p["frequency"]:.0f} Hz'
                    for p in peaks
                ],
                textposition="top center",
                textfont={
                    "size": 10,
                    "color": COLORS["peaks"],
                },
                marker={
                    "color": COLORS["peaks"],
                    "size": 9,
                    "symbol": "circle-open",
                    "line": {
                        "width": 2,
                        "color": COLORS["peaks"],
                    },
                },
                hovertemplate=(
                    "Frecuencia: %{x:.2f} Hz<br>"
                    "Amplitud: %{y:.6f}"
                    "<extra>Pico dominante</extra>"
                ),
            )
        )

    title = (
        "Espectro de amplitud · FFT"
        if processed is None
        else "Comparación espectral · original y procesada"
    )

    _layout(
        figure,
        title,
        "Frecuencia (Hz)",
        "Amplitud",
        height=440,
    )

    figure.update_xaxes(rangemode="tozero")

    return figure


def filter_response_figure(
    frequencies,
    magnitude_db,
    title,
):
    """Respuesta en frecuencia del filtro en decibelios."""

    figure = go.Figure()

    figure.add_trace(
        go.Scattergl(
            x=frequencies,
            y=magnitude_db,
            mode="lines",
            name="Respuesta del filtro",
            line={
                "color": COLORS["response"],
                "width": 2.4,
            },
            hovertemplate=(
                "Frecuencia: %{x:.2f} Hz<br>"
                "Magnitud: %{y:.2f} dB"
                "<extra>Respuesta del filtro</extra>"
            ),
        )
    )

    # Referencia de ganancia unitaria.
    figure.add_hline(
        y=0,
        line_dash="dash",
        line_color="rgba(220,227,245,0.5)",
        line_width=1,
        annotation_text="0 dB · Ganancia unitaria",
        annotation_position="top left",
        annotation_font={
            "color": COLORS["muted"],
            "size": 10,
        },
    )

    _layout(
        figure,
        title,
        "Frecuencia (Hz)",
        "Magnitud (dB)",
        height=390,
    )

    figure.update_xaxes(rangemode="tozero")
    figure.update_yaxes(range=[-100, 5])

    return figure
