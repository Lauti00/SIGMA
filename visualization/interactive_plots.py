
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    row=None,
):
    """Agrega una señal temporal al gráfico."""

    def add_trace(trace):
        if row is None:
            figure.add_trace(trace)
        else:
            figure.add_trace(trace, row=row, col=1)

    times, lows, highs, reduced = _minmax_decimate(
        signal, sample_rate
    )

    group = label.lower()

    if not reduced:
        add_trace(
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

    add_trace(
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

    add_trace(
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
    separate=False,
):
    """Gráfico temporal superpuesto o con señales en paneles separados."""
    signals = [
        ("Original", original, COLORS["original"]),
    ]

    if processed is not None:
        signals.append(
            ("Procesada", processed, COLORS["processed"])
        )

    if reference is not None:
        signals.append(
            ("Referencia limpia", reference, COLORS["response"])
        )

    separate = separate and len(signals) > 1
    if separate:
        figure = make_subplots(
            rows=len(signals),
            cols=1,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[label for label, _, _ in signals],
        )
        for row in range(2, len(signals) + 1):
            figure.update_yaxes(matches="y", row=row, col=1)
        for row, (label, signal, color) in enumerate(signals, start=1):
            _add_time_signal(
                figure,
                signal,
                sample_rate,
                label,
                color,
                show_legend=False,
                row=row,
            )
        figure.update_annotations(
            font={"color": COLORS["text"], "size": 13}
        )
        height = 280 * len(signals) + 140
    else:
        figure = go.Figure()
        for label, signal, color in signals:
            _add_time_signal(
                figure,
                signal,
                sample_rate,
                label,
                color,
            )
        height = 440

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
        height=height,
    )

    range_slider = {
        "visible": True,
        "thickness": 0.065,
        "bgcolor": "rgba(120,140,180,0.06)",
        "bordercolor": "rgba(160,175,210,0.15)",
        "borderwidth": 1,
    }
    if separate:
        figure.update_xaxes(rangeslider={"visible": False})
        figure.update_xaxes(
            rangeslider=range_slider,
            row=len(signals),
            col=1,
        )
    else:
        figure.update_xaxes(rangeslider=range_slider)

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
    separate=False,
):
    """Espectro FFT superpuesto o comparado en paneles separados."""
    spectra = [
        ("Original", (frequencies, amplitude), COLORS["original"], "solid"),
    ]

    if processed is not None:
        spectra.append(
            ("Procesada", processed, COLORS["processed"], "solid")
        )

    if reference is not None:
        spectra.append(
            (
                "Referencia limpia",
                reference,
                COLORS["response"],
                "dash",
            )
        )

    separate = separate and len(spectra) > 1
    if separate:
        figure = make_subplots(
            rows=len(spectra),
            cols=1,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.08,
            subplot_titles=[label for label, _, _, _ in spectra],
        )
        for row in range(2, len(spectra) + 1):
            figure.update_yaxes(matches="y", row=row, col=1)
        figure.update_annotations(
            font={"color": COLORS["text"], "size": 13}
        )
        height = 280 * len(spectra) + 140
    else:
        figure = go.Figure()
        height = 440

    def add_trace(trace, row):
        if separate:
            figure.add_trace(trace, row=row, col=1)
        else:
            figure.add_trace(trace)

    for row, (label, spectrum, color, dash) in enumerate(spectra, start=1):
        spectrum_f, spectrum_a = _peak_preserving_decimate(*spectrum)
        add_trace(
            go.Scattergl(
                x=spectrum_f,
                y=spectrum_a,
                mode="lines",
                name=label,
                showlegend=not separate,
                line={
                    "color": color,
                    "width": 1.7,
                    "dash": dash,
                },
                hovertemplate=(
                    "Frecuencia: %{x:.2f} Hz<br>"
                    "Amplitud: %{y:.6f}"
                    f"<extra>{label}</extra>"
                ),
            ),
            row,
        )

    if peaks:
        add_trace(
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
            ),
            1,
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
        height=height,
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
