import math

import numpy as np
import plotly.graph_objects as go


COLORS = {
    "original": "#4F6BED",
    "processed": "#F08A24",
    "response": "#18A999",
    "peaks": "#D946EF",
}


def _layout(figure, title, x_title, y_title, height=410):
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        height=height,
        margin={"l": 42, "r": 24, "t": 64, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hovermode="x unified",
        uirevision=title,
    )
    figure.update_xaxes(title_text=x_title, showgrid=False, zeroline=False)
    figure.update_yaxes(title_text=y_title, showgrid=True, gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return figure


def _minmax_decimate(signal, sample_rate, max_points=80_000):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size <= max_points:
        times = np.arange(signal.size) / sample_rate
        return times, signal, signal, False

    bucket_size = math.ceil(signal.size / max_points)
    usable = signal.size // bucket_size * bucket_size
    blocks = signal[:usable].reshape(-1, bucket_size)
    times = (np.arange(len(blocks)) * bucket_size + bucket_size / 2) / sample_rate
    return times, blocks.min(axis=1), blocks.max(axis=1), True


def _add_time_signal(figure, signal, sample_rate, label, color, show_legend=True):
    times, lows, highs, reduced = _minmax_decimate(signal, sample_rate)
    if not reduced:
        figure.add_trace(
            go.Scattergl(x=times, y=highs, mode="lines", name=label, line={"color": color, "width": 1.3})
        )
        return

    group = label.lower()
    figure.add_trace(
        go.Scattergl(
            x=times,
            y=highs,
            mode="lines",
            name=label,
            legendgroup=group,
            showlegend=show_legend,
            line={"color": color, "width": 0.8},
            hovertemplate="%{x:.3f} s<br>máx %{y:.4f}<extra>" + label + "</extra>",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=times,
            y=lows,
            mode="lines",
            name=label + " mín.",
            legendgroup=group,
            showlegend=False,
            line={"color": color, "width": 0.5},
            fill="tonexty",
            fillcolor="rgba(79,107,237,0.12)" if label == "Original" else "rgba(240,138,36,0.12)",
            hovertemplate="%{x:.3f} s<br>mín %{y:.4f}<extra>" + label + "</extra>",
        )
    )


def time_comparison_figure(original, sample_rate, processed=None):
    figure = go.Figure()
    _add_time_signal(figure, original, sample_rate, "Original", COLORS["original"])
    if processed is not None:
        _add_time_signal(figure, processed, sample_rate, "Procesada", COLORS["processed"])
    title = "Señal en el tiempo" if processed is None else "Comparación temporal · original y procesada"
    _layout(figure, title, "Tiempo (s)", "Amplitud normalizada")
    figure.update_xaxes(rangeslider={"visible": True, "thickness": 0.08})
    return figure


def _peak_preserving_decimate(frequencies, amplitude, max_points=40_000):
    if len(frequencies) <= max_points:
        return frequencies, amplitude
    bucket_size = math.ceil(len(frequencies) / max_points)
    output_frequencies = []
    output_amplitude = []
    for start in range(0, len(frequencies), bucket_size):
        end = min(start + bucket_size, len(frequencies))
        index = start + int(np.argmax(amplitude[start:end]))
        output_frequencies.append(frequencies[index])
        output_amplitude.append(amplitude[index])
    return np.asarray(output_frequencies), np.asarray(output_amplitude)


def spectrum_figure(frequencies, amplitude, peaks=None, processed=None):
    frequencies, amplitude = _peak_preserving_decimate(frequencies, amplitude)
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=frequencies,
            y=amplitude,
            mode="lines",
            name="Original",
            line={"color": COLORS["original"], "width": 1.3},
        )
    )
    if processed is not None:
        processed_f, processed_a = _peak_preserving_decimate(*processed)
        figure.add_trace(
            go.Scattergl(
                x=processed_f,
                y=processed_a,
                mode="lines",
                name="Procesada",
                line={"color": COLORS["processed"], "width": 1.2},
            )
        )
    if peaks:
        figure.add_trace(
            go.Scattergl(
                x=[peak["frequency"] for peak in peaks],
                y=[peak["amplitude"] for peak in peaks],
                mode="markers+text",
                name="Picos principales",
                text=[f"{peak['frequency']:.0f} Hz" for peak in peaks],
                textposition="top center",
                marker={"color": COLORS["peaks"], "size": 8, "symbol": "circle-open"},
                hovertemplate="%{x:.2f} Hz<br>Amplitud %{y:.4f}<extra>Pico</extra>",
            )
        )
    title = "Espectro de amplitud · FFT"
    if processed is not None:
        title = "Comparación espectral · original y procesada"
    _layout(figure, title, "Frecuencia (Hz)", "Amplitud")
    figure.update_xaxes(rangemode="tozero")
    return figure


def filter_response_figure(frequencies, magnitude_db, title):
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=frequencies,
            y=magnitude_db,
            mode="lines",
            name="Respuesta del filtro",
            line={"color": COLORS["response"], "width": 2},
            hovertemplate="%{x:.1f} Hz<br>%{y:.2f} dB<extra></extra>",
        )
    )
    _layout(figure, title, "Frecuencia (Hz)", "Magnitud (dB)", height=330)
    figure.update_xaxes(rangemode="tozero")
    figure.update_yaxes(range=[-100, 5])
    return figure
