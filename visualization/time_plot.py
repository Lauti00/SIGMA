import matplotlib.pyplot as plt
import numpy as np


def _minmax_envelope(audio, sample_rate, max_points=100_000):
    """Reduce muestras para el gráfico conservando mínimos y máximos locales."""
    bucket_size = max(1, int(np.ceil(len(audio) / max_points)))
    usable_count = (len(audio) // bucket_size) * bucket_size
    blocks = np.asarray(audio[:usable_count]).reshape(-1, bucket_size)
    lows = blocks.min(axis=1)
    highs = blocks.max(axis=1)
    times = (np.arange(len(lows)) * bucket_size + bucket_size / 2) / sample_rate
    return times, lows, highs


def plot_waveform(audio, sample_rate):
    """Crea un gráfico de amplitud en función del tiempo."""
    figure, axis = plt.subplots(figsize=(10, 3.5))
    if len(audio) > 100_000:
        times, lows, highs = _minmax_envelope(audio, sample_rate)
        axis.fill_between(times, lows, highs, linewidth=0, alpha=0.65)
        axis.set_title("Señal temporal (envolvente min/máx para vista rápida)")
    else:
        times = np.arange(len(audio)) / sample_rate
        axis.plot(times, audio, linewidth=0.7)
        axis.set_title("Señal de audio en el dominio temporal")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Amplitud normalizada")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure
