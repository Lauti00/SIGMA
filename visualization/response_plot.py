import matplotlib.pyplot as plt


def plot_filter_response(frequencies, magnitude_db, title):
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(frequencies, magnitude_db, linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("Frecuencia (Hz)")
    axis.set_ylabel("Magnitud (dB)")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=-100, top=5)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure


def plot_signal_comparison(original, processed, sample_rate):
    import numpy as np

    figure, axis = plt.subplots(figsize=(10, 3.5))
    if len(original) > 100_000:
        from visualization.time_plot import _minmax_envelope

        times, original_min, original_max = _minmax_envelope(original, sample_rate)
        _, processed_min, processed_max = _minmax_envelope(processed, sample_rate)
        axis.fill_between(times, original_min, original_max, label="Original", alpha=0.35)
        axis.fill_between(times, processed_min, processed_max, label="Procesada", alpha=0.35)
    else:
        times = np.arange(len(original)) / sample_rate
        axis.plot(times, original, label="Original", linewidth=0.7, alpha=0.7)
        axis.plot(times, processed, label="Procesada", linewidth=0.7, alpha=0.8)
    axis.set_title("Comparación temporal")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Amplitud normalizada")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure
