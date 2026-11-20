import matplotlib.pyplot as plt


def plot_spectrum(frequencies, amplitude, title="Espectro de amplitud (FFT)"):
    """Grafica amplitud frente a frecuencia, en Hz."""
    figure, axis = plt.subplots(figsize=(10, 3.5))
    axis.plot(frequencies, amplitude, linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel("Frecuencia (Hz)")
    axis.set_ylabel("Amplitud")
    axis.set_xlim(left=0)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure
