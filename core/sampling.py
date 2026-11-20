import numpy as np


def time_axis(sample_count, sample_rate):
    """Genera el instante, en segundos, correspondiente a cada muestra."""
    return np.arange(sample_count) / sample_rate
