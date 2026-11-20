import numpy as np
from scipy.signal import freqz, sosfreqz


def fir_response(coefficients, sample_rate, points=2048):
    frequencies, response = freqz(coefficients, worN=points, fs=sample_rate)
    return frequencies, 20 * np.log10(np.maximum(np.abs(response), 1e-8))


def iir_response(sos, sample_rate, points=2048):
    frequencies, response = sosfreqz(sos, worN=points, fs=sample_rate)
    return frequencies, 20 * np.log10(np.maximum(np.abs(response), 1e-8))
