import unittest

import numpy as np

from core.fourier import compute_spectrum, find_dominant_frequencies


class FourierTests(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 8000
        times = np.arange(self.sample_rate) / self.sample_rate
        self.signal = 0.8 * np.sin(2 * np.pi * 440 * times)

    def test_aligned_tone_frequency_and_amplitude(self):
        frequencies, amplitude = compute_spectrum(self.signal, self.sample_rate)
        peak = int(np.argmax(amplitude))
        self.assertAlmostEqual(frequencies[peak], 440.0)
        self.assertAlmostEqual(amplitude[peak], 0.8, places=5)

    def test_windows_preserve_tone_amplitude_and_peak(self):
        for window in ("Hann", "Hamming"):
            frequencies, amplitude = compute_spectrum(self.signal, self.sample_rate, window)
            peak = int(np.argmax(amplitude))
            self.assertAlmostEqual(frequencies[peak], 440.0)
            self.assertAlmostEqual(amplitude[peak], 0.8, delta=0.01)

    def test_hann_and_hamming_reduce_off_bin_leakage(self):
        times = np.arange(self.sample_rate) / self.sample_rate
        off_bin_tone = np.sin(2 * np.pi * 440.5 * times)
        leakage = {}
        for window in ("Rectangular", "Hann", "Hamming"):
            _, amplitude = compute_spectrum(off_bin_tone, self.sample_rate, window)
            peak = int(np.argmax(amplitude))
            outside_main_lobe = np.ones(amplitude.size, dtype=bool)
            outside_main_lobe[max(0, peak - 3) : peak + 4] = False
            leakage[window] = amplitude[outside_main_lobe].sum()

        self.assertLess(leakage["Hann"], leakage["Rectangular"] * 0.1)
        self.assertLess(leakage["Hamming"], leakage["Rectangular"] * 0.3)

    def test_dominant_frequency_finder(self):
        frequencies, amplitude = compute_spectrum(self.signal, self.sample_rate)
        peaks = find_dominant_frequencies(frequencies, amplitude, count=3)
        self.assertTrue(peaks)
        self.assertAlmostEqual(peaks[0]["frequency"], 440.0)

    def test_invalid_signal_and_window_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_spectrum([], self.sample_rate)
        with self.assertRaises(ValueError):
            compute_spectrum(self.signal, self.sample_rate, "Blackman")


if __name__ == "__main__":
    unittest.main()
