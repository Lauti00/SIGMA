import unittest
from pathlib import Path

import numpy as np

from core.audio import read_wav_file
from core.fir import apply_fir, design_fir
from core.fourier import compute_spectrum, find_dominant_frequencies
from core.iir import apply_iir, design_iir


class DemoAudioPipelineTests(unittest.TestCase):
    def test_demo_file_exercises_analysis_and_both_filter_families(self):
        sample_path = Path(__file__).parents[1] / "samples" / "audio_examples" / "demo_tonos_440_2500Hz.wav"
        raw = sample_path.read_bytes()

        class Uploaded:
            def getvalue(self):
                return raw

        sample_rate, original, channels = read_wav_file(Uploaded())
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(channels, 1)
        frequencies, amplitude = compute_spectrum(original, sample_rate, "Hann")
        peaks = find_dominant_frequencies(frequencies, amplitude, count=2)
        peak_frequencies = [peak["frequency"] for peak in peaks]
        self.assertTrue(any(abs(frequency - 440) < 2 for frequency in peak_frequencies))
        self.assertTrue(any(abs(frequency - 2500) < 2 for frequency in peak_frequencies))

        for family in ("FIR", "IIR"):
            with self.subTest(family=family):
                if family == "FIR":
                    coefficients = design_fir("Pasabajo", sample_rate, 128, [1000])
                    processed = apply_fir(original, coefficients)
                else:
                    coefficients = design_iir("Pasabajo", sample_rate, 4, [1000])
                    processed = apply_iir(original, coefficients)
                out_frequencies, out_amplitude = compute_spectrum(processed, sample_rate)
                low_bin = int(np.argmin(abs(out_frequencies - 440)))
                high_bin = int(np.argmin(abs(out_frequencies - 2500)))
                self.assertGreater(out_amplitude[low_bin], 0.2)
                self.assertLess(out_amplitude[high_bin], 0.02)
                self.assertTrue(np.all(np.isfinite(processed)))


if __name__ == "__main__":
    unittest.main()
