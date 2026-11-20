import unittest

import numpy as np

from core.noise import add_noise_at_snr


class NoiseGenerationTests(unittest.TestCase):
    def setUp(self):
        sample_rate = 8000
        times = np.arange(sample_rate, dtype=np.float64) / sample_rate
        self.clean = 0.5 * np.sin(2 * np.pi * 440 * times)
        self.sample_rate = sample_rate

    def test_noise_matches_requested_snr_without_clipping(self):
        for noise_type in ("Ruido blanco", "Tono sinusoidal"):
            with self.subTest(noise_type=noise_type):
                clean, mixture, noise = add_noise_at_snr(
                    self.clean,
                    self.sample_rate,
                    10,
                    noise_type=noise_type,
                    tone_frequency=50,
                    seed=12,
                )
                measured_snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
                self.assertAlmostEqual(measured_snr, 10, places=10)
                self.assertTrue(np.allclose(mixture, clean + noise))
                self.assertLessEqual(np.max(np.abs(mixture)), 0.9900001)

    def test_white_noise_is_reproducible_for_a_fixed_seed(self):
        first = add_noise_at_snr(self.clean, self.sample_rate, 8, seed=99)
        second = add_noise_at_snr(self.clean, self.sample_rate, 8, seed=99)
        for first_signal, second_signal in zip(first, second):
            self.assertTrue(np.array_equal(first_signal, second_signal))

    def test_rejects_silent_reference_and_invalid_tone(self):
        with self.assertRaisesRegex(ValueError, "silenciosa"):
            add_noise_at_snr(np.zeros(32), self.sample_rate, 10)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            add_noise_at_snr(self.clean, self.sample_rate, 10, "Tono sinusoidal", 4000)


if __name__ == "__main__":
    unittest.main()
